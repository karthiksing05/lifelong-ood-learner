"""Composing an out-of-distribution concept from discovered prototypes (poe-diffusion Sections 3.1-3.3).

Given a single OOD query x_q, build an analytic product-of-experts teacher q_T = N(mu_T, Sigma_T):

  1. DISCOVER a candidate pool of Gaussian prototypes.  Two routes, deliberately parallel:
       candidates_ascent  -- the paper's route: gradient ascent on the score of p_t over a grid of
                             noise levels, Laplace covariance at each mode (their Eq. 4 + Appendix D).
       candidates_tree    -- this repository's route: the tree's nodes ARE the modes, so discovery is
                             a closed-form log-likelihood ranking over concepts at every abstraction
                             level.  E2 compares the two; if they tie, discovery was free all along.
  2. SELECT K complementary prototypes by greedy maximization of the per-dimension coverage objective
     F(S) = sum_r max_{j in S} l_{j,r}(x_q), monotone submodular (their Eq. 8-9).
  3. WEIGH per dimension with a temperature-tau softmax of the per-dimension log-likelihoods
     (their Eq. 10) and take the weighted diagonal product of experts (their Eq. 7).

The teacher's forward-noised score is analytic (their Eq. 11-12) and slots into classifier-free
guidance with the variance-aware schedule (their Eq. 13-14) -- see `teacher_eps` and `cfg_weight`,
consumed by ScoreField.sample(guide_eps=..., guide_w=...).
"""
import numpy as np

from .score import cosine_abar


# ── step 1: discovery ─────────────────────────────────────────────────────────
def candidates_tree(snap, xq, top=None, per_depth_cap=None):
    """Closed-form discovery: the candidate pool is the tree's OWN concepts, at every abstraction
    level.  By default ALL of them -- ranking the pool by total query log-likelihood would let the
    majority pixels (the background) crowd out the experts that explain the minority pixels (the
    digit), and finding complementary experts is the submodular selection's job, not the pool's.
    `top`/`per_depth_cap` exist only to bound cost on very large trees.  -> (M, V, meta)."""
    xq = np.ravel(xq)
    M, V, depth = snap["M"], snap["V"], snap["depth"]
    ll = -0.5 * (((xq - M) ** 2) / V + np.log(2 * np.pi * V)).sum(1)
    if top is None and per_depth_cap is None:
        keep = np.arange(len(M))
    else:
        keep = []
        for d in np.unique(depth):
            idx = np.where(depth == d)[0]
            keep.extend(idx[np.argsort(-ll[idx])[:(per_depth_cap or len(idx))]])
        keep = np.array(sorted(keep, key=lambda i: -ll[i])[:(top or len(keep))])
    return M[keep], V[keep], dict(node=keep, depth=depth[keep], loglik=ll[keep])


def candidates_ascent(field, xq, u_grid=(0.15, 0.3, 0.45, 0.6), n_starts=8, iters=80,
                      lr=0.05, seed=0, probes=4, dedup_tol=1.0):
    """The paper's discovery: mode ascent on p_t at several noise levels, Laplace covariance at each
    mode, pulled back to x0-space, deduplicated by L2 distance.  -> (M [P,D], V [P,D], meta)."""
    Ms, Vs, us = [], [], []
    for u in u_grid:
        xt, x0 = field.mode_ascent(xq, u, n_starts=n_starts, iters=iters, lr=lr, seed=seed)
        v0 = field.laplace_diag(xt, u, probes=probes, seed=seed)
        Ms.append(x0); Vs.append(v0); us.extend([u] * len(x0))
    M = np.concatenate(Ms); V = np.concatenate(Vs); us = np.array(us)
    keep = []
    for i in range(len(M)):
        if all(np.sum((M[i] - M[j]) ** 2) > dedup_tol for j in keep):
            keep.append(i)
    keep = np.array(keep)
    return M[keep], V[keep], dict(u=us[keep])


# ── steps 2-3: submodular selection + weighted PoE ────────────────────────────
def loglik_perdim(xq, M, V):
    """l_{j,r}(x_q): contribution of prototype j to the query log-likelihood at dimension r. [P, D]"""
    xq = np.ravel(xq)
    return -0.5 * (((xq - M) ** 2) / V + np.log(2 * np.pi * V))


def submodular_select(L, K=3):
    """Greedy max of F(S) = sum_r max_{j in S} L[j, r].  -> selected indices, in pick order."""
    P = len(L)
    K = min(K, P)
    best = np.full(L.shape[1], -np.inf)
    sel = []
    for _ in range(K):
        gains = np.maximum(L, best).sum(1)
        gains[sel] = -np.inf
        j = int(np.argmax(gains))
        sel.append(j)
        best = np.maximum(best, L[j])
    return np.array(sel)


def poe_teacher(xq, M, V, K=3, tau=0.5):
    """Select + weigh + multiply.  -> dict(mu, var, sel, w) with mu, var [D]."""
    L = loglik_perdim(xq, M, V)
    sel = submodular_select(L, K)
    Ls = L[sel] / tau
    Ls -= Ls.max(0, keepdims=True)
    w = np.exp(Ls)
    w /= w.sum(0, keepdims=True)                     # their Eq. 10, per-dimension
    prec = (w / V[sel]).sum(0)                       # their Eq. 7
    mu = (w * M[sel] / V[sel]).sum(0) / prec
    return dict(mu=mu, var=1.0 / prec, sel=sel, w=w)


# ── the analytic teacher inside classifier-free guidance ─────────────────────
def teacher_eps(teacher):
    """-> guide_eps(x, u): the teacher's noise prediction, their Eq. 12."""
    mu, var = teacher["mu"], teacher["var"]

    def _eps(x, u):
        a = float(cosine_abar(u))
        s = a * var + (1.0 - a)
        return np.sqrt(1.0 - a) * (np.atleast_2d(x) - np.sqrt(a) * mu) / s
    return _eps


def cfg_weight(teacher, w0=1.2, wmax=2.0, wmin=0.8):
    """-> guide_w(u): variance-aware guidance schedule, their Eq. 14, with a floor.

    The floor is an addition to the paper's schedule that our analytic setting needs: their
    unconditional branch is a UNet whose epsilon has O(1) scale, while ours is a mixture score whose
    precision 1/(abar V + (1-abar)) grows large at small noise -- with concept variances ~prior_var
    the prior pulls ~1/prior_var times harder per unit distance than the teacher near u=0, and an
    unfloored schedule lets late denoising steps drag the sample back onto the nearest SEEN slot,
    exactly the failure the guidance exists to prevent."""
    var = teacher["var"]

    def _w(u):
        a = float(cosine_abar(u))
        return float(np.clip(w0 * float(np.mean(a * var + (1.0 - a))), wmin, wmax))
    return _w


def query_only_teacher(xq, var=0.01):
    """The paper's query-only baseline: a single isotropic Gaussian around the query."""
    xq = np.ravel(xq)
    return dict(mu=xq.copy(), var=np.full_like(xq, var), sel=np.array([]), w=None)


def topk_teacher(snap, xq, k=1, tau=1.0):
    """Nearest trained-prototype baseline: PoE of the k best-scoring EFFECTIVE LEAVES (no hierarchy,
    no submodular complementarity) with uniform per-dimension weights softmaxed over total loglik."""
    leaves = np.where(snap["is_leaf"])[0]
    M, V = snap["M"][leaves], snap["V"][leaves]
    ll = -0.5 * (((np.ravel(xq) - M) ** 2) / V + np.log(2 * np.pi * V)).sum(1)
    top = np.argsort(-ll)[:k]
    w = np.exp((ll[top] - ll[top].max()) / max(tau, 1e-6) / V.shape[1])
    w /= w.sum()
    prec = (w[:, None] / V[top]).sum(0)
    mu = (w[:, None] * M[top] / V[top]).sum(0) / prec
    return dict(mu=mu, var=1.0 / prec, sel=leaves[top], w=w)
