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


# The companion paper's Table 6 columns, verbatim, for callers that want their budget.
PAPER_COLORMNIST = dict(u_grid=tuple(round(t / 1000, 4) for t in range(50, 401, 25)),
                        n_starts=128, iters=150, lr=0.1, probes=4, init="hybrid")
# `adam=False` is a DEVIATION from their Table 6 and applies to CelebA only: Adam leaves
# 11.1% of each mode outside [-1,1] here, which clips to flat saturated colour, while on
# Color-MNIST it leaves 0.00%-0.32% and the paper's optimizer is kept.
PAPER_CELEBA = dict(u_grid=tuple(round(t / 1000, 3) for t in range(50, 501, 50)),
                    n_starts=8, iters=100, lr=0.09, probes=4, init="query", adam=False, clamp=True, tweedie=True)

# The paper's Table 8 guidance columns.  These live here, next to Table 6, because they were
# previously copy-pasted into six experiment files as a single constant named `PAPER_POE` holding
# w0=1.2 -- which is the paper's COLOR-MNIST value.  Table 8 gives CelebA **w0 = 0.5**, so every
# CelebA run in this repository (n20, n44, diag_celeba_chain) guided at 2.4x the published strength
# under a constant that claimed to be the paper's.  Over-guidance saturates and adds artifacts, and
# it was applied to `poe-guided` -- the paper's own method -- which is a large part of why that arm
# looked broken on CelebA relative to the published figures.  One definition per dataset, so the
# next caller cannot pick the wrong one by accident.
PAPER_POE_COLORMNIST = dict(schedule="variance", var_scale=1.0, w0=1.2, wmax=2.0, wmin=0.0)
PAPER_POE_CELEBA = dict(schedule="variance", var_scale=1.0, w0=0.5, wmax=2.0, wmin=0.0)


def candidates_ascent(field, xq, u_grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), n_starts=16, iters=150,
                      lr=0.1, seed=0, probes=4, dedup_rmse=0.05, init="hybrid",
                      adam=True, clamp=False, tweedie=False):
    """The paper's discovery route: mode ascent on p_t at several noise levels (their Eq. 4), a
    Laplace covariance at each mode from Hutchinson probes, pulled back to x0-space, deduplicated.

    THE DEFAULTS BELOW ARE NOT THE PAPER'S, and this docstring used to claim they were.  Their
    Table 6 ColorMNIST column is a grid of [50, 400, step 25] -- FIFTEEN noise levels -- with **128
    ascent starts per timestep**.  These defaults use six levels and sixteen starts: 96 chains
    against their 1920, a **20x smaller discovery budget**.  That is visible in the case-study
    figures, where the three selected prototypes are often unrelated to the query: submodular
    selection can only choose from the pool ascent found, and a thin pool gives a bad choice.
    Pass `PAPER_COLORMNIST` (below) to run their budget.

    What IS carried over from their column: 150 ascent iterations, base step 0.1, 4 Hutchinson
    probes, and -- the detail that matters most -- HYBRID INITIALIZATION, half the chains started
    from the noised query and half from pure noise.  A pool grown only from
    the query explores one basin, and composition needs experts that DISAGREE with the query
    (the right background lives in a different basin from the right foreground).

    `dedup_rmse` is a per-dimension RMSE, not a raw squared L2: over D=3072 dimensions two visibly
    different images sit ~190 apart in squared L2, so the old raw threshold of 1.0 deduplicated
    nothing and the pool was mostly copies of one mode.
    """
    Ms, Vs, us = [], [], []
    half = max(1, n_starts // 2)
    for u in u_grid:
        parts = ([("query", n_starts - half), ("random", half)] if init == "hybrid"
                 else [(init, n_starts)])
        for k, (mode, ns) in enumerate(parts):
            if ns <= 0:
                continue
            xt, x0 = field.mode_ascent(xq, u, n_starts=ns, iters=iters, lr=lr, adam=adam,
                                       clamp=clamp, tweedie=tweedie,
                                       seed=seed + 1000 * k, init=mode)
            Ms.append(x0)
            Vs.append(field.laplace_diag(xt, u, probes=probes, seed=seed))
            us.extend([u] * len(x0))
    M = np.concatenate(Ms); V = np.concatenate(Vs); us = np.array(us)
    tol2 = (dedup_rmse ** 2) * M.shape[1]
    keep = []
    for i in range(len(M)):
        if all(np.sum((M[i] - M[j]) ** 2) > tol2 for j in keep):
            keep.append(i)
    keep = np.array(keep)
    return M[keep], V[keep], dict(u=us[keep], n_raw=len(M))


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


def poe_teacher(xq, M, V, K=3, tau=0.05, adaptive_tau=True, extent="mixture"):
    """Select + weigh + multiply.  -> dict(mu, var, sel, w) with mu, var [D].

    ADAPTIVE TEMPERATURE is a documented deviation from the paper's Eq. 10, and it is the single
    change that makes composition work here.  The paper fixes tau=0.5 against experts whose Laplace
    covariances are estimated per-mode from a UNet and therefore vary widely.  Cobweb's variances
    instead sit near `prior_var` almost everywhere, which compresses every per-dimension
    log-likelihood difference by a common factor: two experts differing by a whole background colour
    are separated by only ~0.2 nats per pixel, so a tau=0.5 softmax averages them, and a background
    supported by ONE correct expert loses to two incorrect ones.  Measured on the 40 held-out
    ColorMNIST slots, background accuracy of mu_T runs 0.35 at the paper's tau and 0.97 sharpened.

    So the temperature is set from the data's own scale, exactly as cobweb-poincare sets its profile
    temperature:  T = tau * mean_r std_j l_{j,r}.  That makes the weighting scale-free -- it
    transfers across datasets and prior_var settings without retuning -- and it restores the
    consistency the fixed value breaks: selection maximizes a per-dimension MAX (Eq. 8), so the
    composition should approach a per-dimension argmax rather than a near-uniform blend.
    """
    L = loglik_perdim(xq, M, V)
    sel = submodular_select(L, K)
    Ls = L[sel]
    T = tau * max(float(Ls.std(0).mean()), 1e-6) if adaptive_tau else tau
    Z = Ls / T
    Z -= Z.max(0, keepdims=True)
    w = np.exp(Z)
    w /= w.sum(0, keepdims=True)                     # their Eq. 10, per-dimension
    prec = (w / V[sel]).sum(0)                       # their Eq. 7
    mu = (w * M[sel] / V[sel]).sum(0) / prec
    var_poe = 1.0 / prec

    # EXTENT vs PRECISION -- the distinction that decides whether samples are diverse.
    #
    # Sigma_T from Eq. 7 is a PRODUCT of experts, so it is deliberately sharper than any factor: it
    # answers "how precisely do I know WHERE the concept sits".  Sampling treats it as if it answered
    # "how much does the concept VARY", and those are different quantities -- the variance of an
    # estimator versus the variance of a population.  Measured on ColorMNIST: the product covariance
    # gives sigma 0.019 while the real class varies by 0.120, and 99% of its coordinates are pinned
    # at the Laplace floor, so as a description of extent it is a point mass.  Sampling it therefore
    # returns near-copies of mu_T, which is exactly what the case studies showed.
    #
    # The extent of a concept assembled from K prototypes is the variance of the MIXTURE over them,
    # by the law of total variance:  Var = sum_j w_j [ Sigma_j + (mu_j - mu_T)^2 ]  -- within-expert
    # spread plus between-expert spread, with the same per-dimension weights.  The between term is
    # the one that was missing, and it is the larger of the two by an order of magnitude.
    #
    # `mu` is unchanged: the product is the right operator for locating the concept, and mu_T already
    # probe-reads the correct composition. Only the covariance used for sampling changes.
    between = ((M[sel] - mu[None]) ** 2 * w).sum(0)
    var_mix = (w * V[sel]).sum(0) + between
    var = var_mix if extent == "mixture" else var_poe
    return dict(mu=mu, var=var, var_poe=var_poe, var_mixture=var_mix, between=between,
                sel=sel, w=w, temperature=T)


# ── the analytic teacher inside classifier-free guidance ─────────────────────
def teacher_eps(teacher):
    """-> guide_eps(x, u): the teacher's noise prediction, their Eq. 12."""
    mu, var = teacher["mu"], teacher["var"]

    def _eps(x, u):
        a = float(cosine_abar(u))
        s = a * var + (1.0 - a)
        return np.sqrt(1.0 - a) * (np.atleast_2d(x) - np.sqrt(a) * mu) / s
    return _eps


def stacked_teacher_eps(MU, VAR):
    """-> guide_eps(x, u) where ROW i of x is guided by teacher i.

    Eq. 12 is elementwise in the batch dimension, so K different teachers can drive K blocks of one
    sampling run with no change to the sampler: build MU/VAR with one row per sample and the whole
    batch denoises together.  This is what makes batched consolidation cheap -- the K pools come out
    of a single reverse-diffusion pass instead of K sequential ones.
    """
    MU = np.atleast_2d(MU)
    VAR = np.atleast_2d(VAR)

    def _eps(x, u):
        a = float(cosine_abar(u))
        s = a * VAR + (1.0 - a)
        return np.sqrt(1.0 - a) * (np.atleast_2d(x) - np.sqrt(a) * MU) / s
    return _eps


def stacked_cfg_weight(VAR, w0=1.2, wmax=2.0, wmin=0.0, schedule="variance"):
    """-> guide_w(u) returning a COLUMN of weights, one per sample row.

    Eq. 14 makes the guidance weight a function of the teacher's own covariance, so in a batched run
    where each block of rows has a different teacher the weight differs per row.  Collapsing that to
    one scalar (e.g. the first teacher's) would silently apply one concept's geometry to all of them
    -- and since the whole point of Eq. 14 is that a sharper teacher hands off earlier, that
    collapse would destroy exactly the property being tested.  The returned array is [N, 1], which
    broadcasts against the [N, D] epsilon in the sampler.
    """
    VAR = np.atleast_2d(VAR)

    def _w(u):
        u = float(np.clip(u, 0.0, 1.0))
        if schedule == "const":
            return float(wmax)
        if schedule == "anneal":
            return float(wmin + (wmax - wmin) * (1.0 - u))
        a = float(cosine_abar(u))
        return np.clip(w0 * (a * VAR + (1.0 - a)).mean(1, keepdims=True), wmin, wmax)
    return _w


def candidates_ascent_batch(field, xqs, u_grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), n_starts=16,
                            iters=150, lr=0.1, seed=0, probes=4, dedup_rmse=0.05, init="hybrid"):
    """Discovery for K queries at once.  -> list of (M, V, meta), one per query.

    Two savings over calling `candidates_ascent` K times, and the second is the larger:

      * the query-initialized chains for all K queries ascend in ONE batched call per noise level,
        because mode ascent uses the UNCONDITIONAL score and the chains are independent;
      * the random-initialized half does not depend on the query at all, so it is run ONCE and its
        modes are shared by every query's candidate pool.  Sequential discovery re-derives the same
        query-agnostic modes K times over.
    """
    xqs = [np.ravel(x) for x in xqs]
    K = len(xqs)
    per = [[] for _ in range(K)]
    half = max(1, n_starts // 2) if init == "hybrid" else (n_starts if init == "random" else 0)
    nq = n_starts - half if init != "random" else 0
    for u in u_grid:
        if nq > 0:
            Q = np.stack([xqs[i % K] for i in range(K * nq)])
            xt, x0 = field.mode_ascent(Q, u, n_starts=K * nq, iters=iters, lr=lr, seed=seed,
                                       init="query")
            v0 = field.laplace_diag(xt, u, probes=probes, seed=seed)
            for k in range(K):
                idx = np.arange(k, K * nq, K)
                per[k].append((x0[idx], v0[idx], u))
        if half > 0:
            xt, x0 = field.mode_ascent(None, u, n_starts=half, iters=iters, lr=lr,
                                       seed=seed + 1000, init="random")
            v0 = field.laplace_diag(xt, u, probes=probes, seed=seed)
            for k in range(K):
                per[k].append((x0, v0, u))          # query-agnostic: shared, not recomputed
    out = []
    tol2 = None
    for k in range(K):
        M = np.concatenate([m for m, _, _ in per[k]])
        V = np.concatenate([v for _, v, _ in per[k]])
        us = np.concatenate([np.full(len(m), u) for m, _, u in per[k]])
        tol2 = (dedup_rmse ** 2) * M.shape[1]
        keep = []
        for i in range(len(M)):
            if all(np.sum((M[i] - M[j]) ** 2) > tol2 for j in keep):
                keep.append(i)
        keep = np.array(keep)
        out.append((M[keep], V[keep], dict(u=us[keep], n_raw=len(M))))
    return out


def cfg_weight(teacher, w0=1.2, wmax=2.0, wmin=0.8, schedule="anneal"):
    """-> guide_w(u): how hard the analytic teacher pulls, as a function of noise level.

    Read w as a mixing weight: classifier-free guidance forms
    `(1 - w) * eps_uncond + w * eps_cond`, so w=1 follows the teacher exactly, w<1 lets the
    unconditional manifold contribute detail, and w>1 extrapolates away from it.

    `schedule="variance"` is the paper's Eq. 14, `min(w0 * tr(Sigma_t)/d, wmax)`, whose job is to
    cancel the implicit Sigma_t^-1 in Eq. 12 so that guidance strength is comparable across queries
    regardless of prototype geometry.  A floor `wmin` is added because our unconditional branch is a
    Gaussian mixture, not a UNet: its precision 1/(abar*V + (1-abar)) is large at low noise, so an
    unfloored weight lets the last denoising steps drag the sample onto the nearest SEEN slot.

    `schedule="anneal"` (default) is this repository's schedule and reflects a real difference
    between the two settings.  A convolutional UNet trained on 120 ColorMNIST slots shares weights
    across colours and therefore assigns some density to an unseen (fg, bg) pairing; a Gaussian
    mixture over tree concepts assigns it essentially NONE -- that is what makes the composition
    out-of-distribution for this model class.  So the unconditional field is trustworthy for generic
    image structure at high noise but actively wrong about the composition at low noise, where
    identity is decided.  The schedule therefore anneals from `wmin` at u=1 to `wmax` at u=0,
    handing the late steps to the teacher.  E0's sampler sweep is what selects between the two.

    `schedule="const"` holds w fixed at `wmax`; useful for isolating the mixing weight.
    """
    var = teacher["var"]

    def _w(u):
        u = float(np.clip(u, 0.0, 1.0))
        if schedule == "const":
            return float(wmax)
        if schedule == "anneal":
            return float(wmin + (wmax - wmin) * (1.0 - u))
        a = float(cosine_abar(u))
        return float(np.clip(w0 * float(np.mean(a * var + (1.0 - a))), wmin, wmax))
    return _w


def query_only_teacher(xq, var=0.01):
    """The paper's query-only baseline: a single isotropic Gaussian around the query.

    `fixed_var` marks this teacher as exempt from the PoE covariance correction (see
    `neural.lifelong_neural`).  That correction exists because a Laplace-estimated PoE covariance
    understates class spread; this baseline's variance is not estimated at all, it is DEFINED as "a
    fixed small isotropic spread around the query".  Rescaling it would silently replace the paper's
    baseline with a different, more diffuse one and make the comparison meaningless.
    """
    xq = np.ravel(xq)
    return dict(mu=xq.copy(), var=np.full_like(xq, var), sel=np.array([]), w=None,
                fixed_var=True)


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
