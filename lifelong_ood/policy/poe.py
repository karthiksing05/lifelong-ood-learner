"""Prototype PoE planning: Wang et al.'s prototype discovery and product of experts (arXiv 2605.07078), ported to a flow policy
(plan of 2026-10-01, late night).

EXPERTS over the real action rows [H*7] (normalised, step-major), evaluated at the same noisy chunk on every Euler step:
    policy experts   the frozen policy at the CURRENT observation under a conditioning: the task's instruction (row 0), another
                     prompt, an attention focus on a segment (attn_bias), or a folded token: live, so they carry the scene
    prototype experts  Gaussians N(m, V) in clean action space (nodes of the policy's action-marginal tree), analytic along the
                     flow path x_t = (1 - t) a + t eps:  E[a | x_t] = m + (1-t)V / ((1-t)^2 V + t^2) * (x_t - (1-t) m)
COMPOSITION: a convex mix of the experts' velocities per real dim, beta[k, r] >= 0, sum_k beta[k, r] = 1 (equivalently of their
clean-action estimates: v = (x - a_hat) / t), so the last Euler step returns sum_k beta_k a_hat_k.  The share of the policy family
(the policy experts) at the last steps is what decides how much the policy still generates (a guide at weight 1 replaced it:
ledger p0_audit guide_override), so it is set explicitly (`policy_share`).  The gripper dimension is taken from the policy family
only (blending a binary command makes it chatter).  beta = one-hot on row 0 is the policy bit for bit.

SELECTION (Wang's Eq. 10-12): given a demonstration's frames of one phase as the query, the greedy submodular coverage
F(S) = sum_frames sum_r max_{j in S} log N(q_r; m_jr, s_jr^2) picks K experts, the task's own policy forced in; per-dim weights are
the softmax (temperature tau) of each selected expert's mean per-dim log-likelihood.  The policy's variance is its calibrated
error against demonstrations (not its sampling spread).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch

from . import attn_bias as AB

GRIP = 6                                   # the gripper's action dim (of 7)


def real_dims(H=10, A=7):
    return np.arange(H * A)


def grip_mask(H=10, A=7):
    m = np.zeros(H * A, bool)
    m[GRIP::A] = True
    return m


@dataclass
class ProtoExpert:
    """A Gaussian expert in clean (normalised) action space."""
    m: np.ndarray                      # [70]
    V: np.ndarray                      # [70]
    name: str = ""

    def a_hat(self, x_real, t):
        """E[a | x_t] under N(m, V) on the flow path; x_real [n, 70] torch."""
        m = torch.as_tensor(self.m, dtype=x_real.dtype, device=x_real.device)
        V = torch.as_tensor(self.V, dtype=x_real.dtype, device=x_real.device)
        al = 1.0 - float(t)
        k = al * V / (al * al * V + float(t) ** 2)
        return m + k * (x_real - al * m)


@dataclass
class PolicyRow:
    """One policy expert: a prompt and an optional attention focus {camera: [16,16]} (applied as target mass rho)."""
    prompt: str
    focus: dict | None = None
    rho: float = 0.7
    name: str = ""


@torch.no_grad()
def policy_prefixes(bk, obs, rows):
    return bk.prefix(bk.observations([dict(obs, prompt=r.prompt) for r in rows]))


@contextlib.contextmanager
def _focus(bk, rows, width):
    """Per-row attention focus inside one batched expert pass (rows without a focus get an all-zero map = no bias)."""
    if not any(r.focus for r in rows):
        yield
        return
    maps = [r.focus or {} for r in rows]
    # attn_bias's mass mode biases rows whose map has target tokens; rows with an empty map get no bias (target sum is 0)
    with AB.biased(bk, maps, mode="mass", rho=max(r.rho for r in rows if r.focus), width=width):
        yield


@torch.no_grad()
def sample_poe(bk, pre, rows, protos, beta, seed=0, noise=None, num_steps=None, log=None):
    """Euler on the convex mix of experts.  pre: batched Prefix of the P policy rows (row 0 = the task's own conditioning);
    protos: list of ProtoExpert; beta: callable t -> [P + G, 70] numpy/torch convex weights (or a fixed array).
    -> real rows [1, 70] (normalised).  `log` (list) receives per-step expert clean-action estimates for diagnostics."""
    P, G = int(pre.n), len(protos)
    steps = int(num_steps or bk.num_steps)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = (torch.randn((1, bk.dim_full), generator=g).to(bk.device) if noise is None
         else torch.as_tensor(np.asarray(noise, np.float32), device=bk.device).reshape(1, bk.dim_full))
    real = torch.as_tensor(bk.real, device=bk.device)
    dt = -1.0 / steps
    t = 1.0
    width = int(pre.pad_masks.shape[1])
    while t >= -dt / 2:
        tt = torch.full((P,), t, dtype=torch.float32, device=bk.device)
        with _focus(bk, rows, width):
            V = bk.v_full(x.expand(P, -1), tt, pre)                         # [P, dim_full]
        xr = x[:, real]
        # mix VELOCITIES (affine-equivalent to mixing clean-action estimates, and exact: a one-hot weight on row 0 is the
        # policy's own velocity, so the sample is the policy's bit for bit)
        Vr = [V[k:k + 1, real] for k in range(P)] + [(xr - p.a_hat(xr, t)) / t for p in protos]
        Vr = torch.cat(Vr, 0)                                             # [P+G, 70]
        b = beta(t) if callable(beta) else beta
        b = torch.as_tensor(np.asarray(b, np.float32), device=bk.device)
        v = V[0:1].clone()                                                # padded dims follow the task's own policy row
        v[:, real] = (b * Vr).sum(0, keepdim=True)
        if log is not None:
            log.append(dict(t=t, A=(xr - t * Vr).float().cpu().numpy(), a_mix=(xr - t * v[:, real]).float().cpu().numpy()))
        x = x + dt * v
        t = t + dt
    return x[:, real].float().cpu().numpy()


def make_beta(weights, n_policy, policy_share=0.5, late_t=0.3, early_share=None, exact=False, H=10, A=7):
    """Per-dim convex weights over [P policy rows + G prototypes] from selection weights [P+G, 70] (Wang's per-dim softmax).
    The policy family's share is a FLOOR: on a dim where selection gives it less than `policy_share` (at t <= late_t;
    `early_share` before, default the same) its rows are scaled up to that share and the prototypes down; where prototypes
    explain nothing they keep their (near-zero) selection weight.  `exact=True` sets the share exactly instead (the P0 test).
    The gripper dims come from the policy family only.  -> callable t -> [P+G, 70]."""
    W = np.asarray(weights, np.float64)
    K, D = W.shape
    W = W / np.maximum(W.sum(0, keepdims=True), 1e-12)
    fam = np.zeros(K, bool)
    fam[:n_policy] = True
    gm = grip_mask(H, A)

    def beta(t):
        lam = policy_share if (t <= late_t + 1e-9 or early_share is None) else early_share
        B = W.copy()
        nf = int(fam.sum())
        if (~fam).any() and nf:
            sp = B[fam].sum(0)                                    # [D] the policy family's selected share
            target = np.full(D, lam) if exact else np.maximum(sp, lam)
            tiny_p = sp < 1e-9                                    # no policy weight: split the target equally over policy rows
            Bf = np.where(tiny_p[None], target[None] / nf, B[fam] * (target / np.where(tiny_p, 1.0, sp))[None])
            rest = 1.0 - sp
            tiny_q = rest < 1e-9
            Bq = np.where(tiny_q[None], 0.0, B[~fam] * ((1.0 - target) / np.where(tiny_q, 1.0, rest))[None])
            Bq = np.where((tiny_q & (target < 1.0 - 1e-9))[None], (1.0 - target)[None] / max(1, int((~fam).sum())), Bq)
            B[fam], B[~fam] = Bf, Bq
            Wg = W[fam][:, gm]                                    # gripper: the policy family only
            sg = Wg.sum(0, keepdims=True)
            Bg = np.where(sg < 1e-9, 1.0 / nf, Wg / np.where(sg < 1e-9, 1.0, sg))
            Bp, Bq2 = B[fam], B[~fam]
            Bp[:, gm], Bq2[:, gm] = Bg, 0.0
            B[fam], B[~fam] = Bp, Bq2
        return B.astype(np.float32)
    return beta


def query_nodes(tree, Q, levels=(0.3, 0.4, 0.55), floor=1e-4):
    """Wang's discovery FROM the query: the node each query chunk falls into at each level (max diagonal-Gaussian likelihood
    over the level's modes and extents), as (ProtoExpert, id, count).  Only these prototypes enter the selection pool."""
    out = []
    for u in levels:
        li = int(np.argmin([abs(x - u) for x in tree.u_grid]))
        ids = list(tree.levels[li])
        M = np.stack([np.asarray(tree.nodes[i].mode, np.float64) for i in ids])
        V = np.maximum(np.stack([np.asarray(tree.nodes[i].extent_dim, np.float64) for i in ids]), floor)
        Qd = np.asarray(Q, np.float64)
        ll = -0.5 * (((Qd[:, None, :] - M[None]) ** 2) / V[None] + np.log(V)[None]).sum(-1)       # [n, nodes]
        hit, cnt = np.unique(np.argmax(ll, 1), return_counts=True)
        for h, c in zip(hit, cnt):
            out.append((ProtoExpert(M[h].astype(np.float32), V[h].astype(np.float32), f"node:{u}:{ids[h]}"), f"node:{u}:{ids[h]}", int(c)))
    return out


def proto_product(protos, w):
    """Wang's Eq. 9: the weighted product of diagonal Gaussian prototypes, per dim.  w [G, 70] (each prototype's weight per dim;
    rows need not sum to 1).  -> ProtoExpert(mu_T, Sigma_T): broad prototypes add little precision, so they barely move mu_T."""
    w = np.asarray(w, np.float64)
    M = np.stack([p.m for p in protos]).astype(np.float64)
    V = np.stack([p.V for p in protos]).astype(np.float64)
    prec = (w / V).sum(0)
    prec = np.maximum(prec, 1e-12)
    mu = (w * M / V).sum(0) / prec
    return ProtoExpert(mu.astype(np.float32), (1.0 / prec).astype(np.float32), "+".join(p.name for p in protos))


def compose(W, n_policy, protos, policy_share=0.5, exact=False, late_t=0.3, early_share=None):
    """Selection weights W [P + G, 70] (policy rows first) -> (experts for sample_poe: the G prototypes collapsed into ONE product
    expert, beta over [P rows + product]).  Within the prototype family the weights are Wang's (renormalised); the product's share
    against the policy family follows make_beta (policy share a floor, gripper from the policy family)."""
    W = np.asarray(W, np.float64)
    if not protos:
        return [], make_beta(W[:n_policy], n_policy, policy_share, late_t, early_share, exact)
    wq = W[n_policy:]
    prod = proto_product(protos, wq / np.maximum(wq.sum(0, keepdims=True), 1e-12))
    W2 = np.concatenate([W[:n_policy], wq.sum(0, keepdims=True)])
    return [prod], make_beta(W2, n_policy, policy_share, late_t, early_share, exact)


# ── selection (Wang Eq. 10-12) ─────────────────────────────────────────────────────────────────────────────────────
def loglik(Q, M, S2):
    """Q [n, 70] queries, M [n, 70] expert means at each query's observation, S2 [70] or [n, 70] variances -> [n, 70]."""
    return -0.5 * ((Q - M) ** 2 / S2 + np.log(2 * np.pi * S2))


def select(Q, means, vars_, K=3, forced=(0,), tau=0.5):
    """Greedy submodular coverage over experts.  Q [n, 70] (a phase's demo chunks); means [J, n, 70] (expert j's mean at query
    i's observation; a prototype's is the same for every i); vars_ [J, 70].  The `forced` experts are in from the start and K
    more are added.  -> (selected ids, per-dim weights [len, 70], mean per-dim log-likelihoods L [J, 70])."""
    J = means.shape[0]
    L = np.stack([loglik(Q, means[j], vars_[j]).mean(0) for j in range(J)])
    S = list(forced)

    def F(S_):
        return float(L[S_].max(0).sum()) if S_ else -np.inf
    for _ in range(K):
        best, gain = None, -np.inf
        for j in range(J):
            if j in S:
                continue
            g_ = F(S + [j]) - (F(S) if S else 0.0)
            if g_ > gain:
                best, gain = j, g_
        if best is None:
            break
        S.append(best)
    Ls = L[S]
    W = np.exp((Ls - Ls.max(0, keepdims=True)) / tau)
    W = W / W.sum(0, keepdims=True)
    return S, W, L


@dataclass
class PhaseCode:
    """The stored code of one phase: expert ids (strings: 'node:<level>:<id>', 'prompt:<k>', 'focus:<role>', 'token:<id>'),
    per-dim weights [K, 70] (stored fp16 as [K, 4 x 7]-like summaries), and the policy share."""
    phase: str
    experts: list
    weights: np.ndarray
    policy_share: float
    extra: dict = field(default_factory=dict)

    def nbytes(self):
        return int(sum(len(e.encode()) for e in self.experts) + np.asarray(self.weights, np.float16).nbytes + 4)
