"""DECOMPOSE: which prototypes explain a query, and the product of experts they form.

Zekun's Eq. 8-10 (submodular coverage selection, per-dimension softmax weights, closed-form Gaussian
product) applied to the TREE's nodes instead of a per-query ascent pool: every node's (mode, extent)
is already in x0 space, so a decomposition is ~K_total x D multiply-adds and no forward pass.  The
result is a Code (node ids + weights, the thing ADDRESS stores) and a teacher (mu_T, Sigma_T) whose
analytic eps (Eq. 14) drives the sampler through `backbones.sampler.sample(..., guide=...)`.

Ported from lifelong_ood/compose.py (loglik_perdim, submodular_select, poe_teacher, teacher_eps,
stacked_cfg_weight) with the same measured deviations: adaptive temperature and the MIXTURE extent
for sampling (the product covariance is a precision, not a spread).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from ..backbones.schedule import path_coeffs


def loglik_perdim(xq, M, V):
    """l_{j,r}(x_q): prototype j's contribution to the query log-likelihood at dimension r.  [P, D]"""
    xq = np.ravel(np.asarray(xq, np.float64))
    M, V = np.asarray(M, np.float64), np.asarray(V, np.float64)
    return -0.5 * (((xq - M) ** 2) / V + np.log(2 * np.pi * V))


def submodular_select(L, K=3):
    """Greedy max of F(S) = sum_r max_{j in S} L[j, r].  -> selected indices in pick order."""
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
    return np.array(sel, int)


@dataclasses.dataclass
class Code:
    """What ADDRESS stores: the selected nodes with per-dimension weights summarised as one scalar
    weight each (the per-dim weights are recomputed from the stored nodes at read time)."""
    nodes: list                      # node ids, pick order
    weights: list                    # mean per-dimension weight of each node
    levels: list                     # level index of each node
    mu: np.ndarray = dataclasses.field(default=None, repr=False)
    var: np.ndarray = dataclasses.field(default=None, repr=False)
    temperature: float = 0.0

    def bytes(self):
        return 4 * len(self.nodes)   # uint16 id + fp16 weight per entry

    def entries(self):
        return list(zip(self.nodes, self.weights))


def candidates(tree, levels=None):
    """-> (ids, levels, M [P, D], V [P, D]) of every node at the given levels (default: all)."""
    ids, lv, Ms, Vs = [], [], [], []
    for l in (range(len(tree.u_grid)) if levels is None else levels):
        i, M, V, _ = tree.stats(l)
        if len(i):
            ids += list(i)
            lv += [l] * len(i)
            Ms.append(M)
            Vs.append(V)
    if not ids:
        return [], [], np.zeros((0, 3 * tree.img * tree.img)), np.zeros((0, 3 * tree.img * tree.img))
    return ids, lv, np.concatenate(Ms), np.concatenate(Vs)


def decompose(tree, xq, K=3, tau=0.05, adaptive_tau=True, extent="mixture", levels=None):
    """Select K nodes that jointly explain xq (flat rows in the tree's space) and form their product.
    -> Code with mu / var (x0 space) for the teacher."""
    ids, lv, M, V = candidates(tree, levels)
    if len(ids) == 0:
        raise ValueError("decompose: the tree has no nodes")
    L = loglik_perdim(xq, M, V)
    sel = submodular_select(L, K)
    Ls = L[sel]
    T = tau * max(float(Ls.std(0).mean()), 1e-6) if adaptive_tau else tau
    Z = Ls / T
    Z -= Z.max(0, keepdims=True)
    w = np.exp(Z)
    w /= w.sum(0, keepdims=True)
    prec = (w / V[sel]).sum(0)
    mu = (w * M[sel] / V[sel]).sum(0) / prec
    var_poe = 1.0 / prec
    between = ((M[sel] - mu[None]) ** 2 * w).sum(0)
    var_mix = (w * V[sel]).sum(0) + between
    return Code(nodes=[int(ids[j]) for j in sel], weights=[float(w[k].mean()) for k in range(len(sel))],
                levels=[int(lv[j]) for j in sel], mu=mu.astype(np.float32),
                var=(var_mix if extent == "mixture" else var_poe).astype(np.float32), temperature=float(T))


def teacher_from_code(tree, code, tau=0.05, adaptive_tau=True, extent="mixture"):
    """Rebuild (mu, var) from the STORED entries alone (node ids + scalar weights), so a code can be
    read back without the query: the per-dimension weights become the node's scalar weight."""
    M = np.stack([tree.nodes[i].mode for i in code.nodes]).astype(np.float64)
    V = np.stack([tree.nodes[i].var for i in code.nodes]).astype(np.float64)
    w = np.asarray(code.weights, np.float64)[:, None] * np.ones((1, M.shape[1]))
    w /= w.sum(0, keepdims=True)
    prec = (w / V).sum(0)
    mu = (w * M / V).sum(0) / prec
    between = ((M - mu[None]) ** 2 * w).sum(0)
    var = (w * V).sum(0) + between if extent == "mixture" else 1.0 / prec
    return mu.astype(np.float32), var.astype(np.float32)


def guide(mu, var, w0=1.2, wmax=2.0, wmin=0.0, schedule="variance", path="cosine"):
    """-> (eps_fn, w_fn) for the sampler: Zekun's Eq. 14 (analytic eps of the diffused Gaussian
    N(alpha m, alpha^2 V + sigma^2) along `path`) and Eq. 16 (variance-aware guidance weight)."""
    mu_t = torch.as_tensor(np.asarray(mu, np.float32))
    var_t = torch.as_tensor(np.asarray(var, np.float32))

    def eps_fn(x_rows, u):
        alpha, sigma, alpha2, sigma2 = path_coeffs(float(u), path)
        m, v = mu_t.to(x_rows.device), var_t.to(x_rows.device)
        return sigma * (x_rows - alpha * m) / (alpha2 * v + sigma2)

    def w_fn(x_rows, u):
        u = float(np.clip(u, 0.0, 1.0))
        if schedule == "const":
            return float(wmax)
        if schedule == "anneal":
            return float(wmin + (wmax - wmin) * (1.0 - u))
        _, _, alpha2, sigma2 = path_coeffs(u, path)
        return float(np.clip(w0 * float((alpha2 * np.asarray(var) + sigma2).mean()), wmin, wmax))
    return eps_fn, w_fn
