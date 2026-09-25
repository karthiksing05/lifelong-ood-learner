"""Responsibilities: the label-free gate.  r_k(x_t, u) = pi_k N(x_t; sqrt(a) m_k, a V_k + (1-a) I) normalised
over the nodes of the level nearest u (Proposition 2's r evaluated on the tree), or over the children of
the node chosen at the coarser level (a top-down Cobweb descent).  Sums to one over its scope."""
from __future__ import annotations

import numpy as np

from .nodes import level_gaussian


def _loglik(X, C, S, logpi):
    """diagonal-Gaussian log-likelihoods [n, K] of rows X under centres C [K, D], variances S [K, D]."""
    X = np.asarray(X, np.float64)
    n, D = X.shape
    out = np.empty((n, len(C)), np.float64)
    for k in range(len(C)):
        d = X - C[k][None]
        out[:, k] = -0.5 * (d * d / S[k][None]).sum(1) - 0.5 * np.log(S[k]).sum() + logpi[k]
    return out


def _softmax(L):
    z = L.max(1, keepdims=True)
    e = np.exp(L - z)
    return e / e.sum(1, keepdims=True)


def responsibility(tree, Xt, u, scope="level", parent_ids=None):
    """-> (ids, r [n, K]) for noisy rows Xt at noise level u.

    scope="level": normalise over every node of the level nearest u.
    scope="siblings": normalise over the children of `parent_ids` (one per row); rows whose parent has
    no children fall back to the level scope.
    """
    level = tree.level_of(u)
    ids, C, S, logpi = level_gaussian(tree, level)
    n = len(np.atleast_2d(Xt))
    if len(ids) == 0:
        return ids, np.zeros((n, 0))
    L = _loglik(np.atleast_2d(Xt), C, S, logpi)
    if scope == "siblings" and parent_ids is not None:
        par = np.asarray([tree.nodes[i].parent for i in ids])
        mask = np.zeros_like(L, dtype=bool)
        for row, p in enumerate(np.asarray(parent_ids).reshape(-1)):
            m = par == p
            mask[row] = m if m.any() else True
        L = np.where(mask, L, -np.inf)
    return ids, _softmax(L)


def categorize(tree, Xt_by_level, scope="siblings"):
    """Top-down descent: `Xt_by_level[l]` are the rows noised to level l (coarsest last).  -> path
    [n, n_levels] of node ids (level index order) and the responsibilities used at each level."""
    n_levels = len(tree.u_grid)
    n = len(np.atleast_2d(Xt_by_level[0]))
    path = -np.ones((n, n_levels), int)
    resp = []
    parent = np.full(n, -1)
    for level in reversed(range(n_levels)):
        ids, r = responsibility(tree, Xt_by_level[level], tree.u_grid[level],
                                scope=scope if level < n_levels - 1 else "level", parent_ids=parent)
        if len(ids) == 0:
            resp.append(r)
            continue
        pick = np.asarray(ids)[r.argmax(1)]
        path[:, level] = pick
        parent = pick
        resp.append(r)
    return path, resp[::-1]


def basin(tree, X0, level, seed=0, n_noise=4):
    """Hard assignment of CLEAN rows to a level's nodes, averaging responsibilities over noise draws."""
    from ..backbones.schedule import path_coeffs
    alpha, sigma, _, _ = path_coeffs(tree.u_grid[level], tree.path)
    rng = np.random.default_rng(seed)
    X0 = np.atleast_2d(np.asarray(X0, np.float32))
    acc = None
    for _ in range(n_noise):
        xt = alpha * X0 + sigma * rng.standard_normal(X0.shape).astype(np.float32)
        ids, r = responsibility(tree, xt, tree.u_grid[level])
        acc = r if acc is None else acc + r
    if acc is None or acc.shape[1] == 0:
        return np.zeros(len(X0), int) - 1, np.zeros((len(X0), 0))
    acc /= n_noise
    return np.asarray(ids)[acc.argmax(1)], acc
