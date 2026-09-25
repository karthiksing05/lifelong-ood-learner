"""Cobweb's operators on the tree: incorporate (Welford with a soft share r, Proposition 2), create,
merge (pooled statistics, children re-parented), and the dedup / cap rules the build uses."""
from __future__ import annotations

import numpy as np

from .nodes import quantize_mode


def welford_soft(mean, var, count, x, r):
    """One instance x with share r into (mean, var, count): the ideal denoiser's update with r in
    place of Cobweb's 1/(c+1).  -> (mean', var', count')."""
    r = float(r)
    c2 = count + r
    if c2 <= 0:
        return mean, var, count
    d = x - mean
    mean2 = mean + (r / c2) * d
    var2 = (count / c2) * var + (r / c2) * (count / c2) * d * d if count > 0 else var
    return mean2, var2, c2


def incorporate(tree, nid, x_rows, r=1.0):
    """Update node `nid` with instance `x_rows` (flat, the tree's row space) at share r.  The stored
    form is re-quantised, so what is charged is what is used."""
    n = tree.nodes[nid]
    mean2, var2, c2 = welford_soft(n.mode.astype(np.float64), n.var.astype(np.float64), n.count,
                                   np.asarray(x_rows, np.float64).reshape(-1), r)
    n.count = c2
    _store(tree, n, mean2, var2)
    tree._materialise(n)
    return nid


def _store(tree, n, mean, var):
    """Write (mean, var) into the node's STORED form (fp16 rows on a generic tree; uint8 + channel extents on an
    image tree)."""
    if tree.generic:
        n.q = mean.astype(np.float32).astype(np.float16)
        n.extent_dim = np.clip(var, 1e-4, 1.0).astype(np.float16)
        n.extent = np.asarray([float(n.extent_dim.astype(np.float32).mean())], np.float32)
        return
    n.q = quantize_mode(mean.astype(np.float32), tree.img, n.q.shape[0], tree.lo, tree.hi)
    n.extent = var.reshape(tree.img, tree.img, 3).mean(axis=(0, 1)).astype(np.float32)
    if n.extent_dim is not None:
        n.extent_dim = np.clip(var, 1e-3, 1.0).astype(np.float16)


def create(tree, level, mode_rows, extent_perdim=None, extent_channels=None, count=1.0, parent=-1):
    return tree.add(level, mode_rows, extent_perdim=extent_perdim, extent_channels=extent_channels,
                    count=count, parent=parent)


def merge(tree, i, j):
    """Pool two nodes of one level into node i (Cobweb's merge): count-weighted mean, pooled variance
    sum_sq = S_a + S_b + (mu_b - mu_a)^2 n_a n_b / (n_a + n_b).  Children of j are re-parented to i."""
    a, b = tree.nodes[i], tree.nodes[j]
    assert a.level == b.level, "merge pools siblings of one level"
    na, nb = a.count, b.count
    n = na + nb
    if n <= 0:
        tree.remove(j)
        return i
    ma, mb = a.mode.astype(np.float64), b.mode.astype(np.float64)
    mu = (na * ma + nb * mb) / n
    Sa, Sb = na * a.var.astype(np.float64), nb * b.var.astype(np.float64)
    S = Sa + Sb + (mb - ma) ** 2 * na * nb / n
    a.count = n
    _store(tree, a, mu, S / n)
    tree._materialise(a)
    for m in tree.nodes.values():
        if m.parent == j:
            m.parent = i
    tree.remove(j)
    return i


def dedup(modes, tol=0.05):
    """Greedy: keep a mode unless its per-dim RMSE to a kept one is below tol (compose.candidates_ascent's
    rule).  -> indices kept, and for every mode the index of the kept mode it collapsed onto."""
    M = np.asarray(modes, np.float64)
    kept, owner = [], np.zeros(len(M), int)
    for i in range(len(M)):
        if kept:
            d = np.sqrt(((M[kept] - M[i][None]) ** 2).mean(1))
            j = int(d.argmin())
            if d[j] < tol:
                owner[i] = kept[j]
                continue
        kept.append(i)
        owner[i] = i
    return kept, owner
