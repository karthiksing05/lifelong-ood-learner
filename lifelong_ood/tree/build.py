"""Build the prototype tree label-free from a backbone's OWN samples.

    tree = build(backbone, u_grid=(0.10, 0.20, 0.30, 0.40, 0.55, 0.70), n_samples=512, seed=0)

For each level: noise the samples to u, ascend the score (Adam, 150 steps), take the Tweedie mode,
dedup by per-dim RMSE, cap the level by basin size, fit a diagonal Laplace extent (4 Hutchinson
probes) collapsed to one variance per channel.  Parents link each node to the coarser node whose
Gaussian explains its mode best (the ACS paper's bottom-up agglomeration).  Counts are basin sizes.
"""
from __future__ import annotations

import time

import numpy as np

from ..backbones.sampler import sample
from ..backbones.schedule import cosine_abar
from ..backbones.score_ops import laplace_diag, mode_ascent
from .nodes import Tree, level_gaussian
from .responsibility import _loglik
from .update import dedup

DEFAULT_GRID = (0.10, 0.20, 0.30, 0.40, 0.55, 0.70)
DEFAULT_CAPS = {0.10: 32, 0.20: 24, 0.30: 16, 0.40: 16, 0.55: 8, 0.70: 8}


def build(backbone, u_grid=DEFAULT_GRID, n_samples=512, seed=0, iters=150, lr=0.1, tol=0.05, caps=None,
          samples=None, steps=50, probes=4, batch=256, verbose=True, cls=None, select="dedup"):
    """-> Tree over the backbone's row space.  `samples` (flat rows) may be supplied instead of drawn.
    select="dedup": keep distinct modes (per-dim RMSE > tol), capped by basin size;
    select="kmeans": cluster the level's modes into `cap` centroids (the ACS paper's "group modes within
    a variance threshold"), mode = centroid, extent from the medoid, count = cluster size."""
    t0 = time.time()
    lo, hi = backbone.clip
    if samples is None:
        samples = sample(backbone, n_samples, steps=steps, seed=seed, cls=cls, batch=batch)
    X = np.asarray(samples, np.float32)
    tree = Tree(u_grid, backbone.img, lo, hi)
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    per_level = []
    for level, u in enumerate(tree.u_grid):
        tl = time.time()
        xt, x0 = mode_ascent(backbone, X, u, iters=iters, lr=lr, seed=seed + 1000 * level, init="query",
                             adam=True, tweedie=True, cls=cls, batch=batch)
        x0 = np.clip(x0, lo, hi)
        cap = int(caps.get(u, caps.get(round(u, 2), 10 ** 9)))
        if select == "kmeans":
            from sklearn.cluster import KMeans
            K = min(cap, len(x0))
            km = KMeans(n_clusters=K, n_init=4, random_state=seed + level).fit(x0.astype(np.float64))
            lab = km.labels_
            cents = np.clip(km.cluster_centers_.astype(np.float32), lo, hi)
            medoid = [int(np.where(lab == k)[0][np.argmin(((x0[lab == k] - cents[k]) ** 2).sum(1))]) for k in range(K)]
            counts = np.asarray([(lab == k).sum() for k in range(K)], np.float64)
            ext = laplace_diag(backbone, xt[medoid], u, probes=probes, seed=seed + 7 * level, cls=cls, batch=batch)
            for k in range(K):
                tree.add(level, cents[k], extent_perdim=ext[k], count=float(counts[k]))
            n_distinct = len(set(dedup(x0, tol=tol)[1]))
            kept = list(range(K))
        else:
            kept, owner = dedup(x0, tol=tol)
            counts = np.asarray([(owner == k).sum() for k in kept], np.float64)
            order = np.argsort(-counts)[:cap]
            kept = [kept[i] for i in order]
            counts = counts[order]
            ext = laplace_diag(backbone, xt[kept], u, probes=probes, seed=seed + 7 * level, cls=cls, batch=batch)
            for k, i in enumerate(kept):
                tree.add(level, x0[i], extent_perdim=ext[k], count=float(counts[k]))
            n_distinct = len(set(owner))
        per_level.append(dict(u=u, modes=int(len(x0)), dedup=int(n_distinct), kept=len(kept), secs=time.time() - tl))
        if verbose:
            print(f"[tree] level {level} u={u:.2f}: {len(x0)} chains -> {n_distinct} distinct -> {len(kept)} kept "
                  f"({select}, cap {cap}) in {time.time() - tl:.1f}s", flush=True)
    link_parents(tree)
    tree.build_info = dict(seconds=time.time() - t0, per_level=per_level, n_samples=int(len(X)), seed=int(seed),
                           iters=int(iters), lr=float(lr), tol=float(tol), caps={str(k): int(v) for k, v in caps.items()},
                           select=select)
    if verbose:
        print(f"[tree] {tree.summary()} in {time.time() - t0:.1f}s", flush=True)
    return tree


def link_parents(tree):
    """Parent of node j at level l = the level-(l+1) node k maximising log N(sqrt(a') m_j; sqrt(a') m_k, a' V_k + (1-a') I)."""
    for level in range(len(tree.u_grid) - 1):
        ids_c, M_c, _, _ = tree.stats(level)
        ids_p, C, S, logpi = level_gaussian(tree, level + 1)
        if not ids_c or not ids_p:
            continue
        a = float(cosine_abar(tree.u_grid[level + 1]))
        L = _loglik(np.sqrt(a) * M_c, C, S, np.zeros(len(ids_p)))
        for row, cid in enumerate(ids_c):
            tree.nodes[cid].parent = int(ids_p[int(L[row].argmax())])
    for nid in tree.levels[-1]:
        tree.nodes[nid].parent = -1
    return tree
