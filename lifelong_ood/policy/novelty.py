"""ADDRESS for a policy library: is a demo chunk explained by the prototype tree, and at which depth is it new?

The read is the best node's per-dim NLL of the (noised) chunk at each level (operators.address.best_nll),
thresholded at the q-quantile of the same read on HELD-OUT frames of the library's own tasks (episodes that did
not build the tree: the tree's own frames sit on its nodes and would make every threshold too tight).

    rows_cal = heldout_rows(demos, library_npz)            # chunks of unseen episodes of the library's tasks
    thr = calibrate(tree, rows_cal, levels)                # {level: q95 NLL}
    nll, unexplained, depth = read(tree, rows, levels, thr)

depth = the coarsest level (largest t) at which a chunk is unexplained (-1: explained everywhere): a chunk new only
at fine levels is a variant of a known motion; one new at coarse levels is a new gross motion.
"""
from __future__ import annotations

import numpy as np

from ..operators.address import best_nll
from . import data as D


def heldout_rows(demos, library, per_task=2, n=1500, seed=0):
    """Chunk rows of episodes of the library's tasks that did not build it.  `library`: the r2b npz (dict-like)."""
    rng = np.random.default_rng(seed)
    used = set(np.asarray(library["episode"]).tolist())
    norm = D.Norm.load()
    rows = []
    for tid in sorted(set(np.asarray(library["task"]).tolist())):
        for e in [e for e in demos.episodes(int(tid)) if e not in used][:per_task]:
            ep = demos.load(e)
            rows.append(D.to_rows(ep["actions"][D.chunk_indices(len(ep["actions"]), 10)[0]], norm))
    rows = np.concatenate(rows)
    return rows[rng.choice(len(rows), size=min(n, len(rows)), replace=False)]


def calibrate(tree, rows, levels, q=0.95, seed=11):
    return {int(l): float(np.quantile(best_nll(tree, rows, l, seed=seed)[0], q)) for l in levels}


def read(tree, rows, levels, thr, seed=12):
    nll = {int(l): best_nll(tree, rows, l, seed=seed)[0] for l in levels}
    over = np.stack([nll[int(l)] > thr[int(l)] for l in levels], 1)          # [n, L]
    unexplained = over.any(1)
    lv = np.asarray(list(levels))
    depth = np.where(over.any(1), lv[np.where(over, np.arange(len(lv))[None], -1).max(1)], -1)
    return nll, unexplained, depth
