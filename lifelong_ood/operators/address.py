"""ADDRESS and the membership read: is a query explained by the tree, at which levels, and how deep
does its novelty go?

The read is the LIKELIHOOD level of the best node, not its normalised share.  In 3072 dimensions the
responsibility softmax is nearly a hard assignment (every image is claimed by its nearest node at
r ~ 1, novel or not: p1 measured max r 0.95 on the withheld background), so novelty is read as the
per-dimension negative log-likelihood of the best node at each level, compared with the same read on
HELD-OUT samples of the model itself (never the samples that built the tree).  A level is failed when
the query's best-node NLL exceeds the calibrated quantile of held-out own samples; depth_of_novelty
is the number of failed levels and the coarsest one.  Responsibilities remain the GATE (fold.Gate).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..backbones.schedule import cosine_abar
from ..tree.nodes import level_gaussian
from ..tree.responsibility import _loglik


@dataclasses.dataclass
class Calibration:
    levels: list                  # level indices in the band
    thresholds: dict              # level -> upper bound on the best node's per-dim NLL (held-out own samples)
    own_nll: dict                 # level -> mean best-node NLL of held-out own samples
    q: float

    def to_dict(self):
        return dict(levels=list(self.levels), thresholds={str(k): float(v) for k, v in self.thresholds.items()},
                    own_nll={str(k): float(v) for k, v in self.own_nll.items()}, q=self.q)


def best_nll(tree, X0, level, seed=0, n_noise=4):
    """Clean rows -> (per-dim NLL of the best node [n], its id [n]) at a level, averaged over noise draws."""
    ids, C, S, logpi = level_gaussian(tree, level)
    X0 = np.atleast_2d(np.asarray(X0, np.float32))
    n, D = X0.shape
    if len(ids) == 0:
        return np.full(n, np.inf), np.full(n, -1)
    a = float(cosine_abar(tree.u_grid[level]))
    rng = np.random.default_rng(seed)
    acc = np.zeros((n, len(ids)))
    for _ in range(n_noise):
        xt = np.sqrt(a) * X0 + np.sqrt(1.0 - a) * rng.standard_normal(X0.shape).astype(np.float32)
        acc += _loglik(xt, C, S, np.zeros(len(ids)))
    L = acc / n_noise
    j = L.argmax(1)
    return -L[np.arange(n), j] / D, np.asarray(ids)[j]


def membership(tree, X, levels=None, seed=0, n_noise=4):
    """-> (nll [n, L], assign [n, L]): the best node's per-dim NLL and its id per level."""
    lv = list(range(len(tree.u_grid))) if levels is None else list(levels)
    X = np.atleast_2d(np.asarray(X, np.float32))
    nll = np.zeros((len(X), len(lv)))
    assign = -np.ones((len(X), len(lv)), int)
    for j, l in enumerate(lv):
        nll[:, j], assign[:, j] = best_nll(tree, X, l, seed=seed, n_noise=n_noise)
    return nll, assign


def calibrate(tree, own_heldout, q=0.95, seed=0, n_noise=4, levels=None):
    """Label-free: thresholds are the q-quantile of the best-node NLL of HELD-OUT samples of the model
    (fresh draws, not the tree's build set), per level.  All levels are in the band unless given."""
    lv = list(range(len(tree.u_grid))) if levels is None else list(levels)
    nll, _ = membership(tree, own_heldout, levels=lv, seed=seed, n_noise=n_noise)
    thr = {l: float(np.quantile(nll[:, j], q)) for j, l in enumerate(lv)}
    own = {l: float(nll[:, j].mean()) for j, l in enumerate(lv)}
    return Calibration(levels=lv, thresholds=thr, own_nll=own, q=q)


def explained(tree, cal, X, seed=0, n_noise=4):
    """-> (ok [n] bool, depth [n] int, coarsest_failing_level [n] int or -1, nll [n, L_band])."""
    nll, _ = membership(tree, X, levels=cal.levels, seed=seed, n_noise=n_noise)
    thr = np.asarray([cal.thresholds[l] for l in cal.levels])
    fail = nll > thr[None]
    depth = fail.sum(1)
    coarsest = np.array([max((cal.levels[j] for j in np.where(f)[0]), default=-1) for f in fail])
    return ~fail.any(1), depth, coarsest, nll


def depth_of_novelty(tree, cal, X, seed=0, n_noise=4):
    ok, depth, coarsest, _ = explained(tree, cal, X, seed=seed, n_noise=n_noise)
    return depth, coarsest
