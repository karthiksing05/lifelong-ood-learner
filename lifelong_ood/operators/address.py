"""ADDRESS and the membership read: is a query explained by the tree, at which levels, and how deep
does its novelty go?

Membership at a level = the best node's responsibility for the query noised to that level, averaged
over noise draws (`tree.responsibility.basin`).  It is decided in the DISTINCT band: the levels where
the model's OWN samples are claimed by their nodes at high responsibility (calibrated label-free), not
at the root, where Proposition 3 says r is uninformative.  depth_of_novelty(x) = the number of band
levels at which no node claims x above the calibrated threshold, and the coarsest such level.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..backbones.schedule import cosine_abar
from ..tree.responsibility import basin


@dataclasses.dataclass
class Calibration:
    levels: list                  # level indices in the distinct band
    thresholds: dict              # level -> membership threshold (quantile of own-sample max r)
    own_rate: dict                # level -> mean max r on the model's own samples
    q: float

    def to_dict(self):
        return dict(levels=list(self.levels), thresholds={str(k): float(v) for k, v in self.thresholds.items()},
                    own_rate={str(k): float(v) for k, v in self.own_rate.items()}, q=self.q)


def membership(tree, X, levels=None, seed=0, n_noise=4):
    """-> (maxr [n, L], assign [n, L]) max responsibility and claiming node per level."""
    lv = list(range(len(tree.u_grid))) if levels is None else list(levels)
    X = np.atleast_2d(np.asarray(X, np.float32))
    maxr = np.zeros((len(X), len(lv)))
    assign = -np.ones((len(X), len(lv)), int)
    for j, l in enumerate(lv):
        a, r = basin(tree, X, l, seed=seed, n_noise=n_noise)
        if r.shape[1]:
            maxr[:, j] = r.max(1)
            assign[:, j] = a
    return maxr, assign


def calibrate(tree, own_samples, q=0.05, min_rate=0.95, seed=0, n_noise=4, cap=0.98):
    """Label-free: on the model's OWN samples, a level is in the distinct band when their mean max
    responsibility is >= min_rate; the membership threshold is the q-quantile of own-sample max r,
    capped at `cap` (a node claiming a query at 0.98 has explained it; an uncapped quantile of
    0.99999999 turns rounding into a novelty verdict)."""
    maxr, _ = membership(tree, own_samples, seed=seed, n_noise=n_noise)
    levels, thr, rate = [], {}, {}
    for l in range(len(tree.u_grid)):
        m = float(maxr[:, l].mean())
        rate[l] = m
        if m >= min_rate:
            levels.append(l)
            thr[l] = min(float(np.quantile(maxr[:, l], q)), cap)
    if not levels:                              # never leave the band empty: take the most distinct level
        l = int(np.argmax([rate[k] for k in range(len(tree.u_grid))]))
        levels, thr[l] = [l], min(float(np.quantile(maxr[:, l], q)), cap)
    return Calibration(levels=levels, thresholds=thr, own_rate=rate, q=q)


def explained(tree, cal, X, seed=0, n_noise=4):
    """-> (ok [n] bool, depth [n] int, coarsest_failing_level [n] int or -1, maxr [n, L_band])."""
    maxr, _ = membership(tree, X, levels=cal.levels, seed=seed, n_noise=n_noise)
    thr = np.asarray([cal.thresholds[l] for l in cal.levels])
    fail = maxr < thr[None]
    depth = fail.sum(1)
    coarsest = np.array([max((cal.levels[j] for j in np.where(f)[0]), default=-1) for f in fail])
    return ~fail.any(1), depth, coarsest, maxr


def depth_of_novelty(tree, cal, X, seed=0, n_noise=4):
    ok, depth, coarsest, _ = explained(tree, cal, X, seed=seed, n_noise=n_noise)
    return depth, coarsest
