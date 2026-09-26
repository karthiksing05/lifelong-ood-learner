"""DIMENSION-WISE BORROWING: a demo frame's action chunk rebuilt from K library prototypes, each ACTION DIMENSION's whole
10-step trajectory borrowed from one of them.

    code = encode(tree, row, K=8)          -> (prototype ids [K], choice [7] in 0..K-1)
    mu   = decode(tree, code)              -> the 70-D chunk: dimension d's trajectory = prototype choice[d]'s trajectory in d

The K prototypes are chosen jointly by greedy maximum coverage of the per-dimension-trajectory log-likelihood (Zekun's
submodular selection, Eq. 8, with a group = one action dimension over the chunk's 10 steps), and each group takes the
selected prototype that explains it best: the zero-temperature limit of the per-dimension product of experts (Eq. 9-10).
The executed chunk is a deterministic function of the code and the shared library, so the code is ALL a frame stores:
2K bytes of ids + 7 x ceil(log2 K) bits of choices.  Borrowing whole dimension trajectories keeps each executed motion a
real prototype path per axis: smoother than the human demo (about half its jerk).
"""
from __future__ import annotations

import numpy as np

from ..operators.decompose import candidates, loglik_perdim

H, A7 = 10, 7
GROUPS = [[h * A7 + d for h in range(H)] for d in range(A7)]


class Library:
    """The band levels of the shared tree, cached as arrays."""

    def __init__(self, tree, levels=(2, 3, 4)):
        self.ids, self.levels, self.M, self.V = candidates(tree, list(levels))
        self.ids = np.asarray(self.ids)
        self.index = {int(i): k for k, i in enumerate(self.ids)}


def encode(lib, row, K=8):
    L = loglik_perdim(row, lib.M, lib.V)                                   # [P, 70]
    Lg = np.stack([L[:, g].sum(1) for g in GROUPS], 1)                      # [P, 7]
    best = np.full(A7, -np.inf)
    sel = []
    for _ in range(K):
        gains = np.maximum(Lg, best).sum(1)
        gains[sel] = -np.inf
        j = int(np.argmax(gains))
        sel.append(j)
        best = np.maximum(best, Lg[j])
    choice = np.argmax(Lg[sel], 0).astype(np.int64)
    return np.asarray(lib.ids[sel], np.int64), choice


def decode(lib, ids, choice):
    mu = np.zeros(H * A7, np.float32)
    var = np.zeros(H * A7, np.float32)
    for d, g in enumerate(GROUPS):
        k = lib.index[int(ids[choice[d]])]
        mu[g] = lib.M[k][g]
        var[g] = lib.V[k][g]
    return mu, var


def code_bytes(K):
    return 2 * K + int(np.ceil(A7 * np.ceil(np.log2(max(K, 2))) / 8))
