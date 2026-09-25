"""The ColorMNIST world with the primitives the benchmark adds, rendered from disjoint MNIST pools.

Known values are the repository's palette (four bright foregrounds, four dark backgrounds).  The
benchmark adds three primitives no training row can contain: a WHITE background (outside the hull of
the dark backgrounds: Zekun's Appendix J failure case), a RED foreground (G below every trained
foreground), and an OUTLINE stroke (a new FACTOR nobody labelled: the digit mask minus its 1-px
erosion).  A tuple is (digit, fg, bg, stroke) with fg in 0..4 (4 = red), bg in 0..4 (4 = white) and
stroke in {0 fill, 1 outline}.

Pools: every digit's MNIST masks are split once, by seed, into ARRIVAL, REFERENCE and PROBE pools, so
an arrival image is never in a coverage reference set and never trains a probe.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..data.datasets import BG_COLORS, FG_COLORS, IMG, OOD_PAIRS, colorize, load_mnist, slot_id

FG = dict(FG_COLORS, red=(0.90, 0.15, 0.15))
BG = dict(BG_COLORS, white=(1.0, 1.0, 1.0))
FG_NAMES, BG_NAMES = list(FG), list(BG)
STROKES = ("fill", "outline")
NEW_FG, NEW_BG, NEW_STROKE = FG_NAMES.index("red"), BG_NAMES.index("white"), 1
WITHHELD_BG_NOBG3 = BG_NAMES.index("dark_brown")          # the value the nobg3 backbone never saw
POOLS = ("arrival", "reference", "probe")


def outline_mask(G, width=1):
    """Soft digit masks [n, H, W] -> their outline: the mask minus its binary erosion."""
    from scipy.ndimage import binary_erosion
    G = np.asarray(G, np.float32)
    hard = G > 0.5
    er = np.stack([binary_erosion(h, iterations=width) for h in hard])
    return G * (~er)


def render(G, f, b, stroke=0):
    """masks [n, H, W] -> flat RGB rows [n, H*W*3] in [0, 1]."""
    M = G if int(stroke) == 0 else outline_mask(G)
    return colorize(M, FG[FG_NAMES[int(f)]], BG[BG_NAMES[int(b)]])


def tuple_name(t):
    d, f, b, s = (list(t) + [0])[:4]
    return f"digit {d}, {FG_NAMES[f]} on {BG_NAMES[b]}" + (f", {STROKES[s]}" if s else "")


def trained_slots_nobg3():
    """The 90 (d, f, b) slots the nobg3 backbone trained on: seen pairs, dark_brown withheld."""
    ood = set(OOD_PAIRS)
    out = []
    for d in range(10):
        for f, fn in enumerate(FG_NAMES[:4]):
            for b, bn in enumerate(BG_NAMES[:4]):
                if b != WITHHELD_BG_NOBG3 and (fn, bn) not in ood:
                    out.append((d, f, b, 0))
    return out


def is_new_primitive(t):
    d, f, b, s = (list(t) + [0])[:4]
    return f == NEW_FG or b == NEW_BG or s == NEW_STROKE


class ColorMNISTWorld:
    """Renders any (digit, fg, bg, stroke) from a pool of MNIST masks; deterministic given (seed, pool, index)."""

    def __init__(self, n_per_class=600, seed=0, fractions=(0.3, 0.5, 0.2)):
        self.seed = int(seed)
        G, L = load_mnist(n_per_class=n_per_class, seed=seed)
        self.G, self.L = G.astype(np.float32), np.asarray(L, int)
        rng = np.random.default_rng(10_000 + seed)
        self.pool = {p: {} for p in POOLS}
        self.source = np.arange(len(L))                      # identity of each mask (row in the loaded set)
        for d in range(10):
            idx = rng.permutation(np.where(self.L == d)[0])
            n = len(idx)
            a, r = int(fractions[0] * n), int(fractions[1] * n)
            self.pool["arrival"][d] = idx[:a]
            self.pool["reference"][d] = idx[a:a + r]
            self.pool["probe"][d] = idx[a + r:]

    def mask_ids(self, d, pool, n=None, offset=0):
        ids = self.pool[pool][int(d)]
        return ids[offset:offset + n] if n is not None else ids

    def images(self, t, pool, n=1, offset=0):
        """-> rows [n, 3072] in [0, 1] of tuple t from `pool`, masks `offset:offset+n` of that digit."""
        d, f, b, s = (list(t) + [0])[:4]
        ids = self.mask_ids(d, pool, n, offset)
        return render(self.G[ids], f, b, s), ids

    def fingerprint(self):
        h = hashlib.sha256()
        for p in POOLS:
            for d in range(10):
                h.update(self.pool[p][d].astype(np.int64).tobytes())
        return h.hexdigest()[:16]


def slot_of(t):
    d, f, b, s = (list(t) + [0])[:4]
    return slot_id(d, f, b) if f < 4 and b < 4 and s == 0 else -1
