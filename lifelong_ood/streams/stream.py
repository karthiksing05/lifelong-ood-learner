"""Arrival streams: the populations of docs/PLAN_2026-09.md Section 4, ordered, split and hashed.

    S = build_stream("colormnist", spec="q5-v1", split="dev", seed=0)
    for a in S.arrivals: a.pop, a.tup, a.image (one real row or None for coordinates-only), a.prereq

Populations (ColorMNIST, nobg3 backbone):
  trained      12 trained slots                                  -> must store nothing
  new-comb     12 untrained (fg, bg) pairs of known values        -> ADDRESS
  fold         white background, 8 arrivals, digits 0-3 x fg {0,1}   -> FOLD (outside the hull)
  fold-inhull  dark_brown (bg 3, withheld from training), 8 arrivals, same contexts -> the shallow control
  later        8 white-background compositions in FACTOR-DISJOINT contexts (digits 4-9 x fg {2,3})
  later-coords every remaining (d, f, white) cell, coordinates only (no image)
  second       red foreground, 6 arrivals                        -> a second primitive, other factor
  factor       outline stroke, 6 arrivals                        -> a new FACTOR
  two          white x red (4), outline x white (4)              -> two folded primitives composed
Orderings: interleaved (every `later` / `two` arrival follows >= 4 fold arrivals of each primitive it
needs) or blocked.  Splits: dev seeds draw their images from one world seed, report seeds from another.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from .novel import (NEW_BG, NEW_FG, NEW_STROKE, OOD_PAIRS, WITHHELD_BG_NOBG3, BG_NAMES, FG_NAMES,
                    ColorMNISTWorld, is_new_primitive, trained_slots_nobg3)

POPS = ("trained", "new-comb", "fold", "fold-inhull", "later", "later-coords", "second", "factor", "two")
MIN_FOLDS = 4


@dataclasses.dataclass
class Arrival:
    i: int
    pop: str
    tup: tuple                    # (d, f, b, s), the TRUE coordinates: logged for evaluation only
    mask_id: int                  # identity of the source MNIST mask (-1 = coordinates only)
    offset: int                   # position in the digit's arrival pool
    prereq: tuple                 # primitives this arrival needs folded before it: subset of {"white", "red", "outline"}
    image: np.ndarray = dataclasses.field(default=None, repr=False)   # [3072] in [0, 1], or None

    def primitives(self):
        d, f, b, s = self.tup
        return tuple(p for p, on in (("white", b == NEW_BG), ("red", f == NEW_FG), ("outline", s == NEW_STROKE)) if on)


def colormnist_q5v1(rng):
    """The unordered arrival table for the nobg3 backbone.  -> list of (pop, tup)."""
    rows = []
    trained = trained_slots_nobg3()
    for t in [trained[i] for i in rng.choice(len(trained), 12, replace=False)]:
        rows.append(("trained", t))
    ood = [(FG_NAMES.index(f), BG_NAMES.index(b)) for f, b in OOD_PAIRS if BG_NAMES.index(b) != WITHHELD_BG_NOBG3]
    cells = [(d, f, b, 0) for d in range(10) for f, b in ood]
    for t in [cells[i] for i in rng.choice(len(cells), 12, replace=False)]:
        rows.append(("new-comb", t))
    fold_ctx = [(d, f) for d in range(4) for f in (0, 1)]
    for d, f in fold_ctx:
        rows.append(("fold", (d, f, NEW_BG, 0)))
        rows.append(("fold-inhull", (d, f, WITHHELD_BG_NOBG3, 0)))
    later_ctx = [(d, f) for d in range(4, 10) for f in (2, 3)]
    for d, f in [later_ctx[i] for i in rng.choice(len(later_ctx), 8, replace=False)]:
        rows.append(("later", (d, f, NEW_BG, 0)))
    used = {r[1] for r in rows}
    for d in range(10):
        for f in range(4):
            t = (d, f, NEW_BG, 0)
            if t not in used:
                rows.append(("later-coords", t))
    for k, (d, b) in enumerate([(2, 0), (5, 1), (7, 2), (0, 1), (8, 0), (3, 2)]):
        rows.append(("second", (d, NEW_FG, b, 0)))
    for d, f, b in [(1, 0, 0), (4, 1, 1), (6, 2, 2), (9, 3, 0), (2, 1, 2), (5, 0, 1)]:
        rows.append(("factor", (d, f, b, NEW_STROKE)))
    for d in (1, 4, 6, 9):
        rows.append(("two", (d, NEW_FG, NEW_BG, 0)))
    for d, f in ((0, 2), (3, 3), (7, 0), (8, 1)):
        rows.append(("two", (d, f, NEW_BG, NEW_STROKE)))
    return rows


def _prereq(pop, tup):
    d, f, b, s = tup
    if pop in ("trained", "new-comb", "fold", "fold-inhull", "second", "factor", "later-coords"):
        return ()
    return tuple(p for p, on in (("white", b == NEW_BG), ("red", f == NEW_FG), ("outline", s == NEW_STROKE)) if on)


def _fold_pop_of(prim):
    return {"white": "fold", "red": "second", "outline": "factor"}[prim]


def order(arrivals, rng, mode="interleaved"):
    """Interleaved: a random permutation repaired so each arrival with prerequisites follows the
    MIN_FOLDS-th fold arrival of every primitive it needs.  Blocked: populations in POPS order."""
    if mode == "blocked":
        rank = {p: i for i, p in enumerate(POPS)}
        return sorted(arrivals, key=lambda a: (rank[a.pop], a.i))
    seq = [arrivals[i] for i in rng.permutation(len(arrivals))]
    for _ in range(len(seq) * 3):
        moved = False
        seen = {"white": 0, "red": 0, "outline": 0}
        for pos, a in enumerate(seq):
            ok = all(seen[p] >= MIN_FOLDS for p in a.prereq)
            if not ok:
                seq.pop(pos)
                cnt = dict(seen)
                ins = pos
                for q in range(pos, len(seq)):
                    for p in seq[q].primitives():
                        if seq[q].pop == _fold_pop_of(p):
                            cnt[p] += 1
                    if all(cnt[p] >= MIN_FOLDS for p in a.prereq):
                        ins = q + 1
                        break
                else:
                    ins = len(seq)
                seq.insert(ins, a)
                moved = True
                break
            for p in a.primitives():
                if a.pop == _fold_pop_of(p):
                    seen[p] += 1
        if not moved:
            break
    return seq


@dataclasses.dataclass
class Stream:
    bench: str
    spec: str
    split: str
    seed: int
    ordering: str
    arrivals: list
    world: object
    trained: list
    sha256: str

    def by_pop(self, pop):
        return [a for a in self.arrivals if a.pop == pop]

    def counts(self):
        return {p: len(self.by_pop(p)) for p in POPS}


def build_stream(bench="colormnist", spec="q5-v1", split="dev", seed=0, ordering="interleaved", n_per_class=600):
    """Dev seeds (0-2) and report seeds (3-5) draw images from different world seeds and orderings."""
    if bench != "colormnist" or spec != "q5-v1":
        raise NotImplementedError(f"stream {bench}/{spec}")
    world_seed = (100 if split == "dev" else 200) + int(seed)
    world = ColorMNISTWorld(n_per_class=n_per_class, seed=world_seed)
    rng = np.random.default_rng(1_000 * (0 if split == "dev" else 1) + int(seed))
    rows = colormnist_q5v1(rng)
    per_digit = {d: 0 for d in range(10)}
    arrivals = []
    for i, (pop, tup) in enumerate(rows):
        d = tup[0]
        if pop == "later-coords":
            arrivals.append(Arrival(i, pop, tup, -1, -1, _prereq(pop, tup), None))
            continue
        off = per_digit[d]
        per_digit[d] += 1
        img, ids = world.images(tup, "arrival", n=1, offset=off)
        arrivals.append(Arrival(i, pop, tup, int(ids[0]), off, _prereq(pop, tup), img[0]))
    seq = order(arrivals, rng, ordering)
    for k, a in enumerate(seq):
        a.i = k
    table = [(a.i, a.pop, list(a.tup), a.mask_id) for a in seq]
    sha = hashlib.sha256(json.dumps(table).encode() + world.fingerprint().encode()).hexdigest()
    return Stream(bench, spec, split, int(seed), ordering, seq, world, trained_slots_nobg3(), sha)
