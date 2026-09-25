"""The stream cannot leak: new primitives are in no training slot, dev and report images are disjoint,
later compositions use factor-disjoint contexts, probe / reference pools never hold an arrival image,
every population is non-empty, and the interleaved order respects prerequisites.  numpy only."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.streams.novel import (NEW_BG, NEW_FG, NEW_STROKE, WITHHELD_BG_NOBG3, ColorMNISTWorld,  # noqa: E402
                                        is_new_primitive, outline_mask, render, trained_slots_nobg3)
from lifelong_ood.streams.stream import MIN_FOLDS, POPS, build_stream  # noqa: E402


@pytest.fixture(scope="module")
def streams():
    return {("dev", 0): build_stream(split="dev", seed=0, n_per_class=120),
            ("dev", 1): build_stream(split="dev", seed=1, n_per_class=120),
            ("report", 3): build_stream(split="report", seed=3, n_per_class=120)}


def test_counts_and_populations(streams):
    S = streams[("dev", 0)]
    c = S.counts()
    assert c == {"trained": 12, "new-comb": 12, "fold": 8, "fold-inhull": 8, "later": 8, "later-coords": 24,
                 "second": 6, "factor": 6, "two": 8}
    assert sum(v for p, v in c.items() if p != "later-coords") == 68
    assert all(a.image is not None and a.image.shape == (3072,) for a in S.arrivals if a.pop != "later-coords")
    assert all(a.image is None for a in S.by_pop("later-coords"))


def test_new_primitives_never_appear_in_training_slots(streams):
    trained = set(trained_slots_nobg3())
    assert len(trained) == 90
    assert not any(is_new_primitive(t) for t in trained)
    assert not any(t[2] == WITHHELD_BG_NOBG3 for t in trained)
    S = streams[("dev", 0)]
    for a in S.arrivals:
        if a.pop == "trained":
            assert a.tup in trained
        else:
            assert a.tup not in trained


def test_later_contexts_are_factor_disjoint_from_fold_contexts(streams):
    S = streams[("dev", 0)]
    fold = {(a.tup[0], a.tup[1]) for a in S.by_pop("fold")}
    later = {(a.tup[0], a.tup[1]) for a in S.by_pop("later")}
    assert not ({d for d, _ in fold} & {d for d, _ in later})
    assert not ({f for _, f in fold} & {f for _, f in later})
    assert all(a.tup[2] == NEW_BG for a in S.by_pop("fold") + S.by_pop("later") + S.by_pop("later-coords"))
    assert all(a.tup[1] == NEW_FG for a in S.by_pop("second"))
    assert all(a.tup[3] == NEW_STROKE for a in S.by_pop("factor"))
    assert all(len(a.primitives()) == 2 for a in S.by_pop("two"))


def test_dev_and_report_images_are_disjoint_and_pools_are_disjoint(streams):
    d0, d1, r3 = streams[("dev", 0)], streams[("dev", 1)], streams[("report", 3)]
    assert d0.world.seed != r3.world.seed and d0.sha256 != r3.sha256 != d1.sha256
    # same world seed for dev 0 and dev 1 (different orderings), different images across splits
    src = lambda S: {(a.tup[0], int(S.world.source[a.mask_id])) for a in S.arrivals if a.mask_id >= 0}
    assert src(d0) and src(r3)
    W = d0.world
    for d in range(10):
        A, R, P = set(W.pool["arrival"][d]), set(W.pool["reference"][d]), set(W.pool["probe"][d])
        assert not (A & R) and not (A & P) and not (R & P) and A and R and P
    for a in d0.arrivals:
        if a.mask_id >= 0:
            assert a.mask_id in W.pool["arrival"][a.tup[0]]
    assert [a.tup for a in d0.arrivals] != [a.tup for a in d1.arrivals], "orderings differ across seeds"


def test_interleaved_order_respects_prerequisites_and_blocked_groups(streams):
    S = streams[("dev", 0)]
    seen = {"white": 0, "red": 0, "outline": 0}
    fold_pop = {"white": "fold", "red": "second", "outline": "factor"}
    for a in S.arrivals:
        for p in a.prereq:
            assert seen[p] >= MIN_FOLDS, (a.i, a.pop, a.tup, seen)
        for p in a.primitives():
            if a.pop == fold_pop[p]:
                seen[p] += 1
    pops = [a.pop for a in S.arrivals]
    assert pops != sorted(pops, key=POPS.index), "interleaved is not blocked"
    B = build_stream(split="dev", seed=0, ordering="blocked", n_per_class=120)
    bp = [a.pop for a in B.arrivals]
    assert bp == sorted(bp, key=POPS.index)


def test_renderer_primitives():
    W = ColorMNISTWorld(n_per_class=40, seed=0)
    G = W.G[W.pool["arrival"][3][:2]]
    white = render(G, 0, NEW_BG, 0).reshape(2, 32, 32, 3)
    assert white[:, 0, 0].min() > 0.99, "white background at the corner"
    red = render(G, NEW_FG, 0, 0).reshape(2, 32, 32, 3)
    m = G > 0.9
    assert red[m][:, 0].mean() > 0.85 and red[m][:, 1].mean() < 0.2
    O = outline_mask(G)
    assert O.sum() < G.sum() and (O > 0.5).sum() > 0
    inner = (G > 0.5) & ~(O > 0.5)
    assert inner.sum() > 0, "the outline removes the interior"
