"""Fast synthetic end-to-end smoke test: no MNIST, D=16, runs in seconds.

    python tests/test_smoke.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood import LifelongLearner, ScoreField, make_tree
from lifelong_ood.compose import poe_teacher, candidates_tree


def make_blobs(seed=0, n=80, D=16):
    """Two coarse groups x two fine variants: a known 2-level hierarchy."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((4, D))
    centers[0, :4] = 1.0; centers[1, :4] = 1.0; centers[2, 4:8] = 1.0; centers[3, 4:8] = 1.0
    centers[1, 8] = 0.8; centers[3, 9] = 0.8
    X = np.concatenate([c + 0.07 * rng.standard_normal((n, D)) for c in centers])
    y = np.repeat(np.arange(4), n)
    o = rng.permutation(len(X))
    return X[o], y[o], centers


def main():
    X, y, centers = make_blobs()
    t = make_tree(prior_var=0.01)
    print("backend:", getattr(t, "backend", "numpy"))
    t.fit(X, seed=0)
    snap = t.snapshot(min_count=5)
    assert snap["is_leaf"].sum() >= 4, f"expected >=4 effective leaves, got {snap['is_leaf'].sum()}"
    assert snap["depth"].max() >= 2, "expected a hierarchy, got a flat tree"

    # each true center is near some effective leaf mean
    leaves = snap["M"][snap["is_leaf"]]
    d = np.sqrt(((centers[:, None] - leaves[None]) ** 2).sum(2).min(1) / X.shape[1])
    assert d.max() < 0.15, f"leaf means far from true centers: {d}"

    f = ScoreField(snap, mode="leaves")
    # score at a component mean at tiny noise is ~0
    g = f.score(leaves[:1], 0.01)
    assert np.abs(g).max() < 5.0
    # sampling stays near the data manifold
    S = f.sample(n=8, steps=30, seed=1, clip=(-2, 2))
    dS = np.sqrt(((S[:, None] - centers[None]) ** 2).sum(2).min(1) / X.shape[1])
    assert np.median(dS) < 0.2, f"samples off-manifold: {dS}"

    # mode ascent at coarse noise lands near tree nodes
    xt, x0 = f.mode_ascent(centers[0], u=0.3, n_starts=4, iters=60, seed=0)
    dm = np.sqrt(((x0[:, None] - snap["M"][None]) ** 2).sum(2).min(1) / X.shape[1])
    assert dm.max() < 0.2, f"ascended modes far from any node: {dm}"

    # composition: an OOD query mixing both coarse groups selects experts from both
    xq = centers[0] * 0.5 + centers[2] * 0.5
    M, V, meta = candidates_tree(snap, xq, top=12)
    teacher = poe_teacher(xq, M, V, K=3)
    assert teacher["var"].min() > 0

    # the lifelong loop: graft must leave old loglik untouched (to weight renormalization)
    L = LifelongLearner(tree=t, min_count=5, steps=20, seed=0)
    ll_before = L.loglik(X[:40]).mean()
    teacher, pool = L.learn_concept(xq, "ood_test", n_pool=12)
    ll_after = L.loglik(X[:40]).mean()
    bound = np.log(1 + 12 / len(X)) + 0.05
    assert ll_after > ll_before - X.shape[1] * 0.05 - 1.0, (ll_before, ll_after)
    assert abs(ll_after - ll_before) < 5.0, f"graft drifted old loglik: {ll_before} -> {ll_after}"
    S = L.sample_concept("ood_test", n=6, steps=20)
    assert S.shape == (6, X.shape[1])
    print("smoke: ok  "
          f"(nodes={t.n_nodes}, kept={len(snap['M'])}, depth={snap['depth'].max()}, "
          f"loglik drift {ll_after - ll_before:+.3f})")


if __name__ == "__main__":
    main()
