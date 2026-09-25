"""The prototype tree on an analytic mixture: deterministic given a seed, recovers the planted modes,
responsibilities sum to one and obey Proposition 3's limits, incorporate is Welford with share r,
merge pools exactly, and what is charged is what is used.  numpy + a tiny torch-free double."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.backbones.schedule import cosine_abar                # noqa: E402
from lifelong_ood.tree.build import build, link_parents                # noqa: E402
from lifelong_ood.tree.nodes import Tree, dequantize_mode, mode_res, quantize_mode  # noqa: E402
from lifelong_ood.tree.responsibility import basin, categorize, responsibility  # noqa: E402
from lifelong_ood.tree.update import incorporate, merge, welford_soft  # noqa: E402

IMG = 4
D = 3 * IMG * IMG


class MixtureBackbone:
    """A Backbone-like double whose eps is the exact eps of a K-Gaussian mixture in x0 space."""
    kind, layout, space, clip, device, model = "mixture", "hwc", "model", (-1.0, 1.0), "cpu", None

    def __init__(self, means, s2=0.01):
        import torch
        self.torch = torch
        self.M = np.asarray(means, np.float64)
        self.s2 = float(s2)
        self.img, self.dim = IMG, D

    def eps(self, X, u, cls=None, emb=None, gate=None, batch=256):
        torch = self.torch
        X = X.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(X) else np.asarray(X, np.float64)
        a = float(cosine_abar(float(u)))
        C = np.sqrt(a) * self.M
        v = a * self.s2 + (1.0 - a)
        L = np.stack([-0.5 * ((X - c[None]) ** 2).sum(1) / v for c in C], 1)
        w = np.exp(L - L.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        score = (w @ C - X) / v
        return torch.as_tensor(-np.sqrt(1.0 - a) * score, dtype=torch.float32)


def _planted(seed=0, K=3):
    rng = np.random.default_rng(seed)
    means = np.clip(rng.uniform(-0.8, 0.8, (K, D)), -1, 1)
    means = np.round(means * 8) / 8          # representable at 8-bit after quantisation? no: keeps them spread
    return means


def _samples(means, n=60, s2=0.01, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(len(means), size=n)
    return np.clip(means[k] + np.sqrt(s2) * rng.standard_normal((n, D)), -1, 1).astype(np.float32), k


def test_build_is_deterministic_and_recovers_the_planted_modes():
    means = _planted()
    X, k = _samples(means)
    bk = MixtureBackbone(means)
    grid = (0.10, 0.30, 0.55)
    t1 = build(bk, u_grid=grid, samples=X, seed=3, iters=60, lr=0.05, tol=0.05, caps={0.10: 8, 0.30: 8, 0.55: 8}, verbose=False)
    t2 = build(bk, u_grid=grid, samples=X, seed=3, iters=60, lr=0.05, tol=0.05, caps={0.10: 8, 0.30: 8, 0.55: 8}, verbose=False)
    assert t1.summary() == t2.summary()
    for a, b in zip(t1.nodes.values(), t2.nodes.values()):
        assert np.array_equal(a.q, b.q) and a.count == b.count and a.parent == b.parent
    ids, M, V, c = t1.stats(0)
    assert len(ids) == len(means), t1.summary()
    d = np.sqrt(((M[:, None, :] - means[None]) ** 2).mean(2))
    assert d.min(1).max() < 0.08, "every leaf-level node sits on a planted mode (within quantisation)"
    assert set(d.argmin(1)) == set(range(len(means)))
    assert t1.bytes() == sum(n.bytes() for n in t1.nodes.values()) > 0
    assert all(t1.nodes[i].parent == -1 for i in t1.levels[-1])
    assert all(t1.nodes[i].parent in t1.nodes for i in t1.levels[0])


def test_responsibility_sums_to_one_and_obeys_prop3_limits():
    means = _planted()
    bk = MixtureBackbone(means)
    tree = Tree((0.10, 0.55, 0.95), IMG, -1.0, 1.0)
    for lvl in range(3):
        for m in means:
            tree.add(lvl, m, extent_channels=np.full(3, 0.01, np.float32), count=1.0)
    link_parents(tree)
    X = np.random.default_rng(0).standard_normal((5, D)).astype(np.float32)
    for u in tree.u_grid:
        ids, r = responsibility(tree, X, u)
        assert r.shape == (5, len(means)) and np.allclose(r.sum(1), 1.0, atol=1e-9)
    # high noise: the level's nodes share the instance almost equally (r -> 1/(K) with equal counts)
    a = float(cosine_abar(0.95))
    xt = np.sqrt(a) * means[0] + 0.0 * means[0]
    _, r_hi = responsibility(tree, xt[None], 0.95)
    assert r_hi.max() < 0.6
    # low noise: a point at a mode belongs to that node
    _, r_lo = responsibility(tree, (np.sqrt(float(cosine_abar(0.10))) * means[1])[None], 0.10)
    assert r_lo[0, 1] > 0.999
    # sibling scope conditions on the parent
    path, resp = categorize(tree, [means[2][None] * np.sqrt(float(cosine_abar(u))) for u in tree.u_grid])
    assert path.shape == (1, 3) and all(p in tree.nodes for p in path[0])


def test_basin_assigns_clean_rows_to_their_planted_node():
    means = _planted()
    X, k = _samples(means, n=90)
    bk = MixtureBackbone(means)
    tree = build(bk, u_grid=(0.10, 0.30), samples=X, seed=0, iters=60, lr=0.05, caps={0.10: 8, 0.30: 8}, verbose=False)
    assign, r = basin(tree, X, 0, seed=0)
    ids, M, _, _ = tree.stats(0)
    node_to_planted = {i: int(np.sqrt(((M[j] - means) ** 2).mean(1)).argmin()) for j, i in enumerate(ids)}
    purity = np.mean([node_to_planted[a] == kk for a, kk in zip(assign, k)])
    assert purity > 0.95, purity


def test_incorporate_is_welford_with_share_r_and_merge_pools():
    rng = np.random.default_rng(0)
    mean, var, count = rng.standard_normal(D), np.abs(rng.standard_normal(D)) + 0.1, 5.0
    x = rng.standard_normal(D)
    m1, v1, c1 = welford_soft(mean, var, count, x, 1.0)
    # r = 1 is the textbook Welford / Cobweb update
    assert np.allclose(m1, mean + (x - mean) / 6.0) and c1 == 6.0
    assert np.allclose(v1, (5 / 6) * var + (1 / 6) * (5 / 6) * (x - mean) ** 2)
    m0, v0, c0 = welford_soft(mean, var, count, x, 0.0)
    assert np.array_equal(m0, mean) and np.array_equal(v0, var) and c0 == 5.0
    mh, vh, ch = welford_soft(mean, var, count, x, 0.5)
    assert np.allclose(mh, mean + (0.5 / 5.5) * (x - mean)) and ch == 5.5
    tree = Tree((0.10,), IMG, -1.0, 1.0)
    i = tree.add(0, np.zeros(D), extent_channels=np.full(3, 0.02, np.float32), count=2.0)
    j = tree.add(0, np.ones(D) * 0.5, extent_channels=np.full(3, 0.02, np.float32), count=2.0)
    incorporate(tree, i, np.full(D, 0.25), r=1.0)
    assert tree.nodes[i].count == 3.0 and abs(tree.nodes[i].mode.mean() - 0.25 / 3) < 0.01
    merge(tree, i, j)
    assert j not in tree.nodes and tree.nodes[i].count == 5.0 and len(tree.levels[0]) == 1
    assert abs(tree.nodes[i].mode.mean() - (3 * 0.25 / 3 + 2 * 0.5) / 5) < 0.01


def test_stored_form_is_what_is_used():
    tree = Tree((0.10, 0.55), IMG, -1.0, 1.0)
    rows = np.random.default_rng(0).uniform(-1, 1, D).astype(np.float32)
    i = tree.add(0, rows, extent_channels=np.full(3, 0.01, np.float32))
    j = tree.add(1, rows, extent_channels=np.full(3, 0.01, np.float32))
    assert tree.nodes[i].q.shape == (IMG, IMG, 3) and tree.nodes[j].q.shape == (mode_res(0.55, IMG),) * 2 + (3,)
    assert np.array_equal(tree.nodes[i].mode, dequantize_mode(tree.nodes[i].q, IMG, -1.0, 1.0))
    assert np.abs(tree.nodes[i].mode - rows).max() <= 2.0 / 255 + 1e-6
    assert tree.nodes[i].bytes() == 16 + IMG * IMG * 3 + 12 and tree.nodes[j].bytes() == 16 + 3 * mode_res(0.55, IMG) ** 2 + 12
    k = tree.add(0, rows, extent_perdim=np.full(D, 0.02, np.float32), per_dim=True)
    assert tree.nodes[k].bytes() == 16 + IMG * IMG * 3 + 12 + 2 * D and tree.nodes[k].var.dtype == np.float32
    assert np.allclose(tree.nodes[k].var, 0.02, atol=1e-4)
    t3 = Tree.from_dict(tree.to_dict())
    assert np.array_equal(t3.nodes[k].var, tree.nodes[k].var) and t3.bytes() == tree.bytes()
    d = tree.to_dict()
    t2 = Tree.from_dict(d)
    assert t2.bytes() == tree.bytes() and np.array_equal(t2.nodes[i].mode, tree.nodes[i].mode)
    q = quantize_mode(rows, IMG, IMG, -1.0, 1.0)
    assert q.dtype == np.uint8 and q.shape == (IMG, IMG, 3)
