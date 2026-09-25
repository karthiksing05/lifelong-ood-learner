"""decompose / address on the analytic mixture: a planted mode is explained by itself with a code of
one dominant node, a point between modes is decomposed into both, an off-manifold point fails
membership at the fine levels (depth of novelty > 0), and the teacher's eps is Zekun's Eq. 14."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.backbones.schedule import cosine_abar                # noqa: E402
from lifelong_ood.operators.address import calibrate, depth_of_novelty, explained  # noqa: E402
from lifelong_ood.operators.decompose import Code, decompose, guide, teacher_from_code  # noqa: E402
from lifelong_ood.tree.build import build                              # noqa: E402
from test_tree import IMG, D, MixtureBackbone, _planted, _samples      # noqa: E402


def _tree(seed=0):
    means = _planted(seed)
    X, k = _samples(means, n=90, seed=seed)
    bk = MixtureBackbone(means)
    t = build(bk, u_grid=(0.10, 0.30, 0.55), samples=X, seed=0, iters=60, lr=0.05, caps={0.10: 8, 0.30: 8, 0.55: 8}, verbose=False)
    return means, X, k, bk, t


def test_decompose_explains_a_mode_with_itself_and_a_midpoint_with_two():
    means, X, k, bk, t = _tree()
    ids, M, V, _ = t.stats(0)
    code = decompose(t, means[0], K=3, levels=[0])
    assert isinstance(code, Code) and len(code.nodes) == 3 and code.bytes() == 12
    top = t.nodes[code.nodes[0]].mode
    assert np.sqrt(((top - means[0]) ** 2).mean()) < 0.08
    assert code.weights[0] == max(code.weights) and code.weights[0] > 0.7, code.weights
    # a point between two modes: the FIRST pick is one of its two neighbours (per-dimension coverage
    # is free to spend the second pick on whichever node covers the remaining dimensions best)
    mid = 0.5 * (means[0] + means[1])
    c2 = decompose(t, mid, K=2, levels=[0])
    first = int(np.sqrt(((t.nodes[c2.nodes[0]].mode - means) ** 2).mean(1)).argmin())
    assert first in (0, 1) and len(set(c2.nodes)) == 2
    near = 0.9 * means[0] + 0.1 * means[1]
    c3 = decompose(t, near, K=2, levels=[0])
    assert int(np.sqrt(((t.nodes[c3.nodes[0]].mode - means) ** 2).mean(1)).argmin()) == 0
    mu, var = teacher_from_code(t, code)
    assert mu.shape == (D,) and var.shape == (D,) and np.all(var > 0)
    assert np.sqrt(((mu - means[0]) ** 2).mean()) < 0.1


def test_membership_calibration_and_depth_of_novelty():
    means, X, k, bk, t = _tree()
    Xh, _ = _samples(means, n=60, seed=5)                     # held-out draws of the same mixture
    cal = calibrate(t, Xh, q=0.95)
    assert cal.levels and all(np.isfinite(cal.thresholds[l]) for l in cal.levels)
    ok, depth, coarsest, _ = explained(t, cal, X[:20])
    assert ok.mean() >= 0.85, ok.mean()
    novel = np.clip(means[0] * -1.0, -1, 1)            # the antipode: a colour no node has
    d, c = depth_of_novelty(t, cal, novel[None])
    assert d[0] >= 1 and c[0] >= 0
    d0, _ = depth_of_novelty(t, cal, means[0][None])
    assert d0[0] == 0


def test_guide_is_eq14_and_eq16():
    mu = np.zeros(D, np.float32)
    var = np.full(D, 0.25, np.float32)
    eps_fn, w_fn = guide(mu, var, w0=1.2, wmax=2.0)
    x = torch.ones(2, D)
    u = 0.5
    a = float(cosine_abar(u))
    expect = (1 - a) ** 0.5 * (x - a ** 0.5 * 0.0) / (a * 0.25 + (1 - a))
    assert torch.allclose(eps_fn(x, u), expect)
    assert abs(w_fn(x, u) - min(1.2 * (a * 0.25 + (1 - a)), 2.0)) < 1e-6
