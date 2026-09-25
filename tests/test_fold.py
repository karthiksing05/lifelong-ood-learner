"""FOLD on the tiny UNet: a tree from the model's own samples, an analytic gate that is ~0 on the
model's own samples and ~1 on a novel arrival once its node exists, gated losses that run, and a
per-step gate in the sampler.  CPU, seconds."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.backbones.layout import to_img                       # noqa: E402
from lifelong_ood.backbones.loader import Backbone                    # noqa: E402
from lifelong_ood.backbones.sampler import sample                     # noqa: E402
from lifelong_ood.experts.base import LoRAExpert, TokenExpert         # noqa: E402
from lifelong_ood.neural.unet import UNet                             # noqa: E402
from lifelong_ood.operators.address import calibrate                  # noqa: E402
from lifelong_ood.operators.fold import Gate, anchor_loss_natural, arrival_loss, create_nodes  # noqa: E402
from lifelong_ood.tree.build import build                             # noqa: E402


def _bk(seed=0):
    torch.manual_seed(seed)
    m = UNet(n_classes=3, base=32, mults=(1,), edim=32, attn=(), img=8)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
        m.cemb.weight[m.null_id].zero_()
    m.eval()
    return Backbone(model=m, kind="unet", layout="permuted", img=8, clip=(-1.0, 1.0), space="model",
                    null_id=3, emb_dim=32, n_params=sum(p.numel() for p in m.parameters()), device="cpu")


@pytest.fixture(scope="module")
def world():
    bk = _bk()
    own = sample(bk, 24, steps=4, seed=0)
    tree = build(bk, u_grid=(0.10, 0.40, 0.70), samples=own, seed=0, iters=10, lr=0.05, caps={0.10: 6, 0.40: 4, 0.70: 3},
                 verbose=False)
    cal = calibrate(tree, sample(bk, 16, steps=4, seed=1), q=0.95)   # held-out own samples
    return bk, own, tree, cal


def test_create_nodes_and_gate_polarity(world):
    bk, own, tree, cal = world
    novel = np.full((4, bk.dim), 0.9, np.float32)               # a flat bright image: nothing like the samples
    n0 = len(tree.nodes)
    ids = create_nodes(tree, cal, novel, key="k")
    assert ids and len(tree.nodes) == n0 + len(ids) and all(tree.nodes[i].expert == "k" for i in ids)
    gate = Gate(tree, ids, "analytic")
    lvl = tree.nodes[ids[0]].level
    u = tree.u_grid[lvl]
    from lifelong_ood.backbones.schedule import cosine_abar
    a = float(cosine_abar(u))
    rng = np.random.default_rng(0)
    xt_new = np.sqrt(a) * novel + np.sqrt(1 - a) * rng.standard_normal(novel.shape).astype(np.float32)
    xt_own = np.sqrt(a) * own[:8] + np.sqrt(1 - a) * rng.standard_normal((8, bk.dim)).astype(np.float32)
    w_new = gate.weights(torch.as_tensor(xt_new), u).mean()
    w_own = gate.weights(torch.as_tensor(xt_own), u).mean()
    # polarity: the node claims the arrival and mostly not the model's own samples.  The leak on own
    # samples at a coarse level is what the stop rule's S2 measures; it is not zero for a 3-node level.
    assert w_new > 0.9 and w_own < 0.35 and w_new - w_own > 0.5, (w_new, w_own)
    assert Gate(tree, ids, "forced").weights(torch.zeros(3, bk.dim), u).tolist() == [1, 1, 1]
    assert Gate(tree, ids, "label").weights(torch.zeros(3, bk.dim), u, flag=[1, 0, 1]).tolist() == [1, 0, 1]


def test_gated_losses_run_and_move_only_the_expert(world):
    bk, own, tree, cal = world
    novel = np.full((4, bk.dim), 0.9, np.float32)
    ids = create_nodes(tree, cal, novel, key="k2")
    for factory in (lambda: LoRAExpert.create(bk, "k2", rank=1), lambda: TokenExpert.create(bk, "k2t", rank=1)):
        e = factory()
        gate = Gate(tree, ids, "analytic")
        x0 = to_img(torch.as_tensor(novel), 8, "permuted")
        xa = to_img(torch.as_tensor(own[:8]), 8, "permuted")
        g = torch.Generator().manual_seed(0)
        base_sd = {k: v.clone() for k, v in bk.model.state_dict().items() if ".base." in k or not any(t in k for t in ("A.", "B.", "dgamma", "dbeta"))}
        opt = torch.optim.Adam(e.parameters(), lr=1e-2)
        for _ in range(3):
            loss = arrival_loss(bk.model, e, gate, x0, "permuted", gen=g) + 10.0 * anchor_loss_natural(bk.model, e, gate, xa, "permuted", gen=g)
            opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss)
        for k, v in bk.model.state_dict().items():
            if k in base_sd:
                assert torch.equal(v, base_sd[k]), f"base weight {k} moved"
        S = sample(bk, 3, steps=3, seed=1, step_gate=gate.step_gate(e))
        assert S.shape == (3, bk.dim)
        e.drop()
