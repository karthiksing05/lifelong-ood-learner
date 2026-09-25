"""Every expert type is an EXACT no-op when off, a fresh one is exact even when on, keys are isolated,
bytes are measured, and state dicts round-trip.  CPU, 8x8 UNet, seconds."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.backbones.loader import Backbone                    # noqa: E402
from lifelong_ood.backbones.sampler import sample                     # noqa: E402
from lifelong_ood.experts.base import (Expert, FiLMExpert, LoRAExpert, NonparamExpert,  # noqa: E402
                                       TokenExpert, band_names, module_resolutions, tensor_bytes)
from lifelong_ood.neural.unet import UNet                             # noqa: E402


def _bk(seed=0):
    torch.manual_seed(seed)
    m = UNet(n_classes=3, base=32, mults=(1, 2), edim=32, attn=(), img=8)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
        m.cemb.weight[m.null_id].zero_()      # the null token is the zero vector, as in training
    m.eval()
    return Backbone(model=m, kind="unet", layout="permuted", img=8, clip=(-1.0, 1.0), space="model",
                    null_id=3, emb_dim=32, n_params=sum(p.numel() for p in m.parameters()), device="cpu")


def _out(bk, e=None, w=1.0, emb=None):
    """Conditioning = the NULL token (cls=None) unless a token expert supplies `emb`; a zero token
    equals the null row of the class table, so a fresh TokenExpert is exact against this base."""
    X = np.random.default_rng(0).standard_normal((3, bk.dim)).astype(np.float32)
    gate = None if e is None else e.on(w)
    return bk.eps(X, 0.4, cls=None, emb=emb, gate=gate)


def _perturb(e, scale=0.5):
    with torch.no_grad():
        for p in e.parameters():
            p.normal_(0, scale)


FACTORIES = {
    "lora-all": lambda bk, k: LoRAExpert.create(bk, k, rank=1),
    "lora-band8": lambda bk, k: LoRAExpert.create(bk, k, rank=1, band=8),
    "film-8ch": lambda bk, k: FiLMExpert.create(bk, k, channels=8),
    "film-all": lambda bk, k: FiLMExpert.create(bk, k, channels="all"),
    "token": lambda bk, k: TokenExpert.create(bk, k),
    "token+emb-lora": lambda bk, k: TokenExpert.create(bk, k, rank=1),
}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_fresh_expert_is_exact_and_off_is_exact_after_training(name):
    bk = _bk()
    base = _out(bk)
    e = FACTORIES[name](bk, "k")
    emb = e.emb()
    assert torch.equal(_out(bk, e, 1.0, emb=emb), base), "a fresh expert must be an exact no-op when on"
    assert torch.equal(_out(bk), base)
    _perturb(e)
    on = _out(bk, e, 1.0, emb=emb)
    assert not torch.equal(on, base), "a trained expert must move the output when on"
    assert torch.equal(_out(bk), base), "off must be bit-identical to the base"
    half = _out(bk, e, 0.5, emb=emb)
    if name.startswith("lora") or name.startswith("film"):
        assert not torch.equal(half, on) and not torch.equal(half, base)
    e.drop()
    if name.startswith("lora") or name.startswith("film"):
        assert torch.equal(_out(bk, e, 1.0), base), "after drop the key no longer exists"


@pytest.mark.parametrize("name", ["lora-all", "film-8ch"])
def test_keys_are_isolated(name):
    bk = _bk()
    a, b = FACTORIES[name](bk, "a"), FACTORIES[name](bk, "b")
    _perturb(a)
    base = _out(bk)
    assert torch.equal(_out(bk, b, 1.0), base), "training a must not touch b"
    assert not torch.equal(_out(bk, a, 1.0), base)
    both = bk.eps(np.random.default_rng(0).standard_normal((3, bk.dim)).astype(np.float32), 0.4, cls=None,
                  gate=Expert.several([a, b], {"a": 1.0, "b": 1.0}))
    assert torch.equal(both, _out(bk, a, 1.0)), "b is fresh (zero), so a+b == a exactly"


def test_bytes_are_measured_not_assumed():
    bk = _bk()
    e = LoRAExpert.create(bk, "k", rank=1)
    assert e.bytes() == 4 * sum(p.numel() for p in e.parameters()) + 2 * len(e.banks)
    f = FiLMExpert.create(bk, "f", channels=8)
    assert f.bytes() == sum(8 * len(w.idx["f"]) + 2 * len(w.idx["f"]) for w in f.wrappers) + 2 * len(f.wrappers)
    t = TokenExpert.create(bk, "t")
    assert t.bytes() == 4 * 32
    n = NonparamExpert("n", np.zeros(bk.dim), np.ones(bk.dim))
    assert n.bytes() == 0 and n.parameters() == []
    band = LoRAExpert.create(bk, "b8", rank=1, band=8)
    assert 0 < band.bytes() < e.bytes()


def test_band_selection_follows_resolution():
    bk = _bk()
    LoRAExpert.create(bk, "warm", rank=1)                # wraps the model
    res = module_resolutions(bk.model)
    assert res["stem"] == 8 and res["out.2"] == 8 and res["down.1.0.0.c1"] == 4 and res["mid1.c1"] == 4
    assert res["up.0.0.0.c1"] == 4 and res["up.1.0.0.c1"] == 8
    n8, n4 = set(band_names(bk.model, 8)), set(band_names(bk.model, 4))
    assert n8 and n4 and not (n8 & n4)
    assert "stem" in n8 and "mid1.c1" in n4


def test_state_dict_round_trip():
    bk = _bk()
    e = LoRAExpert.create(bk, "k", rank=1)
    _perturb(e)
    on = _out(bk, e, 1.0)
    sd = e.state_dict()
    e.drop()
    e2 = LoRAExpert.create(bk, "k", rank=1)
    assert torch.equal(_out(bk, e2, 1.0), _out(bk))
    e2.load_state_dict(sd)
    assert torch.equal(_out(bk, e2, 1.0), on)


def test_nonparam_guide_forces_the_node():
    bk = _bk()
    m = np.full(bk.dim, 0.3, np.float32)
    n = NonparamExpert("n", m, np.full(bk.dim, 1e-3, np.float32))
    S = sample(bk, 4, steps=4, seed=0, guide=n.guide(None))
    assert np.abs(S - m).mean() < 0.05, "with r forced to 1 and a tight extent the sampler lands on the mode"


def test_per_row_responsibility_gate():
    bk = _bk()
    e = LoRAExpert.create(bk, "k", rank=1)
    _perturb(e)
    X = np.random.default_rng(0).standard_normal((3, bk.dim)).astype(np.float32)
    base = bk.eps(X, 0.4, cls=1)
    on = bk.eps(X, 0.4, cls=1, gate=e.on(1.0))
    mixed = bk.eps(X, 0.4, cls=1, gate=e.on(torch.tensor([0.0, 1.0, 0.0])))
    assert torch.equal(mixed[0], base[0]) and torch.allclose(mixed[1], on[1], atol=1e-6) and torch.equal(mixed[2], base[2])
