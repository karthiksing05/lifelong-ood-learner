"""The sampler contract: ONE sampler, driven by the Backbone handle, deterministic, and moved only by
an expert that is on.  Runs on CPU with an 8x8 UNet in seconds."""
import ast
import inspect
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.backbones import loader, sampler                    # noqa: E402
from lifelong_ood.backbones.loader import Backbone, load_backbone     # noqa: E402
from lifelong_ood.backbones.sampler import ddim_sample, sample        # noqa: E402
from lifelong_ood.experts.lora import add_node, inject_keyed_lora, keys_on  # noqa: E402
from lifelong_ood.neural.unet import UNet                             # noqa: E402

NEW_PACKAGES = ["backbones", "experts", "operators", "tree", "streams"]


def _tiny(seed=0, n_classes=3):
    torch.manual_seed(seed)
    m = UNet(n_classes=n_classes, base=32, mults=(1,), edim=32, attn=(), img=8)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m.eval()


def _handle(model, layout="permuted"):
    return Backbone(model=model, kind="unet", layout=layout, img=8, clip=(-1.0, 1.0), space="model",
                    null_id=model.null_id, emb_dim=model.emb_dim,
                    n_params=sum(p.numel() for p in model.parameters()), device="cpu")


def test_exactly_one_public_sampler_in_the_new_packages():
    found = {}
    for pkg in NEW_PACKAGES:
        d = os.path.join(ROOT, "lifelong_ood", pkg)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(d, f), encoding="utf-8").read())
            for n in tree.body:
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_") and "sample" in n.name:
                    found.setdefault(n.name, []).append(f"{pkg}/{f}")
    assert set(found) == {"sample", "ddim_sample"}, found
    assert found["sample"] == ["backbones/sampler.py"] and found["ddim_sample"] == ["backbones/sampler.py"]
    # the legacy alias and the public sampler share one core
    assert "_ddim_core(" in inspect.getsource(sample) and "_ddim_core(" in inspect.getsource(ddim_sample)


def test_sampler_takes_conventions_only_from_the_handle():
    params = inspect.signature(sample).parameters
    for banned in ("layout", "img", "clip", "device"):
        assert banned not in params, f"sample() must not accept {banned}; it comes from the Backbone"
    assert list(params)[:2] == ["backbone", "n"]


def test_deterministic_given_seed_and_equal_to_the_legacy_call():
    bk = _handle(_tiny())
    a = sample(bk, 4, steps=3, seed=7)
    b = sample(bk, 4, steps=3, seed=7)
    c = sample(bk, 4, steps=3, seed=8)
    assert a.shape == (4, 3 * 8 * 8) and np.array_equal(a, b) and not np.array_equal(a, c)
    legacy = ddim_sample(bk.model, 4, steps=3, seed=7, layout="permuted", img=8, device="cpu", clip=(-1.0, 1.0))
    assert np.array_equal(a, legacy)


def test_samples_move_iff_a_nontrivial_expert_is_on():
    model = _tiny()
    bk = _handle(model)
    base = sample(bk, 4, steps=3, seed=1)
    st = {"on": set()}
    banks = inject_keyed_lora(model, st)
    add_node(banks, "k", rank=2)                    # B is zero-initialised: an exact no-op even when on
    assert np.array_equal(sample(bk, 4, steps=3, seed=1, gate=lambda: keys_on(st, {"k"})()), base)
    with torch.no_grad():
        for w in banks:
            b = w.B["k"]
            (b.weight if hasattr(b, "weight") else b).normal_(0, 0.5)   # B is a layer in KeyedLoRA
    on = sample(bk, 4, steps=3, seed=1, gate=lambda: keys_on(st, {"k"})())
    off = sample(bk, 4, steps=3, seed=1)
    assert not np.array_equal(on, base) and np.array_equal(off, base)


def test_eps_matches_a_direct_forward():
    bk = _handle(_tiny())
    X = np.random.default_rng(0).standard_normal((5, bk.dim)).astype(np.float32)
    e = bk.eps(X, 0.5, cls=1)
    from lifelong_ood.backbones.layout import to_flat, to_img
    xt = to_img(torch.from_numpy(X), 8, "permuted")
    ref = to_flat(bk.model(xt, torch.full((5,), 0.5), c=torch.full((5,), 1, dtype=torch.long)), "permuted")
    assert torch.allclose(e, ref, atol=1e-6)


def test_loader_records_layout_and_sizes(tmp_path):
    model = _tiny()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    arch = dict(base=32, mults=(1,), edim=32, attn=(), img=8)
    for layout, expect in ((None, "permuted"), ("hwc", "hwc")):
        ck = dict(ema=sd, n_classes=3, arch=arch, remap={0: 0, 1: 1, 2: 2})
        if layout:
            ck["layout"] = layout
        p = tmp_path / f"tiny_{expect}.pt"
        torch.save(ck, p)
        bk = load_backbone(str(p), device="cpu")
        assert bk.kind == "unet" and bk.layout == expect and bk.img == 8 and bk.clip == (-1.0, 1.0)
        assert bk.space == "model" and bk.null_id == 3 and bk.emb_dim == 32
        assert bk.n_params == sum(v.numel() for v in sd.values())
        assert bk.sha256 and len(bk.sha256) == 64 and bk.meta["remap"] == {0: 0, 1: 1, 2: 2}
        assert bk.class_table().shape == (3, 32)
        assert np.array_equal(sample(bk, 2, steps=2, seed=0), sample(load_backbone(str(p), device="cpu"), 2, steps=2, seed=0))


def test_row_space_round_trip():
    bk = _handle(_tiny())
    X01 = np.random.default_rng(1).random((3, bk.dim)).astype(np.float32)
    assert np.allclose(bk.rows_to_data(bk.rows_from_data(X01)), X01, atol=1e-6)
    ve = Backbone(model=bk.model, kind="ve", layout="hwc", img=8, clip=(0.0, 1.0), space="data", null_id=3,
                  emb_dim=32, n_params=1, device="cpu")
    assert np.array_equal(ve.rows_from_data(X01), X01)
