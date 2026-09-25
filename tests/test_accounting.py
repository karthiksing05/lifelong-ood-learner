"""charge() measures every byte and refuses hidden teacher copies.  CPU, seconds."""
import copy
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                       # noqa: E402
from lifelong_ood.accounting import HiddenStateError, charge, code_bytes, tensor_bytes  # noqa: E402
from lifelong_ood.backbones.loader import Backbone                    # noqa: E402
from lifelong_ood.experts.base import LoRAExpert                      # noqa: E402
from lifelong_ood.neural.unet import UNet                             # noqa: E402


def _bk():
    torch.manual_seed(0)
    m = UNet(n_classes=3, base=32, mults=(1,), edim=32, attn=(), img=8).eval()
    return Backbone(model=m, kind="unet", layout="permuted", img=8, clip=(-1.0, 1.0), space="model",
                    null_id=3, emb_dim=32, n_params=sum(p.numel() for p in m.parameters()), device="cpu")


def test_charge_measures_experts_samples_and_teacher_copies():
    bk = _bk()
    base_params = sum(p.numel() for p in bk.model.parameters())
    e = LoRAExpert.create(bk, "k", rank=1)
    anchors = [np.zeros(bk.dim, np.float32) for _ in range(8)]
    teacher = copy.deepcopy(bk.model)
    bill = charge(experts=[e], samples=anchors, teachers=[teacher], base=bk.model, strict=False)
    expect_teacher = sum(tensor_bytes(p) for p in teacher.parameters()) + sum(tensor_bytes(b) for b in teacher.buffers())
    assert bill["experts"] == e.bytes()
    assert bill["samples"] == 8 * bk.dim * 4
    assert bill["teacher_copies"] == expect_teacher and expect_teacher >= 4 * base_params
    assert bill["total"] == e.bytes() + 8 * bk.dim * 4 + expect_teacher
    assert bill["codes"] == 0 and bill["tree"] == 0


def test_strict_mode_raises_on_a_hidden_teacher_copy():
    bk = _bk()
    with pytest.raises(HiddenStateError):
        charge(teachers=[copy.deepcopy(bk.model)], base=bk.model, strict=True)
    with pytest.raises(HiddenStateError):
        charge(teachers=[{k: v.clone() for k, v in bk.model.state_dict().items()}], base=bk.model)


def test_a_teacher_that_aliases_the_base_costs_nothing():
    bk = _bk()
    bill = charge(teachers=[bk.model, bk.model.state_dict()], base=bk.model, strict=True)
    assert bill["teacher_copies"] == 0 and bill["total"] == 0


def test_codes_and_tree_and_extras():
    class T:
        def bytes(self):
            return 1234
    bill = charge(tree=T(), codes=[[(3, 0.5), (7, 0.5)], 10, [(1, 1.0)]], extras=[("fisher", 100)])
    assert code_bytes([(3, 0.5), (7, 0.5)]) == 8
    assert bill["tree"] == 1234 and bill["codes"] == 8 + 10 + 4 and bill["total"] == 1234 + 22 + 100
    assert bill["items"] == {"fisher": 100}


def test_uint8_samples_are_charged_as_held():
    assert charge(samples=[np.zeros((4, 12), np.uint8)])["samples"] == 48
    assert charge(samples=[torch.zeros(4, 12, dtype=torch.float16)])["samples"] == 96


def test_the_bill_is_a_valid_ledger_bytes_block(tmp_path):
    bill = charge(experts=[16])
    p = ledger.record("unit", metrics={}, bytes=bill, config={}, seed=0, bench="b", script="s", command="c",
                      ledger_dir=tmp_path)
    assert p.exists()
