"""KeyedLoRA: each node's delta is its own, the empty key set is the frozen model exactly, and a
node's training cannot move another node's output."""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.neural.cobweb_update import (add_node, drop_node, inject_keyed_lora,  # noqa: E402
                                               keys_on)


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1),
                         nn.Flatten(), nn.Linear(3 * 8 * 8, 5))


def test_keyed_lora():
    base = _net().eval()
    x = torch.randn(4, 3, 8, 8)
    y0 = base(x).detach()
    m = copy.deepcopy(base)
    st = {"on": frozenset()}
    banks = inject_keyed_lora(m, st)
    assert len(banks) == 3
    pa, pb = add_node(banks, "a", 1), add_node(banks, "b", 2)
    # fresh deltas are exact no-ops, on or off
    for keys in ((), ("a",), ("b",), ("a", "b")):
        with keys_on(st, keys)():
            assert torch.equal(m(x), y0)
    # train node a only
    opt = torch.optim.SGD(pa, lr=0.1)
    for _ in range(20):
        with keys_on(st, ("a",))():
            loss = ((m(x) - 1.0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with keys_on(st, ("a",))():
        ya = m(x).detach()
    assert not torch.allclose(ya, y0)
    # off, and b alone, are still the frozen model exactly; b's params never received a gradient
    assert torch.equal(m(x), y0)
    with keys_on(st, ("b",))():
        assert torch.equal(m(x), y0)
    assert all(q.grad is None or torch.count_nonzero(q.grad) == 0 for q in pb)
    # both on = a's delta (b's is zero); dropping a restores the frozen model under a
    with keys_on(st, ("a", "b"))():
        assert torch.allclose(m(x), ya)
    drop_node(banks, "a")
    with keys_on(st, ("a",))():
        assert torch.equal(m(x), y0)
    # the context manager restores the previous key set
    assert st["on"] == frozenset()


if __name__ == "__main__":
    test_keyed_lora()
    print("ok")
