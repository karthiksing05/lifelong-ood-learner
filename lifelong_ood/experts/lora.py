"""Rank-r LoRA experts on every Conv/Linear, switched on by a key set (KeyedLoRA) or one boolean (GatedLoRA).

A KeyedLoRA is an exact no-op when its key is off (`tests/test_keyed_lora.py`); one adapter per node so a
later node cannot overwrite an earlier one (C2's FLO fault).  The gate is about to become a per-key
WEIGHT (the tree's responsibility r_k) instead of a set membership, so `out += w_k * B_k(A_k(x))`.
"""
import contextlib

import numpy as np
import torch
import torch.nn as nn


class GatedLoRA(nn.Module):
    """A low-rank delta on a frozen layer, applied only while `state["on"]` is True.

    A is Kaiming, B is zero, so a fresh delta is an EXACT no-op -- the same initialisation the
    adapter bank uses.  Gated rather than path-keyed: one delta per new factor VALUE, switched on
    for every concept that uses the value and off for everything else.
    """

    def __init__(self, base, rank, state):
        super().__init__()
        self.base, self.state = base, state
        if isinstance(base, nn.Conv2d):
            self.A = nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding,
                               dilation=base.dilation, bias=False)
            self.B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        else:
            self.A = nn.Linear(base.in_features, rank, bias=False)
            self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        out = self.base(x)
        return out + self.B(self.A(x)) if self.state["on"] else out


def inject_gated_lora(model, rank, state, exclude=()):
    """Wrap every nn.Linear / nn.Conv2d of `model` in place, except modules whose qualified name
    contains any of `exclude` (on a pretrained UNet, adapting the noise-level path -- `time_emb` --
    destroys the model, RESULTS 12).  -> the list of new parameters."""
    params = []
    for mname, mod in list(model.named_modules()):
        if isinstance(mod, GatedLoRA):
            continue
        for cname, child in list(mod.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if any(e in full for e in exclude):
                continue
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                w = GatedLoRA(child, rank, state).to(child.weight.device)
                setattr(mod, cname, w)
                params += list(w.A.parameters()) + list(w.B.parameters())
    return params


def gate_on(state):
    """-> a zero-argument callable returning a context manager that switches `state` on."""
    @contextlib.contextmanager
    def _on():
        prev = state["on"]
        state["on"] = True
        try:
            yield
        finally:
            state["on"] = prev
    return _on


class KeyedLoRA(nn.Module):
    """A BANK of low-rank deltas on one frozen layer, one per node key; the deltas whose keys are in
    `state["on"]` (a set) are added, and with the set empty the layer is the frozen one exactly.

    `GatedLoRA` holds one delta, so every node that switched it on trained the same weights: in C2 a
    node created on the wrong factor and the right node would share one adapter.  Here each value
    node -- or each open hypothesis of a Cobweb leaf -- owns its delta, and a concept's forward pass
    switches on exactly the nodes in its coordinates.
    """

    def __init__(self, base, state):
        super().__init__()
        self.base, self.state = base, state
        self.A, self.B = nn.ModuleDict(), nn.ModuleDict()

    def add(self, key, rank):
        base = self.base
        if isinstance(base, nn.Conv2d):
            A = nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding,
                          dilation=base.dilation, bias=False)
            B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        else:
            A = nn.Linear(base.in_features, rank, bias=False)
            B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(A.weight, a=5 ** 0.5)
        nn.init.zeros_(B.weight)
        dev = base.weight.device
        self.A[key], self.B[key] = A.to(dev), B.to(dev)
        return list(self.A[key].parameters()) + list(self.B[key].parameters())

    def drop(self, key):
        del self.A[key]
        del self.B[key]

    def forward(self, x):
        out = self.base(x)
        on = self.state["on"]
        items = on.items() if isinstance(on, dict) else ((k, 1.0) for k in on)
        for k, w in items:
            if k not in self.A:
                continue
            Ak = self.A[k]
            if x.dtype == Ak.weight.dtype:
                d = self.B[k](Ak(x))
            else:                                       # a bf16 base (pi0.5's expert): fp32 adapter, delta cast back
                d = self.B[k](Ak(x.to(Ak.weight.dtype))).to(out.dtype)
            if torch.is_tensor(w):                      # a per-row responsibility, [B]
                w = w.to(d.dtype).reshape(-1, *([1] * (d.dim() - 1)))
                out = out + w * d
            elif float(w) != 1.0:
                out = out + float(w) * d
            else:                                       # w == 1: bit-identical to the unweighted gate
                out = out + d
        return out


def inject_keyed_lora(model, state, exclude=()):
    """Wrap every nn.Linear / nn.Conv2d of `model` (bar `exclude`) in an empty KeyedLoRA bank.
    -> the list of banks; `add_node` / `drop_node` then grow and shrink every bank together.
    Idempotent: a model already wrapped returns its existing banks (`model._keyed_banks`)."""
    if getattr(model, "_keyed_banks", None) is not None:
        if model._keyed_state is not state:
            raise RuntimeError("inject_keyed_lora: the model is already wrapped with a different gate state")
        return model._keyed_banks
    banks = []
    for mname, mod in list(model.named_modules()):
        if isinstance(mod, (GatedLoRA, KeyedLoRA)):
            continue
        for cname, child in list(mod.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if any(e in full for e in exclude):
                continue
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                w = KeyedLoRA(child, state)
                w.name = full
                setattr(mod, cname, w)
                banks.append(w)
    model._keyed_banks, model._keyed_state = banks, state
    return banks


def add_node(banks, key, rank):
    """A fresh (exact no-op) delta for node `key` in every bank.  -> its parameters."""
    return [q for b in banks for q in b.add(key, rank)]


def drop_node(banks, key):
    for b in banks:
        b.drop(key)


def keys_on(state, keys):
    """-> a zero-argument callable returning a context manager that switches exactly `keys` on.
    `keys` is a set (weight 1 each) or a dict key -> weight (a float or a per-row tensor)."""
    keys = dict(keys) if isinstance(keys, dict) else frozenset(keys)

    @contextlib.contextmanager
    def _on():
        prev = state["on"]
        state["on"] = keys
        try:
            yield
        finally:
            state["on"] = prev
    return _on


