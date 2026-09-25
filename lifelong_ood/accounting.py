"""One function charges every byte a method keeps: `charge(...)` -> the `bytes` block a ledger row needs.

The audit of 2026-09-25 found that the c-series kept 8 self-generated samples and a `copy.deepcopy`
of the 6.8M-parameter backbone per stored concept and charged neither.  Here nothing is charged by a
formula: every tensor is measured (`numel * element_size`), codes are counted at their stored width,
a teacher that aliases the frozen base costs nothing and a teacher that does not costs its full size,
and in strict mode (the protocol for our method) a non-aliasing teacher copy RAISES.
"""
from __future__ import annotations

import numpy as np
import torch

from .ledger import BYTES_KEYS

CODE_ID_BYTES, CODE_W_BYTES = 2, 2         # uint16 node id + fp16 weight per (level, node) entry


class HiddenStateError(RuntimeError):
    pass


def tensor_bytes(t):
    if torch.is_tensor(t):
        return int(t.numel() * t.element_size())
    a = np.asarray(t)
    return int(a.size * a.dtype.itemsize)


def code_bytes(code):
    """A code is a sequence of (node_id, weight) entries (one or more per level), an int, or an object
    with .bytes()."""
    if hasattr(code, "bytes"):
        return int(code.bytes())
    if isinstance(code, (int, np.integer)):
        return int(code)
    return int(len(list(code)) * (CODE_ID_BYTES + CODE_W_BYTES))


def _params_of(obj):
    if isinstance(obj, torch.nn.Module):
        return list(obj.parameters()) + list(obj.buffers())
    if isinstance(obj, dict):
        return [v for v in obj.values() if torch.is_tensor(v)]
    raise TypeError(f"teacher of type {type(obj).__name__} is not a module or a state dict")


def teacher_bytes(teacher, base=None):
    """0 when every tensor of `teacher` aliases a tensor of `base` (a view on the frozen model, not a
    copy); otherwise the full measured size."""
    ts = _params_of(teacher)
    if base is not None:
        ptrs = {p.data_ptr() for p in _params_of(base)}
        if ts and all(t.data_ptr() in ptrs for t in ts):
            return 0
    return int(sum(tensor_bytes(t) for t in ts))


def charge(experts=(), tree=None, codes=(), samples=(), teachers=(), base=None, strict=True, extras=()):
    """-> dict with BYTES_KEYS (experts, tree, codes, samples, teacher_copies, total) plus `items`.

    experts   objects with .bytes() (lifelong_ood.experts) or ints
    tree      an object with .bytes() or an int (the prototype tree, charged once)
    codes     stored ADDRESS codes (see code_bytes)
    samples   stored arrays / tensors (anchors, replay, exemplars, query images), charged as held
    teachers  persisted model copies or state dicts; aliasing `base` costs 0; else full, or raise (strict)
    extras    (name, bytes) pairs for anything else a baseline stores (e.g. an EWC Fisher)
    """
    items = {}
    e = int(sum(int(x.bytes()) if hasattr(x, "bytes") else int(x) for x in experts))
    t = int(tree.bytes()) if hasattr(tree, "bytes") else int(tree or 0)
    c = int(sum(code_bytes(x) for x in codes))
    s = int(sum(tensor_bytes(x) for x in samples))
    tc = 0
    for i, th in enumerate(teachers):
        b = teacher_bytes(th, base)
        if b and strict:
            raise HiddenStateError(f"teacher #{i} is a {b:,}-byte copy that does not alias the base; the "
                                   "protocol forbids persisted teacher copies (use base + frozen experts), "
                                   "or pass strict=False for a baseline that keeps one")
        tc += b
    x = int(sum(int(b) for _, b in extras))
    for name, b in extras:
        items[str(name)] = int(b)
    out = dict(experts=e, tree=t, codes=c, samples=s, teacher_copies=tc, total=e + t + c + s + tc + x)
    assert set(BYTES_KEYS) <= set(out)
    out["items"] = items
    return out
