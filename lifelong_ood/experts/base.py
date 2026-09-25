"""The expert interface: what a fold may add to a backbone for one node key, charged to the byte.

    e = LoRAExpert.create(backbone, key, rank=1, band=None)     # rank-1 on every Linear / Conv2d
    e = LoRAExpert.create(backbone, key, rank=1, band=32)       # only the modules at 32 px
    e = FiLMExpert.create(backbone, key, channels)              # (dgamma, dbeta) on chosen GN channels
    e = TokenExpert.create(backbone, key)                       # a conditioning vector (+ optional emb LoRA)
    e = NonparamExpert(key, mode, extent)                       # Prop-2 correction, zero weights

    with e.on()():            samples / reads see the expert at weight 1
    with e.on(r)():           ... at weight r (a float or a per-row tensor: the responsibility)
    e.bytes()                 every byte the expert stores, measured (numel * element_size + indices)
    e.parameters()            what an optimiser may move
    e.state_dict() / load_state_dict()

Every expert is an EXACT no-op when it is off, and a fresh one is an exact no-op when it is on
(tests/test_experts_exact_noop.py).  Experts on one model share one gate state, so several nodes can be
on at once with their own weights; that is how a composition of folded prototypes is sampled.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn

from .lora import KeyedLoRA, add_node, drop_node, inject_keyed_lora, keys_on

DEFAULT_EXCLUDE = ("temb", "cemb", "attr")      # the noise-level path and the class table are never adapted


def tensor_bytes(t):
    return int(t.numel() * t.element_size())


def _gate_state(model):
    st = getattr(model, "_keyed_state", None)
    if st is None:
        st = {"on": frozenset()}
        inject_keyed_lora(model, st, exclude=DEFAULT_EXCLUDE)
    return st


class Expert:
    """Base class; subclasses set `kind` and implement parameters / bytes / state_dict."""
    kind = "abstract"

    def __init__(self, key, model):
        self.key, self.model = str(key), model
        self.state = _gate_state(model)

    def on(self, w=1.0):
        """-> zero-arg callable returning a context manager: this expert on at weight w, every other off."""
        return keys_on(self.state, {self.key: w})

    @staticmethod
    def several(experts, weights=None):
        """Several experts on at once: `weights` is key -> weight (default 1 each)."""
        exps = list(experts)
        st = exps[0].state
        for e in exps[1:]:
            if e.state is not st:
                raise ValueError("experts on different models cannot share one gate")
        return keys_on(st, dict(weights) if weights is not None else {e.key: 1.0 for e in exps})

    def parameters(self):
        raise NotImplementedError

    def bytes(self):
        return sum(tensor_bytes(p) for p in self.parameters())

    def state_dict(self):
        return {f"p{i}": p.detach().cpu().clone() for i, p in enumerate(self.parameters())}

    def load_state_dict(self, sd):
        with torch.no_grad():
            for i, p in enumerate(self.parameters()):
                p.copy_(sd[f"p{i}"].to(p.device))

    def emb(self):
        return None

    def drop(self):
        pass

    def describe(self):
        return f"{self.kind}(key={self.key}, params={sum(p.numel() for p in self.parameters()):,}, bytes={self.bytes():,})"


# ── resolution bands on our UNet ─────────────────────────────────────────────
def module_resolutions(model):
    """qualified name -> the spatial resolution (px) that module operates at, for `neural.unet.UNet`.
    Returns {} for models without that structure (bands then need explicit name lists)."""
    img = int(getattr(model, "img", 0) or 0)
    if img == 0 or not hasattr(model, "down"):
        return {}
    res = {}
    L = len(model.down)
    for n, _ in model.named_modules():
        parts = n.split(".")
        if not n:
            continue
        head = parts[0]
        if head == "stem" or head == "out":
            res[n] = img
        elif head == "down" and len(parts) > 1 and parts[1].isdigit():
            res[n] = img // (2 ** int(parts[1]))
        elif head == "downsample" and len(parts) > 1 and parts[1].isdigit():
            res[n] = img // (2 ** int(parts[1]))
        elif head in ("mid1", "mida", "mid2"):
            res[n] = img // (2 ** (L - 1))
        elif head == "up" and len(parts) > 1 and parts[1].isdigit():
            res[n] = img // (2 ** (L - 1 - int(parts[1])))
        elif head == "upsample" and len(parts) > 1 and parts[1].isdigit():
            res[n] = img // (2 ** (L - 1 - int(parts[1])))
    return res


def band_names(model, band):
    """Names of the LoRA banks at resolution `band` (an int in px, an iterable of px, or a callable on
    the qualified name).  Requires the model to be wrapped (banks carry `.name`)."""
    banks = getattr(model, "_keyed_banks", None) or []
    if callable(band):
        return [b.name for b in banks if band(b.name)]
    bands = {int(band)} if isinstance(band, (int, np.integer)) else {int(x) for x in band}
    res = module_resolutions(model)
    return [b.name for b in banks if res.get(b.name) in bands]


class LoRAExpert(Expert):
    """Rank-r deltas on every wrapped Linear / Conv2d, or only on the banks named in `band`."""
    kind = "lora"

    def __init__(self, key, model, banks, rank):
        super().__init__(key, model)
        self.banks, self.rank = list(banks), int(rank)

    @classmethod
    def create(cls, backbone, key, rank=1, band=None, exclude=DEFAULT_EXCLUDE):
        model = backbone.model if hasattr(backbone, "model") else backbone
        st = getattr(model, "_keyed_state", None) or {"on": frozenset()}
        all_banks = inject_keyed_lora(model, st, exclude=exclude)
        if band is None:
            banks = all_banks
        else:
            names = set(band_names(model, band))
            banks = [b for b in all_banks if b.name in names]
            if not banks:
                raise ValueError(f"LoRAExpert: no banks in band {band!r}")
        add_node(banks, str(key), rank)
        return cls(key, model, banks, rank)

    def parameters(self):
        return [q for b in self.banks for q in list(b.A[self.key].parameters()) + list(b.B[self.key].parameters())]

    def bytes(self):
        return super().bytes() + 2 * len(self.banks)      # the bank ids the expert lives on (uint16 each)

    def drop(self):
        drop_node(self.banks, self.key)


class KeyedFiLM(nn.Module):
    """A GroupNorm with per-key (dgamma, dbeta) on a subset of channels, weighted by the gate."""

    def __init__(self, base, state):
        super().__init__()
        self.base, self.state = base, state
        self.idx = {}
        self.dgamma, self.dbeta = nn.ParameterDict(), nn.ParameterDict()

    def add(self, key, channels):
        c = torch.as_tensor(np.asarray(channels, np.int64)).reshape(-1)
        dev = self.base.weight.device
        self.idx[key] = c.to(dev)
        self.dgamma[key] = nn.Parameter(torch.zeros(len(c), device=dev))
        self.dbeta[key] = nn.Parameter(torch.zeros(len(c), device=dev))
        return [self.dgamma[key], self.dbeta[key]]

    def drop(self, key):
        del self.idx[key], self.dgamma[key], self.dbeta[key]

    def forward(self, x):
        out = self.base(x)
        on = self.state["on"]
        items = on.items() if isinstance(on, dict) else ((k, 1.0) for k in on)
        for k, w in items:
            if k not in self.idx:
                continue
            idx = self.idx[k]
            g, b = self.dgamma[k], self.dbeta[k]
            shape = (1, -1) + (1,) * (out.dim() - 2)
            if torch.is_tensor(w):
                w = w.to(out.dtype).reshape(-1, *([1] * (out.dim() - 1)))
            sub = out[:, idx]
            out = out.clone()
            out[:, idx] = sub * (1.0 + w * g.reshape(shape)) + w * b.reshape(shape)
        return out


class FiLMExpert(Expert):
    """Per-channel affine on chosen GroupNorm channels: 8 bytes per channel of parameters plus 2 bytes
    per channel of index (uint16)."""
    kind = "film"

    def __init__(self, key, model, wrappers):
        super().__init__(key, model)
        self.wrappers = list(wrappers)

    @staticmethod
    def wrap_groupnorms(model, state):
        if getattr(model, "_film_wrappers", None) is not None:
            return model._film_wrappers
        ws = []
        for mname, mod in list(model.named_modules()):
            if isinstance(mod, KeyedFiLM):
                continue
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.GroupNorm):
                    w = KeyedFiLM(child, state)
                    w.name = f"{mname}.{cname}" if mname else cname
                    setattr(mod, cname, w)
                    ws.append(w)
        model._film_wrappers = ws
        return ws

    @classmethod
    def create(cls, backbone, key, channels="all"):
        """`channels`: "all", an int k (the first k channels of every norm; the fold operator passes
        chosen indices), or a dict wrapper-name -> index array (norms absent from the dict get none)."""
        model = backbone.model if hasattr(backbone, "model") else backbone
        st = _gate_state(model)
        ws = cls.wrap_groupnorms(model, st)
        used = []
        for w in ws:
            n = w.base.num_channels
            if channels == "all":
                idx = np.arange(n)
            elif isinstance(channels, (int, np.integer)):
                idx = np.arange(min(int(channels), n))
            else:
                if w.name not in channels:
                    continue
                idx = np.asarray(channels[w.name], np.int64)
            if len(idx) == 0:
                continue
            w.add(str(key), idx)
            used.append(w)
        if not used:
            raise ValueError("FiLMExpert: no channels selected")
        return cls(key, model, used)

    def parameters(self):
        return [q for w in self.wrappers for q in (w.dgamma[self.key], w.dbeta[self.key])]

    def bytes(self):
        return super().bytes() + sum(2 * len(w.idx[self.key]) for w in self.wrappers) + 2 * len(self.wrappers)

    def drop(self):
        for w in self.wrappers:
            w.drop(self.key)


class TokenExpert(Expert):
    """A prototype token: one conditioning vector fed through `emb_override`, optionally with a rank-r
    LoRA on the ResBlock `.emb` projections so the token can reach channels the frozen path cannot."""
    kind = "token"

    def __init__(self, key, model, token, inner=None):
        super().__init__(key, model)
        self.token, self.inner = token, inner

    @classmethod
    def create(cls, backbone, key, rank=0, init=None):
        model = backbone.model if hasattr(backbone, "model") else backbone
        d = int(backbone.emb_dim if hasattr(backbone, "emb_dim") else model.emb_dim)
        dev = next(model.parameters()).device
        tok = nn.Parameter(torch.zeros(d, device=dev) if init is None else torch.as_tensor(init, dtype=torch.float32, device=dev).clone())
        inner = LoRAExpert.create(backbone, f"{key}#emb", rank=rank, band=lambda n: n.endswith(".emb")) if rank > 0 else None
        e = cls(key, model, tok, inner)
        return e

    def on(self, w=1.0):
        if self.inner is None:
            return keys_on(self.state, {})
        return keys_on(self.state, {self.inner.key: w})

    def emb(self):
        return self.token

    def parameters(self):
        return [self.token] + (self.inner.parameters() if self.inner is not None else [])

    def bytes(self):
        return tensor_bytes(self.token) + (self.inner.bytes() if self.inner is not None else 0)

    def drop(self):
        if self.inner is not None:
            self.inner.drop()


class NonparamExpert(Expert):
    """Proposition 2 applied literally: no weights.  The node's Gaussian (mode m, diagonal extent S in
    x0 space) supplies the conditional eps of Zekun's Eq. 14, and the sampler mixes it with the base
    prediction at the responsibility:  eps = e0 + r (eps_node - e0).  Bytes are the node's, charged
    under `tree`, so `bytes()` is 0 here."""
    kind = "nonparam"

    def __init__(self, key, mode, extent, model=None):
        self.key = str(key)
        self.model = model
        self.state = _gate_state(model) if model is not None else {"on": frozenset()}
        self.mode = np.asarray(mode, np.float32).reshape(-1)
        self.extent = np.asarray(extent, np.float32).reshape(-1)

    def parameters(self):
        return []

    def bytes(self):
        return 0

    def guide(self, r_fn=None):
        """-> (eps_fn, w_fn) for `backbones.sampler.sample(..., guide=...)`.  `r_fn(x_rows, u)` gives
        the per-row responsibility; None forces the node (r = 1)."""
        from ..backbones.schedule import cosine_abar
        m, S = self.mode, self.extent

        def eps_fn(x_rows, u):
            a = float(cosine_abar(float(u)))
            mt = torch.as_tensor(m, dtype=torch.float32, device=x_rows.device)
            St = torch.as_tensor(S, dtype=torch.float32, device=x_rows.device)
            return (1.0 - a) ** 0.5 * (x_rows - a ** 0.5 * mt) / (a * St + (1.0 - a))

        def w_fn(x_rows, u):
            return 1.0 if r_fn is None else r_fn(x_rows, u)
        return eps_fn, w_fn

    def state_dict(self):
        return {"mode": torch.from_numpy(self.mode), "extent": torch.from_numpy(self.extent)}

    def load_state_dict(self, sd):
        self.mode, self.extent = sd["mode"].numpy(), sd["extent"].numpy()
