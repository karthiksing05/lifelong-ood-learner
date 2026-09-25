"""ADAPT: low-rank experts on pi0.5's ACTION EXPERT, trained by flow matching with the VLM prefix cached.

Knowledge insulation (arXiv 2505.23705) means the action loss never shaped the VLM: an expert that lives in the
action expert reproduces pi0.5's own training regime and leaves the prefix untouched, so the prefix of every
training observation is computed ONCE (no_grad) and each step runs only the 300M expert with the adapters.

    state, banks = inject(bk)                               # KeyedLoRA banks on the expert's Linear layers
    params = add_node(banks, "task35", rank=16)
    bank = PrefixBank(bk, obs_dicts)                        # cached prefixes, gathered by row
    train(bk, state, "task35", params, bank, chunks_full, steps=1000)   # flat LoRA (few-shot baseline)
    train(..., anchor=(anchor_bank, anchor_chunks))         # + the r ~ 0 anchor term (fold)
    save(banks, "task35", path); load(bk, path) -> (state, key)

Loss = openpi's: t ~ Beta(1.5, 1) * 0.999 + 0.001, x_t = t eps + (1 - t) a, u = eps - a, mean (v - u)^2 over the
full padded chunk.  The anchor term = mean (v_on - v_off)^2 on library observations, with the base policy's own
velocity (expert off) as the target: no teacher copy.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..experts.lora import add_node, inject_keyed_lora, keys_on


def inject(bk, state=None, exclude=("lm_head", "embed_tokens")):
    state = state if state is not None else {"on": frozenset()}
    banks = inject_keyed_lora(bk.expert, state, exclude=exclude)
    return state, banks


def n_params(params):
    return int(sum(p.numel() for p in params))


class PrefixBank:
    """Prefixes of many observations cached as ONE batched Prefix (built in blocks), gathered by row index."""

    def __init__(self, bk, obs_dicts, block=32):
        from .backbone import Prefix
        self.bk = bk
        parts = [bk.prefix(bk.observations(obs_dicts[i:i + block])) for i in range(0, len(obs_dicts), block)]
        kv0 = parts[0].kv
        self.state = torch.cat([p.state for p in parts])
        self.pad = torch.cat([p.pad_masks for p in parts])
        self.k = [torch.cat([p.kv.key_cache[l] for p in parts]) for l in range(len(kv0.key_cache))]
        self.v = [torch.cat([p.kv.value_cache[l] for p in parts]) for l in range(len(kv0.value_cache))]
        self.n = len(self.state)
        self._Prefix = Prefix
        self._cache_type = type(kv0)

    def take(self, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.state.device, dtype=torch.long)
        kv = self._cache_type()
        for l in range(len(self.k)):
            kv.update(self.k[l].index_select(0, idx), self.v[l].index_select(0, idx), l)
        return self._Prefix(state=self.state.index_select(0, idx), pad_masks=self.pad.index_select(0, idx), kv=kv, n=len(idx))

    def bytes(self):
        return int(sum(t.numel() * t.element_size() for t in self.k + self.v))


def _flow_batch(n, H, A, gen, device):
    b = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
    t = (b.sample((n,)) * 0.999 + 0.001).to(device)
    eps = torch.randn((n, H, A), generator=gen).to(device)
    return t, eps


def train(bk, state, key, params, bank, chunks_full, steps=1000, lr=1e-4, batch=32, seed=0, anchor=None,
          anchor_weight=1.0, log_every=100, log=print, weight_decay=0.0):
    """Flow-matching fine-tune of the expert `key` on (bank rows, chunks_full [N, H, A] padded, normalised).
    anchor = (anchor_bank, anchor_chunks_full): the r ~ 0 half, target = the base velocity (expert off).
    -> dict of loss curves."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    A = torch.as_tensor(np.asarray(chunks_full, np.float32), device=bk.device)
    on = keys_on(state, {key})
    hist = dict(step=[], loss=[], anchor=[])
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = rng.integers(0, bank.n, size=min(batch, bank.n))
        pre = bank.take(idx)
        a = A[idx]
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        with on():
            v = bk.v_full(xt, t, pre)
        loss = ((v - u) ** 2).mean()
        la = torch.zeros((), device=bk.device)
        if anchor is not None:
            abank, achunks = anchor
            j = rng.integers(0, abank.n, size=min(batch, abank.n))
            apre = abank.take(j)
            aa = torch.as_tensor(np.asarray(achunks[j], np.float32), device=bk.device)
            ta, ea = _flow_batch(len(j), bk.H, bk.A, gen, bk.device)
            xa = ta[:, None, None] * ea + (1 - ta[:, None, None]) * aa
            with torch.no_grad():
                v_off = bk.v_full(xa, ta, apre)
            with on():
                v_on = bk.v_full(xa, ta, apre)
            la = ((v_on - v_off) ** 2).mean()
        tot = loss + anchor_weight * la
        opt.zero_grad(set_to_none=True)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s % log_every == 0 or s == 1:
            hist["step"].append(s); hist["loss"].append(float(loss)); hist["anchor"].append(float(la))
            log(f"[adapt] {key} step {s}/{steps}: flow {float(loss):.4f} anchor {float(la):.5f} ({time.time() - t0:.0f}s)")
    return hist


def save(banks, key, path, meta=None):
    sd = {b.name: dict(A=b.A[key].weight.detach().cpu().half(), B=b.B[key].weight.detach().cpu().half()) for b in banks if key in b.A}
    torch.save(dict(key=key, banks=sd, meta=meta or {}), path)
    return int(sum(v["A"].numel() + v["B"].numel() for v in sd.values()) * 2)


def load(bk, path, state=None):
    """Inject banks (idempotent) and load a saved expert.  -> (state, key, banks)."""
    z = torch.load(path, map_location="cpu")
    key = z["key"]
    state, banks = inject(bk, state)
    by_name = {b.name: b for b in banks}
    rank = next(iter(z["banks"].values()))["A"].shape[0]
    add_node(banks, key, rank)
    for name, w in z["banks"].items():
        b = by_name[name]
        b.A[key].weight.data.copy_(w["A"].to(b.A[key].weight.dtype))
        b.B[key].weight.data.copy_(w["B"].to(b.B[key].weight.dtype))
    return state, key, banks
