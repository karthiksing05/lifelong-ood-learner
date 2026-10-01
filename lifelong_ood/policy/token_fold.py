"""FACTOR TOKENS: the overhaul's fold (step 4 of Read -> Compose -> Fold -> Consolidate).

A factor token is a learned key and value per layer, appended to the prompt's cache that the action expert attends to
(prefix tuning: Li & Liang 2021).  Every weight of the policy stays frozen; the token changes the policy's conditional only
where a code addresses it, so tasks that do not use it are untouched by construction.  Gradients flow only through the
action expert (the prompt's cache is computed once per frame, adapt.PrefixBank), which makes a token cheap to fit.  The
user's mechanism study found composition carried by the key / value split, which is where the token lives.

Size on pi0.5: 18 layers x 1 KV head x 256 dims x (key, value) x n tokens, fp16: 18.4 KB per token.  Fitted by flow matching
on (prompt cache, target chunk) pairs: a composition's own successful rollouts (self-distillation: the composition becomes
readable from one prompt) or, for a primitive no composition reaches, the arrival's frames (never executed).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .backbone import Prefix


class KVToken(nn.Module):
    """n learned key / value entries per layer.  k, v: [L, n_kv_heads, n, head_dim]."""

    def __init__(self, k, v):
        super().__init__()
        self.k = nn.Parameter(torch.as_tensor(k, dtype=torch.float32).clone())
        self.v = nn.Parameter(torch.as_tensor(v, dtype=torch.float32).clone())

    @property
    def n(self):
        return int(self.k.shape[2])

    def nbytes(self, dtype_bytes=2):
        return int(self.k.numel() + self.v.numel()) * dtype_bytes

    @staticmethod
    def from_prefix(pre, positions, row=0):
        """Initialise from the cache entries at `positions` of one prompt's prefix (e.g. a word's own tokens, or the mean of
        the prompt's language tokens), so the token starts as something the policy already reads."""
        pos = torch.as_tensor(np.atleast_1d(positions), device=pre.kv.key_cache[0].device)
        k = torch.stack([kk[row].index_select(1, pos) for kk in pre.kv.key_cache])             # [L, H, n, d]
        v = torch.stack([vv[row].index_select(1, pos) for vv in pre.kv.value_cache])
        return KVToken(k.float(), v.float())

    def state_dict_fp16(self):
        return {"k": self.k.detach().half().cpu(), "v": self.v.detach().half().cpu()}


def attach(pre, token):
    """The prompt's prefix with the token's entries appended to every layer's cache and marked valid.  The action tokens'
    positions shift by token.n (prefix_offsets count valid prefix entries), as a slightly longer prompt does."""
    B = int(pre.n)
    kv = type(pre.kv)()
    for l in range(len(pre.kv.key_cache)):
        kk, vv = pre.kv.key_cache[l], pre.kv.value_cache[l]
        tk = token.k[l].to(kk.dtype)[None].expand(B, -1, -1, -1)
        tv = token.v[l].to(vv.dtype)[None].expand(B, -1, -1, -1)
        kv.update(torch.cat([kk, tk], dim=2), torch.cat([vv, tv], dim=2), l)
    pad = torch.cat([pre.pad_masks, torch.ones((B, token.n), dtype=pre.pad_masks.dtype, device=pre.pad_masks.device)], dim=1)
    return Prefix(state=pre.state, pad_masks=pad, kv=kv, n=B)


def fit(bk, token, bank, chunks_full, steps=600, lr=1e-2, batch=16, seed=0, anchor=None, anchor_weight=1.0, log_every=100,
        log=print):
    """Flow matching on (bank rows, chunks_full [N, H, A] normalised, padded) with the token attached; only the token learns.
    anchor = (bank_a, chunks_a): rows where the token must NOT change the policy (target: the velocity without it)."""
    from .adapt import _flow_batch
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(token.parameters(), lr=lr)
    A = torch.as_tensor(np.asarray(chunks_full, np.float32), device=bk.device)
    hist = dict(step=[], loss=[], anchor=[])
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = rng.integers(0, bank.n, size=min(batch, bank.n))
        pre = attach(bank.take(idx), token)
        a = A[idx]
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        v = bk.v_full(xt, t, pre)
        loss = ((v - u) ** 2).mean()
        la = torch.zeros((), device=bk.device)
        if anchor is not None:
            abank, achunks = anchor
            j = rng.integers(0, abank.n, size=min(batch, abank.n))
            pa = abank.take(j)
            aa = torch.as_tensor(np.asarray(achunks, np.float32)[j], device=bk.device)
            ta, ea = _flow_batch(len(j), bk.H, bk.A, gen, bk.device)
            xa = ta[:, None, None] * ea + (1 - ta[:, None, None]) * aa
            with torch.no_grad():
                v0 = bk.v_full(xa, ta, pa)
            la = ((bk.v_full(xa, ta, attach(pa, token)) - v0) ** 2).mean()
        (loss + anchor_weight * la).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % log_every == 0 or s == steps:
            hist["step"].append(s)
            hist["loss"].append(float(loss))
            hist["anchor"].append(float(la))
            log(f"[token] step {s}: flow {float(loss):.4f} anchor {float(la):.4f} ({time.time() - t0:.0f}s)")
    return hist


class TokenHooks:
    """Server hooks: the task's own prompt with factor token(s) appended to its cache; the policy samples unguided."""

    def __init__(self, paths, device="cuda"):
        self.tokens = []
        for p in paths:
            d = torch.load(p, map_location=device)["token"]
            self.tokens.append(KVToken(d["k"].float(), d["v"].float()).to(device))

    def transform_prefix(self, pre):
        for tok in self.tokens:
            pre = attach(pre, tok)
        return pre

    def __call__(self, obs, pre, meta=None):
        return None, None
