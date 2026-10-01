"""BIND: an additive bias on the ACTION EXPERT's attention logits over the image tokens of a referent (plan of 2026-10-01,
evening; docs/METHOD.md).

The frozen policy's action expert attends, at every layer and every Euler step, to the cached prefix: 256 agent-view tokens
(columns 0-255, a 16 x 16 grid of 14-px patches of the 224 image, row-major), 256 wrist tokens (256-511), a zeroed third
image (512-767, masked) and the language (768-967); appended factor tokens extend it.  Binding raises the attention the
expert pays to the tokens covering the instruction's referent, the robot counterpart of Attend-and-Excite, so the policy's
own next chunk is computed with the referent in view of every layer.  The policy still produces every action.

Two modes, applied by patching `eager_attention_forward` of transformers' Gemma for the expert's attention modules only:
    fixed  logits += beta * m                 (m in [0, 1] per prefix column)
    mass   logits += beta_r * 1[m >= 0.5], beta_r = log[rho (1 - p) / (p (1 - rho))] clipped at 0, per row (layer, head,
           query), where p is the row's current attention mass on the referent: the referent receives mass rho (or keeps
           more, never less).  rho is comparable across layers and heads; beta is not.
beta = 0 / rho = None leaves the original function in place, so the policy is reproduced bit for bit.

    gate = bias_gate(bk, maps={"agent": m16}, mode="mass", rho=0.5)       # m16: [16, 16] coverage of the referent
    rows = bk.sample(1, pre, seed=s, step_gate=gate)
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

CAMERA_COLS = {"agent": 0, "wrist": 256}       # first prefix column of each camera's 256 tokens
GRID = 16


def _gemma():
    from transformers.models.gemma import modeling_gemma
    return modeling_gemma


def expert_attentions(bk):
    """The action expert's 18 attention modules (the VLM's are never touched)."""
    return [layer.self_attn for layer in bk.model.paligemma_with_expert.gemma_expert.model.layers]


def prefix_map(maps, width, device=None):
    """{camera: [16, 16] in [0, 1]} -> [width] float32 over the prefix columns (zeros elsewhere)."""
    m = torch.zeros(width, dtype=torch.float32, device=device)
    for cam, g in (maps or {}).items():
        g = torch.as_tensor(np.asarray(g, np.float32), device=device).reshape(-1)
        assert g.numel() == GRID * GRID, f"{cam}: expected a 16 x 16 map"
        c0 = CAMERA_COLS[cam]
        m[c0:c0 + GRID * GRID] = g
    return m


class _State:
    """What the patched attention reads: the referent map, the mode, the layers, and an optional log."""

    def __init__(self):
        self.ids = set()
        self.map = None          # [B or 1, P] float32
        self.mode = "fixed"
        self.beta = 0.0
        self.rho = None
        self.layers = None       # None = all
        self.log = None          # list to append (layer, mass before, mass after) to, or None


_STATE = _State()
_ORIG = {}


def _patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    st = _STATE
    orig = _ORIG["f"]
    active = (st.map is not None and id(module) in st.ids
              and (st.layers is None or module.layer_idx in st.layers)
              and ((st.mode == "fixed" and st.beta != 0.0) or (st.mode == "mass" and st.rho is not None) or st.log is not None))
    if not active:
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    G = _gemma()
    key_states = G.repeat_kv(key, module.num_key_value_groups)
    value_states = G.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : key_states.shape[-2]]
    w = w.float()
    K = w.shape[-1]
    m = st.map
    if m.shape[-1] < K:                                     # suffix (action) columns and any columns past the map: no bias
        m = torch.nn.functional.pad(m, (0, K - m.shape[-1]))
    m = m[..., :K][:, None, None, :]                       # [B or 1, 1, 1, K]
    p0 = None
    target = (m >= 0.5)
    if st.mode == "mass" or st.log is not None:
        # log-space masses: a head whose referent mass underflows still reaches rho exactly
        lse_t = torch.logsumexp(w.masked_fill(~target, float("-inf")), dim=-1, keepdim=True)
        lse_n = torch.logsumexp(w.masked_fill(target, float("-inf")), dim=-1, keepdim=True)
        p0 = torch.exp(lse_t - torch.logaddexp(lse_t, lse_n))
    if st.mode == "fixed" and st.beta != 0.0:
        w = w + float(st.beta) * m
    elif st.mode == "mass" and st.rho is not None:
        rho = float(st.rho)
        b = (np.log(rho) - np.log(1.0 - rho) + lse_n - lse_t).clamp(min=0.0)       # mass after = rho (or more, never less)
        b = torch.where(torch.isfinite(b), b, torch.zeros_like(b))                   # no referent column: no bias
        w = w + b * target.float()
    attn = torch.softmax(w, dim=-1, dtype=torch.float32)
    if st.log is not None:
        p1 = (attn * target.float()).sum(-1)
        st.log.append((int(module.layer_idx), float(p0.mean()), float(p1.mean())))
    attn = attn.to(query.dtype)
    out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
    return out, attn


def install(bk):
    """Patch transformers' Gemma eager attention once (idempotent); register the expert's modules."""
    G = _gemma()
    if "f" not in _ORIG:
        _ORIG["f"] = G.eager_attention_forward
        G.eager_attention_forward = _patched
    _STATE.ids = {id(a) for a in expert_attentions(bk)}


@contextlib.contextmanager
def biased(bk, maps, mode="mass", rho=None, beta=0.0, layers=None, log=None, width=None, batch=1):
    """Within the block, the action expert attends with the referent bias.  `maps` = {camera: [16,16]} (one for all rows) or
    a list of such dicts (one per row)."""
    install(bk)
    W = int(width or 968)
    if isinstance(maps, (list, tuple)):
        m = torch.stack([prefix_map(x, W, bk.device) for x in maps])
    else:
        m = prefix_map(maps, W, bk.device)[None].expand(int(batch), W)
    saved = (_STATE.map, _STATE.mode, _STATE.beta, _STATE.rho, _STATE.layers, _STATE.log)
    _STATE.map, _STATE.mode, _STATE.beta, _STATE.rho = m, mode, float(beta), (None if rho is None else float(rho))
    _STATE.layers = None if layers is None else set(int(x) for x in layers)
    _STATE.log = log
    try:
        yield
    finally:
        _STATE.map, _STATE.mode, _STATE.beta, _STATE.rho, _STATE.layers, _STATE.log = saved


def bias_gate(bk, maps, pre, mode="mass", rho=None, beta=0.0, layers=None, log=None):
    """A `step_gate` for PolicyBackbone.sample: every Euler step runs inside `biased(...)` (the prefix width is read from
    `pre`, so appended tokens are never biased)."""
    width = int(pre.pad_masks.shape[1])

    def step_gate(x_rows, t):
        return lambda: biased(bk, maps, mode=mode, rho=rho, beta=beta, layers=layers, log=log, width=width,
                              batch=int(x_rows.shape[0]))
    return step_gate


def is_identity(mode, rho, beta):
    return (mode == "fixed" and float(beta) == 0.0) or (mode == "mass" and rho is None)
