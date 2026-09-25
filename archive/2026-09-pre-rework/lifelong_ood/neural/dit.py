"""A small DiT over latent patches, exposing the SAME conditioning interface as our UNet.

The point of this file is a portability test, not a better model.  Everything the three-tier
architecture does is defined over a **class embedding table**: `factorize` fits
`c ~ mu + sum_f V_f[value]` to it, ADDRESS composes a new point in that basis, EXTEND fits a new
direction, and the router reads epsilon-error under candidate conditionings.  None of that mentions
convolutions.  If it is genuinely architectural, it should survive being moved onto a transformer
that conditions through adaLN modulation instead of an additive time-embedding bias -- which is the
family vision-language-action models are built from.

So this DiT deliberately keeps the interface identical:

    self.cemb      nn.Embedding(n_classes + 1, edim), null row at self.null_id
    forward(x, u, c=None, emb_override=None)

`emb_override` is the load-bearing one: it is how every downstream tool supplies a composed
conditioning vector that corresponds to no trained class.  `factorize`, `compose`, `route_signals`
and `conditional_losses` then run on this backbone with no changes at all, and any that need changes
are exactly the places the architecture was secretly UNet-specific.

The conditioning path is DiT's own (Peebles & Xie): class embedding is ADDED to the timestep
embedding and the sum drives adaLN-Zero -- per-block scale/shift on the norms and a zero-initialised
gate on each residual branch.  That is a different mechanism from the UNet's additive bias, which is
the whole reason the test is interesting.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(u, dim):
    """u in [0, 1] -> [N, dim] sinusoidal, the same parameterisation the UNet uses."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=u.device).float() / half)
    a = u.float().reshape(-1, 1) * 1000.0 * freqs.reshape(1, -1)
    out = torch.cat([torch.cos(a), torch.sin(a)], -1)
    return F.pad(out, (0, dim - out.shape[-1])) if out.shape[-1] < dim else out


class DiTBlock(nn.Module):
    """Attention + MLP, both gated by adaLN-Zero off the conditioning vector."""

    def __init__(self, d, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(approximate="tanh"), nn.Linear(h, d))
        # zero-init so every block starts as the identity -- "adaLN-Zero"
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, e):
        sa, ba, ga, sm, bm, gm = self.ada(e).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sa[:, None]) + ba[:, None]
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + ga[:, None] * a
        h = self.n2(x) * (1 + sm[:, None]) + bm[:, None]
        return x + gm[:, None] * self.mlp(h)


class DiT(nn.Module):
    """Latent DiT with the UNet's conditioning interface.

    `in_ch`/`size` describe the LATENT grid (e.g. 4 x 8 x 8 from `latent.ConvAE`), not pixels.
    """

    def __init__(self, n_classes, in_ch=4, size=8, patch=2, d=256, depth=6, heads=4,
                 edim=256, **_ignored):
        super().__init__()
        self.n_classes, self.null_id = n_classes, n_classes
        self.in_ch, self.size, self.patch, self.edim = in_ch, size, patch, edim
        n_patch = (size // patch) ** 2
        self.x_embed = nn.Linear(in_ch * patch * patch, d)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d))
        nn.init.normal_(self.pos, std=0.02)
        # SAME interface as unet.UNet: a table with one extra row for the CFG null token
        self.cemb = nn.Embedding(n_classes + 1, edim)
        nn.init.normal_(self.cemb.weight, std=0.02)
        with torch.no_grad():
            self.cemb.weight[self.null_id].zero_()
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.c_proj = nn.Linear(edim, d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_f[1].weight); nn.init.zeros_(self.ada_f[1].bias)
        self.out = nn.Linear(d, in_ch * patch * patch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.d = d

    def class_table(self):
        return self.cemb.weight[:self.n_classes].detach()

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // p) * (W // p), C * p * p)

    def _unpatchify(self, t):
        B, N, _ = t.shape
        p, C = self.patch, self.in_ch
        g = int(N ** 0.5)
        t = t.reshape(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5)
        return t.reshape(B, C, g * p, g * p)

    def forward(self, x, u, c=None, emb_override=None):
        """`emb_override` supplies the class embedding directly -- how a COMPOSED concept is used."""
        u = torch.as_tensor(u, device=x.device, dtype=torch.float32).reshape(-1)
        if u.numel() == 1:
            u = u.expand(len(x))
        e = self.t_mlp(timestep_embedding(u, self.d))
        if emb_override is not None:
            ce = torch.as_tensor(emb_override, dtype=torch.float32, device=x.device)
            ce = ce.reshape(-1, self.edim)
            if len(ce) == 1:
                ce = ce.expand(len(x), -1)
        else:
            cid = (torch.full((len(x),), self.null_id, device=x.device, dtype=torch.long)
                   if c is None else torch.as_tensor(c, device=x.device, dtype=torch.long))
            ce = self.cemb(cid)
        # DiT's own path: class embedding ADDED to the timestep embedding, sum drives adaLN
        e = e + self.c_proj(ce)
        h = self.x_embed(self._patchify(x)) + self.pos
        for b in self.blocks:
            h = b(h, e)
        s, bshift = self.ada_f(e).chunk(2, dim=-1)
        h = self.nf(h) * (1 + s[:, None]) + bshift[:, None]
        return self._unpatchify(self.out(h))
