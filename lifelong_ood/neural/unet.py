"""A small class-conditional UNet for 32x32 RGB, in the shape the poe-diffusion paper's backbone has.

The paper uses `google/ddpm-cifar10-32` with a class-embedding table sized |C|+1 to reserve a row for
the null token used by classifier-free-guidance dropout.  This is the same architecture at a smaller
width so it trains in under an hour on one GPU instead of six: 3 resolutions (32/16/8), two residual
blocks per level, self-attention at 16x16 and 8x8, sinusoidal time embedding added inside every
block, and the class embedding added to the time embedding (so a null class is exactly "no class
signal", which is what CFG needs).

Only `eps(x, u, c)` matters downstream: everything in `compose.py` speaks the same
epsilon-prediction interface as the analytic field, so the two backbones are interchangeable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(u, dim, max_period=10000.0):
    """u in [0, 1] (our shared noise coordinate) -> sinusoidal embedding, scaled to the usual 1000."""
    t = torch.as_tensor(u, dtype=torch.float32).reshape(-1) * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    a = t[:, None].to(freqs.device) * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], -1)
    return F.pad(emb, (0, dim - emb.shape[-1])) if dim % 2 else emb


class ResBlock(nn.Module):
    def __init__(self, cin, cout, edim, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(32, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(edim, cout)
        self.n2 = nn.GroupNorm(32, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(e))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.h = heads
        self.n = nn.GroupNorm(32, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                           v.transpose(-1, -2))
        return x + self.out(a.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    """Class-conditional epsilon-predictor.  `n_classes` EXCLUDES the null token, which is appended
    as the last row of the embedding table (index `n_classes`), exactly as the paper describes."""

    def __init__(self, n_classes, base=64, mults=(1, 2, 2), edim=256, attn=(16, 8), dropout=0.1,
                 in_ch=3):
        super().__init__()
        self.n_classes = n_classes
        self.null_id = n_classes
        self.temb = nn.Sequential(nn.Linear(base, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.base = base
        self.cemb = nn.Embedding(n_classes + 1, edim)
        nn.init.normal_(self.cemb.weight, std=0.02)
        with torch.no_grad():
            self.cemb.weight[self.null_id].zero_()       # null token = no class signal

        chs = [base * m for m in mults]
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        skip_ch, cur, res = [base], base, 32
        for i, c in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(2):
                blocks.append(nn.ModuleList([ResBlock(cur, c, edim, dropout),
                                             Attention(c) if res in attn else nn.Identity()]))
                cur = c
                skip_ch.append(cur)
            self.down.append(blocks)
            if i < len(chs) - 1:
                self.downsample.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))
                skip_ch.append(cur)
                res //= 2
        self.mid1 = ResBlock(cur, cur, edim, dropout)
        self.mida = Attention(cur)
        self.mid2 = ResBlock(cur, cur, edim, dropout)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, c in reversed(list(enumerate(chs))):
            blocks = nn.ModuleList()
            for _ in range(3):
                blocks.append(nn.ModuleList([ResBlock(cur + skip_ch.pop(), c, edim, dropout),
                                             Attention(c) if res in attn else nn.Identity()]))
                cur = c
            self.up.append(blocks)
            if i > 0:
                self.upsample.append(nn.Conv2d(cur, cur, 3, padding=1))
                res *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, cur), nn.SiLU(),
                                 nn.Conv2d(cur, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, u, c=None, emb_override=None):
        """`emb_override` supplies the class embedding directly instead of looking it up.

        The lifelong adapter bank uses it to condition on a concept whose embedding is a trainable
        parameter living outside this module's table: passing the tensor through keeps it in the
        autograd graph, which writing it into `cemb.weight` would not.
        """
        e = self.temb(timestep_embedding(u, self.base).to(x.device))
        if e.shape[0] == 1 and x.shape[0] > 1:
            e = e.expand(x.shape[0], -1)
        if emb_override is not None:
            e = e + emb_override.reshape(1, -1).to(e.dtype)
        else:
            if c is None:
                c = torch.full((x.shape[0],), self.null_id, device=x.device, dtype=torch.long)
            e = e + self.cemb(torch.as_tensor(c, device=x.device, dtype=torch.long))
        h = self.stem(x)
        hs = [h]
        for i, blocks in enumerate(self.down):
            for rb, at in blocks:
                h = at(rb(h, e))
                hs.append(h)
            if i < len(self.down) - 1:
                h = self.downsample[i](h)
                hs.append(h)
        h = self.mid2(self.mida(self.mid1(h, e)), e)
        for i, blocks in enumerate(self.up):
            for rb, at in blocks:
                h = at(rb(torch.cat([h, hs.pop()], 1), e))
            if i < len(self.up) - 1:
                h = self.upsample[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)
