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
    as the last row of the embedding table (index `n_classes`), exactly as the paper describes.

    TWO CONDITIONING MODES, and the second exists because the first is untrainable on CelebA.

    * `cond_mode="class"` (default, ColorMNIST) — one free embedding row per compositional class.
      This is the paper's setup and the one that makes the ideal-words result meaningful: the factor
      structure is *discovered* in an entangled space (R^2 0.826), not built in.
    * `cond_mode="factored"` (CelebA) — the conditioning vector is the SUM of one learned embedding
      per (attribute, value) pair: `e_c = sum_k attr[k, v_k]`.  RESULTS 5o measured why this is
      needed: a per-class row on CelebA is estimated from as few as 25 images and 2.3x the data with
      140 epochs moved conditional/ceiling 34% -> 33%, so the weakness is the conditioning design,
      not the compute.  Under factoring every attribute embedding is estimated from *every* image
      carrying that value — tens of thousands rather than tens.

      The honest consequence, stated at the point where it is created: on CelebA the factor basis is
      ARCHITECTURAL, not discovered, so a held-out combination is expressible in conditioning space
      by construction.  What CelebA then tests is the thing ColorMNIST structurally cannot — whether
      composing known factor values *renders correctly* when the factors overlap the same pixels.

    Both modes resolve a class id to ONE embedding vector added to the time embedding, so every
    downstream consumer (adapter bank, inversion, ideal-words, the score field) is unchanged.
    """

    def __init__(self, n_classes, base=64, mults=(1, 2, 2), edim=256, attn=(16, 8), dropout=0.1,
                 in_ch=3, img=32, cond_mode="class", class_attrs=None, n_attr_vals=2):
        # `img` is the input resolution the level bookkeeping starts from.  It was hardcoded to 32,
        # which silently placed attention at the wrong levels for any other size -- at 64x64 the
        # first level then ran full attention over 4096 tokens and ran out of memory.  Attention
        # levels are named in PIXELS (`attn=(16, 8)`), so they stay meaningful across resolutions.
        super().__init__()
        self.n_classes = n_classes
        self.null_id = n_classes
        self.temb = nn.Sequential(nn.Linear(base, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.base = base
        self.edim = edim
        self.cond_mode = cond_mode
        if cond_mode == "factored":
            if class_attrs is None:
                raise ValueError("cond_mode='factored' needs class_attrs [n_classes, n_attrs]")
            ca = torch.as_tensor(class_attrs, dtype=torch.long)
            self.n_attrs = int(ca.shape[1])
            self.n_attr_vals = int(n_attr_vals)
            self.register_buffer("class_attrs", ca)
            self.attr = nn.Parameter(torch.empty(self.n_attrs, self.n_attr_vals, edim))
            nn.init.normal_(self.attr, std=0.02)
        else:
            self.cemb = nn.Embedding(n_classes + 1, edim)
            nn.init.normal_(self.cemb.weight, std=0.02)
            with torch.no_grad():
                self.cemb.weight[self.null_id].zero_()   # null token = no class signal

        chs = [base * m for m in mults]
        self.img = int(img)
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        skip_ch, cur, res = [base], base, int(img)
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

    @property
    def emb_dim(self):
        return self.edim

    def attr_embedding(self, v):
        """Attribute-value vector(s) -> conditioning embedding, `sum_k attr[k, v_k]`.

        This is basis composition made architectural: an unseen combination of *known* attribute
        values costs zero new parameters, which is the CelebA analogue of `extend_basis` (n11).
        Accepts one row `[n_attrs]` or a batch `[B, n_attrs]`.
        """
        if self.cond_mode != "factored":
            raise RuntimeError("attr_embedding needs cond_mode='factored'")
        v = torch.as_tensor(v, dtype=torch.long, device=self.attr.device)
        one = v.dim() == 1
        v = v.reshape(1, -1) if one else v
        idx = torch.arange(self.n_attrs, device=v.device)
        e = self.attr[idx[None].expand(len(v), -1), v].sum(1)      # [B, edim]
        return e[0] if one else e

    def class_embeddings(self):
        """-> [n_classes, edim], the conditioning vector of every trained class, in either mode.

        Experiments that factorize the conditioning space (ideal-words) read this rather than
        `cemb.weight`, so they work on both backbones.  In `factored` mode the returned matrix is by
        construction a sum of attribute vectors, so factorizing it recovers R^2 = 1 trivially — a
        CelebA experiment should use `attr` directly and say so, not report that R^2.
        """
        if self.cond_mode == "factored":
            with torch.no_grad():
                return self.attr_embedding(self.class_attrs)
        return self.cemb.weight[:self.n_classes].detach()

    def forward(self, x, u, c=None, emb_override=None):
        """`emb_override` supplies the class embedding directly instead of looking it up.

        The lifelong adapter bank uses it to condition on a concept whose embedding is a trainable
        parameter living outside this module's table: passing the tensor through keeps it in the
        autograd graph, which writing it into `cemb.weight` would not.

        Accepts either one vector `[d]`, shared by the whole batch, or a per-instance batch `[B, d]`.
        The per-instance form is what basis extension needs (n11): one new factor direction is
        learned under many different contexts at once, so every element of the batch carries a
        different combination of the frozen factor vectors plus the single vector being learned.
        """
        e = self.temb(timestep_embedding(u, self.base).to(x.device))
        if e.shape[0] == 1 and x.shape[0] > 1:
            e = e.expand(x.shape[0], -1)
        if emb_override is not None:
            eo = emb_override.to(e.dtype)
            e = e + (eo.reshape(1, -1) if eo.dim() == 1 else eo.reshape(x.shape[0], -1))
        else:
            if c is None:
                c = torch.full((x.shape[0],), self.null_id, device=x.device, dtype=torch.long)
            c = torch.as_tensor(c, device=x.device, dtype=torch.long)
            if self.cond_mode == "factored":
                # null id has no attribute row; it conditions on the ZERO vector, which is what CFG
                # needs ("no class signal") and is distinct from every realisable attribute sum.
                live = c != self.null_id
                ce = torch.zeros(len(c), self.edim, device=x.device, dtype=e.dtype)
                if bool(live.any()):
                    ce[live] = self.attr_embedding(self.class_attrs[c[live]]).to(e.dtype)
                e = e + ce
            else:
                e = e + self.cemb(c)
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
