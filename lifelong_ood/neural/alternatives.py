"""Three alternatives to a per-concept LoRA adapter, behind the same consolidation interface.

What the LoRA is actually doing is narrower than it looks.  The composition is read in EMBEDDING
space (`compose_factored`) and the manifold generates from it; the adapter only fits the residual
between what the model produces under that embedding and what the target really looks like.  RESULTS
5n measured this: zeroing the private embedding drops a composition 0.38 -> 0.04, while tripling the
adapter path barely moves it.  So the adapter is a local density correction, not the compositional
mechanism -- which is exactly why it is worth asking what else could play that role.

Each class here replaces the delta and nothing else.  Composition, sampling and consolidation are
untouched, so a difference in results is attributable to the parameterization.

  HyperDelta   the delta is GENERATED from the concept embedding by a shared network, so nothing is
               stored per concept.  This is the direct consequence of 5n: if the embedding carries
               the semantics, generate the correction from it instead of storing one per concept.
               It is also the one option that keeps zero growth AND conditionality, which 5r
               identified as the two properties that actually matter.
  OrthoDelta   the delta is an orthogonal transform (Cayley), so it cannot change the norm of what
               it acts on.  Aimed squarely at 5r's failure: merged deltas corrupt the base, and
               norm distortion is the destructive part.
  GuidanceHead a classifier on noisy images, used to steer sampling by the gradient of the target
               attributes.  Costs parameters per ATTRIBUTE rather than per concept, composes by
               summing gradients, leaves the base model untouched -- and needs no conditioning
               channel at all, which is what blocked the pretrained model in 5u.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..score import cosine_abar


# ── 1. hypernetwork: the delta is generated, never stored ────────────────────
class HyperDelta(nn.Module):
    """FiLM-style channel modulation whose gains are produced from the concept embedding.

    For every wrapped layer the base output is scaled by `1 + gamma_i(c)`, with
    `gamma_i(c) = P_i @ h(c)` for a shared trunk `h` and a per-layer projection `P_i`.  The parameter
    count is fixed by the ARCHITECTURE, not by how many concepts have been learned, so a stream of
    any length costs the same.

    The catch, and the thing the experiment has to measure: `h` and `P` are shared, so learning a new
    concept moves them for every earlier concept too.  Preservation cannot be exact here the way it
    is for a frozen adapter -- it has to be earned by the trunk generalizing across concepts rather
    than memorizing one.  That is the trade being tested, not a flaw to be hidden.
    """

    def __init__(self, wrapped, emb_dim, hidden=64, device="cuda"):
        super().__init__()
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.SiLU(), nn.Linear(128, self.hidden)).to(device)
        self.heads = nn.ParameterList()
        self.outs = []
        for w in wrapped:
            b = w.base
            o = b.out_channels if hasattr(b, "out_channels") else b.out_features
            self.outs.append(int(o))
            self.heads.append(nn.Parameter(torch.zeros(int(o), self.hidden, device=device)))
        self._emb = None
        self._h = None

    def n_params(self):
        return sum(q.numel() for q in self.parameters())

    def set_concept(self, emb):
        """Point the modulation at a concept.  The trunk is evaluated LAZILY.

        Caching the computed gains here instead would keep one autograd graph alive across training
        steps, and the second backward then fails -- the gains have to be rebuilt from the embedding
        on each forward, not stored.
        """
        self._emb = None if emb is None else emb.reshape(1, -1)
        self._h = None

    def gamma(self, i):
        """Rebuild the gain from the embedding every time it is asked for.

        Caching the trunk output across a forward pass would be cheaper, but any cached activation
        survives into the next training step and the second backward then fails.  The trunk is two
        small linears on a single vector, so recomputation costs almost nothing next to the UNet.
        """
        if self._emb is None:
            return None
        return F.linear(self.trunk(self._emb), self.heads[i]).reshape(-1)


class HyperHook:
    """Applies `HyperDelta`'s modulation to a wrapped layer's output."""

    def __init__(self, hyper, index):
        self.hyper, self.index = hyper, index

    def __call__(self, mod, inp, out):
        g = self.hyper.gamma(self.index)
        if g is None:
            return out
        shape = (1, -1, 1, 1) if out.dim() == 4 else (1, -1)
        return out * (1 + g.reshape(shape).to(out.dtype))


# ── 2. orthogonal delta: cannot change the norm of what it acts on ───────────
class OrthoDelta(nn.Module):
    """Block-diagonal orthogonal transform on output channels, via the Cayley map.

    `R = (I - S)(I + S)^-1` with `S` skew-symmetric is orthogonal for any `S`, so the transform is
    a rotation: it moves information between channels without amplifying or shrinking it.  5r found
    that folding LoRA deltas into the base corrupts it; a rotation is the natural candidate for a
    delta that stays well-behaved under repeated composition, because rotations compose into
    rotations.

    Blocks keep it affordable -- a full rotation over 512 channels would be 512x512 per layer.
    """

    def __init__(self, wrapped, block=16, device="cuda"):
        super().__init__()
        self.block = int(block)
        self.S = nn.ParameterList()
        self.shapes = []
        for w in wrapped:
            b = w.base
            o = int(b.out_channels if hasattr(b, "out_channels") else b.out_features)
            nb = max(1, o // self.block)
            bs = o // nb
            self.shapes.append((nb, bs, o))
            self.S.append(nn.Parameter(torch.zeros(nb, bs, bs, device=device)))

    def n_params(self):
        return sum(q.numel() for q in self.parameters())

    def rotation(self, i):
        nb, bs, o = self.shapes[i]
        S = self.S[i]
        S = S - S.transpose(1, 2)                      # skew-symmetric
        I = torch.eye(bs, device=S.device).expand(nb, bs, bs)
        return torch.linalg.solve(I + S, I - S)        # Cayley -> orthogonal


class OrthoHook:
    """One hook per layer, consulting a shared registry for which concept's rotation is active.

    Registering a hook per concept per layer and toggling them individually gets unmanageable as
    concepts accumulate; a single hook that reads `registry["active"]` keeps activation O(1) and
    mirrors how `PathLoRA` already switches concepts by switching a shared path list.
    """

    def __init__(self, registry, index):
        self.registry, self.index = registry, index

    def __call__(self, mod, inp, out):
        act = self.registry.get("active")
        if act is None:
            return out
        self.ortho = act
        nb, bs, o = self.ortho.shapes[self.index]
        R = self.ortho.rotation(self.index).to(out.dtype)
        if out.dim() == 4:
            B, C, H, W = out.shape
            y = out[:, :nb * bs].reshape(B, nb, bs, H * W)
            y = torch.einsum("nij,bnjk->bnik", R, y).reshape(B, nb * bs, H, W)
            return torch.cat([y, out[:, nb * bs:]], 1) if nb * bs < C else y
        B, C = out.shape
        y = out[:, :nb * bs].reshape(B, nb, bs)
        y = torch.einsum("nij,bnj->bni", R, y).reshape(B, nb * bs)
        return torch.cat([y, out[:, nb * bs:]], 1) if nb * bs < C else y


# ── 3. classifier guidance: no per-concept parameters at all ─────────────────
class GuidanceHead(nn.Module):
    """A small classifier on NOISY images, used to steer sampling toward target attributes.

    Cost scales with the number of ATTRIBUTES, not with the number of concepts, so a stream of
    compositions over a fixed vocabulary is free after the head is trained once.  Composition is
    literally addition -- the guidance for (digit d, foreground f, background b) is the sum of three
    log-probability gradients -- which makes it the most direct statement of the compositional
    hypothesis anywhere in this repo.

    It also needs no conditioning channel, which is what blocked the pretrained model in 5u: this
    works on an unconditional backbone unchanged.
    """

    def __init__(self, img=32, sizes=(10, 4, 4), base=64, device="cuda"):
        super().__init__()
        self.sizes = list(sizes)
        c = base
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
        self.u_emb = nn.Linear(1, c * 4).to(device)
        self.heads = nn.ModuleList([nn.Linear(c * 4, s).to(device) for s in sizes])
        self.img = int(img)
        self.device = device

    def forward(self, x, u):
        h = self.net(x) + self.u_emb(u.reshape(-1, 1))
        return [hd(h) for hd in self.heads]

    def fit(self, X, labels, epochs=8, batch=64, lr=2e-4, seed=0, verbose=True):
        """Train on images noised across the whole schedule, so guidance is valid at every step."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        Xt = torch.from_numpy(np.asarray(X, np.float32)).to(self.device)
        Xt = Xt.reshape(-1, 3, self.img, self.img)
        Y = torch.from_numpy(np.asarray(labels, np.int64)).to(self.device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(epochs):
            perm = torch.randperm(len(Xt), generator=g)
            tot, nb = 0.0, 0
            for i0 in range(0, len(Xt), batch):
                idx = perm[i0:i0 + batch].to(self.device)
                x0, y = Xt[idx], Y[idx]
                u = torch.rand(len(x0), device=self.device).clamp_(1e-3, 1.0)
                ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(self.device)
                ab = ab.float().view(-1, 1, 1, 1)
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
                outs = self(xt, u)
                loss = sum(F.cross_entropy(o, y[:, k]) for k, o in enumerate(outs))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss); nb += 1
            if verbose:
                print(f"      guidance head ep {ep + 1}/{epochs}: {tot / max(nb, 1):.4f}", flush=True)
        for q in self.parameters():
            q.requires_grad_(False)
        return self

    @torch.enable_grad()
    def grad_logp(self, x, u, target):
        """d/dx sum_k log p(target_k | x, u) -- the composition, as a sum of gradients."""
        x = x.detach().requires_grad_(True)
        outs = self(x, u)
        lp = sum(F.log_softmax(o, -1)[:, int(t)].sum() for o, t in zip(outs, target))
        return torch.autograd.grad(lp, x)[0]


def sample_guided(field, head, target, n=16, steps=50, seed=0, scale=1.0, clip=(-1, 1), eta=0.0):
    """DDIM on the frozen base model, steered by the summed attribute gradients.

    No adapter, no concept embedding, no consolidation -- the base manifold is untouched and the
    composition exists only in the guidance term.  That makes this the cleanest possible test of
    whether composition needs to be learned into weights at all.

    This delegates to `BaseScoreField.sample`, the sampler every other experiment uses, rather than
    re-implementing DDIM.  A hand-rolled loop here produced samples that scored at chance even with
    the guidance switched OFF (digit 0.09 at scale 0), which would have been misreported as a failure
    of classifier guidance rather than of the loop around it.  Expressed through `guide_eps` with
    weight 1, the composed epsilon is exactly `eps - scale * sqrt(1-abar) * grad log p(target)`.
    """
    from .ddpm import _as_img

    def guide_eps(X, u):
        e = field.eps(X, u)
        xt = _as_img(X, field.device, field.img)
        ut = torch.full((len(xt),), float(u), device=field.device)
        with torch.enable_grad():
            gr = head.grad_logp(xt, ut, target)
        gr = gr.reshape(len(xt), -1).double().cpu().numpy()
        return e - scale * float(1 - cosine_abar(np.array([u]))[0]) ** 0.5 * gr

    return field.sample(n=n, steps=steps, seed=seed, clip=clip, eta=eta,
                        guide_eps=guide_eps, guide_w=lambda u: 1.0)
