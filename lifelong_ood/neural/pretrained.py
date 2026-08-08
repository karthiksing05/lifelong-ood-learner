"""Wrap a PRETRAINED diffusers UNet as a `BaseScoreField`, and give it a conditioning slot.

`google/ddpm-celebahq-256` is the paper's benchmark domain at a resolution and quality our own
backbone cannot reach (113.7M parameters, 256x256, trained on CelebA-HQ).  Testing on it answers a
question our ColorMNIST results cannot: does any of this survive contact with a real pretrained
model, or is it an artifact of a small model we trained ourselves?

The obstacle is that it is **unconditional**.  Its config has no `class_embed_type`, no
`num_class_embeds` and no `cross_attention_dim`, and `class_embedding` is None -- there is nothing to
condition on, so `emb_override` has nothing to override and the ideal-words basis has no embedding
table to factorize.  Everything this repo does in conditioning space is, on the face of it,
inapplicable.

What makes it applicable is that the conditioning slot our own UNet has is not exotic: a vector
added to the time embedding before it reaches the residual blocks.  Any DDPM has a time embedding,
so the slot can be ADDED to a pretrained model without retraining it -- `time_embedding` is wrapped
so its output gains a settable vector.  The pretrained weights are untouched and the model's
unconditional behaviour is exactly recovered by setting the vector to zero.

That gives a fair test.  There is no pretrained basis to extend -- the model never learned classes --
so the basis has to be BUILT from data, one direction per attribute value, which is `extend_basis`
used for its other purpose.  Whether that works is the transfer question.
"""
import contextlib

import numpy as np
import torch

from ..score import BaseScoreField, cosine_abar


@contextlib.contextmanager
def _null_ctx():
    yield


class _CondTimeEmbedding(torch.nn.Module):
    """Wraps the pretrained time embedding so its output can be offset by a learned vector.

    Additive on the time embedding is exactly where our own UNet accepts `emb_override`, so a concept
    vector learned here means the same thing it means there.  `extra=None` reproduces the pretrained
    model bit-for-bit.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        # kept OUT of nn.Module's attribute machinery on purpose.  The conditioning vector handed in
        # here is often the bank's `nn.Parameter`, and a plain `self.extra = p` would REGISTER it as
        # a parameter of this module -- after which assigning an ordinary tensor raises, and the
        # vector would silently join any optimizer built over the model's parameters.
        object.__setattr__(self, "extra", None)   # [d] or [B, d], set per forward by the field

    def set_extra(self, v):
        object.__setattr__(self, "extra", v)

    def forward(self, *a, **kw):
        e = self.inner(*a, **kw)
        if self.extra is None:
            return e
        x = self.extra.to(e.dtype)
        return e + (x.reshape(1, -1) if x.dim() == 1 else x.reshape(e.shape[0], -1))


class CondUNetShim(torch.nn.Module):
    """Presents a pretrained diffusers UNet through THIS repo's UNet interface.

    Existence of this class is the fix N15 asked for.  That experiment conditioned the pretrained
    model with a single additive vector on its 512-d time embedding and found the bottleneck was the
    conditioning CHANNEL, not the composer: fitting one combination directly (`per-combo`, 0.167)
    barely beat the untouched model (`uncond`, 0.146).  One vector into a 113.7M-parameter network
    cannot move it, whatever it is optimized against.

    The capacity has to come from the same place it comes from for every method that works on
    pretrained diffusion models -- a weight delta -- and this repository already has one: the
    insert-only LoRA bank.  `inject()` walks an arbitrary `nn.Module`, so it attaches to a diffusers
    UNet as readily as to ours; what was missing is the three-attribute interface the bank and the
    lifelong learner expect (`emb_dim`, `null_id`, and `forward(x, u, emb_override=...)`).  That is
    all this class supplies.  Preservation is unaffected and still structural: the pretrained weights
    are frozen, ancestors are sealed, and a concept is a new leaf.

    `u` is this repo's normalized noise level in [0, 1]; the wrapped model is queried at the trained
    timestep whose cumulative alpha is closest, so it always sees a noise level it was trained for.
    """

    def __init__(self, unet, timestep_fn, emb_dim):
        super().__init__()
        self.unet = unet
        self._timestep_fn = timestep_fn
        self._emb_dim = int(emb_dim)
        self.null_id = 0                     # nothing to look up; the bank only needs the attribute
        self.cond_mode = "class"

    @property
    def emb_dim(self):
        return self._emb_dim

    def n_params(self):
        return sum(p.numel() for p in self.unet.parameters())

    def forward_t(self, x, ts, emb_override=None):
        """Query at NATIVE trained timesteps.  `ts` is an int or a [B] LongTensor.

        The sampler walks the model's own timestep grid, so it must not round-trip an integer
        timestep through u and back -- that mapping is only approximately invertible and would query
        a neighbouring noise level at every step.
        """
        if not torch.is_tensor(ts):
            ts = torch.full((x.shape[0],), int(ts), device=x.device, dtype=torch.long)
        self.unet.time_embedding.set_extra(emb_override)
        try:
            return self.unet(x, ts.to(x.device)).sample
        finally:
            self.unet.time_embedding.set_extra(None)

    def forward(self, x, u, emb_override=None, **kw):
        """x: [B, 3, img, img]; u: scalar or [B] in [0, 1].  -> predicted epsilon."""
        uu = u if torch.is_tensor(u) else torch.as_tensor(u)
        if uu.numel() > 1:
            # a batch may carry one noise level per row; map each to its own trained timestep
            ts = torch.as_tensor([self._timestep_fn(float(v)) for v in uu.reshape(-1)],
                                 device=x.device, dtype=torch.long)
        else:
            ts = torch.full((x.shape[0],), int(self._timestep_fn(float(uu.reshape(-1)[0]))),
                            device=x.device, dtype=torch.long)
        return self.forward_t(x, ts, emb_override)


class PretrainedScoreField(BaseScoreField):
    """A diffusers `UNet2DModel` behind the same `eps(x, u, ...)` primitive everything else uses.

    `u` is this repo's normalized noise level in [0, 1] under the cosine schedule; diffusers models
    are trained on integer timesteps under their own schedule.  The two are reconciled by mapping u
    to the timestep whose cumulative alpha is closest, so the model is always queried at the noise
    level it was trained for rather than at a rescaled index.

    `self.model` is a `CondUNetShim`, not the raw diffusers module, so that an adapter bank injected
    into it is active for BOTH training and sampling -- a bank attached to one and bypassed by the
    other would silently evaluate the frozen model.
    """

    def __init__(self, model, scheduler=None, img=256, device="cuda", batch=4):
        unet = model.to(device).eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        self.device = device
        self.img = int(img)
        self.batch = int(batch)
        self.scheduler = scheduler
        unet.time_embedding = _CondTimeEmbedding(unet.time_embedding)
        # the timestep whose trained alpha_bar matches each queried u
        if scheduler is not None:
            self._abar = scheduler.alphas_cumprod.detach().cpu().numpy()
        else:
            self._abar = None
        # read the width BEFORE any LoRA injection replaces linear_2 with a wrapper
        emb_dim = int(unet.time_embedding.inner.linear_2.out_features)
        self.model = CondUNetShim(unet, self._timestep, emb_dim).to(device).eval()

    @property
    def dim(self):
        return 3 * self.img * self.img

    @property
    def unet(self):
        return self.model.unet

    @property
    def emb_dim(self):
        return self.model.emb_dim

    def _timestep(self, u):
        """Map this repo's u in [0,1] to the trained timestep with the closest alpha_bar."""
        if self._abar is None:
            return int(round(float(u) * 999))
        target = float(cosine_abar(np.array([u]))[0])
        return int(np.argmin(np.abs(self._abar - target)))

    def abar(self, u):
        """The model's OWN cumulative alpha at the timestep `u` maps to.

        Not `cosine_abar(u)`.  This model was trained on a linear beta schedule, and using this
        repo's cosine value in the DDIM update while the epsilon came from the matched NATIVE
        timestep makes the x0 estimate inconsistent with the prediction -- visible as a colour cast
        on otherwise correct faces, which is how the mismatch was found.
        """
        if self._abar is None:
            return float(cosine_abar(np.array([u]))[0])
        return float(self._abar[self._timestep(u)])

    @torch.no_grad()
    def eps(self, X, u, branch=None, emb=None):
        X = np.atleast_2d(np.asarray(X, np.float64))
        out = np.empty_like(X)
        ev = (None if emb is None else
              (emb if torch.is_tensor(emb) else
               torch.as_tensor(np.asarray(emb, np.float32), device=self.device)))
        # `branch` selects a consolidated concept's adapter path, exactly as on our own backbone
        ctx = (self.bank.activate(branch) if (branch is not None and getattr(self, "bank", None))
               else _null_ctx())
        with ctx:
            for i0 in range(0, len(X), self.batch):
                xb = torch.from_numpy(X[i0:i0 + self.batch]).float().to(self.device)
                xb = xb.reshape(-1, 3, self.img, self.img)
                e = self.model(xb, float(u), emb_override=ev)
                out[i0:i0 + self.batch] = e.reshape(len(xb), -1).double().cpu().numpy()
        return out

    @torch.no_grad()
    def sample(self, n=8, steps=50, seed=0, clip=(-1, 1), emb=None, branch=None, eta=0.0,
               guide_eps=None, guide_w=None):
        """DDIM on the model's NATIVE timestep grid.

        An earlier version walked this repo's cosine u-grid and mapped each u to the nearest trained
        timestep.  That distorts the trajectory even with the right per-step alphas -- the samples
        came out systematically dark and green-shifted against the diffusers reference (channel means
        R 0.338 / G 0.365 / B 0.177 against 0.531 / 0.477 / 0.232).  A pretrained model should be
        integrated on the schedule it was trained on; the u-grid stays as the interface for `eps`,
        where the caller genuinely wants a specific noise level.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(n, 3, self.img, self.img, generator=g).to(self.device)
        T = len(self._abar) if self._abar is not None else 1000
        ts_seq = np.linspace(T - 1, 0, steps + 1).round().astype(int)
        for i in range(steps):
            t, tn = int(ts_seq[i]), int(ts_seq[i + 1])
            ab = float(self._abar[t]) if self._abar is not None else float(cosine_abar(np.array([t / T]))[0])
            abn = float(self._abar[tn]) if self._abar is not None else float(cosine_abar(np.array([tn / T]))[0])
            ev = (None if emb is None else
                  (emb if torch.is_tensor(emb) else
                   torch.as_tensor(np.asarray(emb, np.float32), device=self.device)))
            ctx = (self.bank.activate(branch) if (branch is not None and getattr(self, "bank", None))
                   else _null_ctx())
            with ctx:
                e = torch.empty_like(x)
                for i0 in range(0, n, self.batch):
                    xb = x[i0:i0 + self.batch]
                    e[i0:i0 + self.batch] = self.model.forward_t(xb, t, emb_override=ev)
            if guide_eps is not None:
                # classifier-free-style guidance toward an externally composed epsilon, so score
                # composition (composable diffusion, a PoE teacher) works on a pretrained backbone
                # too.  `guide_eps` follows the repo convention: flat float64 rows, u in [0, 1].
                uu = float(t) / max(T - 1, 1)
                gflat = np.asarray(guide_eps(x.reshape(n, -1).double().cpu().numpy(), uu))
                ge = torch.from_numpy(gflat).float().to(self.device).reshape_as(x)
                w = 1.0 if guide_w is None else float(np.asarray(guide_w(uu)).reshape(-1)[0])
                e = e + w * (ge - e)
            x0 = (x - (1 - ab) ** 0.5 * e) / max(ab ** 0.5, 1e-8)
            if clip is not None:
                x0 = x0.clamp(*clip)
            x = abn ** 0.5 * x0 + (1 - abn) ** 0.5 * e
        return x.reshape(n, -1).double().cpu().numpy()


def load_pretrained(name="google/ddpm-celebahq-256", device="cuda", batch=4):
    """-> (field, img).  Downloads on first use into HF_HOME."""
    from diffusers import DDPMScheduler, UNet2DModel
    model = UNet2DModel.from_pretrained(name)
    try:
        sched = DDPMScheduler.from_pretrained(name)
    except Exception:
        sched = None
    img = int(model.config.sample_size)
    return PretrainedScoreField(model, sched, img=img, device=device, batch=batch), img
