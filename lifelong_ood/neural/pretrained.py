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

from ..backbones.schedule import cosine_abar
from ..score import BaseScoreField


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
    # MODEL space [-1,1] -- diffusers UNet2DModel checkpoints are trained on [-1,1] images. This
    # class already overrode `sample`'s clip default to (-1,1) for exactly this reason; declaring it
    # makes the same fact available to `default_clip` and to `instrument.assert_space`.
    data_space = "model"

    """A diffusers `UNet2DModel` behind the same `eps(x, u, ...)` primitive everything else uses.

    `u` is this repo's normalized noise level in [0, 1] under the cosine schedule; diffusers models
    are trained on integer timesteps under their own schedule.  The two are reconciled by mapping u
    to the timestep whose cumulative alpha is closest, so the model is always queried at the noise
    level it was trained for rather than at a rescaled index.

    `self.model` is a `CondUNetShim`, not the raw diffusers module, so that an adapter bank injected
    into it is active for BOTH training and sampling -- a bank attached to one and bypassed by the
    other would silently evaluate the frozen model.
    """

    def __init__(self, model, scheduler=None, img=256, device="cuda", batch=4, layout="hwc"):
        unet = model.to(device).eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        self.device = device
        self.img = int(img)
        self.batch = int(batch)
        # DEFAULT "hwc" here, unlike our own backbones, and the asymmetry is deliberate.  A model we
        # trained ourselves learned in whatever space we fed it, so its default must stay "permuted"
        # or every existing checkpoint breaks.  A PRETRAINED model did not: its weights are fixed and
        # were fitted to real images, so the only correct thing to hand it is a real image.
        self.layout = str(layout)
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

    def _u_for_timestep(self, t):
        """The u whose cosine alpha_bar matches THIS model's alpha_bar at timestep `t`.

        The inverse of `_timestep`, and it has to exist: guidance callbacks take u on this repo's
        cosine grid, but the sampler walks the model's own integer grid.  Handing a callback
        `t/(T-1)` would send it to a different noise level than the one being denoised -- which
        renders as pure noise, since the composed epsilon then describes the wrong marginal.
        """
        if self._abar is None:
            return float(t) / 999.0
        grid = np.linspace(1e-4, 1.0, 2001)
        return float(grid[int(np.argmin(np.abs(cosine_abar(grid) - float(self._abar[int(t)]))))])

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

    def _to_chw(self, t):
        """flat rows -> (N, 3, H, W).  `layout="hwc"` transposes properly; "permuted" reshapes."""
        if self.layout == "hwc":
            return t.reshape(-1, self.img, self.img, 3).permute(0, 3, 1, 2).contiguous()
        return t.reshape(-1, 3, self.img, self.img)

    def _to_flat(self, t):
        """(N, 3, H, W) -> flat rows in this repository's HWC convention.  Inverse of `_to_chw`."""
        if self.layout == "hwc":
            return t.permute(0, 2, 3, 1).reshape(len(t), -1)
        return t.reshape(len(t), -1)

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
                # TRUE CHW.  This was a plain `.reshape(-1, 3, H, W)` of an HWC row, i.e. a pixel
                # PERMUTATION -- see tests/diag_layout.py.  On OUR backbones that merely costs the
                # convolutional prior, because training and sampling share the permutation.  Here it
                # is worse in kind: `google/ddpm-celebahq-256` has FIXED weights learned on real
                # images, so handing it scrambled input is not a lost prior but wrong input to a
                # frozen function.  The paper's entire CelebA setup is built on this backbone.
                xb = self._to_chw(xb)
                e = self.model(xb, float(u), emb_override=ev)
                out[i0:i0 + self.batch] = self._to_flat(e).double().cpu().numpy()
        return out

    @torch.no_grad()
    def sample(self, n=8, steps=50, seed=0, clip=(-1, 1), emb=None, branch=None, eta=0.0,
               guide_eps=None, guide_w=None, u_start=1.0, x_init=None):
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
        # `u_start` / `x_init` -- the prototype-distance dial (§20), and what `sample_teacher` needs
        # to run at all.  Their absence here is why the pretrained track (§12) could only ever use
        # adapters: the analytic-teacher route calls sample() with both, so `poe-guided` -- the
        # companion paper's OWN method -- could not be run on a pretrained backbone.  Since this
        # field integrates on the model's NATIVE timestep grid, u_start is converted to a start
        # INDEX on that grid rather than used as a cosine-u directly.
        t_start = int(round(float(np.clip(u_start, 0.0, 1.0)) * (T - 1)))
        ts_seq = np.linspace(t_start, 0, steps + 1).round().astype(int)
        if x_init is not None:
            xi = np.atleast_2d(np.asarray(x_init, np.float32))
            xi = np.repeat(xi, n, 0) if len(xi) == 1 else xi[:n]
            xi = self._to_chw(torch.from_numpy(xi).to(self.device))
            ab0 = (float(self._abar[t_start]) if self._abar is not None
                   else float(cosine_abar(np.array([t_start / T]))[0]))
            x = ab0 ** 0.5 * xi + (1 - ab0) ** 0.5 * x
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
                uu = self._u_for_timestep(t)
                gflat = np.asarray(guide_eps(self._to_flat(x).double().cpu().numpy(), uu))
                ge = self._to_chw(torch.from_numpy(gflat).float().to(self.device))
                w = 1.0 if guide_w is None else float(np.asarray(guide_w(uu)).reshape(-1)[0])
                e = e + w * (ge - e)
            x0 = (x - (1 - ab) ** 0.5 * e) / max(ab ** 0.5, 1e-8)
            if clip is not None:
                x0 = x0.clamp(*clip)
            x = abn ** 0.5 * x0 + (1 - abn) ** 0.5 * e
        return self._to_flat(x).double().cpu().numpy()


def load_pretrained(name="google/ddpm-celebahq-256", device="cuda", batch=4, layout="hwc"):
    """-> (field, img).  Downloads on first use into HF_HOME.

    `layout` defaults to "hwc" -- a real image -- because these weights are FIXED and were fitted to
    real images.  "permuted" reproduces the historical behaviour for comparison only.
    """
    from diffusers import DDPMScheduler, UNet2DModel
    model = UNet2DModel.from_pretrained(name)
    try:
        sched = DDPMScheduler.from_pretrained(name)
    except Exception:
        sched = None
    img = int(model.config.sample_size)
    return PretrainedScoreField(model, sched, img=img, device=device, batch=batch,
                                layout=layout), img

def load_paper_backbone(path, device="cuda", batch=4):
    """Rebuild the fine-tuned CelebA backbone n47 wrote -> (field, img, cemb, remap).

    This is the companion paper's Table 5 CelebA column: `google/ddpm-celebahq-256` fine-tuned at
    128px with a class-embedding table of |C|+1 rows added to the time embedding, on the model's own
    LINEAR beta schedule with T=1000.

    The schedule matters and is easy to get wrong here.  Everything else in this repository runs on
    a cosine abar, so a `u` means one thing on our backbones and another on this one; the field is
    therefore built with the model's OWN `DDPMScheduler`, whose `alphas_cumprod` is the linear ladder
    the fine-tune was trained against.  Passing our cosine u-grid to it would silently query the
    wrong noise levels.

    `cemb` is returned rather than installed, because conditioning here goes through the time
    embedding via `CondUNetShim.forward(..., emb_override=...)`, exactly as for any other concept
    vector in this repository -- so a class row and a learned concept are the same kind of object.
    """
    import torch as _t
    from diffusers import DDPMScheduler, UNet2DModel
    ck = _t.load(path, map_location="cpu", weights_only=False)
    unet = UNet2DModel.from_pretrained(ck["model"])
    unet.config.sample_size = int(ck["img"])
    unet.sample_size = int(ck["img"])
    try:
        sched = DDPMScheduler.from_pretrained(ck["model"])
    except Exception:
        sched = None
    # Build the field FIRST and load the weights after.  `PretrainedScoreField.__init__` wraps
    # `time_embedding` in `_CondTimeEmbedding` itself and then reads `time_embedding.inner.linear_2`
    # to size the conditioning; wrapping here as well double-wraps it, so `inner` becomes another
    # wrapper and the read fails with AttributeError.  n47 saves its state dict from an already
    # wrapped model, so the checkpoint's keys are the post-wrap ones and they line up exactly once
    # the field has done its wrapping.
    field = PretrainedScoreField(unet, sched, img=int(ck["img"]), device=device, batch=batch,
                                 layout=ck.get("layout", "hwc"))
    missing, unexpected = field.model.unet.load_state_dict(ck["ema"], strict=False)
    hard = [k for k in missing if "time_embedding" not in k]
    if hard:
        raise RuntimeError(f"load_paper_backbone: {len(hard)} weights missing from the checkpoint, "
                           f"e.g. {hard[:3]} -- this is not the model the checkpoint was written "
                           f"from")
    return field, int(ck["img"]), ck["ema_cemb"].numpy(), ck["remap"]
