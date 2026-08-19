"""Present a Score-SDE VARIANCE-EXPLODING model through our VARIANCE-PRESERVING interface.

`eurecom-ds/scoresdeve-ema-shapes3d-64` is the pretrained unconditional Shapes3D backbone this
project needs -- a model that already renders the domain, so conditioning can be fine-tuned into it
the way N47 does for CelebA instead of learned from scratch alongside the images (which failed three
times: samples excellent, label ignored).  But it speaks a different diffusion dialect, and getting
that wrong is silent: feeding it a VP timestep produced eps-loss pinned at exactly 1.0000 for 1200
steps, which is what predicting zero scores.

THE CORRESPONDENCE IS EXACT, so no approximation is involved:

    VP:        x_t = sqrt(a) x0 + sqrt(1-a) eps
    /sqrt(a):  x_t / sqrt(a) = x0 + sigma eps,      sigma = sqrt((1-a)/a)

so a VP sample at abar=a IS a VE sample at that sigma once divided by sqrt(a).

WHAT THE NETWORK RETURNS WAS MEASURED, NOT ASSUMED (`tests/diag_ve_parameterisation.py`).  Four
candidate readings against two input ranges, scored by how well each recovers a KNOWN eps:

    input [0,1], eps = -output * sigma      MSE 0.0002 - 0.0005 across sigma 0.05 .. 20
    every other combination                 MSE 0.64 .. 441

So: images in [0,1], the sigma itself as the timestep argument, and the output is a SCORE.

`sigma_max = 90` is the model's training ceiling; VP `u -> 1` implies sigma -> 300+, well outside it,
so sigma is clamped and the corresponding abar floor is reported rather than hidden.
"""
import numpy as np
import torch
import torch.nn as nn

from ..score import cosine_abar

SIGMA_MAX = 90.0          # the checkpoint's ScoreSdeVeScheduler sigma_max
SIGMA_MIN = 0.01


class VEToVPUNet(nn.Module):
    """A VE `UNet2DModel` wearing our UNet's interface: `forward(x, u, c=None, emb_override=None)`.

    `x` is [B, 3, H, W] in [0, 1] and `u` in [0, 1] is our cosine noise level; the return is an
    EPSILON prediction, so `conditional_losses`, `compose`, `route_signals` and the samplers all work
    against it unchanged.

    Conditioning is added to the time embedding by `_CondTimeEmbedding`, which is the same operation
    diffusers' `class_embed_type` performs and the same place our own UNet accepts `emb_override`.
    """

    def __init__(self, unet, n_classes, edim=None):
        super().__init__()
        from .pretrained import _CondTimeEmbedding
        self.unet = unet
        d = int(edim or unet.time_embedding.linear_2.out_features)
        self.edim = d
        self.n_classes, self.null_id = n_classes, n_classes
        self.cemb = nn.Embedding(n_classes + 1, d)
        nn.init.normal_(self.cemb.weight, std=0.02)
        with torch.no_grad():
            self.cemb.weight[self.null_id].zero_()
        if not isinstance(unet.time_embedding, _CondTimeEmbedding):
            unet.time_embedding = _CondTimeEmbedding(unet.time_embedding)

    def class_table(self):
        return self.cemb.weight[:self.n_classes].detach()

    @property
    def emb_dim(self):
        """Alias for `edim`, because `HierarchicalAdapterBank` asks for this name.

        A pure naming mismatch, and it stopped the storage loop dead on this backbone --
        `_register` does `torch.zeros(self.model.emb_dim)` and every other conditioning module here
        (`UNet`, `DiT`, `CondUNetShim`) exposes `emb_dim` while this one exposes `edim`. It is the
        smallest possible instance of the contract problem: what a backbone must implement to work
        with the bank is written down nowhere, so each new family discovers it by traceback.
        """
        return self.edim

    @staticmethod
    def sigma_of(u):
        """our cosine u -> the VE sigma, clamped to the checkpoint's trained range."""
        a = np.clip(np.asarray(cosine_abar(np.atleast_1d(np.asarray(u, np.float64)))),
                    1.0 / (1.0 + SIGMA_MAX ** 2), 1.0)
        return np.sqrt((1.0 - a) / a), a

    def forward(self, x, u, c=None, emb_override=None):
        dev = x.device
        u = torch.as_tensor(u, device=dev, dtype=torch.float32).reshape(-1)
        if u.numel() == 1:
            u = u.expand(len(x))
        sig_np, a_np = self.sigma_of(u.detach().cpu().numpy())
        sig = torch.as_tensor(sig_np, device=dev, dtype=torch.float32).clamp(SIGMA_MIN, SIGMA_MAX)
        a = torch.as_tensor(a_np, device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
        if emb_override is not None:
            ce = torch.as_tensor(emb_override, dtype=torch.float32, device=dev).reshape(-1, self.edim)
            if len(ce) == 1:
                ce = ce.expand(len(x), -1)
        else:
            cid = (torch.full((len(x),), self.null_id, device=dev, dtype=torch.long)
                   if c is None else torch.as_tensor(c, device=dev, dtype=torch.long))
            ce = self.cemb(cid)
        self.unet.time_embedding.set_extra(ce)
        out = self.unet(x / a.sqrt(), sig).sample          # VE input; sigma as the timestep
        return -out * sig.view(-1, 1, 1, 1)                 # measured: eps = -score * sigma


def load_ve_field(path, device="cuda"):
    """Load an n57 checkpoint as a VP score field.  -> (field-like object, ck).

    Returns something with `.model`, `.device`, `.img` and `.sample(...)`, which is the surface every
    downstream tool touches (`conditional_losses`, `sample_factored_pool`, `route_signals`).  The
    point of the bridge is that none of them need to know the backbone is variance-exploding.
    """
    import torch
    from diffusers import UNet2DModel

    ck = torch.load(path, map_location="cpu", weights_only=False)
    unet = UNet2DModel.from_pretrained(ck["model"], subfolder="unet")
    unet.config.sample_size = ck["img"]
    unet.sample_size = ck["img"]
    m = VEToVPUNet(unet, n_classes=ck["n_classes"])
    missing = m.unet.load_state_dict({k: v for k, v in ck["ema"].items()}, strict=False)
    with torch.no_grad():
        m.cemb.weight.copy_(ck["ema_cemb"])
    m = m.to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)

    class _Field:
        """The minimal score-field surface, in VP terms, over a VE backbone."""

        def __init__(self, model, img, device):
            self.model, self.img, self.device = model, img, device
            self.layout = "hwc"
            # DATA space [0,1] -- the VE backbone was trained on [0,1] images, which is why this
            # class's `sample` clamps to (0.0, 1.0) rather than to either generic range. Declared
            # explicitly so `instrument.assert_space` and `score.default_clip` agree with it; the
            # generic data-space clip (-0.2, 1.2) would have changed every Shapes3D number.
            self.data_space = "data"
            self.clip_default = (0.0, 1.0)

        @torch.no_grad()
        def sample(self, n=16, steps=50, seed=0, clip=(0.0, 1.0), branch=None,
                   emb_override=None, guidance=1.0, **unsupported):
            """DDIM over the VE backbone in VP terms.

            `**unsupported` ACCEPTS the wider keyword surface every other field in this repository
            offers, and is deliberately not a silent swallow. Five separate runs died here on
            TypeErrors -- `eta`, then `u_start` from inside `sample_concept` -- because this
            sampler's signature is narrower than `BaseScoreField`'s and callers reasonably assume
            otherwise. Patching each caller moved the fault rather than fixing it.

            An argument at its DEFAULT is a no-op and is ignored. An argument at a NON-default value
            would change the result, so it raises: silently dropping `eta=0.5` or a `guide_eps`
            would produce a plausible number computed by a different method, which is the exact
            failure mode this file's own history is made of.
            """
            NOOP = {"u_start": 1.0, "x_init": None, "eta": 0.0,
                    "guide_eps": None, "guide_w": None, "mcmc": None}
            for k, v in unsupported.items():
                if k not in NOOP:
                    raise TypeError(f"_Field.sample got an unexpected keyword {k!r}")
                if v is not None and v != NOOP[k]:
                    raise NotImplementedError(
                        f"_Field.sample cannot honour {k}={v!r} -- this VE backbone implements "
                        f"neither guidance callbacks nor stochastic/partial trajectories. Passing "
                        f"it would silently change the method behind the number.")
            g = torch.Generator(device=self.device).manual_seed(seed)
            sh = (n, 3, self.img, self.img)
            us = np.linspace(1.0, 0.0, steps + 1)
            a0 = float(cosine_abar(np.array([us[0]]))[0])
            # The VP prior at u=1 is CENTRED AT ZERO, not at the data mean.
            #
            #     x_t = sqrt(a) x0 + sqrt(1-a) eps   ->   a -> 0   ->   x_t -> eps ~ N(0, 1)
            #
            # The x0 term VANISHES, so there is no 0.5 offset even though the data lives in [0,1].
            # The first version wrote `0.5 + sqrt(1-a0) * randn`, which starts every trajectory in
            # the same biased basin; under deterministic DDIM (eta=0) they then funnel to one
            # washed-out mean scene regardless of conditioning. That is exactly what the Shapes3D
            # case study showed -- the SAME cyan-cube-on-orange image in every block and every arm,
            # with `ours` equal to the unconditional floor to three decimals.
            x = ((1 - a0) ** 0.5) * torch.randn(sh, generator=g, device=self.device)
            emb = None
            if emb_override is not None:
                emb = torch.as_tensor(np.asarray(emb_override, np.float32),
                                      device=self.device).reshape(1, -1)
            elif branch is not None:
                emb = self.model.cemb.weight[int(branch)].reshape(1, -1)
            for i in range(steps):
                u, un = float(us[i]), float(us[i + 1])
                a, an = (float(cosine_abar(np.array([u]))[0]),
                         float(cosine_abar(np.array([un]))[0]))
                e = self.model(x, torch.full((n,), u, device=self.device),
                               emb_override=None if emb is None else emb.expand(n, -1))
                x0 = (x - (1 - a) ** 0.5 * e) / (a ** 0.5)
                if clip is not None:
                    x0 = x0.clamp(*clip)
                x = (an ** 0.5) * x0 + ((1 - an) ** 0.5) * e
            return x.permute(0, 2, 3, 1).reshape(n, -1).cpu().numpy()

        # ── the discovery surface ────────────────────────────────────────────
        # Added so `compose.candidates_ascent` / `poe_teacher` -- the companion paper's Eq. 4 and
        # Eq. 7-10 -- can run on this backbone at all.  Without them the shim exposed only
        # `sample`, so the paper's discovery had NEVER been run on Shapes3D: N70's "mu_T" was a
        # 4-start single-noise-level unweighted mean of modes standing in for it, which is not a
        # fair test of their method and must not be reported as one.
        #
        # These are ports, not re-uses.  `NeuralScoreField`'s versions operate in MODEL space
        # [-1,1] (`to_model` is a range transform); this field works in DATA space [0,1] -- its
        # sampler clips x0 to (0,1) and `n61`'s loader passes rows through unchanged for exactly
        # that reason.  Calling the inherited methods would be a silent convention mismatch of the
        # kind that has cost this repository three results, so the arithmetic is repeated here
        # against this field's own convention rather than shared.

        def _as_chw(self, X):
            X = np.atleast_2d(np.asarray(X, np.float32))
            return torch.as_tensor(X, device=self.device).reshape(-1, self.img, self.img,
                                                                  3).permute(0, 3, 1, 2)

        @torch.no_grad()
        def eps(self, X, u, branch=None, emb=None):
            """Epsilon prediction for flat DATA-space rows.  -> [N, D] in the same layout."""
            x = self._as_chw(X)
            e = None
            if emb is not None:
                e = torch.as_tensor(np.asarray(emb, np.float32),
                                    device=self.device).reshape(1, -1).expand(len(x), -1)
            elif branch is not None:
                e = self.model.cemb.weight[int(branch)].reshape(1, -1).expand(len(x), -1)
            out = self.model(x, torch.full((len(x),), float(u), device=self.device),
                             emb_override=e)
            return out.permute(0, 2, 3, 1).reshape(len(x), -1).cpu().numpy()

        def score(self, X, u, branch=None):
            a = float(cosine_abar(np.array([float(u)]))[0])
            return -self.eps(X, u, branch) / np.sqrt(max(1.0 - a, 1e-12))

        def laplace_diag(self, xt, u, probes=4, delta=5e-3, seed=0, var_floor=1e-4, var_cap=1.0):
            """Diagonal Laplace covariance via Hutchinson finite differences, pulled to x0-space."""
            rng = np.random.default_rng(seed)
            xt = np.atleast_2d(np.asarray(xt, np.float32))
            a = float(cosine_abar(np.array([float(u)]))[0])
            h = np.zeros_like(xt)
            for _ in range(probes):
                v = rng.choice([-1.0, 1.0], size=xt.shape)
                h += v * (self.score(xt + delta * v, u)
                          - self.score(xt - delta * v, u)) / (2 * delta)
            h /= probes
            neg = h < -1e-9
            st = np.full_like(h, var_cap)
            np.divide(-1.0, h, out=st, where=neg)
            return np.clip((st - (1.0 - a)) / a, var_floor, var_cap)

        def mode_ascent(self, xq, u, n_starts=16, iters=100, lr=0.05, seed=0, init="query",
                        clamp=False, k_sigma=3.0, tweedie=False, adam=True):
            """Adam ascent on log p_t.  -> (x_t rows, x_0 rows), both flat in this field's layout."""
            a = float(cosine_abar(np.array([float(u)]))[0])
            g = torch.Generator(device=self.device).manual_seed(int(seed))
            xq_t = self._as_chw(xq)[:1]
            half = n_starts // 2 if init == "hybrid" else (n_starts if init == "query" else 0)
            parts = []
            if half:
                e = torch.randn((half, *xq_t.shape[1:]), generator=g, device=self.device)
                parts.append((a ** 0.5) * xq_t.expand(half, -1, -1, -1) + ((1 - a) ** 0.5) * e)
            if n_starts - half:
                parts.append(torch.randn((n_starts - half, *xq_t.shape[1:]), generator=g,
                                         device=self.device))
            xt = torch.cat(parts).clone().requires_grad_(True)
            opt = (torch.optim.Adam([xt], lr=lr) if adam
                   else torch.optim.SGD([xt], lr=lr))
            uu = torch.full((len(xt),), float(u), device=self.device)
            for _ in range(iters):
                with torch.enable_grad():
                    ep = self.model(xt, uu)
                    loss = (ep.detach() * xt).sum() / max((1 - a) ** 0.5, 1e-6)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if clamp:
                    with torch.no_grad():
                        xt.clamp_(-k_sigma, 1.0 + k_sigma)
            with torch.no_grad():
                ep = self.model(xt, uu)
                x0 = (xt - ((1 - a) ** 0.5) * ep) / (a ** 0.5)
            flat = lambda z: z.detach().permute(0, 2, 3, 1).reshape(len(z), -1).cpu().numpy()
            return flat(xt), flat(x0)

    return _Field(m, int(ck["img"]), device), ck
