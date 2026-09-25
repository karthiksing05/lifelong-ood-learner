"""The Backbone handle: one object that carries a model AND the conventions a caller cannot be trusted
to pass (layout, image size, the clip of its x0 space, which row space it speaks, the null token).

Three families load through `load_backbone(spec)`:

  * our UNet checkpoints (`results/cache/ddpm_*.pt`, keys ema / n_classes / arch / remap / layout):
    rows in MODEL space [-1, 1], clip (-1, 1), layout from the checkpoint ("permuted" if unrecorded);
  * the Shapes3D VE bridge (n57 checkpoints, keys model / ema / ema_cemb / img): rows in DATA space
    [0, 1], clip (0, 1), layout "hwc";
  * a diffusers `UNet2DModel` by name (`google/ddpm-cifar10-32`): rows in MODEL space, layout "hwc",
    queried at the trained timestep whose alpha-bar is closest to our cosine u.

Every downstream function takes the handle, never the bare model, so the sampler / reads / ascent
cannot be run in the wrong layout or range -- the two conventions whose silent mismatch cost the
previous repository three results (docs/REVIEW_2026-09.md, RESULTS 7bb / 7ee).
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import torch

from .layout import to_data, to_flat, to_img, to_model
from .schedule import cosine_abar


@dataclasses.dataclass
class Backbone:
    model: torch.nn.Module
    kind: str                       # "unet" | "ve" | "diffusers"
    layout: str                     # "permuted" | "hwc"
    img: int
    clip: tuple                     # clip of x0 in the row space
    space: str                      # "model" ([-1,1] rows) | "data" ([0,1] rows)
    null_id: int
    emb_dim: int
    n_params: int
    device: str
    path: str | None = None
    sha256: str | None = None
    meta: dict = dataclasses.field(default_factory=dict)

    # ── row-space conversions ────────────────────────────────────────────────
    @property
    def dim(self):
        return 3 * self.img * self.img

    def rows_from_data(self, X01):
        """[0,1] flat data rows -> rows in this backbone's space (float32)."""
        X = np.asarray(X01, np.float32)
        return to_model(X).astype(np.float32) if self.space == "model" else np.clip(X.reshape(len(X), -1), 0.0, 1.0)

    def rows_to_data(self, Xr):
        """rows in this backbone's space -> [0,1] flat data rows (float32)."""
        X = np.asarray(Xr, np.float32)
        return to_data(X).astype(np.float32) if self.space == "model" else np.clip(X.reshape(len(X), -1), 0.0, 1.0)

    def class_table(self):
        """[n_classes, emb_dim] tensor of the trained class embeddings, or None (unconditional)."""
        cemb = getattr(self.model, "cemb", None)
        if cemb is None:
            return None
        n = int(getattr(self.model, "n_classes", cemb.weight.shape[0] - 1))
        return cemb.weight[:n].detach()

    # ── one forward, with explicit conditioning ──────────────────────────────
    @torch.no_grad()
    def eps(self, X, u, cls=None, emb=None, gate=None, batch=256):
        """eps-prediction of flat rows X (numpy or torch, this backbone's space) at noise level u.
        `u` is a float or a per-row array.  -> torch [n, dim] on the device."""
        from .sampler import _cond, _gate
        Xt = torch.as_tensor(np.asarray(X, np.float32) if not torch.is_tensor(X) else X,
                             dtype=torch.float32, device=self.device)
        n = len(Xt)
        E = None if emb is None else torch.as_tensor(np.asarray(emb, np.float32) if not torch.is_tensor(emb) else emb,
                                                     dtype=torch.float32, device=self.device)
        uu = torch.as_tensor(np.asarray(u, np.float32), device=self.device).reshape(-1)
        if uu.numel() == 1:
            uu = uu.expand(n)
        out = torch.empty(n, self.dim, device=self.device)
        self.model.eval()
        with _gate(gate):
            for i0 in range(0, n, batch):
                xb = to_img(Xt[i0:i0 + batch], self.img, self.layout)
                m = len(xb)
                c, eo = _cond(self.model, m, self.device, emb=E, cls=cls, i0=i0)
                out[i0:i0 + m] = to_flat(self.model(xb, uu[i0:i0 + m], c=c, emb_override=eo), self.layout)
        return out

    def describe(self):
        return (f"Backbone(kind={self.kind}, layout={self.layout}, img={self.img}, space={self.space}, "
                f"clip={self.clip}, n_params={self.n_params:,}, emb_dim={self.emb_dim}, device={self.device})")


def _sha(path):
    try:
        from ..ledger import sha256_file
        return sha256_file(path)
    except Exception:
        return None


def _n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def load_backbone(spec, device="cuda", sha=True):
    """-> Backbone.  `spec` is a checkpoint path (.pt) or a diffusers model name."""
    if isinstance(spec, Backbone):
        return spec
    spec = str(spec)
    if spec.endswith(".pt") or os.path.exists(spec):
        ck = torch.load(spec, map_location="cpu", weights_only=False)
        if "arch" in ck and "ema" in ck and "ema_cemb" not in ck:
            return _load_unet(spec, ck, device, sha)
        if "ema_cemb" in ck and "model" in ck:
            return _load_ve(spec, ck, device, sha)
        raise ValueError(f"load_backbone: {spec} has keys {sorted(ck)[:8]}, not a known checkpoint family")
    return _load_diffusers(spec, device)


def _load_unet(path, ck, device, sha):
    from ..neural.unet import UNet
    model = UNet(n_classes=ck["n_classes"], **ck.get("arch", {}))
    model.load_state_dict(ck["ema"])
    model = model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    meta = {k: ck[k] for k in ("remap", "concepts", "dataset", "n_per_slot", "n_classes") if k in ck}
    return Backbone(model=model, kind="unet", layout=ck.get("layout", "permuted"), img=int(model.img),
                    clip=(-1.0, 1.0), space="model", null_id=int(model.null_id), emb_dim=int(model.emb_dim),
                    n_params=_n_params(model), device=device, path=path, sha256=_sha(path) if sha else None,
                    meta=meta)


def _load_ve(path, ck, device, sha):
    from diffusers import UNet2DModel
    from ..neural.ve_bridge import VEToVPUNet          # the bridge class; its _Field wrapper is legacy
    unet = UNet2DModel.from_pretrained(ck["model"], subfolder="unet")
    unet.config.sample_size = ck["img"]
    unet.sample_size = ck["img"]
    m = VEToVPUNet(unet, n_classes=ck["n_classes"])
    m.unet.load_state_dict({k: v for k, v in ck["ema"].items()}, strict=False)
    with torch.no_grad():
        m.cemb.weight.copy_(ck["ema_cemb"])
    m = m.to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    meta = {k: ck[k] for k in ("remap", "split", "n_classes", "model") if k in ck}
    return Backbone(model=m, kind="ve", layout=ck.get("layout", "hwc"), img=int(ck["img"]), clip=(0.0, 1.0),
                    space="data", null_id=int(m.null_id), emb_dim=int(m.edim), n_params=_n_params(m),
                    device=device, path=path, sha256=_sha(path) if sha else None, meta=meta)


def _load_diffusers(name, device):
    from diffusers import DDPMScheduler, UNet2DModel
    from ..neural.pretrained import CondUNetShim, _CondTimeEmbedding
    unet = UNet2DModel.from_pretrained(name)
    sched = DDPMScheduler.from_pretrained(name)
    abar = np.asarray(sched.alphas_cumprod, np.float64)
    if not isinstance(unet.time_embedding, _CondTimeEmbedding):
        unet.time_embedding = _CondTimeEmbedding(unet.time_embedding)
    edim = int(unet.time_embedding.inner.linear_2.out_features)

    def t_of_u(u):
        return int(np.argmin(np.abs(abar - float(cosine_abar(np.array([float(u)]))[0]))))
    m = CondUNetShim(unet, t_of_u, edim).to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return Backbone(model=m, kind="diffusers", layout="hwc", img=int(unet.config.sample_size), clip=(-1.0, 1.0),
                    space="model", null_id=0, emb_dim=edim, n_params=_n_params(m), device=device, path=name,
                    sha256=None, meta=dict(schedule="linear", T=int(len(abar)), abar=abar))
