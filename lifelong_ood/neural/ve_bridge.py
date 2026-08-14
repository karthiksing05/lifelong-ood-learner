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
