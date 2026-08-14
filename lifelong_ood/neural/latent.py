"""A small convolutional autoencoder, so a latent-space diffusion model can be trained on ColorMNIST.

This exists for one question: **is the three-tier architecture a property of the METHOD or of the
UNet we happened to use?**  Everything measured so far runs on a pixel-space UNet whose class
embedding is added to the time embedding.  A DiT conditions through adaLN modulation driven by a
class embedding table, operates on LATENT patches rather than pixels, and is the family that
vision-language-action models are built from.  If the factor basis still emerges there and ADDRESS
still composes, the claim is architectural rather than incidental.

Latent diffusion needs an autoencoder first, and this is deliberately the plainest one that works:

  * DETERMINISTIC, not a VAE.  A KL term buys a smooth prior that matters for text-to-image sampling
    and costs a hyperparameter and a scale factor we would then have to defend.  Nothing here needs
    a prior over latents -- the diffusion model learns the latent distribution itself.
  * 32x32x3 -> 8x8x4, a 6x compression.  Small enough that a DiT over 16 patches trains in minutes,
    large enough that the reconstruction keeps digit shape AND both colours, which is what the probes
    read.
  * `scale` is stored with the weights.  Latents are normalised to unit variance before diffusion
    (the standard latent-diffusion move) and the factor is computed once on the training set rather
    than hardcoded, because it depends on the AE that was actually fit.

Reconstruction is checked by the SAME probes the composition metrics use, not by MSE: an AE with a
respectable MSE that loses the background colour would silently destroy one third of every
composition claim downstream.

    from lifelong_ood.neural.latent import ConvAE, train_ae, encode, decode
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    """32x32x3 <-> 8x8x`zdim`, deterministic."""

    def __init__(self, zdim=4, base=64, img=32, ch=3):
        super().__init__()
        self.zdim, self.img, self.ch = zdim, img, ch
        self.enc = nn.Sequential(
            nn.Conv2d(ch, base, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base, base, 4, 2, 1), nn.SiLU(),          # 32 -> 16
            nn.Conv2d(base, base * 2, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 4, 2, 1), nn.SiLU(),  # 16 -> 8
            nn.Conv2d(base * 2, zdim, 3, 1, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(zdim, base * 2, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),        # 8 -> 16
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),        # 16 -> 32
            nn.Conv2d(base * 2, base, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base, ch, 3, 1, 1),
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        return self.decode(self.encode(x))


def train_ae(X01, epochs=40, batch=256, lr=2e-3, zdim=4, base=64, img=32, ch=3,
             device="cuda", seed=0, log_every=10, fg_weight=1.0):
    """Fit the AE on [N, D] rows in [0, 1].  -> (model, scale, history)."""
    torch.manual_seed(seed)
    X = torch.from_numpy(np.asarray(X01, np.float32).reshape(-1, img, img, ch)
                         ).permute(0, 3, 1, 2).contiguous()
    m = ConvAE(zdim=zdim, base=base, img=img, ch=ch).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]].to(device)
            rec = m(xb)
            if fg_weight > 1.0:
                # Upweight pixels far from the image's own modal colour. The foreground stroke is a
                # few hundred pixels against ~2800 of background, so plain MSE is minimised almost
                # entirely by the background and a factor carried by the stroke can be lost at an
                # excellent MSE -- measured: mse 0.00044 with the fg probe at 0.835 against 1.000.
                med = xb.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
                w = 1.0 + (fg_weight - 1.0) * (xb - med).abs().amax(1, keepdim=True).gt(0.15).float()
                # NORMALISE to mean 1. Without this the loss is ~8x larger than the unweighted one,
                # which is an 8x learning-rate change in disguise: the first attempt diverged and
                # collapsed to predicting black (mse 0.075 = the data variance, every probe exactly
                # at chance). Weighting should change WHERE the error is counted, not how big it is.
                loss = ((w / w.mean()) * (rec - xb) ** 2).mean()
            else:
                loss = F.mse_loss(rec, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        sched.step()
        hist.append(tot / len(X))
        if log_every and (ep + 1) % log_every == 0:
            print(f"    ae epoch {ep + 1}/{epochs}: mse {hist[-1]:.5f}", flush=True)
    # normalise latents to unit variance -- computed on THIS AE's training set, never hardcoded
    with torch.no_grad():
        Z = torch.cat([m.encode(X[i:i + 512].to(device)).cpu() for i in range(0, len(X), 512)])
    scale = float(1.0 / (Z.std() + 1e-8))
    return m, scale, hist


@torch.no_grad()
def encode(m, X01, scale=1.0, img=32, ch=3, device="cuda", batch=512):
    """[N, D] pixels in [0, 1] -> [N, zdim, 8, 8] scaled latents."""
    X = torch.from_numpy(np.asarray(X01, np.float32).reshape(-1, img, img, ch)
                         ).permute(0, 3, 1, 2).contiguous()
    out = [m.encode(X[i:i + batch].to(device)).cpu() * scale for i in range(0, len(X), batch)]
    return torch.cat(out)


@torch.no_grad()
def decode(m, Z, scale=1.0, device="cuda", batch=512):
    """[N, zdim, 8, 8] scaled latents -> [N, D] pixels in [0, 1]."""
    Z = torch.as_tensor(Z, dtype=torch.float32)
    out = []
    for i in range(0, len(Z), batch):
        x = m.decode(Z[i:i + batch].to(device) / scale).cpu()
        out.append(x.permute(0, 2, 3, 1).reshape(len(x), -1))
    return torch.cat(out).numpy()
