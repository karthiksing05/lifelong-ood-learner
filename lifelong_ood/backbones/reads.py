"""Denoising-error reads on common random numbers: the eps-profile over noise levels and the
per-candidate losses the router decides from.  Reads are only trustworthy in the band where the
conditioning, not the input, carries the prediction (u ~ 0.8-0.95 on ColorMNIST); pooling raw losses
over levels is decided at low noise, where a new value's pixels swamp it (REVIEW Part 4).
"""
import numpy as np
import torch

from .layout import to_flat, to_img
from .sampler import _cond, _gate
from .schedule import cosine_abar


@torch.no_grad()
def eps_profile(model, X, u_grid, emb=None, cls=None, n_noise=4, seed=0, layout="permuted",
                img=32, device="cuda", batch=256, gate=None, per_image=False):
    """Mean eps-prediction error of MODEL-space rows `X` at each u -> [len(u)] or [n, len(u)].

    The (u, epsilon) draws are seeded HERE, independent of the caller's RNG, so two calls that differ
    only in the model (before / after an update) or the conditioning see identical noise, and their
    difference is the model's rather than the draw's.
    """
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    n = len(Xt)
    E = None if emb is None else torch.as_tensor(np.asarray(emb, np.float32), device=device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.zeros(n, len(u_grid), dtype=torch.float64)
    model.eval()
    with _gate(gate):
        for j, u in enumerate(u_grid):
            a = float(cosine_abar(float(u)))
            for _ in range(n_noise):
                eps = torch.randn(tuple(Xt.shape), generator=g).to(device)
                for i0 in range(0, n, batch):
                    xb, eb = Xt[i0:i0 + batch], eps[i0:i0 + batch]
                    m = len(xb)
                    c, eo = _cond(model, m, device, emb=E, cls=cls, i0=i0)
                    xt = to_img(a ** 0.5 * xb + (1.0 - a) ** 0.5 * eb, img, layout)
                    pred = to_flat(model(xt, torch.full((m,), float(u), device=device), c=c,
                                         emb_override=eo), layout)
                    out[i0:i0 + m, j] += ((pred - eb) ** 2).mean(1).double().cpu()
    out /= n_noise
    return out.numpy() if per_image else out.mean(0).numpy()


@torch.no_grad()
def candidate_losses(model, X, u_grid, embs=None, cls=None, n_noise=2, seed=0, layout="permuted",
                     img=32, device="cuda", batch=1024, gate=None, per_u=False):
    """Mean eps-loss of the SAME rows `X` under each of C candidate conditionings -> [C]
    (or [C, len(u_grid)] with `per_u`, the profile a per-noise-level criterion needs).

    Every candidate sees the identical (u, epsilon) draw for each image -- the router's correctness
    requirement: with independent draws the differences between conditionings are swamped by the
    variance of the draw.  `embs` [C, d] or `cls` [C] ids.
    """
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    n = len(Xt)
    E = None if embs is None else torch.as_tensor(np.asarray(embs, np.float32), device=device)
    Cc = None if cls is None else torch.as_tensor(np.asarray(cls), device=device, dtype=torch.long)
    C = len(E) if E is not None else len(Cc)
    per = max(1, batch // max(n, 1))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    tot = torch.zeros(C, len(u_grid), dtype=torch.float64)
    model.eval()
    with _gate(gate):
        for ju, u in enumerate(u_grid):
            a = float(cosine_abar(float(u)))
            for _ in range(n_noise):
                eps = torch.randn(tuple(Xt.shape), generator=g).to(device)
                xt = to_img(a ** 0.5 * Xt + (1.0 - a) ** 0.5 * eps, img, layout)
                for c0 in range(0, C, per):
                    m = min(per, C - c0)
                    xb = xt.repeat(m, 1, 1, 1)                       # candidate-major blocks
                    eo = None if E is None else E[c0:c0 + m].repeat_interleave(n, 0)
                    cb = None if Cc is None else Cc[c0:c0 + m].repeat_interleave(n)
                    pred = to_flat(model(xb, torch.full((m * n,), float(u), device=device), c=cb,
                                         emb_override=eo), layout)
                    err = ((pred - eps.repeat(m, 1)) ** 2).mean(1).view(m, n).mean(1)
                    tot[c0:c0 + m, ju] += err.double().cpu()
    prof = (tot / n_noise).numpy()
    return prof if per_u else prof.mean(1)


