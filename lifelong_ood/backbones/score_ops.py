"""Score-field operations on a Backbone, in torch: the score, mode ascent (a prototype = a mode of
p_t, Zekun's Eq. 6 with Adam, returned in x0 space by Tweedie), and the diagonal Laplace extent
(four Hutchinson probes of the score's Jacobian, pulled back to x0 space).

Ported from the numpy `BaseScoreField.mode_ascent` / `laplace_diag` (archived) with the same
settings the paper uses: Adam, no clamp, Tweedie for the x0 mode.  Everything runs batched on the
device; inputs and outputs are flat rows in the backbone's row space.
"""
import numpy as np
import torch

from .schedule import cosine_abar


def score(backbone, X, u, cls=None, emb=None, gate=None, batch=256):
    """grad_x log p_u(x) = -eps / sqrt(1 - abar) for flat rows X.  -> torch [n, dim]."""
    a = float(cosine_abar(float(u)))
    return -backbone.eps(X, u, cls=cls, emb=emb, gate=gate, batch=batch) / float(np.sqrt(1.0 - a))


@torch.no_grad()
def mode_ascent(backbone, X, u, iters=150, lr=0.1, seed=0, init="query", adam=True, tweedie=True,
                cls=None, emb=None, gate=None, batch=256, n_starts=None):
    """Ascend log p_u from noised copies of X (init="query") or from pure noise.

    -> (x_t modes [n, dim], x0 modes [n, dim]) as float32 numpy, n = len(X) or n_starts.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = float(cosine_abar(float(u)))
    dev = backbone.device
    if init == "query" and X is not None:
        Q = torch.as_tensor(np.asarray(X, np.float32), device=dev)
        if n_starts is not None:
            Q = Q[torch.arange(int(n_starts)) % len(Q)]
        x = a ** 0.5 * Q + (1.0 - a) ** 0.5 * torch.randn(Q.shape, generator=g).to(dev)
    else:
        n = int(n_starts or len(X))
        x = torch.randn((n, backbone.dim), generator=g).to(dev)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for i in range(1, int(iters) + 1):
        gsc = score(backbone, x, u, cls=cls, emb=emb, gate=gate, batch=batch)
        if adam:
            m = b1 * m + (1 - b1) * gsc
            v = b2 * v + (1 - b2) * gsc * gsc
            x = x + lr * (m / (1 - b1 ** i)) / (torch.sqrt(v / (1 - b2 ** i)) + eps)
        else:
            x = x + lr * (1.0 - a) * gsc
    if tweedie:
        x0 = (x + (1.0 - a) * score(backbone, x, u, cls=cls, emb=emb, gate=gate, batch=batch)) / a ** 0.5
    else:
        x0 = x / a ** 0.5
    return x.float().cpu().numpy(), x0.float().cpu().numpy()


@torch.no_grad()
def laplace_diag(backbone, Xt, u, probes=4, delta=5e-3, seed=0, var_floor=1e-4, var_cap=1.0,
                 cls=None, emb=None, gate=None, batch=256):
    """Diagonal Laplace extent of p_u at x_t, pulled back to x0 space:
    Sigma_t = -1/diag(H) where H is the score's Jacobian (Hutchinson finite differences);
    Sigma_0 = (Sigma_t - (1 - abar)) / abar, clipped to [var_floor, var_cap].  -> float32 numpy [n, dim]."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    xt = torch.as_tensor(np.asarray(Xt, np.float32), device=backbone.device)
    a = float(cosine_abar(float(u)))
    h = torch.zeros_like(xt)
    for _ in range(int(probes)):
        vsg = (torch.randint(0, 2, xt.shape, generator=g).float() * 2 - 1).to(xt.device)
        sp = score(backbone, xt + delta * vsg, u, cls=cls, emb=emb, gate=gate, batch=batch)
        sm = score(backbone, xt - delta * vsg, u, cls=cls, emb=emb, gate=gate, batch=batch)
        h += vsg * (sp - sm) / (2 * delta)
    h /= probes
    st = torch.full_like(h, float(var_cap))
    neg = h < -1e-9
    st[neg] = -1.0 / h[neg]
    return torch.clamp((st - (1.0 - a)) / a, var_floor, var_cap).float().cpu().numpy()
