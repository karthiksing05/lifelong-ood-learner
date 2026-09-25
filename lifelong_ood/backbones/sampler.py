"""THE sampler: DDIM (eta = 0) on the cosine schedule with explicit conditioning and an explicit gate.

Ten sampler entry points diverged in the previous repository (`score.py:95`, `pretrained.py:251`,
`ve_bridge.py:142`, `instrument.py:199`, `inversion.py:267/355`, `idealwords.py:294`, ...) and
`instrument.py` records the runs lost to their differing conventions.  This is the one that survives
(it was `cobweb_update.ddim_sample`); the `Backbone` handle (loader.py) will supply layout / img / clip
so callers cannot pass the wrong ones.  `tests/test_sampler_contract.py` asserts there is exactly one.
"""
import contextlib

import numpy as np
import torch

from .layout import to_flat, to_img
from .schedule import cosine_abar


def _gate(gate):
    return gate() if gate is not None else contextlib.nullcontext()


def _cond(model, m, device, emb=None, cls=None, i0=0):
    """-> (c, emb_override) for rows i0:i0+m.  `emb` is [d] or [N, d]; `cls` an int or [N] ids."""
    if emb is not None:
        E = emb if emb.dim() == 2 else emb[None].expand(m, -1)
        return None, E[i0:i0 + m] if emb.dim() == 2 else E
    if cls is None:
        return torch.full((m,), model.null_id, device=device, dtype=torch.long), None
    c = torch.as_tensor(cls, device=device, dtype=torch.long)
    if c.dim() == 0:
        return c.expand(m).clone(), None
    return c[i0:i0 + m], None


@torch.no_grad()
def ddim_sample(model, n, steps=50, seed=0, emb=None, cls=None, layout="permuted", img=32,
                device="cuda", batch=256, gate=None, clip=(-1.0, 1.0)):
    """DDIM (eta = 0) on the cosine schedule -> flat MODEL-space rows [n, 3*img*img] in [-1, 1].

    Mirrors `BaseScoreField.sample`: the starting noise is `np.random.default_rng(seed)` drawn in the
    flat row space, the u grid is linspace(1, 0, steps + 1), and x0 is clipped to the model's space
    at every step.  So a number sampled here is comparable with a table sampled through the field,
    but each step is ONE forward pass and the conditioning is explicit.

    `clip` is the space the model's x0 lives in: (-1, 1) for this repository's UNets, (0, 1) for the
    VE bridge, whose rows are data-space.  Passing the wrong one clamps half the image away at every
    step -- the fault behind RESULTS 7bb's first false negative.
    """
    D = 3 * img * img
    noise = np.random.default_rng(seed).standard_normal((n, D)).astype(np.float32)
    us = np.linspace(1.0, 0.0, steps + 1)
    ab = cosine_abar(us)
    E = None if emb is None else torch.as_tensor(np.asarray(emb, np.float32), device=device)
    out = np.empty((n, D), np.float32)
    model.eval()
    with _gate(gate):
        for i0 in range(0, n, batch):
            x = to_img(torch.from_numpy(noise[i0:i0 + batch]).to(device), img, layout)
            m = len(x)
            c, eo = _cond(model, m, device, emb=E, cls=cls, i0=i0)
            for i in range(steps):
                a, an = float(ab[i]), float(ab[i + 1])
                e = model(x, torch.full((m,), float(us[i]), device=device), c=c, emb_override=eo)
                x0 = ((x - (1.0 - a) ** 0.5 * e) / a ** 0.5).clamp(*clip)
                x = an ** 0.5 * x0 + (1.0 - an) ** 0.5 * e
            out[i0:i0 + m] = to_flat(x.clamp(*clip), layout).cpu().numpy()
    return out


sample = ddim_sample  # the public name; the Backbone-handle signature lands with loader.py
