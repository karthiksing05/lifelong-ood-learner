"""The noise schedule and the numeric-range conventions, in one place.

Every module used to reach these through `lifelong_ood.score`, which imports the analytic tree, so the
schedule dragged the dead track into every backbone.  Nothing here imports anything of ours.
"""
import numpy as np


def cosine_abar(u, s=0.008):
    """Continuous cosine schedule: u in [0, 1] -> abar in (0, 1], abar(0) = 1.  numpy in, numpy out."""
    u = np.asarray(u, np.float64)
    f = np.cos((u + s) / (1 + s) * np.pi / 2) ** 2
    f0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return np.clip(f / f0, 1e-5, 1.0)


def abar_of(u):
    """torch u -> torch abar on the same device, through the repository's cosine schedule."""
    import torch
    a = cosine_abar(u.detach().cpu().numpy().astype(np.float64))
    return torch.as_tensor(a, dtype=torch.float32, device=u.device)


# Which numeric range a field's x-rows live in.  Declared per class rather than inferred, because
# inferring it is what failed: a DATA-range clamp of (-0.2, 1.2) applied to a MODEL-space trajectory
# clamps away the negative half at every reverse step (RESULTS 7ee, VLA_PATH lesson 2-bis).
CLIP_BY_SPACE = {"model": (-1.0, 1.0), "data": (-0.2, 1.2)}


def default_clip(field):
    """The clip a field should use when the caller does not say.  -> (lo, hi).

    A field may override with `clip_default` when its own convention is neither generic range -- the
    VE bridge clamps to (0.0, 1.0)."""
    explicit = getattr(field, "clip_default", None)
    if explicit is not None:
        return tuple(explicit)
    return CLIP_BY_SPACE[getattr(field, "data_space", "model")]
