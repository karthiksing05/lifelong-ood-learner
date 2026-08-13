"""Inception-V3 pool3 features -- the space the companion paper's metrics are defined in.

Every number in their Tables 1-2 (FID, precision, recall, F1) is computed on 2048-d Inception-V3
pool3 activations.  Ours were computed by `evals._feat`, a 2x box-downsample of the raw pixels, whose
docstring argues that "absolute values are therefore NOT comparable to the paper's; the ordering
between methods, which is what the comparison rests on, is."

That argument is load-bearing and it is not safe.  The single quantity the whole comparison now turns
on is RECALL -- whether generated samples cover the class -- and recall is a k-NN manifold estimate
whose answer depends entirely on which directions the metric considers close.  Downsampled pixels
call two images of the same digit at different stroke positions FAR APART, because pixel distance is
dominated by spatial alignment; Inception features call them near, because it was trained to be
invariant to exactly that.  So the pixel space systematically under-reports recall for a generator
that produces correct-but-not-aligned images, which is the failure mode under test.  Reproducing a
paper's recall in a different metric space and attributing the difference to its method is not a
reproduction.

The weights are already on this machine from a previous project
(`~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth`), so this needs no network.  Note
this is the torchvision port, not the original TF-Inception graph that `pytorch-fid` uses; FID values
from the two differ by a small constant offset, so ours remain non-comparable to published FIDs on an
absolute scale.  Precision/recall/F1 are k-NN quantities in the feature space and do not inherit that
offset in the same way -- they are the numbers this module exists to make honest.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

_WEIGHTS = os.path.expanduser("~/.cache/torch/hub/checkpoints/"
                              "inception_v3_google-0cc3c7bd.pth")
_MODEL = None
# ImageNet statistics: the network expects its own normalization, not [-1, 1] or [0, 1]
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def available():
    """-> True if features can be computed here.  Callers fall back to `evals._feat` if not."""
    try:
        import torchvision  # noqa: F401
    except Exception:
        return False
    return os.path.exists(_WEIGHTS)


def _load(device):
    global _MODEL
    if _MODEL is None:
        from torchvision.models import inception_v3
        m = inception_v3(weights=None, aux_logits=True, init_weights=False)
        m.load_state_dict(torch.load(_WEIGHTS, map_location="cpu"))
        m.fc = torch.nn.Identity()          # pool3, 2048-d, the FID/precision-recall layer
        _MODEL = m.eval().to(device)
        for p in _MODEL.parameters():
            p.requires_grad_(False)
    return _MODEL


def features(X, img=None, device=None, batch=64):
    """[N, D] images in [0, 1], flat HWC rows -> [N, 2048] pool3 activations.

    `img` is the side length; inferred from the row width assuming 3 channels when omitted.  Images
    are bilinearly resized to 299 because that is the resolution the network's receptive fields were
    trained for -- feeding 32x32 directly makes every activation a boundary artifact.
    """
    X = np.atleast_2d(np.asarray(X, np.float32))
    n, d = X.shape
    if img is None:
        img = int(round(np.sqrt(d / 3.0)))
    ch = d // (img * img)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = _load(device)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, n, batch):
            T = torch.from_numpy(X[i:i + batch]).to(device)
            T = T.reshape(-1, img, img, ch).permute(0, 3, 1, 2)
            if ch == 1:
                T = T.repeat(1, 3, 1, 1)
            T = T.clamp(0, 1)
            T = F.interpolate(T, size=(299, 299), mode="bilinear", align_corners=False)
            out.append(m((T - mean) / std).float().cpu().numpy())
    return np.concatenate(out)
