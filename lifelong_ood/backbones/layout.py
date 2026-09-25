"""Flat rows <-> image tensors, and [0,1] <-> [-1,1], defined once.

Flat rows in this repository are HWC.  There are two ways to hand one to a UNet:

  "permuted"  a plain `.reshape(-1, 3, H, W)`.  This is NOT an image: for 64x64x3 the UNet's three
              "channels" are image ROWS 0-21 / 21-42 / 42-63 and its width axis is the COLOUR axis.
              Training and sampling share the permutation, so images still DISPLAY correctly, which
              is why it survived until 2026-08-11 (fixing it cut held-out loss 28.7%).
  "hwc"       a real transpose; the UNet sees an image.

Every checkpoint records the layout it was trained in and MUST be sampled in that layout.  The
previous two copies of this pair (`ddpm._as_img` / `cobweb_update.to_img`) agreed by discipline only.
"""
import os

import numpy as np

IMG, CH = 32, 3
DIM = CH * IMG * IMG
LAYOUT_DEFAULT = os.environ.get("LIFELONG_LAYOUT", "permuted")


def to_model(x01):
    """[0,1] pixels, flat or image-shaped -> [-1,1] flat.  Resolution-agnostic."""
    x = np.asarray(x01, np.float64)
    return 2.0 * x.reshape(len(x) if x.ndim > 1 else 1, -1) - 1.0


def to_data(xm):
    """[-1,1] flat -> [0,1] flat, clipped.  Resolution-agnostic."""
    x = np.asarray(xm, np.float64)
    return np.clip((x.reshape(len(x) if x.ndim > 1 else 1, -1) + 1.0) / 2.0, 0.0, 1.0)


def to_img(T, img, layout):
    """torch flat rows (HWC convention) -> the (N, 3, H, W) tensor the checkpoint was trained on."""
    if layout == "hwc":
        return T.reshape(-1, img, img, CH).permute(0, 3, 1, 2).contiguous()
    return T.reshape(-1, CH, img, img)


def to_flat(T, layout):
    """Inverse of `to_img`."""
    if layout == "hwc":
        return T.permute(0, 2, 3, 1).reshape(len(T), -1)
    return T.reshape(len(T), -1)


def as_img(X, device, img=IMG, layout=None):
    """numpy flat rows -> image tensor on `device`, defaulting to LAYOUT_DEFAULT when unspecified."""
    import torch
    X = np.atleast_2d(np.asarray(X, np.float32))
    return to_img(torch.from_numpy(X).to(device), img, layout or LAYOUT_DEFAULT)


def as_flat(T, layout=None):
    """(N, C, H, W) -> flat rows in the repository's HWC convention.  Inverse of `as_img`."""
    return to_flat(T, layout or LAYOUT_DEFAULT)
