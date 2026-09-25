"""A factor VALUE a pretrained backbone has never rendered, made from real images of the dataset.

The storage ladder's hardest tier is a genuinely new value: RESULTS 5q Phase B showed conditioning
cannot produce one, and on ColorMNIST the test needs a backbone trained with background 3 withheld.
On Shapes3D every backbone worth using is pretrained on all 480,000 images, so every one of the ten
floor hues has been rendered.  A new value therefore has to be one Shapes3D does not contain at all.

GREY FLOOR.  Every Shapes3D floor is a saturated hue.  Setting the floor's saturation to zero while
keeping its brightness keeps the scene's geometry, lighting and shading exactly -- only the colour of
one factor changes -- and produces a floor no Shapes3D image has.

The floor is found by HUE: its pixels share the floor hue (index/10 of the colour wheel) under
varying brightness.  Rows are rejected where that cannot separate it: a floor hue equal to the
object's (the object would be recoloured too) or to the wall's (pinned to index 0 in the split the
backbones use, so the wall would be).  `grey_floor` returns which rows it kept.
"""
import numpy as np


def rgb_to_hsv(rgb):
    """[..., 3] in [0, 1] -> [..., 3] hue/saturation/value, hue in [0, 1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-8
    rr = nz & (mx == r)
    gg = nz & (mx == g) & ~rr
    bb = nz & ~rr & ~gg
    h[rr] = ((g - b)[rr] / d[rr]) % 6.0
    h[gg] = (b - r)[gg] / d[gg] + 2.0
    h[bb] = (r - g)[bb] / d[bb] + 4.0
    s = np.where(mx > 1e-8, d / np.maximum(mx, 1e-8), 0.0)
    return np.stack([h / 6.0, s, mx], -1)


def hsv_to_rgb(hsv):
    """Inverse of `rgb_to_hsv`."""
    h, s, v = hsv[..., 0] * 6.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.empty(hsv.shape, dtype=np.float64)
    for k, (a, b, c) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v),
                                   (v, p, q)]):
        m = i == k
        out[..., 0][m], out[..., 1][m], out[..., 2][m] = a[m], b[m], c[m]
    return out


def floor_mask(X, floor, img=64, tol=0.04, sat_min=0.25):
    """-> [N, img, img] bool, pixels whose hue matches each row's floor hue (index/10)."""
    rgb = np.asarray(X, np.float64).reshape(-1, img, img, 3)
    hsv = rgb_to_hsv(rgb)
    target = (np.asarray(floor, np.float64) / 10.0)[:, None, None]
    dh = np.abs(hsv[..., 0] - target)
    dh = np.minimum(dh, 1.0 - dh)
    return (dh < tol) & (hsv[..., 1] > sat_min)


def grey_floor(X, floor, obj, img=64, wall=0, tol=0.04, sat_min=0.25):
    """Recolour each row's floor to grey (saturation 0, brightness kept).

    X [N, img*img*3] HWC rows in [0, 1]; floor/obj hue indices.  -> (X_grey [n_kept, D], keep [N]).
    Rows whose floor hue equals the object's or the wall's are dropped, since hue cannot isolate
    their floor.
    """
    floor, obj = np.asarray(floor, int), np.asarray(obj, int)
    keep = (floor != obj) & (floor != wall)
    rgb = np.asarray(X, np.float64)[keep].reshape(-1, img, img, 3)
    m = floor_mask(rgb.reshape(len(rgb), -1), floor[keep], img, tol, sat_min)
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.where(m, 0.0, hsv[..., 1])
    out = np.clip(hsv_to_rgb(hsv), 0.0, 1.0).reshape(len(rgb), -1).astype(np.float32)
    return out, keep
