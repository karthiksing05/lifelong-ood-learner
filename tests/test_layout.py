"""Training and sampling must interpret a flat image vector the SAME way.

This exists because they briefly did not.  The uint8 fast path added for CelebA transposed flat
rows HWC -> NCHW, while every other consumer in the repository (`_as_img`, the float branch of
`train_ddpm`, the sampler) *reinterprets* the same flat row as CHW.  Both conventions are
self-consistent; mixing them is not.  The result was a backbone trained on upright faces and
sampled through a different permutation -- tiled mosaic samples, per-attribute controllability
0.000 on eight of eight attributes, and three hours of GPU spent before a figure made it obvious.

Nothing here needs a checkpoint or a GPU: it compares the two code paths on the same bytes.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.neural.ddpm import CH, _as_img, to_model              # noqa: E402

IMG = 8
D = CH * IMG * IMG
ok = True

rng = np.random.default_rng(0)
flat8 = rng.integers(0, 256, (5, D), dtype=np.uint8)
flat01 = flat8.astype(np.float32) / 255.0

# --- the two branches of train_ddpm, on the same data -----------------------
float_path = torch.from_numpy(to_model(flat01).astype(np.float32)).reshape(-1, CH, IMG, IMG)
u8_path = torch.from_numpy(np.ascontiguousarray(flat8)).reshape(-1, CH, IMG, IMG)
u8_path = u8_path.float().div_(127.5).sub_(1.0)
err = float((float_path - u8_path).abs().max())
print(f"  train float path vs uint8 path: max |delta| = {err:.3e}")
if err > 2e-2:                       # 1/255 quantisation only
    print("    FAIL: the two training paths disagree on layout or scaling")
    ok = False

# --- training layout vs the layout the sampler feeds the model --------------
sampler_view = _as_img(to_model(flat01), "cpu", img=IMG)
err2 = float((float_path - sampler_view).abs().max())
print(f"  train layout vs sampler (_as_img) layout: max |delta| = {err2:.3e}")
if err2 > 1e-6:
    print("    FAIL: the model is trained on one permutation and sampled through another")
    ok = False

# --- round trip: to_model/to_data must be exact inverses --------------------
from lifelong_ood.neural.ddpm import to_data                            # noqa: E402
err3 = float(np.abs(to_data(to_model(flat01)) - flat01).max())
print(f"  to_data(to_model(x)) round trip: max |delta| = {err3:.3e}")
if err3 > 1e-6:
    print("    FAIL: the [0,1] <-> [-1,1] conversion is not an involution")
    ok = False


# --- the layout OPTION itself must round-trip in both modes -----------------
from lifelong_ood.neural.ddpm import _as_flat                            # noqa: E402
for lay in ("permuted", "hwc"):
    T = _as_img(to_model(flat01), "cpu", img=IMG, layout=lay)
    back = _as_flat(T, lay).numpy()
    e = float(np.abs(back - to_model(flat01)).max())
    print(f"  _as_img/_as_flat round trip, layout={lay:9s}: max |delta| = {e:.3e}")
    if e > 1e-6:
        print("    FAIL: the layout conversion is not invertible")
        ok = False

# --- "hwc" must actually be a real image, and "permuted" must not ----------
# The whole point of the option: under `hwc`, channel 0 is the RED channel of the whole picture;
# under `permuted` it is the top third of the image with the colours interleaved.  If these two
# ever agree, the option has stopped meaning anything and RESULTS 30a/30b lose their mechanism.
img_hwc = _as_img(to_model(flat01), "cpu", img=IMG, layout="hwc").numpy()
true_chw = to_model(flat01).reshape(-1, IMG, IMG, CH).transpose(0, 3, 1, 2)
e = float(np.abs(img_hwc - true_chw).max())
print(f"  layout=hwc equals a true HWC->CHW transpose: max |delta| = {e:.3e}")
if e > 1e-6:
    print("    FAIL: layout='hwc' is not producing a true image")
    ok = False
img_perm = _as_img(to_model(flat01), "cpu", img=IMG, layout="permuted").numpy()
if float(np.abs(img_perm - true_chw).max()) < 1e-6:
    print("    FAIL: 'permuted' and 'hwc' agree -- the option is a no-op and the layout "
          "mechanism in RESULTS 30a/30b cannot be what it claims")
    ok = False

# --- the PRETRAINED field must default to a real image ---------------------
# Not a stylistic preference: those weights are frozen and were fitted to real images, so a
# permuted row is wrong input to a fixed function.  Under the old default the field returned a
# 3x3 tiled greyscale mosaic (RESULTS 30b).
import inspect                                                          # noqa: E402
from lifelong_ood.neural import pretrained as _pre                      # noqa: E402
d = inspect.signature(_pre.PretrainedScoreField.__init__).parameters["layout"].default
print(f"  PretrainedScoreField default layout = {d!r}")
if d != "hwc":
    print("    FAIL: a pretrained backbone must be fed a real image by default")
    ok = False


# --- no consolidation path may use a plain reshape --------------------------
# The failure this catches: a pretrained field defaults to layout="hwc" while an adapter is trained
# through a plain `.reshape(-1, CH, img, img)` (i.e. "permuted").  The model is then SAMPLED in one
# layout and its adapter TRAINED in another -- which is the original sin this whole file exists for,
# reappearing at a new boundary.  It produced rainbow noise from the pretrained bank while
# preservation still reported 1.00, because bit-identity does not care what the input looks like.
import re                                                               # noqa: E402
BAD = re.compile(r"reshape\(-1,\s*(?:CH|3),\s*(?:field|self)\.(?:field\.)?img")
for mod in ("lifelong_neural.py", "inversion.py", "idealwords.py"):
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "lifelong_ood", "neural", mod)
    hits = [l.strip() for l in open(p, encoding="utf-8").read().splitlines()
            if BAD.search(l) and not l.strip().startswith("#")]
    print(f"  {mod:22s} plain-reshape sites: {len(hits)}")
    for h in hits:
        print(f"      {h[:96]}")
    if hits:
        print("    FAIL: this path ignores the field's layout")
        ok = False

print(f"\nlayout: {'ok -- every path agrees' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
