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

print(f"\nlayout: {'ok -- every path agrees' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
