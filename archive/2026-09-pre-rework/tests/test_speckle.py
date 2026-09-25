"""`speckle()` must stay bounded, and must actually rank roughness.

This exists because the isolated-pixel half of the metric divided by a denominator that is nearly
zero on real ColorMNIST (~2e-4 of pixels), and the same sweep reported `isolated_ratio` of 20345 on
two rows and 0.2 on three others for image sets that are statistically identical
(`slurm/3673660.out`).  A metric that swings five orders of magnitude across equivalent inputs is
worse than no metric: it was being read as evidence.

The denominator is now floored.  What this asserts:

  1. the ratio stays bounded when the reference has essentially no isolated pixels -- the case that
     used to explode;
  2. `tv_ratio` orders a rough image set above a smooth one, i.e. the half of the metric that
     carries the signal still works;
  3. an image set compared against ITSELF scores exactly 1.0, so the normalization is honest.

Nothing here needs a checkpoint or a GPU.

    python tests/test_speckle.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.evals import speckle                                  # noqa: E402

IMG = 32
ok = True


def check(cond, label, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))


rng = np.random.default_rng(0)
# a SMOOTH reference: a horizontal ramp, repeated.  Isolated-pixel count is exactly zero, which is
# the degenerate denominator the floor exists for.
ramp = np.linspace(0, 1, IMG, dtype=np.float32)
smooth = np.tile(np.repeat(ramp[None, :], IMG, 0).reshape(1, -1), (12, 3))[:, :IMG * IMG * 3]
# a flat mid-grey reference with EXACTLY zero isolated pixels, and a copy with unambiguous dots:
# a dot at 1.0 on 0.5 grey differs from all four neighbours by 0.5, well past the 0.25 threshold.
grey = np.full((12, IMG * IMG * 3), 0.5, np.float32)
dotted = grey.copy().reshape(12, IMG, IMG, 3)
dotted[:, 7::9, 5::11, :] = 1.0
dotted = dotted.reshape(12, -1)
# and a genuinely rough set: white noise, which has high total variation everywhere
rough = rng.random((12, IMG * IMG * 3)).astype(np.float32)

print("[speckle] bounded ratio, and does tv_ratio rank roughness?")

r_dot = speckle(dotted, grey, img=IMG)
check(np.isfinite(r_dot["isolated_ratio"]) and r_dot["isolated_ratio"] < 1e4,
      "isolated_ratio bounded against a zero-isolate reference",
      f"got {r_dot['isolated_ratio']:.4f}")
# and the detector must actually fire, or `bounded` would pass trivially on a dead metric
check(r_dot["isolated"] > 0, "isolated fires on unambiguous dots",
      f"got {r_dot['isolated']:.5f} of pixels")

r_self = speckle(smooth, smooth, img=IMG)
check(abs(r_self["tv_ratio"] - 1.0) < 1e-9, "tv_ratio of a set against ITSELF is exactly 1.0",
      f"got {r_self['tv_ratio']:.6f}")

# white noise against a ramp lands at ~4.4x: the threshold only has to be clearly above 1, and is
# deliberately not tuned to the observed value.
r_rough = speckle(rough, smooth, img=IMG)
check(r_rough["tv_ratio"] > 2.0, "tv_ratio flags a rough set against a smooth reference",
      f"got {r_rough['tv_ratio']:.2f}")
check(r_rough["tv_ratio"] > speckle(dotted, smooth, img=IMG)["tv_ratio"],
      "tv_ratio orders broadband roughness above sparse dots",
      f"{r_rough['tv_ratio']:.2f} > {speckle(dotted, smooth, img=IMG)['tv_ratio']:.2f}")

# every quantity finite for a degenerate all-constant input (tv = 0 on both sides)
flat = np.full((4, IMG * IMG * 3), 0.5, np.float32)
r_flat = speckle(flat, flat, img=IMG)
check(all(np.isfinite(v) for v in r_flat.values()), "finite on an all-constant input",
      str({k: round(v, 4) for k, v in r_flat.items()}))

print(f"\n[speckle] {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
