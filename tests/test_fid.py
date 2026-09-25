"""`fid.py` must read images the DISPLAY way, and KID must be usable where FID is not.

Three things are asserted, in increasing order of how much they matter:

  1. **Sanity.** A set scored against itself is ~0 for both estimators, and both grow monotonically
     as the samples are corrupted.  A distance that does not do this is not a distance.

  2. **Layout.** Inception must see an upright image.  This repository's flat vectors are HWC and the
     model pipeline reinterprets them as CHW; feeding that reinterpretation to Inception produces a
     confident number computed on a permuted image.  A transposed copy of a set must therefore score
     FAR from the original -- if it scores near zero, the module is not actually looking at the
     image.  (`tests/test_layout.py` exists because this permutation already cost three hours once.)

  3. **The estimator choice.** The module claims FID is unusable at the sample counts this repository
     uses per composition and KID is not.  That is demonstrated, not asserted: the SAME pair of
     distributions is scored at several n, and FID's value moves with n while KID's does not.

    python tests/test_fid.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.evals.fid import fid, inception_features, kid                     # noqa: E402

IMG = 32
ok = True


def check(cond, label, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))


rng = np.random.default_rng(0)


def blobs(n, shift=0.0, noise=0.0):
    """Smooth low-frequency colour images -- structured enough that Inception features move."""
    yy, xx = np.mgrid[0:IMG, 0:IMG] / IMG
    out = np.empty((n, IMG, IMG, 3), np.float32)
    for i in range(n):
        p = rng.random(6)
        for c in range(3):
            out[i, :, :, c] = (0.5 + 0.4 * np.sin(2 * np.pi * (p[c] + xx * (1 + p[3])))
                               * np.cos(2 * np.pi * (p[(c + 1) % 3] + yy * (1 + p[4]))) + shift)
    if noise:
        out += rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).reshape(n, -1)


N = 256
A, B = blobs(N), blobs(N)

print("[fid] 1. sanity: self-distance ~0, and monotone under corruption")
FA = inception_features(A)
check(FA.shape == (N, 2048), "inception_features returns pool3", f"shape {FA.shape}")

k_self = kid(A, A, features=(FA, FA))["kid"]
f_self = fid(A, A, features=(FA, FA), min_n=N)["fid"]
check(f_self < 1e-3, "FID of a set against ITSELF is ~0", f"got {f_self:.2e}")

FB = inception_features(B)
ks = [kid(blobs(N, noise=s), B, features=(inception_features(blobs(N, noise=s)), FB))["kid"]
      for s in (0.0, 0.15, 0.35)]
check(ks[0] < ks[1] < ks[2], "KID grows monotonically with added noise",
      " < ".join(f"{v:.4f}" for v in ks))

# A self-comparison of an UNBIASED estimator fluctuates around zero and is legitimately NEGATIVE
# about half the time -- that is what unbiasedness means here, not a bug.  So the assertion is
# scale-free: the self-value must be negligible against a difference the estimator should see,
# rather than below some absolute constant picked by hand.
check(abs(k_self) < ks[2] / 50, "KID of a set against ITSELF is negligible vs a real difference",
      f"self {k_self:+.2e} vs corrupted {ks[2]:.4f} ({ks[2] / abs(k_self):.0f}x larger)")

print("\n[fid] 2. layout: Inception must see an UPRIGHT image")
# the CHW reinterpretation the model pipeline uses -- a real permutation of the same bytes
chw = A.reshape(N, IMG, IMG, 3).transpose(0, 3, 1, 2).reshape(N, -1)
k_perm = kid(chw, A, features=(inception_features(chw), FA))["kid"]
check(k_perm > 20 * max(abs(k_self), 1e-6),
      "a CHW-permuted copy scores FAR from the original",
      f"permuted {k_perm:.4f} vs self {k_self:.2e}")

print("\n[fid] 3. the estimator choice, on two halves of ONE distribution (true distance = 0)")
# This is the case that matters and the case where FID is worst.  Both sets are drawn from the SAME
# distribution, so the true distance is exactly zero and EVERY number FID reports is pure bias.  (The
# earlier version of this test compared two far-apart distributions, where the bias is small next to
# a large real difference -- which flatters FID and demonstrates nothing.)
pool = blobs(2048)
FP = inception_features(pool)
half = len(pool) // 2
print(f"    {'n':>6s}{'FID':>12s}{'KID':>12s}   (true distance is 0)")
fids, kids = [], []
for n in (32, 64, 128, 256, 512, 1024):
    X, Y = FP[:n], FP[half:half + n]
    f = fid(None, None, features=(X, Y), strict=False)["fid"]
    k = kid(None, None, features=(X, Y), subset_size=n, n_subsets=40)["kid"]
    fids.append(f); kids.append(k)
    print(f"    {n:6d}{f:12.3f}{k:12.5f}")
check(fids[0] > 50, "FID reports a LARGE distance between identical distributions at small n",
      f"n=32 -> FID {fids[0]:.1f}, which is entirely bias")
check(fids[0] > 5 * fids[-1], "and that FID shrinks steadily as n grows -- it tracks n, not the data",
      f"{fids[0]:.1f} at n=32 -> {fids[-1]:.1f} at n=1024")
check(max(abs(k) for k in kids) < 0.01, "KID stays ~0 at EVERY n, which is why it is used per-composition",
      "max |KID| = " + f"{max(abs(k) for k in kids):.5f}")

print(f"\n[fid] {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
