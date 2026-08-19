"""In-sample vs leave-one-class-out R^2 of the ideal-words fit, for every backbone on disk.

`factorize` reports its own R^2 and that number is quoted throughout this repository as evidence
that a conditional diffusion model's class table organises compositionally.  It is an IN-SAMPLE
fit statistic, and how much it is worth depends entirely on how over-determined the fit is:

    ColorMNIST   120 classes, ~19 free vectors  -- heavily over-determined
    CelebA grid  9 classes (7 once five cells are held out), 5 free vectors -- barely determined

At the second ratio an R^2 near 0.9 is most of the way to arithmetic and says little about
structure.  The honest quantity is leave-one-class-out: refit without a class and predict it, which
is exactly the question ADDRESS asks of a held-out combination.

This runs on the embedding tables alone -- no sampling, no GPU -- so it is cheap enough to be a
standing check rather than an experiment.

    python tests/diag_basis_cv.py
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                              # noqa: E402

from lifelong_ood.neural.idealwords import cv_r2, factorize, slot_factors  # noqa: E402


def labels_for(ck, path):
    """Factor rows for a checkpoint's SEEN classes, in `remap` order."""
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    ds = ck.get("dataset", "")
    if ds == "celeba_grid":
        from lifelong_ood.data.celeba import load_celeba_grid
        sp = ck.get("split", {})
        d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                             n_per_cell=int(sp.get("n_per_slot", 3000)),
                             n_holdout=int(sp.get("n_holdout") or (12 - len(seen))),
                             max_rows=sp.get("max_rows"))
        F, slot = np.asarray(d["factors"]), np.asarray(d["slot"])
        return C, np.stack([F[slot == int(s)][0] for s in seen])
    if "colormnist" in os.path.basename(path):
        return C, slot_factors(seen)
    return None, None


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, "*.pt"))):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:                                              # noqa: BLE001
            continue
        if not isinstance(ck, dict) or "remap" not in ck or "ema" not in ck:
            continue
        if "cemb.weight" not in ck["ema"]:
            continue
        try:
            C, lab = labels_for(ck, p)
        except Exception as e:                                         # noqa: BLE001
            print(f"[cv] {os.path.basename(p):34s} skipped: {type(e).__name__}: {e}")
            continue
        if C is None:
            continue
        fac = factorize(C, lab)
        r2cv, n_cv = cv_r2(C, lab)
        n_free = 1 + sum(len(v) for v in fac["V"])
        rows.append((os.path.basename(p), len(C), n_free, float(fac["r2"]), r2cv, n_cv))

    print(f"\n[cv] {'backbone':34s}{'classes':>9s}{'free':>6s}{'ratio':>7s}"
          f"{'R2':>8s}{'R2 cv':>8s}{'n cv':>6s}")
    for name, n, k, r2, r2cv, n_cv in rows:
        print(f"[cv] {name:34s}{n:>9d}{k:>6d}{n / max(k, 1):>7.1f}{r2:>8.3f}{r2cv:>8.3f}{n_cv:>6d}")

    print(f"\n[cv] {'=' * 74}")
    print("[cv] The ratio column is the point. Where classes outnumber free vectors several times")
    print("[cv] over, in-sample R^2 and cross-validated R^2 agree and the in-sample number is a")
    print("[cv] real statement about the table. Where the ratio approaches 1, they diverge and only")
    print("[cv] the cross-validated number should be quoted -- an additive model with almost as")
    print("[cv] many parameters as observations fits anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
