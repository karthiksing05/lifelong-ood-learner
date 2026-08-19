"""WHY does ADDRESS barely beat retrieval on the CelebA grid? Measure where the composed point lands.

Once the range bug was fixed, the case study made the answer visible: `ours (ADDRESS)` and
`nearest-class` draw nearly the same faces in every held-out cell, and their joint accuracies agree
to the sample (0.17/0.17, 0.00/0.00, 0.17/0.17, 0.33/0.33 on the 8-seen backbone). N75 puts the
pooled margin at +0.018 +/- 0.011. Those are the same fact stated twice, and neither says why.

The geometric question, which §7dd already asks on ColorMNIST: how far is the composed address from
the nearest TRAINED embedding, measured against the spacing between trained rows? On ColorMNIST it
is 0.930, i.e. **0.88x** the median nearest-neighbour spacing -- a genuinely new point, roughly as
far from its neighbour as trained rows are from each other, which is why sampling it gives something
`nearest-class` does not.

If that ratio is small here, then `compose(fac, tgt)` is landing essentially ON a trained row, and
ADDRESS *is* retrieval on this dataset -- not as a measurement artifact but as geometry. That would
explain the whole §7ee result, and it is predictable from the embedding table alone, before
generating a single image.

The number is bounded below by the grid: with 12 cells over 3 factors, every held-out cell is one
factor value away from a seen cell, so its composed point differs from that neighbour by ONE factor
vector. ColorMNIST's 120 cells give the same Hamming-1 structure but far more factor values to
separate them. So the prediction is that the ratio falls with table size, and that is checkable
against both backbones here.

    python tests/diag_grid_geometry.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                              # noqa: E402

from lifelong_ood.neural.idealwords import compose, factorize, slot_factors  # noqa: E402


def rows_for(ck, path):
    """(C, labels, held_labels) for a checkpoint, or None if it is not a factored grid."""
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    if ck.get("dataset") == "celeba_grid":
        from lifelong_ood.data.celeba import load_celeba_grid
        sp = ck.get("split", {})
        d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                             n_per_cell=int(sp.get("n_per_slot", 3000)),
                             n_holdout=int(sp.get("n_holdout") or (12 - len(seen))),
                             max_rows=sp.get("max_rows"))
        F, slot = np.asarray(d["factors"]), np.asarray(d["slot"])
        lab = np.stack([F[slot == int(s)][0] for s in seen])
        held = [tuple(int(v) for v in F[np.where(slot == int(h))[0][0]])
                for h in d["held"] if (slot == int(h)).any()]
        return C, lab, held
    if "colormnist" in os.path.basename(path):
        lab = slot_factors(seen)
        # ColorMNIST's held-out slots are the ones absent from `remap`, over the full 10x4x4 design
        allc = {(d_, f_, b_) for d_ in range(10) for f_ in range(4) for b_ in range(4)}
        held = sorted(allc - {tuple(int(v) for v in r) for r in lab})
        return C, lab, held[:12]
    return None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=None)
    args = p.parse_args()

    paths = args.ckpts or sorted(
        glob.glob(os.path.join(CACHE, "ddpm_celeba_grid*.pt"))
        + [os.path.join(CACHE, "ddpm_colormnist.pt")])

    print(f"[geo] {'backbone':30s}{'cells':>7s}{'spacing':>10s}{'d(c,near)':>12s}"
          f"{'ratio':>8s}{'cos':>8s}")
    for path in paths:
        if not os.path.exists(path) or "smoke" in path:
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:                                             # noqa: BLE001
            continue
        if not isinstance(ck, dict) or "remap" not in ck:
            continue
        C, lab, held = rows_for(ck, path)
        if C is None or not held:
            continue
        fac = factorize(C, lab)
        # median nearest-neighbour spacing among TRAINED rows -- the only scale that makes a
        # distance in embedding space interpretable
        D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(len(C)) * 1e9
        spacing = float(np.median(D.min(1)))
        ds, ratios, coss = [], [], []
        for tgt in held:
            c = compose(fac, tgt)
            dd = np.linalg.norm(C - c[None], axis=1)
            j = int(np.argmin(dd))
            ds.append(float(dd[j]))
            ratios.append(float(dd[j] / max(spacing, 1e-9)))
            coss.append(float(C[j] @ c / max(np.linalg.norm(C[j]) * np.linalg.norm(c), 1e-9)))
        print(f"[geo] {os.path.basename(path):30s}{len(C):>7d}{spacing:>10.4f}"
              f"{float(np.mean(ds)):>12.4f}{float(np.mean(ratios)):>8.2f}"
              f"{float(np.mean(coss)):>8.3f}")

    print(f"\n[geo] {'=' * 70}")
    print("[geo] `ratio` is the number that matters: distance from the composed address to the")
    print("[geo] nearest TRAINED row, over the median spacing between trained rows. Near 1.0 the")
    print("[geo] composed point is as far from its neighbour as trained rows are from each other,")
    print("[geo] so sampling it gives something `nearest-class` cannot. Near 0 it lands ON a")
    print("[geo] trained row and ADDRESS is retrieval -- by geometry, not by measurement error,")
    print("[geo] and knowable from the embedding table before generating anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
