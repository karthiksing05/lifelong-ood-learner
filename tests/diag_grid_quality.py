"""How bad is CelebA-grid sample QUALITY, against what, and what would fix it?

The case studies look poor next to the pretrained-backbone figures, and "looks poor" is not
actionable.  This puts a number on it and separates the two candidate causes.

WHY THE GAP IS EXPECTED, AND WHY IT CANNOT BE FIXED THE OBVIOUS WAY.  `google/ddpm-celebahq-256` is
~114M parameters trained on 30k images at 256px for hundreds of GPU-hours; `ddpm_celeba_grid.pt` is
**9.3M parameters, 21k images, 64px, 41 minutes**.  The obvious fix -- fine-tune the pretrained
backbone -- is exactly what destroys the density holdout, because its pretraining saw every
combination including the held-out ones.  That is the whole reason these backbones are trained from
scratch (RESULTS 7ee).  So quality and the density holdout are in tension, and the only honest
lever is compute and data.

Three quantities, each against a scale bar:

  FID vs real          the standard, computed against real images of the SEEN cells
  precision            what fraction of samples land in the real manifold -- REAL images scored the
                       same way are the ceiling, and they measure ~0.83, not 1.0
  vendi                diversity, read against real images' own vendi rather than against zero

and the comparison that matters is ACROSS BACKBONES, since the question is whether more training
and more data move it.  Run it on every `ddpm_celeba_grid*` checkpoint on disk; the ladder is the
answer to "what would fix it".

    python tests/diag_grid_quality.py
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

from lifelong_ood.data.celeba import load_celeba_grid                 # noqa: E402
from lifelong_ood.evals import precision_recall, vendi_score          # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet, to_data       # noqa: E402


def fid_like(A, B):
    """Frechet distance between two pixel-feature Gaussians.

    Pixel features, not Inception: the point here is a RELATIVE comparison between backbones on
    identical data, and a number that needs no external weights can be recomputed anywhere.  It is
    not comparable to published FIDs and is not labelled as one.
    """
    from scipy.linalg import sqrtm
    A = np.asarray(A, np.float64).reshape(len(A), -1)
    B = np.asarray(B, np.float64).reshape(len(B), -1)
    # project to a manageable dimension with a FIXED random basis, so every backbone is measured
    # through the same lens
    rng = np.random.default_rng(0)
    P = rng.standard_normal((A.shape[1], 64)) / np.sqrt(A.shape[1])
    A, B = A @ P, B @ P
    mA, mB = A.mean(0), B.mean(0)
    CA, CB = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cov = sqrtm(CA @ CB)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(((mA - mB) ** 2).sum() + np.trace(CA + CB - 2 * cov))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted(glob.glob(os.path.join(CACHE, "ddpm_celeba_grid*.pt")))
    print(f"[q] {'backbone':32s}{'params':>9s}{'imgs':>8s}{'FID*':>9s}"
          f"{'prec':>7s}{'vendi':>7s}")
    ref_printed = False
    for path in paths:
        if "smoke" in path:
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:                                             # noqa: BLE001
            continue
        if "ema" not in ck or "cemb.weight" not in ck.get("ema", {}):
            continue
        sp = ck.get("split", {})
        seen_n = len(ck["remap"])
        d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                             n_per_cell=int(sp.get("n_per_slot", 3000)),
                             n_holdout=int(sp.get("n_holdout") or (12 - seen_n)),
                             max_rows=sp.get("max_rows"))
        X = np.asarray(d["X"], np.float32)
        seen_mask = np.asarray(d["seen"], bool)
        R = X[seen_mask][:400]

        field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                                 layout=ck.get("layout"))
        field.model.load_state_dict(ck["ema"])
        field.model.eval()
        n_par = sum(v.numel() for v in ck["ema"].values() if v.dtype.is_floating_point)

        # UNCONDITIONAL samples: quality is a property of the backbone, not of the conditioning,
        # and the unconditional arm is the one with no composition in it to confound the reading.
        S = np.asarray(to_data(field.sample(n=args.n, steps=args.steps, seed=args.seed,
                                            clip=(-1, 1), branch=None)), np.float32)
        pr = precision_recall(S, R)
        print(f"[q] {os.path.basename(path):32s}{n_par / 1e6:>8.1f}M"
              f"{int(seen_mask.sum()):>8d}{fid_like(S, R):>9.2f}"
              f"{pr['precision']:>7.3f}{float(vendi_score(S)):>7.2f}")

        if not ref_printed:
            # the ceiling for every column: REAL images scored exactly as the samples were
            H = X[seen_mask][400:400 + args.n]
            prr = precision_recall(H, R)
            print(f"[q] {'-- REAL images (ceiling)':32s}{'':>9s}{'':>8s}"
                  f"{fid_like(H, R):>9.2f}{prr['precision']:>7.3f}"
                  f"{float(vendi_score(H)):>7.2f}")
            ref_printed = True

    print(f"\n[q] {'=' * 70}")
    print("[q] FID* is a pixel-feature Frechet distance through a FIXED random projection -- good")
    print("[q] for comparing these backbones to each other on identical data, NOT comparable to a")
    print("[q] published FID. Read every column against the REAL row, which is the ceiling.")
    print("[q] The pretrained CelebA-HQ backbone is not in this table on purpose: it saw every")
    print("[q] held-out combination in pretraining, so it cannot be a density-holdout backbone at")
    print("[q] any quality. The only levers here are model size, data and training length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
