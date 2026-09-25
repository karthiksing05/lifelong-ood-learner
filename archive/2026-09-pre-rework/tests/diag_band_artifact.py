"""The CelebA banding artifact: is it the layout permutation?  A falsifiable prediction.

Samples show horizontal seams and rectangular patches -- coherent faces cut by band boundaries.
The permuted layout predicts EXACTLY that, and predicts WHERE.

Under `.reshape(-1, 3, H, W)` of an HWC row, the UNet's three "channels" are image ROWS
[0, H/3), [H/3, 2H/3), [2H/3, H).  A convolution's first layer treats channels independently and
mixes them only through learned weights, so the three bands are free to drift apart -- and where
they meet, at rows H/3 and 2H/3, there is nothing enforcing continuity.  For 64px that is rows
**21 and 43**.

So the prediction is sharp: in samples from a `permuted` backbone, the row-to-row difference should
SPIKE at rows 21 and 43 relative to its neighbours, and real images should show no such spike.  If
the spike is absent, the permutation is not what produces the banding and this line of investigation
is wrong.

Reports a "seam score": the row-difference at the two band boundaries divided by the median
row-difference elsewhere.  1.0 = no seam.  Run on every CelebA backbone available, plus real data
as the control, plus the `hwc` backbone as the counter-control.

    python tests/diag_band_artifact.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.data.celeba import as_float, load_celeba                          # noqa: E402
from lifelong_ood.neural import load_field, to_data                            # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")


def seam_profile(S, img):
    """Mean |row[i+1]-row[i]| over samples, as a length-(img-1) profile."""
    A = np.asarray(S, np.float32).reshape(len(S), img, img, 3)
    return np.abs(np.diff(A, axis=1)).mean(axis=(0, 2, 3))


def seam_score(prof, img):
    """Row-difference AT the band boundaries / median elsewhere.  1.0 means no seam."""
    b = [int(round(img / 3.0)) - 1, int(round(2 * img / 3.0)) - 1]
    b = [i for i in b if 0 <= i < len(prof)]
    other = np.array([v for i, v in enumerate(prof) if all(abs(i - j) > 1 for j in b)])
    return float(np.mean(prof[b]) / max(np.median(other), 1e-9)), b


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = 64
    data = load_celeba(n_per_class=40, img=img, seed=0, min_count=25, max_rows=25000)
    R = np.asarray(data["X"])[:256].astype(np.float32)

    rows = []
    prof_real = seam_profile(R, img)
    s_real, bnds = seam_score(prof_real, img)
    print(f"[bands] band boundaries for {img}px: rows {bnds[0]+1} and {bnds[1]+1}\n")
    print(f"[bands] {'REAL CelebA (control)':44s} seam score {s_real:.3f}")
    rows.append(("real", s_real, prof_real))

    cands = [("ddpm_celeba_factored.pt", "our shipped 64px backbone (permuted)"),
             ("ddpm_celeba_layout_permuted.pt", "12-epoch pair: permuted"),
             ("ddpm_celeba_layout_hwc.pt", "12-epoch pair: HWC"),
             ("ddpm_celeba_permuted.pt", "60-epoch pair: permuted"),
             ("ddpm_celeba_hwc.pt", "60-epoch pair: HWC (the replacement backbone)")]
    for ckname, label in cands:
        p = os.path.join(CACHE, ckname)
        if not os.path.exists(p):
            print(f"[bands] {label:44s} (missing {ckname})")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        field = load_field(p, device=dev, batch=16)
        S = to_data(field.sample(n=32, steps=50, seed=0, clip=(-1, 1))).astype(np.float32)
        prof = seam_profile(S, img)
        sc, _ = seam_score(prof, img)
        print(f"[bands] {label:44s} seam score {sc:.3f}   (layout={ck.get('layout','permuted')})")
        rows.append((label, sc, prof))
        del field
        torch.cuda.empty_cache()

    print("\n[bands] A seam score well above 1 means the row-difference SPIKES exactly at the band\n"
          "        boundaries -- the permutation's signature.  Near 1 for `permuted` would refute\n"
          "        the layout explanation for the banding.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for label, sc, prof in rows:
        ax.plot(np.arange(1, len(prof) + 1), prof, lw=1.6, label=f"{label}  (seam {sc:.2f})")
    for b in bnds:
        ax.axvline(b + 1, color="#B03A2E", ls=(0, (4, 3)), lw=1.2, alpha=0.8)
    ax.set_xlabel("row boundary  i  (difference between row i and row i+1)")
    ax.set_ylabel("mean |row difference|")
    ax.set_title(f"Where do CelebA samples discontinue?  Dashed lines = the permuted layout's band\n"
                 f"boundaries (rows {bnds[0]+1}, {bnds[1]+1}), where its three 'channels' meet.",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    o = os.path.join(FIG, "extra", "f24_band_artifact.png")
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"[bands] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
