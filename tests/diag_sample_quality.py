"""Are the generated scenes actually good, or did they only LOOK good in one grid of eight?

I described the Shapes3D backbone's samples as "excellent" on the strength of a single visual
inspection of eight images.  That is not verification, and the number that superficially contradicts
it -- precision 0.000 in N52 -- was measured in PIXEL space against 24 images of ONE composition,
which a well-formed but mis-conditioned sample fails by construction.  Neither observation settles
image quality.

This measures it the way the composition tables do: Inception-V3 pool3 features, k-NN precision and
recall (Kynkaanniemi) plus FID, against a LARGE reference set of real images, on UNCONDITIONAL
samples -- so image quality is separated from whether the label works.

The calibration is the point.  The same measurement runs on ColorMNIST, whose backbone is known to
work, so "good" is defined against a model that does rather than against my impression.  Precision
and recall are relative to each model's own data, so they compare across datasets; FID does not, and
is reported per-model only.

Also writes a 48-sample grid, because a claim about how images look should be checkable by looking.

    python tests/diag_sample_quality.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.evals import paper_metrics                           # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data, to_model     # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")
FIG = os.path.join(HERE, "figures", "additional")

CASES = [
    ("ddpm_colormnist.pt", "colormnist", 32),
    ("ddpm_shapes3d.pt", "shapes3d", 64),
    ("ddpm_shapes3d_cond.pt", "shapes3d", 64),
]


def real_images(kind, n):
    if kind == "colormnist":
        from lifelong_ood.data.datasets import load_colormnist
        d = load_colormnist(n_per_slot=8, seed=0)
    else:
        from lifelong_ood.data.factored import load_factored
        d = load_factored("shapes3d", n_per_slot=200, seed=0, value_stride=2, n_holdout=25)
    X = np.asarray(d["X"], np.float32)
    seen = np.asarray(d["seen"], bool)
    rng = np.random.default_rng(0)
    idx = rng.permutation(np.where(seen)[0])[:n]
    return np.stack([to_model(X[i]).ravel() for i in idx])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_gen, n_ref = 64, 512
    print(f"{'backbone':30s} {'img':>4} {'precision':>10} {'recall':>8} {'F1':>7} {'FID':>9}")
    print("-" * 74)
    grids = []
    for name, kind, img in CASES:
        p = os.path.join(CACHE, name)
        if not os.path.exists(p):
            print(f"{name:30s} {'-':>4} {'(absent -- still training?)':>10}")
            continue
        field = load_field(p, device=dev)
        # UNCONDITIONAL: the null token, i.e. the model's own marginal, no label involved
        S = to_data(field.sample(n=n_gen, steps=50, seed=0, clip=(-1, 1)))
        S = np.asarray(S, np.float32)
        R = real_images(kind, n_ref)
        m = paper_metrics(S, R, img=img, k=3, device=dev, space="inception")
        print(f"{name:30s} {img:>4} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>7.3f} {m['fid']:>9.2f}   [{m.get('space')}]", flush=True)
        grids.append((name, img, S[:16], R[:16]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG, exist_ok=True)
        rows = len(grids) * 2
        fig, axes = plt.subplots(rows, 16, figsize=(16, 1.1 * rows))
        axes = np.atleast_2d(axes)
        r = 0
        for name, img, S, R in grids:
            for lbl, A in (("generated", S), ("real", R)):
                for k in range(16):
                    axes[r][k].axis("off")
                    axes[r][k].imshow(np.clip(A[k].reshape(img, img, 3), 0, 1))
                axes[r][0].set_ylabel(f"{name[:18]}\n{lbl}", rotation=0, ha="right",
                                      va="center", fontsize=6)
                axes[r][0].axis("on")
                axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
                r += 1
        fig.suptitle("Unconditional samples vs real images, per backbone. Image quality only -- the "
                     "label is not involved.\nPrecision/recall are Inception-space and comparable "
                     "across datasets; FID is not.", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(FIG, "diag_sample_quality.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n[quality] wrote {out}")
    except Exception as e:                                             # noqa: BLE001
        print(f"[quality] plot failed: {e}")

    print("\n[quality] ColorMNIST is the calibration: its backbone demonstrably conditions and")
    print("[quality] composes, so its precision/recall is what 'the images are fine' looks like")
    print("[quality] in this measurement. Shapes3D near it = my visual claim holds; far below")
    print("[quality] it = the samples are worse than they looked and stage-1 has a second problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
