"""Does the layout permutation cost MORE on CelebA than on Color-MNIST?  It should, and by how much.

N45 measured the permutation at -28.7% held-out loss on Color-MNIST and noted that Color-MNIST is
the easiest possible case for a scrambled CNN: a near-binary stroke on flat colour, almost all low
frequency, exactly the content a permuted network can still fit by brute force.  CelebA is the
opposite -- dense local texture is most of the signal -- so the mechanism predicts a LARGER effect.

The two backbones were trained by `slurm/jobs/n45_celeba.sbatch`, identical in every respect except
`layout`.  This scores both on the SAME held-out split and renders samples from each, because the
loss is the rigorous number and the pictures are the legible one.

    python tests/diag_layout_celeba.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.celeba import as_float, load_celeba_comp                     # noqa: E402
from lifelong_ood.evals import speckle                                         # noqa: E402
from lifelong_ood.neural import load_field, to_data                            # noqa: E402
from lifelong_ood.neural.ddpm import train_ddpm                                # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
ARMS = [("permuted (every backbone to date)", "ddpm_celeba_layout_permuted.pt"),
        ("hwc (a real image)", "ddpm_celeba_layout_hwc.pt")]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = 64
    data = load_celeba_comp(img=img, max_rows=25000, min_count=60, n_holdout=24)
    sp = data["split"]
    rng = np.random.default_rng(0)
    hold = rng.permutation(len(data["X8"]))[:2000]
    X = as_float(data["X8"][hold])
    y = np.asarray(sp["combo_of"])[hold]
    R = as_float(data["X8"][:512]).astype(np.float32)

    out = []
    panels = {}
    for name, ckname in ARMS:
        p = os.path.join(CACHE, ckname)
        if not os.path.exists(p):
            print(f"[layout-celeba] missing {p} -- run slurm/jobs/n45_celeba.sbatch")
            return 1
        ck = torch.load(p, map_location="cpu", weights_only=False)
        lay = ck.get("layout", "permuted")
        field = load_field(p, device=dev, batch=32)
        # remap the held-out labels into the checkpoint's own class indexing
        remap = ck["remap"]
        keep = np.array([i for i, c in enumerate(y) if int(c) in remap])
        yy = np.array([remap[int(c)] for c in y[keep]], np.int64)
        loss = train_ddpm(field.model, X[keep], yy, epochs=1, batch=32, device=dev,
                          seed=1, eval_only=True, layout=lay)
        S = to_data(field.sample(n=8, steps=50, seed=0, clip=(-1, 1))).astype(np.float32)
        sk = speckle(S, R, img=img)
        out.append((name, lay, float(loss), float(sk.get("tv_ratio", np.nan))))
        panels[name] = S
        print(f"[layout-celeba] {name:36s} layout={lay:9s} held-out loss {float(loss):.5f}   "
              f"tv vs real {sk.get('tv_ratio', float('nan')):.3f}", flush=True)
        del field
        torch.cuda.empty_cache()

    a, b = out[0], out[1]
    print(f"\n[layout-celeba] hwc is {100 * (a[2] - b[2]) / a[2]:+.1f}% on held-out loss "
          f"({a[2]:.5f} -> {b[2]:.5f})")
    print("[layout-celeba] Color-MNIST measured -28.7%.  A LARGER effect here is the mechanism's own "
          "prediction: locality matters more where texture carries the signal.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("real", R[:8])] + [(n, panels[n]) for n, _ in ARMS]
    fig, axes = plt.subplots(len(rows), 8, figsize=(12, 1.7 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, S) in enumerate(rows):
        for c in range(8):
            axes[r][c].axis("off")
            if c < len(S):
                axes[r][c].imshow(np.clip(np.asarray(S[c], np.float32).reshape(img, img, 3), 0, 1))
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s_ in axes[r][0].spines.values():
            s_.set_visible(False)
    fig.suptitle(f"CelebA, same UNet / data / seed / 12 epochs — only the input layout differs.\n"
                 f"held-out loss {a[2]:.4f} → {b[2]:.4f}  ({100 * (a[2] - b[2]) / a[2]:+.1f}%); "
                 f"Color-MNIST measured -28.7%", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    o = os.path.join(FIG, "f23_layout_celeba.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[layout-celeba] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
