"""Is §30d's "the bank renders nothing on a pretrained backbone" a DISPLAY bug?

§30d has been open for a while with three explanations killed by measurement: the perturbation
magnitude (a sweep at n26's own rank/lr/budget leaves samples near the control), the pool (it is real
CelebA photographs), and `add_concept`'s `mu` (tree placement, not conditioning).  The remaining
listed candidate was "the rendering call itself", untested.

Reading the two conventions against each other makes it the obvious suspect:

    n25_pretrained_case.show()   reshape(3, img, img).transpose(1, 2, 0)      <- CHW-flat
    PretrainedScoreField         layout="hwc" by default; `_to_flat` does
                                 permute(0, 2, 3, 1).reshape(n, -1)           <- HWC-flat

so `n26` renders HWC-flat rows as though they were CHW-flat.  That is a channel/row transposition,
and the reported symptom -- "rainbow noise with fine horizontal striping" -- is what one looks like.
It is the same fault as RESULTS §30b (which was about the model's INPUT) on the model's OUTPUT.

If this is right it makes a strong prediction that has nothing to do with the adapter:

    THE UNTOUCHED BACKBONE, rendered by `show`, IS ALSO NOISE.

and it also explains why the controlled sweep found nothing wrong: `tv` is computed on the flat
arrays, and the control is transposed identically, so the RATIO stays near 1 while both pictures are
scrambled.  A metric that cannot see the defect reported no defect.

This samples the untouched model once and renders it three ways.  No training, no adapter -- if the
control is already broken, the bank was never the problem.

    python tests/diag_pretrained_render.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")


def seam(A):
    """Mean |difference| across horizontal neighbours minus across vertical ones, normalized.

    A CHW-read of HWC data interleaves the three channels down the rows, so neighbouring ROWS stop
    being correlated while neighbouring COLUMNS still are.  This asymmetry is the fingerprint, and
    it needs no reference image: a real photograph is close to isotropic (~1.0), a transposed one
    is not.
    """
    A = np.asarray(A, np.float32)
    dv = np.abs(np.diff(A, axis=0)).mean()
    dh = np.abs(np.diff(A, axis=1)).mean()
    return float(dv / max(dh, 1e-8))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from lifelong_ood.data.celeba import as_float, load_celeba
    from lifelong_ood.neural.pretrained import load_pretrained
    field, img = load_pretrained("google/ddpm-celebahq-256", device=dev, batch=4)
    print(f"[render] field layout = {getattr(field, 'layout', '?')}, img = {img}\n")

    S = field.sample(n=4, steps=50, seed=0, clip=(-1, 1))
    S01 = (np.asarray(S, np.float32) + 1.0) / 2.0

    reads = {
        "CHW  (what n26's `show` does)": lambda r: r.reshape(3, img, img).transpose(1, 2, 0),
        "HWC  (what the field returns)": lambda r: r.reshape(img, img, 3),
    }
    real = np.asarray(as_float(load_celeba(n_per_class=2, img=img, seed=0, min_count=25,
                                           max_rows=2000)["X"][:4]), np.float32)
    # `as_float` hands back the repository's model-space convention, which is not always [0,1];
    # rendering it raw clipped the whole reference row to black while the gradient statistic below
    # stayed correct, so the panel silently lost its control.  Normalize for DISPLAY only.
    if real.min() < -0.01 or real.max() > 1.01:
        real = (real - real.min()) / max(real.max() - real.min(), 1e-8)
    print(f"  {'read':34s} {'vertical/horizontal gradient ratio':>36s}")
    print(f"  {'real CelebA photographs (the bar)':34s} "
          f"{np.mean([seam(r.reshape(img, img, 3)) for r in real]):36.3f}")
    panels = {}
    for name, fn in reads.items():
        imgs = [np.clip(fn(r), 0, 1) for r in S01]
        panels[name] = imgs
        print(f"  {name:34s} {np.mean([seam(a) for a in imgs]):36.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(panels) + ["real CelebA"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9, 2.4 * len(rows)))
    for r, name in enumerate(rows):
        imgs = (panels[name] if name in panels
                else [x.reshape(img, img, 3) for x in real])
        for c in range(4):
            axes[r][c].imshow(np.clip(imgs[c], 0, 1))
            axes[r][c].axis("off")
        axes[r][0].set_title(name, fontsize=8, loc="left")
    fig.suptitle("The UNTOUCHED pretrained backbone, read two ways. No adapter, no consolidation.\n"
                 "If the top row is noise, §30d's 'the bank renders nothing' was never about the "
                 "bank.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "extra", "f30_pretrained_render.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[render] wrote {out}")
    print("[render] a ratio near 1 is an image; a ratio far from 1 in the CHW row means the flat\n"
          "         rows are HWC and every figure that used `show` on them was scrambled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
