"""Is the UNet being fed a pixel PERMUTATION?  Yes -- and it destroys the convolutional prior.

Every flat vector in this repository is HWC as the loaders write it (`rgb_img` and `celeba.rgb`
both display via `.reshape(img, img, 3)`).  Every consumer -- `_as_img`, the uint8 training branch,
`pretrained.py:217`, the samplers -- turns that flat row into a tensor with a PLAIN
`.reshape(-1, 3, img, img)`.  A plain reshape of an HWC row is not an image: it is a fixed
permutation of the pixels.

`ddpm.py` records this deliberately, arguing it is harmless because "training and sampling apply
the SAME permutation [so] the model simply learns in the permuted space".  That argument is sound
for a permutation-equivariant model.  **A CNN is not one.**  Under this reshape, for 64x64x3:

    what the UNet calls channel 0/1/2   = image ROWS 0-21 / 21-42 / 42-63  (not R/G/B)
    its horizontal axis                 = the COLOUR axis (adjacent pixels are R,G,B of one pixel)
    its vertical axis                   = a jump of ~22 pixels ACROSS the image

so a single 3x3 convolution -- the operation the entire architecture is built out of -- reads real
pixels from columns 3, 24 and 45 simultaneously.  There is no spatial locality left for it to
exploit, at any scale, and downsampling compounds it.  The model can still fit low-frequency
structure by brute force, which is why faces appear at all and why the simpler Color-MNIST looks
acceptable; what it cannot do is render coherent local texture, and the residue is precisely the
blocky, banded, ~22-pixel-quilted artifact seen in f19.

This script proves it three ways and needs no training:

  1. INDEX ALGEBRA   what a 3x3 conv actually touches (exact, no model involved).
  2. PICTURE         a real face rendered as the UNet sees it.
  3. CONVENTION      which layout the PRETRAINED google/ddpm-celebahq-256 expects -- it was trained
                     on true CHW, so if we hand it permuted rows the damage is far worse than a
                     lost prior, and it would explain n44's rough samples on a photorealistic
                     backbone.  Sampled natively and displayed BOTH ways; whichever is a face wins.

    python tests/diag_layout.py            # 1 and 2 only, CPU
    python tests/diag_layout.py --pretrained   # adds 3, needs a GPU
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")


def index_algebra(H=64, W=64, C=3):
    """Exactly which real pixels a 3x3 conv in model space reads.  Pure arithmetic."""
    h, w, c = np.meshgrid(np.arange(H), np.arange(W), np.arange(C), indexing="ij")
    flat = h * W * C + w * C + c                 # HWC, as the loaders write it
    ch, y, x = flat // (H * W), (flat % (H * W)) // W, flat % W   # as the UNet reinterprets it
    inv = {}
    for a in range(H):
        for b in range(W):
            for d in range(C):
                inv[(ch[a, b, d], y[a, b, d], x[a, b, d])] = (a, b, d)
    print("  UNet 'channel' 0 = image rows %d-%d" % (h[ch == 0].min(), h[ch == 0].max()))
    print("  UNet 'channel' 1 = image rows %d-%d" % (h[ch == 1].min(), h[ch == 1].max()))
    print("  UNet 'channel' 2 = image rows %d-%d" % (h[ch == 2].min(), h[ch == 2].max()))
    print("\n  a 3x3 conv at model-space (ch0, y=10, x=10) reads real (row, col, colour):")
    for dy in (-1, 0, 1):
        print("     ", [inv[(0, 10 + dy, 10 + dx)] for dx in (-1, 0, 1)])
    a, b = inv[(0, 10, 10)], inv[(0, 11, 10)]
    print(f"\n  -> vertical neighbours in model space are {abs(b[1] - a[1])} pixels apart "
          f"HORIZONTALLY in the image; horizontal neighbours are the same pixel's colour channels.")
    return inv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[layout] 1. INDEX ALGEBRA (64x64x3)\n")
    index_algebra()

    print("\n[layout] 2. PICTURE -- a real face as the UNet sees it")
    from lifelong_ood.data.celeba import as_float, load_celeba_comp
    data = load_celeba_comp(img=64, max_rows=64, min_count=1, n_holdout=1)
    img = 64
    X = as_float(data["X8"][:4]).astype(np.float32)

    panels = [("real image\n(flat row shown as HWC — correct)",
               [x.reshape(img, img, 3) for x in X]),
              ("what the UNet actually convolves\n(the same row reshaped to 3x64x64)",
               [np.transpose(x.reshape(3, img, img), (1, 2, 0)) for x in X])]
    rows = list(panels)

    if args.pretrained:
        print("\n[layout] 3. CONVENTION -- what does the pretrained backbone expect?")
        import torch
        from diffusers import DDPMPipeline
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DDPMPipeline.from_pretrained(args.model).to(dev)
        with torch.no_grad():
            out = pipe(batch_size=4, num_inference_steps=50,
                       generator=torch.Generator(dev).manual_seed(0), output_type="np")
        S = np.asarray(out.images, np.float32)            # native, correct: (n, H, W, 3)
        n, Hn = S.shape[0], S.shape[1]
        print(f"          sampled {n} images natively at {Hn}x{Hn} from {args.model}")
        # Our pipeline would flatten this and re-reshape it.  Show both readings.
        flatCHW = np.stack([np.transpose(s, (2, 0, 1)).ravel() for s in S])   # true CHW flattened
        rows.append(("pretrained sample, displayed as our\npipeline would (flat -> HWC)",
                     [np.clip(f.reshape(Hn, Hn, 3), 0, 1) for f in flatCHW]))
        rows.append(("pretrained sample, native layout\n(the model's own output)",
                     [np.clip(s, 0, 1) for s in S]))

    fig, axes = plt.subplots(len(rows), 4, figsize=(9.0, 2.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(rows):
        for c in range(4):
            axes[r][c].axis("off")
            if c < len(imgs):
                axes[r][c].imshow(np.clip(imgs[c], 0, 1))
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s_ in axes[r][0].spines.values():
            s_.set_visible(False)
    fig.suptitle("The UNet is trained on a pixel permutation, not an image.\n"
                 "A plain .reshape(3,H,W) of an HWC row makes 'channels' out of horizontal bands "
                 "and the width axis out of colour.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "extra", "f20_layout.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[layout] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
