"""Which CelebA attributes are worth conditioning on at all?

This repository's CelebA track uses a fixed list of 14 attributes (`celeba.ATTRS`), and the
compositional classes are their joint settings.  That choice was never measured, and it is the
likely reason the CelebA numbers look weak:

  * an attribute the BACKBONE cannot learn still multiplies the class count, so every class gets
    fewer images and the conditioning is diluted;
  * an attribute the PROBE cannot read caps every score that includes it -- real held-out images
    score mean-attribute 0.827, so 0.827 is the ceiling, not 1.0;
  * an attribute that is nearly constant carries no compositional information at all.

Measured on the ATTRS list, `Bald` has base rate **0.000**, `Sideburns` 0.020, `Gray_Hair` 0.038 and
`Wearing_Necklace` 0.046, while `No_Beard` is 0.917 -- five of fourteen are close to useless, and
three of those are the ones whose probes were already known to be blind (Oval_Face TPR 0.066, Bald
0.206, Gray_Hair 0.321).

This scores a wider candidate pool on the two criteria that matter and prints the subset that passes
both, so the attribute set becomes a measurement rather than an inherited constant.  A usable
attribute needs a **balanced base rate** (a rare attribute cannot be learned from few positives, and
its accuracy is dominated by the negative class) and a **readable probe** (TPR, not plain accuracy,
which a constant predictor maximises on a rare attribute).

    python tests/diag_celeba_attrs.py --n 8000
"""
import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lifelong_ood.celeba import ATTRS, _parquet_shards, find_celeba          # noqa: E402
from lifelong_ood.evals import SoftmaxProbe                                   # noqa: E402

# ATTRS plus the salient, well-known ones it omits.  Eyeglasses in particular is both balanced and
# highly visible, and is what the pretrained case study (N25) uses.
CANDIDATES = sorted(set(ATTRS) | {
    "Eyeglasses", "Young", "Wearing_Hat", "Heavy_Makeup", "Mouth_Slightly_Open",
    "Bangs", "Pale_Skin", "Wearing_Lipstick", "High_Cheekbones", "Attractive",
    "Arched_Eyebrows", "Bags_Under_Eyes", "Chubby", "Goatee", "Straight_Hair",
})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000, help="images to score")
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min-base", type=float, default=0.15)
    p.add_argument("--min-tpr", type=float, default=0.60)
    p.add_argument("--min-bal", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image
    root = find_celeba()
    have = set(pq.ParquetFile(_parquet_shards(root)[0]).schema_arrow.names)
    cand = [a for a in CANDIDATES if a in have]
    print(f"[diag] scoring {len(cand)} candidate attributes on {args.n} images at "
          f"{args.img}px\n", flush=True)

    X, Y, n = [], [], 0
    for sp in _parquet_shards(root):
        f = pq.ParquetFile(sp)
        for b in f.iter_batches(batch_size=512, columns=["image"] + cand):
            d = b.to_pydict()
            for k in range(len(d["image"])):
                im = Image.open(io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
                w, h = im.size
                t = max(0, (h - w) // 2)
                im = im.crop((0, t, w, t + w)).resize((args.img, args.img), Image.BICUBIC)
                X.append((np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1).ravel())
                Y.append([1 if d[a][k] else 0 for a in cand])
                n += 1
            if n >= args.n:
                break
        if n >= args.n:
            break
    X, Y = np.stack(X), np.array(Y)
    cut = int(0.9 * len(X))

    print(f"    {'attribute':22s}{'base':>7s}{'TPR':>7s}{'TNR':>7s}{'bal':>7s}  {'in ATTRS':>9s}  ok")
    keep = []
    for i, a in enumerate(cand):
        pr = SoftmaxProbe(2, seed=args.seed, epochs=args.epochs).fit(X[:cut], Y[:cut, i])
        yv, pv = Y[cut:, i], pr.predict(X[cut:])
        tpr = float((pv[yv == 1] == 1).mean()) if (yv == 1).any() else 0.0
        tnr = float((pv[yv == 0] == 0).mean()) if (yv == 0).any() else 0.0
        bal = 0.5 * (tpr + tnr)
        base = float(Y[:, i].mean())
        ok = (tpr >= args.min_tpr and bal >= args.min_bal
              and args.min_base <= base <= 1 - args.min_base)
        if ok:
            keep.append((a, base, tpr, bal))
        print(f"    {a:22s}{base:7.3f}{tpr:7.2f}{tnr:7.2f}{bal:7.2f}"
              f"{('yes' if a in ATTRS else ''):>9s}  {'YES' if ok else ''}", flush=True)

    keep.sort(key=lambda r: -r[3])
    print(f"\n[diag] {len(keep)} attributes pass both criteria "
          f"(base in [{args.min_base}, {1 - args.min_base}], TPR >= {args.min_tpr}, "
          f"balanced >= {args.min_bal}):")
    for a, base, tpr, bal in keep:
        print(f"      {a:22s} base {base:.2f}  TPR {tpr:.2f}  balanced {bal:.2f}")
    print(f"\n[diag] of the {len(ATTRS)} attributes currently used, "
          f"{sum(1 for a, *_ in keep if a in ATTRS)} pass.")
    top = [a for a, *_ in keep[:4]]
    print(f"[diag] a 4-attribute set -> {2 ** 4} compositional classes instead of "
          f"{2 ** len(ATTRS)} possible: {top}")


if __name__ == "__main__":
    main()
