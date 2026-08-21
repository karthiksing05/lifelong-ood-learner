"""Which of CelebA's FORTY attributes can a probe actually read at 64px?

`diag_celeba_design.py` answers the arithmetic half of "can CelebA carry a Shapes3D-shaped table":
yes, up to 153 classes over 11 free vectors, ratio 13.9 against the flagship's 18.2.  But it
optimises ratio with no readability constraint and therefore reaches that ratio through
`Attractive`, `Arched_Eyebrows`, `Pointy_Nose` and `Oval_Face` -- subjective or fine-grained labels
a 64px probe has little chance with.

That matters more on CelebA than anywhere else in this repository, because CelebA's factors are
BINARY, so the only route to more classes is more factors, and joint accuracy is roughly the PRODUCT
of per-factor accuracies:

    3 factors at 0.85 each -> joint 0.61        10 factors at 0.85 each -> joint 0.20

Shapes3D escapes this by having 3 factors with 10, 10 and 4 VALUES -- ratio 18.2 without ever
multiplying more than three probabilities.  So the CelebA design has to maximise ratio SUBJECT TO
the joint staying above the floor, and that needs a per-attribute readability number for all forty.

**AND THE MARGIN RULE IS THE WRONG RULE HERE, WHICH THIS SCRIPT FOUND THE HARD WAY.**  `margin =
val accuracy - base rate` is the guard `probe_readable` applies library-wide, and 5e-sexies is the
retraction that made it non-negotiable: `No_Beard` is 83% positive, so a probe answering "yes"
always scores 0.83 and has read nothing.  That is sound for a BALANCED attribute.  It is
arithmetically broken for a rare one:

    Eyeglasses   val 0.975   base 0.944   margin 0.031  ->  "unreadable"
    Wearing_Hat  val 0.985   base 0.951   margin 0.034  ->  "unreadable"

**16 of CelebA's 40 attributes have a base rate above 0.90, so a +0.10 margin is not merely hard for
them, it is impossible** -- the score is capped at 1.0.  The first run of this script concluded
"7 of 40 readable" and returned the IDENTICAL seven in pixel and Inception space, which is the
tell: a criterion that gives the same answer in two very different feature spaces is measuring
itself, not the spaces.

So readability is reported three ways, and the bar is set on the base-rate-immune ones:

    margin    val - base.  Kept for comparability with 7ee and `probe_readable`; NOT the bar.
    balanced  (TPR + TNR) / 2.  Chance is 0.5 whatever the base rate.
    auroc     threshold-free, from the probe's own logit margin.  This is the bar.

    python tests/diag_celeba_attr_readability.py --max-rows 30000
"""
import argparse
import io as _io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood import inception                                    # noqa: E402
from lifelong_ood.data.celeba import _parquet_shards, find_celeba     # noqa: E402
from lifelong_ood.evals import SoftmaxProbe                           # noqa: E402


def load_images_and_all_attrs(img=64, max_rows=None):
    """-> (names[40], X [N, img*img*3] in [0,1], A [N, 40] uint8).  Centre-crop, the CelebA norm."""
    import pyarrow.parquet as pq
    from PIL import Image
    shards = _parquet_shards(find_celeba())
    names = [c for c in pq.ParquetFile(shards[0]).schema.names
             if c not in ("bytes", "path", "celeb_id", "image")]
    Xs, As, n = [], [], 0
    for sp in shards:
        f = pq.ParquetFile(sp)
        for batch in f.iter_batches(batch_size=512, columns=["image"] + names):
            d = batch.to_pydict()
            for k in range(len(d["image"])):
                im = Image.open(_io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
                w, h = im.size
                top = max(0, (h - w) // 2)
                im = im.crop((0, top, w, top + w)).resize((img, img), Image.BILINEAR)
                Xs.append((np.asarray(im, np.float32) / 255.0).ravel())
                As.append([1 if d[a][k] else 0 for a in names])
                n += 1
                if max_rows and n >= max_rows:
                    return names, np.stack(Xs), np.array(As, np.uint8)
    return names, np.stack(Xs), np.array(As, np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-rows", type=int, default=30000)
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--margin", type=float, default=0.10,
                   help="legacy bar, reported for comparability only -- see the docstring")
    p.add_argument("--auroc", type=float, default=0.75,
                   help="the actual bar: base-rate-immune, so a 5%%-positive attribute can pass")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names, X, A = load_images_and_all_attrs(args.img, args.max_rows)
    print(f"[read] {len(X)} images at {args.img}px, {len(names)} attributes", flush=True)

    spaces = {"pixel": X}
    if inception.available():
        spaces["inception"] = inception.features(X, img=args.img)
        print(f"[read] inception features {spaces['inception'].shape}", flush=True)
    else:
        print("[read] inception unavailable -- pixel only", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    ntr = int(0.8 * len(X))
    tr, va = perm[:ntr], perm[ntr:]

    out = {}
    for space, FEAT in spaces.items():
        print(f"\n[read] === {space} ===")
        print(f"[read] {'attribute':22s}{'val':>7}{'base':>7}{'margin':>8}{'bal':>7}"
              f"{'auroc':>8}   readable")
        rows = []
        for j, nm in enumerate(names):
            y = A[:, j].astype(int)
            base = float(max(y[va].mean(), 1 - y[va].mean()))
            pr = SoftmaxProbe(2, seed=args.seed)
            pr.fit(FEAT[tr], y[tr])
            yh = pr.predict(FEAT[va])
            yv = y[va]
            acc = float((yh == yv).mean())
            tpr = float((yh[yv == 1] == 1).mean()) if (yv == 1).any() else float("nan")
            tnr = float((yh[yv == 0] == 0).mean()) if (yv == 0).any() else float("nan")
            bal = 0.5 * (tpr + tnr)
            # AUROC from the probe's own logit margin -- threshold-free, so a rare attribute is
            # judged on whether it can be RANKED, not on whether an argmax happens to fire.
            Xb = np.concatenate([np.asarray(FEAT[va], np.float32),
                                 np.ones((len(va), 1), np.float32)], 1)
            sc = (Xb @ pr.Wb)[:, 1] - (Xb @ pr.Wb)[:, 0]
            order = np.argsort(sc)
            ranks = np.empty(len(sc), float)
            ranks[order] = np.arange(1, len(sc) + 1)
            npos, nneg = int((yv == 1).sum()), int((yv == 0).sum())
            auroc = (float((ranks[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
                     if npos and nneg else float("nan"))
            rows.append(dict(attr=nm, val=acc, base=base, margin=acc - base,
                             balanced=bal, auroc=auroc, pos_rate=float(yv.mean())))
            print(f"[read] {nm:22s}{acc:>7.3f}{base:>7.3f}{acc - base:>8.3f}{bal:>7.3f}"
                  f"{auroc:>8.3f}   {'YES' if auroc >= args.auroc else ''}")
        good = [r["attr"] for r in rows if r["auroc"] >= args.auroc]
        legacy = [r["attr"] for r in rows if r["margin"] >= args.margin]
        print(f"[read] {space}: legacy margin>={args.margin} would admit {len(legacy)}: {legacy}")
        out[space] = dict(rows=rows, readable=good)
        print(f"[read] {space}: {len(good)} of {len(names)} readable at AUROC >= {args.auroc}")
        print(f"[read]   {good}")

    o = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "celeba_attr_readability.json")
    with open(o, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[read] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
