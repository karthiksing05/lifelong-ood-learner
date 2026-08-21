"""N81 — CelebA from scratch at 64px, on the 45-cell factored design.

The configuration where BOTH halves are known to work independently, which is what the previous
three runs established the hard way:

  the CONDITIONING works from scratch at 64px.  7ee's retired grid reads `nearest-class` 0.279 and
  `address` 0.297 against a 0.117 floor -- all far above zero, so a trained class reproduces its own
  combination.  On the pretrained 256px transfer (N79/N80) the same quantity reads 0.000-0.031
  against a 0.240 ceiling, with or without backbone adaptation and with or without guidance: an
  additive class embedding into the time embedding of `google/ddpm-celebahq-256` never became
  class-discriminative at that budget.  That is a fact about the transfer, not about CelebA.

  the TABLE works at this design.  The grid is 12 cells over 5 free vectors (basis ratio 1.8), which
  7ee blamed for everything downstream.  `diag_celeba_table_design.py` chose hair(4) x Male x
  Smiling x Bangs x Wearing_Lipstick: 45 cells, 39 seen over 8 free vectors, ratio 4.9, at
  E[joint] 0.299.  `hair` is categorical rather than three binaries -- same free-vector cost, one
  accuracy in the joint instead of three, and no incoherent blond-AND-black cells.

So this is the grid's recipe (`UNet` base 64, mults (1,2,2,2), cond_mode class, 64px, hwc) on the
better table.  Nothing here is new machinery; the two halves have simply never been run together.

**The holdout is a CONDITIONING holdout**: every image stays in training and a held-out cell lacks
only a row in the class table.  7jj's framing, and the one that made the pretrained comparison
meaningful.

    sbatch --export=ALL,SCRIPT="experiments/n81_celeba_factored_train.py --epochs 50" slurm/jobs/gpu.sbatch
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from common import CACHE                                              # noqa: E402

from lifelong_ood.data.celeba import load_celeba_factored             # noqa: E402
from lifelong_ood.neural.ddpm import train_ddpm                       # noqa: E402
from lifelong_ood.neural.unet import UNet                             # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--min-count", type=int, default=200)
    p.add_argument("--n-per-class", type=int, default=3000)
    p.add_argument("--n-holdout", type=int, default=6)
    p.add_argument("--binaries", default="Male:Smiling:Bangs:Wearing_Lipstick",
                   help="COLON-separated; sbatch splits --export on commas")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(CACHE, "ddpm_celeba_fac64.pt"))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.max_rows, args.min_count = 1, 4000, 20
        args.n_holdout, args.out = 2, args.out.replace(".pt", "_quick.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    d = load_celeba_factored(img=args.img, seed=args.seed, min_count=args.min_count,
                             n_holdout=args.n_holdout, max_rows=args.max_rows,
                             n_per_class=args.n_per_class, binaries=args.binaries.split(":"))
    X, slot, seen = np.asarray(d["X"], np.float32), np.asarray(d["slot"]), np.asarray(d["seen"])
    seen_slots = sorted(set(int(s) for s in slot[seen]))
    remap = {int(s): i for i, s in enumerate(seen_slots)}
    n_classes = len(remap)
    n_free = 1 + sum(int(sz) - 1 for sz in d["sizes"])
    print(f"[n81] loaded in {time.time() - t0:.0f}s: {len(X)} images at {args.img}px | "
          f"{len(d['live'])} cells, {n_classes} seen / {len(d['held'])} held out | "
          f"factors {d['names']} sizes {d['sizes']}", flush=True)
    print(f"[n81] BASIS ratio {n_classes} seen / {n_free} free = {n_classes / n_free:.1f} "
          f"(the retired grid 1.8; Shapes3D flagship 14.4)", flush=True)

    tr = np.where(seen)[0]
    y = np.array([remap[int(s)] for s in slot[tr]])
    arch = dict(base=args.base, img=args.img, cond_mode="class", mults=(1, 2, 2, 2))
    model = UNet(n_classes=n_classes, **arch).to(dev)
    print(f"[n81] UNet {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M params, "
          f"{len(tr)} training images over {n_classes} classes "
          f"({len(tr) // max(n_classes, 1)} per class)", flush=True)

    def _save(ema, ep):
        torch.save(dict(ema=ema, n_classes=n_classes, arch=arch, remap=remap,
                        img=args.img, layout="hwc", dataset="celeba_factored",
                        sizes=[int(v) for v in d["sizes"]], names=list(d["names"]),
                        held=[int(h) for h in d["held"]], epochs_done=ep,
                        split=dict(loader="load_celeba_factored", img=args.img, seed=args.seed,
                                   min_count=args.min_count, n_per_class=args.n_per_class,
                                   n_holdout=args.n_holdout, max_rows=args.max_rows,
                                   binaries=args.binaries)),
                   args.out)

    def on_epoch(ep, ema, info=None):
        # THE CHECK THAT SHOULD HAVE GATED N79/N80: is the class table becoming class-specific, or
        # is it drifting onto one common vector?  `shared` near 1 means every class embeds to the
        # same point and samples come out identical whatever the label -- which is what happened on
        # the pretrained backbone, unnoticed for three runs because the raw and centred cosines each
        # hid half of it.
        W = ema.get("cemb.weight")
        if W is not None:
            T = W[:n_classes].float()
            sh = float(T.mean(0).norm() / T.norm(dim=1).mean())
            Wc = torch.nn.functional.normalize(T - T.mean(0, keepdim=True), dim=1)
            cos = float((Wc @ Wc.T).fill_diagonal_(0).abs().mean())
            print(f"[n81] epoch {ep}: table shared {sh:.2f} class-specific {1 - sh:.2f} | "
                  f"centred |cos| {cos:.3f}"
                  f"{'  <- COLLAPSING' if sh > 0.90 else ''}", flush=True)
        if ep % 5 == 0 or ep == args.epochs:
            _save(ema, ep)

    ema = train_ddpm(model, X[tr], y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     device=dev, seed=args.seed, layout="hwc", on_epoch=on_epoch)
    _save(ema, args.epochs)
    print(f"[n81] wrote {args.out} after {args.epochs} epochs, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
