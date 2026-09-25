"""N63 -- train a scene-conditional DDPM on MULTI-OBJECT scenes.

The formulation `docs/CLEVR_VERDICT.md` names as the next one, and the one a VLA actually faces.
Every result so far conditions on ONE composition per image.  Here an image holds TWO objects and
the label is the unordered SET of them, so the question becomes the storage question:

    does a scene of k known objects cost k addresses, or does the scene itself become a
    composition over addresses?

**The conditioning is deliberately NOT compositional.**  The model gets one free embedding row per
SCENE -- 36 seen rows out of 45 possible pairs -- and is never told that a scene contains objects.
This is the same choice the ColorMNIST backbone makes for attributes, and it is what makes the
result meaningful: if a freely-learned scene table turns out to be additive in object membership,
that is a FINDING about what conditional diffusion training discovers, not a property built in by
the parameterisation.  Summing per-object embeddings would answer the question by construction and
measure nothing.  (`n1_train --cond factored` is the built-in-by-construction arm for attributes;
the analogue here is left for N64 to compare against, fitted post hoc.)

N64 reads this checkpoint and asks:
  1. does the scene table factorise into per-object directions (R^2 against `object_membership`)?
  2. does ADDRESS place a HELD-OUT PAIR of known objects correctly?
  3. do per-object DETECTORS -- not a whole-image classifier -- find both objects in the samples?

    python experiments/n63_multiobject_train.py --n-per-scene 400 --epochs 60
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE                                               # noqa: E402

from lifelong_ood.data.multiobject import (N_OBJ, load_multiobject,    # noqa: E402
                                           scene_pairs)
from lifelong_ood.neural import UNet, train_ddpm                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-scene", type=int, default=400)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-holdout", type=int, default=9)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--layout", choices=["permuted", "hwc"], default="hwc",
                   help="hwc is the CORRECT one and the default here.  The ColorMNIST backbone "
                        "predates that fix and keeps `permuted` for continuity; a NEW dataset has "
                        "no such debt, and three results were already lost to layout faults.")
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_per_scene, args.epochs, args.base = 8, 2, 32
    if args.out is None:
        args.out = os.path.join(CACHE, "ddpm_multiobject" + ("_smoke" if args.quick else "") + ".pt")

    d = load_multiobject(n_per_scene=args.n_per_scene, seed=args.seed, n_holdout=args.n_holdout)
    X, scene, seen = d["X"], d["scene"], d["seen"]
    Xs, ss = X[seen], scene[seen]
    uniq = np.array(sorted(set(int(s) for s in ss)))
    remap = {int(s): i for i, s in enumerate(uniq)}
    y = np.array([remap[int(s)] for s in ss])
    P = scene_pairs()
    print(f"[n63] {len(uniq)} seen scenes of {len(P)}; held out "
          f"{[P[h] for h in d['held']]}", flush=True)
    print(f"[n63] {len(Xs)} training images, {N_OBJ} objects in the vocabulary", flush=True)

    # Every object must still appear among the SEEN scenes, or the experiment silently becomes an
    # EXTEND test on new vocabulary rather than an ADDRESS test on a new arrangement.
    present = np.zeros(N_OBJ, int)
    for s in uniq:
        a, b = P[int(s)]
        present[a] += 1
        present[b] += 1
    assert present.min() > 0, f"object starved by the holdout: {present}"
    print(f"[n63] per-object seen-scene counts {present.tolist()} (min {present.min()})", flush=True)

    arch = dict(base=args.base)
    model = UNet(n_classes=len(uniq), **arch)
    print(f"[n63] UNet {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M params", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    ema = train_ddpm(model, Xs, y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     p_drop=args.p_drop, seed=args.seed, layout=args.layout,
                     device="cuda" if torch.cuda.is_available() else "cpu")
    torch.save(dict(ema=ema, n_classes=len(uniq), arch=arch, remap=remap, concepts={},
                    dataset="multiobject", img=d["img"], epochs_done=args.epochs,
                    layout=args.layout, held=d["held"], n_obj=N_OBJ,
                    split=dict(loader="load_multiobject", n_per_scene=args.n_per_scene,
                               seed=args.seed, n_holdout=args.n_holdout, img=d["img"])),
               args.out)
    print(f"[n63] wrote {args.out} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
