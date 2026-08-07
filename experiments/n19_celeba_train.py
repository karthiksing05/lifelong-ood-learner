"""N19 — train the CelebA backbone with FACTORED conditioning, the fix RESULTS 5o asked for.

5o left CelebA measured and unusable: conditioning reached 34% of the real-data ceiling, and 2.3x
the data with 140 epochs moved that to 33%, so the weakness was diagnosed as the conditioning
DESIGN rather than compute.  A per-class embedding row over a 14-bit attribute combination is
estimated from as few as 25 images and shares nothing with the combination that differs from it on
one bit.

This trains the same UNet with `cond_mode="factored"`: one embedding per (attribute, value), summed.
Three things change as a result.

  * Every attribute vector is estimated from every image carrying that value -- 478 (Bald) to 21663
    (No_Beard) rather than 25 -- so the conditioning is trainable.
  * Training is no longer restricted to combinations with enough support to own a row, so ~100k
    images are usable instead of ~5k.
  * A held-out combination costs ZERO new parameters: `attr_embedding(combo)` is a sum of vectors
    the backbone already has.  That is basis extension (n11) made architectural.

STATED PLAINLY BECAUSE IT CHANGES WHAT CELEBA PROVES: on ColorMNIST the factor basis is DISCOVERED
in an entangled per-class space (ideal-words R^2 0.826).  Here it is SUPPLIED.  CelebA therefore does
not re-test "can the basis be found"; it tests the thing ColorMNIST structurally cannot -- whether
composing known factor values *renders correctly* when the factors overlap the same pixels, and
whether the new-vocabulary/new-combination split survives that overlap.

Two regimes, mirroring n1_train's ColorMNIST flags exactly:
  --n-holdout N        hold out N whole attribute COMBINATIONS (the free case)
  --holdout-attr NAME  hold out an attribute's ON value ENTIRELY (the vocabulary case, = --holdout-bg)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                       # noqa: E402

from lifelong_ood.celeba import (ATTRS, IMG_CELEBA, as_float, class_name,  # noqa: E402
                                 load_celeba_comp, nearest_seen)
from lifelong_ood.evals import attr_accuracy, fit_attr_probes, save_grid, save_json  # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet, to_data, train_ddpm  # noqa: E402


def build_probes(data, seed, epochs, cache=True):
    """Attribute probes fit on REAL images, cached — every CelebA experiment scores with these."""
    # keyed by HOW MANY real images the probes saw as well as by seed: a smoke run fits them on a
    # few thousand rows, and a cache that ignored that would silently serve those to the full run
    path = os.path.join(CACHE, f"attr_probes_img{data['img']}_n{len(data['X8'])}_s{seed}.pkl")
    if cache and os.path.exists(path):
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data["X8"]))[:20000]
    d = dict(X=as_float(data["X8"][idx]), attrs=data["attrs"][idx], img=data["img"],
             seen=np.ones(len(idx), bool))     # probes are fit on real images, all of them
    probes = fit_attr_probes(d, seed=seed, epochs=epochs)
    if cache:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(probes, fh)
    return probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img", type=int, default=IMG_CELEBA)
    p.add_argument("--max-rows", type=int, default=None, help="parquet rows to scan; None = all")
    p.add_argument("--max-train", type=int, default=None, help="cap training images")
    p.add_argument("--min-count", type=int, default=60)
    p.add_argument("--n-holdout", type=int, default=24)
    p.add_argument("--holdout-attr", default=None, choices=[None] + ATTRS)
    p.add_argument("--cond", choices=["factored", "class"], default="factored")
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.base, args.max_rows, args.max_train = 2, 32, 4000, 2000
        args.min_count, args.n_holdout = 5, 3
    # NEVER default to a name another run owns.  RESULTS 5p is the story of a celeba run silently
    # overwriting the colormnist backbone; the smoke run of THIS script then did it again to
    # ddpm_celeba.pt.  Every distinguishing flag goes in the filename, and --quick gets its own.
    tag = "celeba_factored" if args.cond == "factored" else "celeba_classcond"
    if args.holdout_attr is not None:
        tag += f"_no{args.holdout_attr}"
    if args.quick:
        tag += "_smoke"
    args.out = args.out or os.path.join(CACHE, f"ddpm_{tag}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[n19] device {dev} | cond={args.cond} | holdout_attr={args.holdout_attr}", flush=True)

    data = load_celeba_comp(img=args.img, max_rows=args.max_rows, min_count=args.min_count,
                            n_holdout=args.n_holdout, holdout_attr=args.holdout_attr,
                            seed=args.seed, cache_dir=CACHE)
    sp = data["split"]
    tr = data["train_idx"]
    if args.max_train and len(tr) > args.max_train:
        tr = np.random.default_rng(args.seed).permutation(tr)[:args.max_train]
    A, combo_of = data["attrs"], sp["combo_of"]

    # class table: every distinct combination present in TRAINING.  In factored mode these rows cost
    # no parameters (they index into the 14x2 attribute table), which is exactly the saving.
    tr_combos = np.unique(combo_of[tr])
    class_attrs = sp["all_combos"][tr_combos]
    remap = {int(c): i for i, c in enumerate(tr_combos)}
    y = np.array([remap[int(c)] for c in combo_of[tr]], np.int64)
    print(f"[n19] {len(tr)} training images | {len(tr_combos)} distinct combos | "
          f"{len(sp['hold_ids'])} held-out combos", flush=True)
    if args.holdout_attr:
        k = sp["holdout_attr_idx"]
        print(f"[n19] attribute {args.holdout_attr} held out ENTIRELY: "
              f"{int((A[:, k] == 1).sum())} images never shown", flush=True)

    arch = dict(base=args.base, img=args.img, mults=(1, 2, 2, 2), cond_mode=args.cond)
    if args.cond == "factored":
        arch["class_attrs"] = class_attrs
    model = UNet(n_classes=len(tr_combos), **arch)
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = args.img, 3 * args.img * args.img
    nparam = sum(q.numel() for q in model.parameters())
    ncond = (model.attr.numel() if args.cond == "factored" else model.cemb.weight.numel())
    print(f"[n19] UNet {nparam / 1e6:.1f}M params ({ncond} in the conditioning head, "
          f"vs {len(tr_combos) * 256} for a per-class table)", flush=True)

    t0 = time.time()
    ema = train_ddpm(model, data["X8"][tr], y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     device=dev, seed=args.seed, tf32=True)
    mins = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(ema=ema, n_classes=len(tr_combos), arch=arch, remap=remap,
                    class_attrs=class_attrs, concepts={}, dataset="celeba", img=args.img,
                    hold_ids=sp["hold_ids"], hold_combos=sp["hold_combos"],
                    holdout_attr=args.holdout_attr, seed=args.seed), args.out)
    print(f"[n19] trained in {mins:.1f} min -> {args.out}", flush=True)

    # ── evaluation: conditional accuracy against the REAL-DATA ceiling ───────
    model.load_state_dict(ema)
    field = NeuralScoreField(model, device=dev, batch=64)
    probes = build_probes(data, args.seed, 8 if args.quick else 30)
    readable = [a for a in probes["_attrs"] if probes[f"{a}_val_acc"] >= 0.65]
    print(f"[n19] probes readable at >=0.65: {len(readable)}/{len(probes['_attrs'])}", flush=True)

    rng = np.random.default_rng(args.seed)
    picks = rng.permutation(len(tr_combos))[:4 if args.quick else 12]
    rows = []
    n = 4 if args.quick else 16
    # NOTE: `attrs=` is deliberately NOT passed.  `attr_accuracy` indexes `combo[k]` by position in
    # the attribute list it is given, so handing it a filtered subset would silently score every
    # attribute against the wrong bit.  Its own `min_val_acc` filter drops unreadable attributes
    # while keeping the indices aligned, which is the same exclusion done safely.
    for i in picks:
        combo = class_attrs[i]
        S = to_data(field.sample(n=n, steps=50, seed=args.seed, clip=(-1, 1), branch=int(i)))
        a_gen = attr_accuracy(probes, S.astype(np.float32), combo)
        real = np.where(combo_of == tr_combos[i])[0][:n]
        a_real = attr_accuracy(probes, as_float(data["X8"][real]), combo)
        rows.append(dict(cls=int(i), gen=a_gen, real=a_real))
    S0 = to_data(field.sample(n=n, steps=50, seed=args.seed, clip=(-1, 1)))
    a_unc = np.mean([attr_accuracy(probes, S0.astype(np.float32),
                                   class_attrs[i]).get("mean_attr", float("nan"))
                     for i in picks])

    cond_pa = float(np.mean([r["gen"].get("mean_attr", float("nan")) for r in rows]))
    real_pa = float(np.mean([r["real"].get("mean_attr", float("nan")) for r in rows]))
    cond_j = float(np.mean([r["gen"]["joint"] for r in rows]))
    real_j = float(np.mean([r["real"]["joint"] for r in rows]))
    save_grid(to_data(field.sample(n=32, steps=50, seed=args.seed, clip=(-1, 1))),
              os.path.join(RESULTS, f"n19_uncond_{tag}.png"),
              title=f"CelebA DDPM unconditional ({args.cond} conditioning)")
    out = dict(train_minutes=mins, params=int(nparam), cond_params=int(ncond),
               n_classes=len(tr_combos), n_train=len(tr), epochs=args.epochs, img=args.img,
               cond_mode=args.cond, holdout_attr=args.holdout_attr,
               n_holdout_combos=int(len(sp["hold_ids"])), readable=readable,
               per_attr=dict(real=real_pa, cond=cond_pa, uncond=float(a_unc)),
               joint=dict(real=real_j, cond=cond_j),
               frac_of_ceiling=float(cond_j / real_j) if real_j > 0 else float("nan"),
               margin_per_attr=float(cond_pa - a_unc))
    save_json(out, os.path.join(RESULTS, f"n19_train_{tag}__seed{args.seed}.json"))
    print(f"[n19] per-attribute  real {real_pa:.3f} | conditional {cond_pa:.3f} | "
          f"uncond {a_unc:.3f}  (margin {cond_pa - a_unc:+.3f})")
    print(f"[n19] joint          real {real_j:.3f} | conditional {cond_j:.3f} | "
          f"{100 * cond_j / max(real_j, 1e-9):.0f}% of ceiling")


if __name__ == "__main__":
    main()
