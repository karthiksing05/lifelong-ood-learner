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
                                 set_attrs,
                                 load_celeba_comp, nearest_seen)
from lifelong_ood.evals import attr_accuracy, fit_attr_probes, save_grid, save_json  # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet, to_data, train_ddpm  # noqa: E402


def build_probes(data, seed, epochs, cache=True, layout="hwc"):
    """Attribute probes fit on REAL images, cached — every CelebA experiment scores with these.

    Keyed by how many real images the probes saw, by seed, AND by pixel LAYOUT.  A probe is a linear
    model over flat pixels, so it is only valid for the flat ordering it was fitted on; the layout
    A/B loaded HWC-fitted probes and applied them to CHW-flat images, which showed up as REAL
    held-out images scoring 0.557 where the same images score 0.770 under matched probes.  A
    mismatched probe makes every arm look equally bad, which is the most misleading failure a
    comparison can have.
    """
    path = os.path.join(CACHE,
                        f"attr_probes_img{data['img']}_n{len(data['X8'])}_{layout}_s{seed}.pkl")
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
        # write-then-rename: several CelebA backbones finish within minutes of each other and every
        # one of them reaches this line with the same cache key, so a plain open(..., "wb") would
        # let one job read another's half-written pickle
        import pickle
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(probes, fh)
        os.replace(tmp, path)
    return probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--save-every", type=int, default=5,
                   help="write a resumable snapshot to <out>.partial every N epochs; 0 disables. "
                        "At 128px an epoch is ~17 min and the run does not fit comfortably inside "
                        "one wall limit, so the default is on.")
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
    p.add_argument("--holdout-pair", nargs=2, default=None, metavar=("ATTR_A", "ATTR_B"),
                   help="two attribute names, SPACE separated (e.g. --holdout-pair Wavy_Hair "
                        "Blond_Hair).  Space, not comma: `sbatch --export=ALL,SCRIPT=...` splits "
                        "its value on commas, so a comma-separated pair silently arrives truncated "
                        "to one name.  "
                        "Withholds every combination in which BOTH are on, so each attribute is "
                        "still seen thousands of times alone and only the CO-OCCURRENCE is novel. "
                        "This is the direct analogue of ColorMNIST's held-out (fg, bg) pairs and is "
                        "a far stronger compositional demand than holding out 24 of ~1200 "
                        "combinations individually.")
    p.add_argument("--chw", action="store_true",
                   help="store images CHW-flat so that _as_img's reshape yields a TRUE image.\n"
                        "This repository's loaders write flat vectors HWC and every consumer "
                        "reinterprets them as CHW, which is self-consistent but permutes the "
                        "pixels: at 64x64 a model 'row' of 64 entries is only 21 real pixels with "
                        "RGB interleaved along the width, and one real row is split across three "
                        "model rows.  ColorMNIST tolerates that at 32x32; whether CelebA does is "
                        "an empirical question, and this flag is how it gets asked.  Probes are "
                        "fit on the same flat convention, so scoring stays valid either way; only "
                        "the sample-grid image is drawn wrong under --chw.")
    p.add_argument("--cond", choices=["factored", "class"], default="factored")
    p.add_argument("--attrs", nargs="+", default=None,
                   help="conditioning attributes.  Default is the inherited 14, of which "
                        "tests/diag_celeba_attrs.py shows only 2 are both balanced and "
                        "probe-readable; a short measured set gives far more images per "
                        "compositional class and a probe ceiling near 1.0.")
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.attrs:
        set_attrs(args.attrs)          # in place, so every module sees it
        print(f"[n19] conditioning on {len(args.attrs)} attributes: {args.attrs}",
              flush=True)
    if args.quick:
        args.epochs, args.base, args.max_rows, args.max_train = 2, 32, 4000, 2000
        args.min_count, args.n_holdout = 5, 3
    # NEVER default to a name another run owns.  RESULTS 5p is the story of a celeba run silently
    # overwriting the colormnist backbone; the smoke run of THIS script then did it again to
    # ddpm_celeba.pt.  Every distinguishing flag goes in the filename, and --quick gets its own.
    pair = tuple(args.holdout_pair) if args.holdout_pair else None
    tag = "celeba_factored" if args.cond == "factored" else "celeba_classcond"
    if args.holdout_attr is not None:
        tag += f"_no{args.holdout_attr}"
    if pair:
        tag += f"_nopair{pair[0]}-{pair[1]}"
    if args.chw:
        tag += "_chw"
    if args.quick:
        tag += "_smoke"
    args.out = args.out or os.path.join(CACHE, f"ddpm_{tag}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[n19] device {dev} | cond={args.cond} | holdout_attr={args.holdout_attr}", flush=True)

    data = load_celeba_comp(img=args.img, max_rows=args.max_rows, min_count=args.min_count,
                            n_holdout=args.n_holdout, holdout_attr=args.holdout_attr,
                            holdout_pair=pair, seed=args.seed, cache_dir=CACHE)
    if args.chw:
        n_img = len(data["X8"])
        data["X8"] = np.ascontiguousarray(
            data["X8"].reshape(n_img, args.img, args.img, 3).transpose(0, 3, 1, 2)
        ).reshape(n_img, -1)
        print(f"[n19] images stored CHW-flat: the model now sees upright images", flush=True)
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
    if pair:
        ka, kb = sp["holdout_pair_idx"]
        print(f"[n19] PAIR {pair[0]} x {pair[1]} never co-occur in training: "
              f"{int(((A[:, ka] == 1) & (A[:, kb] == 1)).sum())} images withheld, but "
              f"{int((A[:, ka] == 1).sum())} and {int((A[:, kb] == 1).sum())} images carry each "
              f"attribute alone", flush=True)

    # one more downsampling level at 128px, so the deepest block still reaches the 16x16 and
    # 8x8 resolutions the attention layers are placed at; without it a 128px model attends
    # only at 32x32 and above and never forms a global view of the face.
    mults = (1, 2, 2, 2, 4) if args.img >= 128 else (1, 2, 2, 2)
    arch = dict(base=args.base, img=args.img, mults=mults, cond_mode=args.cond)
    if args.cond == "factored":
        arch["class_attrs"] = class_attrs
    model = UNet(n_classes=len(tr_combos), **arch)
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = args.img, 3 * args.img * args.img
    nparam = sum(q.numel() for q in model.parameters())
    ncond = (model.attr.numel() if args.cond == "factored" else model.cemb.weight.numel())
    print(f"[n19] UNet {nparam / 1e6:.1f}M params ({ncond} in the conditioning head, "
          f"vs {len(tr_combos) * 256} for a per-class table)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def package(ema, epochs_done):
        """The checkpoint dict, identical in shape whether written mid-run or at the end."""
        return dict(ema=ema, n_classes=len(tr_combos), arch=arch, remap=remap,
                    class_attrs=class_attrs, concepts={}, dataset="celeba", img=args.img,
                    hold_ids=sp["hold_ids"], hold_combos=sp["hold_combos"],
                    holdout_attr=args.holdout_attr, holdout_pair=pair,
                    min_count=args.min_count, n_holdout=args.n_holdout, seed=args.seed,
                    epochs_done=epochs_done, epochs_planned=args.epochs,
                    attr_names=list(ATTRS))            # so downstream uses the SAME set

    def checkpoint(ep, ema):
        """Write every `--save-every` epochs, so a wall-clock kill costs one interval, not the run.

        `train_ddpm` copies the non-float buffers into `ema` only when it returns, so a mid-run
        snapshot has to take them from the live model itself -- otherwise the partial checkpoint is
        missing exactly the tensors `load_state_dict` refuses to do without.  Written to a temporary
        name and renamed, so a kill DURING the write cannot leave a truncated file behind.
        """
        if not args.save_every or ep % args.save_every or ep == args.epochs:
            return
        snap = {k: v.detach().clone().cpu() for k, v in ema.items()}
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                snap[k] = v.detach().clone().cpu()
        tmp = args.out + ".partial"
        torch.save(package(snap, ep), tmp + ".tmp")
        os.replace(tmp + ".tmp", tmp)
        print(f"[n19] checkpoint at epoch {ep}/{args.epochs} -> {tmp}", flush=True)

    t0 = time.time()
    ema = train_ddpm(model, data["X8"][tr], y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     device=dev, seed=args.seed, tf32=True, on_epoch=checkpoint)
    mins = (time.time() - t0) / 60
    torch.save(package(ema, args.epochs), args.out)
    if os.path.exists(args.out + ".partial"):
        os.remove(args.out + ".partial")               # the finished run supersedes it
    print(f"[n19] trained in {mins:.1f} min -> {args.out}", flush=True)

    # ── evaluation: conditional accuracy against the REAL-DATA ceiling ───────
    model.load_state_dict(ema)
    field = NeuralScoreField(model, device=dev, batch=64)
    probes = build_probes(data, args.seed, 8 if args.quick else 30,
                          layout="chw" if args.chw else "hwc")
    from lifelong_ood.evals import probe_usable
    readable = [a for a in probes["_attrs"] if probe_usable(probes, a)]
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
