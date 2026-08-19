"""N1 — train the neural backbone: a class-conditional DDPM on the 120 SEEN ColorMNIST slots.

This is the paper's backbone setup at a size that trains in under an hour instead of six: the class
label is the compositional slot id remapped to 0..119, the embedding table reserves one extra row
for the CFG null token, and the 40 held-out slots are never seen in any form.

Everything afterwards (N2 composition, N3 lifelong) loads the EMA checkpoint this writes.  Sanity
numbers printed at the end: unconditional sample validity and per-class conditional accuracy, both
read by the same probes the analytic track uses, so the two backbones are directly comparable.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.datasets import load_colormnist, slot_id             # noqa: E402
from lifelong_ood.evals import fit_probes, save_grid, save_json        # noqa: E402
from lifelong_ood.data.factored import decode_slot                          # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet, to_data, train_ddpm  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--layout", choices=["permuted", "hwc"], default="permuted",
                   help="how a flat HWC row is handed to the UNet.  `permuted` is a plain\n"
                        "reshape(3,H,W), which is NOT an image: it makes the UNet's channels\n"
                        "out of horizontal bands and its width axis out of colour, so a 3x3\n"
                        "conv reads pixels ~22 columns apart and the convolutional prior is\n"
                        "gone.  Every backbone before 2026-08-11 used it, so it stays the\n"
                        "default and the checkpoint records it; `hwc` transposes properly.\n"
                        "See tests/diag_layout.py and figures/f20_layout.png.")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="defaults to results/cache/ddpm_<dataset>.pt -- do NOT default this "
                        "to a fixed name, a celeba run once overwrote the colormnist backbone")
    p.add_argument("--dataset", choices=["colormnist", "celeba", "celeba_grid",
                                         "shapes3d"],
                   default="colormnist",
                   help="celeba is the paper's second benchmark, and the one whose attributes "
                        "OVERLAP spatially -- ColorMNIST's separability is load-bearing in several "
                        "of this repo's conclusions and CelebA is how they get tested")
    p.add_argument("--img", type=int, default=None)
    p.add_argument("--value-stride", type=int, default=1,
                   help="shapes3d only: keep every k-th VALUE of factors with >=8 values. "
                        "Reduces the class count without dropping a factor -- 300 sparse classes "
                        "left conditioning unlearned (embedding pairwise cos 0.007).")
    p.add_argument("--n-holdout", type=int, default=100,
                   help="shapes3d only: how many COMBINATIONS to hold out")
    p.add_argument("--fix-nuisance", nargs="*", default=None,
                   help="shapes3d only: pin these nuisance factors to one value. wall_hue is a "
                        "full-field colour change, so leaving it free makes within-class variance "
                        "swamp the label -- the cause of three failed runs.")
    p.add_argument("--init-from", default=None,
                   help="start from this checkpoint's UNet weights and train CONDITIONING into it "
                        "-- the N47/CelebA recipe. A model that already renders the domain only has "
                        "to learn the label channel; asking one network to learn images AND "
                        "conditioning at once is what failed on shapes3d from scratch (its null gap "
                        "went NEGATIVE: the null token explained training images better than their "
                        "own class embedding, so the model did best ignoring the label).")
    p.add_argument("--p-drop", type=float, default=0.1,
                   help="CFG null-token dropout. Lower it when conditioning is what needs learning: "
                        "every dropped label is a step that teaches the unconditional path instead.")
    p.add_argument("--max-rows", type=int, default=25000,
                   help="celeba parquet rows to scan; more rows -> more images per class, "
                        "which is the first thing to raise when conditioning is weak (5o)")
    p.add_argument("--holdout-bg", type=int, default=None,
                   help="colormnist: hold out an entire BACKGROUND VALUE from training, not just "
                        "(fg,bg) pairs.  The backbone then has never seen that factor value at all, "
                        "which is what makes basis extension (n11) a real test rather than a "
                        "re-derivation of something the embedding space already encodes")
    p.add_argument("--holdout-fg", type=int, default=None,
                   help="hold out an entire FOREGROUND value, same idea as --holdout-bg")
    p.add_argument("--holdout-digits", type=int, nargs="*", default=None,
                   help="hold out entire DIGIT values.  Several held-out values give n12 a STREAM "
                        "of new vocabulary to absorb, which is what measuring cumulative drift "
                        "under weight reconsolidation needs")
    p.add_argument("--extra-holdout", type=int, default=0,
                   help="hold out this many ADDITIONAL slots beyond the benchmark's default, "
                        "chosen at random under --seed.  Drives the held-out-severity sweep: "
                        "compositional generalization is reported to degrade as coverage of the "
                        "combination grid thins, so a single split cannot be read as the result. "
                        "Every individual attribute VALUE is kept present somewhere in training, "
                        "so this thins combinations without ever removing a primitive.")
    p.add_argument("--all-slots", action="store_true",
                   help="train on ALL 160 slots including the held-out ones.  This is not a "
                        "method -- it is the NATIVE ORACLE for the parity evaluation (n23): the "
                        "backbone a slot would have had if it had never been held out.  The whole "
                        "claim 'a consolidated concept behaves as if it had been learned natively' "
                        "is unmeasurable without it.  Never use this checkpoint as a method's "
                        "backbone; its filename carries `_oracle` so a mix-up is visible.")
    p.add_argument("--cond", choices=["class", "factored"], default="class",
                   help="conditioning scheme.  `class` is one free embedding row per "
                        "compositional slot -- the shipped ColorMNIST backbone, whose "
                        "conditionals are ENTANGLED per combination.  `factored` sums one "
                        "learned embedding per (factor, value), so the conditioning is "
                        "compositional BY CONSTRUCTION.  Same data, same UNet, same recipe: "
                        "the only difference is the conditioning scheme, which is what §13 "
                        "could not isolate because it compared two different datasets.")
    p.add_argument("--snapshot-epochs", type=int, nargs="*", default=None,
                   help="also write <out>_ep<N>.pt at each of these epochs: a competence "
                        "ladder from ONE run, which is what N35 sweeps over")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_per_slot, args.epochs, args.base = 10, 2, 32
    if args.out is None:
        suffix = "_oracle" if args.all_slots else ""
        suffix += "_smoke" if args.quick else ""
        args.out = os.path.join(CACHE, f"ddpm_{args.dataset}{suffix}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[n1] device {dev}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if dev == 'cuda' else ''}", flush=True)

    if args.dataset == "celeba":
        from lifelong_ood.data.celeba import IMG_CELEBA, load_celeba
        img = args.img or IMG_CELEBA
        data = load_celeba(n_per_class=args.n_per_slot, img=img, seed=args.seed,
                           min_count=25, max_rows=args.max_rows)
        data["slot"] = data["cls"]
        globals()["_IMG"] = img
    elif args.dataset == "celeba_grid":
        # CelebA as a DENSE GRID with a DENSITY holdout -- the ColorMNIST setup, on faces.
        #
        # `--dataset celeba` uses `build_classes`, which keeps the 74 most frequent 14-attribute
        # combinations and admits a held-out one only if it differs from a seen combination on
        # exactly one attribute.  Almost none qualify, so the split lands at 51 seen / 2 held out,
        # and two held-out classes cannot carry a composition claim (N69 returned INCONCLUSIVE on
        # exactly that n).  More importantly, every CelebA result so far used a backbone FINE-TUNED
        # from `google/ddpm-celebahq-256`, so held-out faces were already in the pretrained density
        # and "held out" meant only "no embedding row" -- which is why mode ascent returns the query
        # there, and why `nearest-class` is so strong on Shapes3D (same pretraining story).
        #
        # This trains from scratch on the seen cells ONLY, so a held-out combination is absent from
        # the model the way it is on ColorMNIST.  The grid is hair(3) x Male(2) x Smiling(2); its
        # size is bounded by measurement, not preference -- only four attributes are readable at
        # 64px (Smiling +0.364, Male +0.266, Brown_Hair +0.131, Blond_Hair +0.112 over base rate;
        # everything else under +0.10 for a CNN probe AND a linear one), and the two hair colours
        # are mutually exclusive so they collapse into one categorical factor.
        from lifelong_ood.data.celeba import load_celeba_grid
        img = args.img or 64
        # `--n-holdout` defaults to 100 because it was added for shapes3d, which has 100 slots.
        # Passed through unclamped it held out EIGHT of eleven cells and left three seen classes,
        # since the loader holds out as many as it can while keeping every factor value present.
        # A grid of twelve wants three held out, so the shapes3d default is treated as "unset".
        n_hold = args.n_holdout if args.n_holdout <= 6 else 3
        data = load_celeba_grid(img=img, seed=args.seed, n_per_cell=args.n_per_slot,
                                n_holdout=n_hold, max_rows=args.max_rows)
        globals()["_IMG"] = img
        print(f"[n1] celeba_grid: {len(data['live'])} live cells over {data['names']} "
              f"sizes {data['sizes']}; held out {data['held']}; "
              f"{int(np.sum(data['seen']))} training images, "
              f"{int(np.sum(~np.asarray(data['seen'])))} never shown", flush=True)
    elif args.dataset == "shapes3d":
        # Third dataset, same recipe -- no new trainer, no new architecture. `load_shapes3d` already
        # emits factored.py's contract, and the only thing n1 needs from a dataset is X/slot/seen.
        # Composition factors are (floor_hue, object_hue, shape); wall_hue/scale/orientation are
        # nuisance and supply the within-composition variation that handwriting supplies on
        # ColorMNIST -- see lifelong_ood/shapes3d.py for why that split is required rather than
        # convenient.
        from lifelong_ood.data.shapes3d import load_shapes3d
        img = args.img or 64
        data = load_shapes3d(n_per_slot=args.n_per_slot, seed=args.seed, img=img,
                             value_stride=args.value_stride, n_holdout=args.n_holdout,
                             fix_nuisance=args.fix_nuisance)
        globals()["_IMG"] = img
        print(f"[n1] shapes3d: {len(np.unique(data['slot']))} compositions over "
              f"{data['names']} sizes {data['sizes']}, {len(data['X'])} images at {img}px",
              flush=True)
    else:
        data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    seen = data["seen"]
    if args.all_slots:
        seen = np.ones(len(data["slot"]), bool)
        data["seen"] = seen
        print(f"[n1] ORACLE: training on ALL {len(np.unique(data['slot']))} slots — this is the "
              f"native-parity reference, not a method", flush=True)
    if args.holdout_bg is not None and args.dataset == "colormnist":
        seen = seen & (data["bg"] != args.holdout_bg)
        print(f"[n1] holding out background value {args.holdout_bg} ENTIRELY: "
              f"{int((data['bg'] == args.holdout_bg).sum())} images, "
              f"{len(np.unique(data['slot'][data['bg'] == args.holdout_bg]))} slots never shown",
              flush=True)
        data["seen"] = seen
    if args.holdout_fg is not None and args.dataset == "colormnist":
        seen = seen & (data["fg"] != args.holdout_fg)
        print(f"[n1] holding out foreground value {args.holdout_fg} ENTIRELY", flush=True)
        data["seen"] = seen
    if args.holdout_digits and args.dataset == "colormnist":
        seen = seen & ~np.isin(data["digit"], np.asarray(args.holdout_digits))
        print(f"[n1] holding out digits {args.holdout_digits} ENTIRELY", flush=True)
        data["seen"] = seen
    if args.extra_holdout > 0:
        # thin the combination grid without ever deleting a primitive: a slot is only eligible to be
        # dropped if each of its three attribute values still appears in some OTHER slot that stays.
        # Otherwise the task stops being "compose known parts in a new arrangement" and silently
        # becomes "invent an unseen part", which is a different claim measured elsewhere (n11/n21).
        rng = np.random.default_rng(args.seed)
        cur = np.unique(data["slot"][seen])
        dropped = []
        for s in rng.permutation(cur):
            if len(dropped) >= args.extra_holdout:
                break
            keep = seen & (data["slot"] != s) & ~np.isin(data["slot"], dropped)
            ok = all((data[k][keep] == data[k][data["slot"] == s][0]).any()
                     for k in ("digit", "fg", "bg"))
            if ok:
                dropped.append(int(s))
        seen = seen & ~np.isin(data["slot"], dropped)
        data["seen"] = seen
        print(f"[n1] SEVERITY: {len(dropped)} extra slots held out "
              f"(every attribute value still present): {sorted(dropped)}", flush=True)

    Xs, slots = data["X"][seen], data["slot"][seen]
    uniq = np.unique(slots)
    remap = {int(s): i for i, s in enumerate(uniq)}          # 120 seen slots -> 0..119
    y = np.array([remap[int(s)] for s in slots])
    print(f"[n1] {len(Xs)} images, {len(uniq)} seen classes, {int((~seen).sum())} held-out images "
          f"never shown", flush=True)

    img = data.get("img", 32)
    # deeper stack at higher resolution so the spatial size still reaches the attention levels
    arch = dict(base=args.base, img=img, cond_mode=args.cond)
    if img >= 64:
        arch["mults"] = (1, 2, 2, 2)
    if args.cond == "factored":
        # one embedding per (factor, value), summed -- so the conditioning is compositional by
        # construction rather than one entangled row per combination.  `class_attrs` maps each
        # trained slot to its (digit, fg, bg) values; n_attr_vals is the widest factor.
        from lifelong_ood.neural.idealwords import slot_factors
        ca = slot_factors(uniq)
        arch["class_attrs"] = ca
        arch["n_attr_vals"] = int(ca.max()) + 1
        print(f"[n1] FACTORED conditioning: {ca.shape[1]} factors, "
              f"{arch['n_attr_vals']} values wide -> "
              f"{ca.shape[1] * arch['n_attr_vals']} embedding rows instead of {len(uniq)}",
              flush=True)
    model = UNet(n_classes=len(uniq), **arch)
    if args.init_from:
        # Load the DOMAIN and relearn the LABEL. Everything except the class table is copied; the
        # table is left fresh because its old rows correspond to a different class partition and
        # carrying them over would start conditioning from a wrong answer rather than from none.
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = src.get("ema", src)
        keep = {k: v for k, v in sd.items()
                if not k.startswith("cemb.") and k in model.state_dict()
                and model.state_dict()[k].shape == v.shape}
        missing = model.load_state_dict(keep, strict=False)
        print(f"[n1] init-from {os.path.basename(args.init_from)}: copied {len(keep)} tensors, "
              f"class table left fresh ({len(missing.missing_keys)} keys not copied)", flush=True)
    if img != 32:
        # CelebA runs at a larger resolution; the score field and UNet are resolution-agnostic
        # except for these two module-level constants, which the loaders set.
        import lifelong_ood.neural.ddpm as _d
        _d.IMG, _d.DIM = img, 3 * img * img
        import lifelong_ood.neural.unet as _u
        _u.IMG = img
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[n1] UNet {nparam / 1e6:.1f}M params, {args.epochs} epochs", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def package(ema, epochs_done):
        # `split` records EXACTLY how the training data was built.  Without it an evaluation has to
        # guess, and guessing wrong is silent: a diag scored two CelebA backbones with labels from a
        # different loader (`load_celeba_comp`, per-combination) than training used (`load_celeba`,
        # per-class), so both arms were mis-conditioned and the comparison between them meant
        # nothing.  The only tell was a held-out loss ten times the training loss.
        return dict(ema=ema, n_classes=len(uniq), arch=arch,
                    remap=remap, concepts={}, n_per_slot=args.n_per_slot,
                    dataset=args.dataset, img=img, epochs_done=epochs_done,
                    layout=args.layout,
                    split=dict(loader=("load_celeba_grid" if args.dataset == "celeba_grid"
                                       else "load_celeba" if args.dataset == "celeba"
                                       else "load_colormnist"),
                               n_per_slot=args.n_per_slot, img=img, seed=args.seed,
                               min_count=(25 if args.dataset == "celeba" else None),
                               # RECORDED, because a reader that re-derives the split has to
                               # reproduce the HOLDOUT too. Without it `n73` fell back to a
                               # hardcoded 3 and would have scored an `n_holdout=5` backbone
                               # against the wrong held-out cells -- silently, since the arms all
                               # still run.  Readers additionally assert seen == sorted(remap).
                               n_holdout=(n_hold if args.dataset == "celeba_grid" else None),
                               max_rows=getattr(args, "max_rows", None)))

    # A LADDER OF COMPETENCE FROM ONE RUN.  N35 asks how much backbone training the method needs,
    # which naively means training one backbone per rung -- but every rung is a PREFIX of the longest
    # run, so snapshotting the EMA mid-training gives the whole ladder for the price of the top rung
    # and removes the between-run backbone variance (RESULTS 5p measured up to 0.07 F1 between two
    # runs of the SAME command) from the comparison entirely.  The rungs then differ ONLY in training
    # length, which is the variable under study.
    snaps = sorted({int(e) for e in (args.snapshot_epochs or []) if 0 < int(e) < args.epochs})

    def on_epoch(ep, ema_live):
        if ep not in snaps:
            return
        snap = {k: v.detach().clone().cpu() for k, v in ema_live.items()}
        for k, v in model.state_dict().items():          # non-float buffers: folded in only on
            if not v.dtype.is_floating_point:            # return, so take them from the live model
                snap[k] = v.detach().clone().cpu()
        path = f"{os.path.splitext(args.out)[0]}_ep{ep}.pt"
        torch.save(package(snap, ep), path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"[n1] snapshot at epoch {ep} -> {path}", flush=True)

    t0 = time.time()
    ema = train_ddpm(model, Xs, y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     device=dev, seed=args.seed, on_epoch=on_epoch if snaps else None,
                     layout=args.layout, p_drop=args.p_drop)
    mins = (time.time() - t0) / 60
    torch.save(package(ema, args.epochs), args.out)
    print(f"[n1] trained in {mins:.1f} min -> {args.out}", flush=True)

    # ── sanity: does it generate valid compositions? ─────────────────────────
    model.load_state_dict(ema)
    # LAYOUT MUST MATCH TRAINING.  `train_ddpm` is given `args.layout`; building the eval field
    # without it silently falls back to `permuted`, so a model trained with `--layout hwc` is
    # sampled through a different pixel ordering.  The samples then come out as TILED MOSAICS of
    # miniature faces and every probe reads them as noise -- the celeba_grid run scored joint 0.111,
    # exactly 1/9 chance over its nine seen classes, and the checkpoint was fine.  This is the same
    # fault class as RESULTS 30d ("a renderer reading HWC rows as CHW") reappearing at the eval
    # boundary rather than the render one.
    field = NeuralScoreField(model, device=dev, layout=args.layout)
    n = 16 if args.quick else 64
    S = to_data(field.sample(n=n, steps=50, seed=args.seed, clip=(-1, 1)))
    save_grid(S[:32], os.path.join(RESULTS, f"n1_uncond_{args.dataset}.png"),
              title=f"DDPM unconditional samples ({args.dataset})")
    rng = np.random.default_rng(args.seed)
    # `uniq` is the SEEN class count, which is 120 on ColorMNIST but nine on the CelebA grid --
    # asking for 16 without replacement then raises rather than sampling what exists.
    n_pick = min(len(uniq), 8 if args.quick else 16)
    picks = rng.choice(uniq, n_pick, replace=False)
    hits, grids, probe_val = [], [], {}

    if args.dataset == "celeba":
        from lifelong_ood.data.celeba import class_name
        from lifelong_ood.evals import attr_accuracy, fit_attr_probes
        probes = fit_attr_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
        probe_val = {a: probes[f"{a}_val_acc"] for a in probes["_attrs"]}
        usable = [a for a, v in probe_val.items() if v >= 0.65]
        print(f"[n1] attribute probes readable at >=0.65: {len(usable)}/{len(probe_val)} "
              f"({', '.join(usable[:6])}{'...' if len(usable) > 6 else ''})", flush=True)
        for s in picks:
            combo = data["combos"][int(s)]
            Sc = to_data(field.sample(n=8, steps=50, seed=args.seed, clip=(-1, 1),
                                      branch=remap[int(s)])).astype(np.float32)
            a = attr_accuracy(probes, Sc, combo)
            hits.append(a["joint"] if a["joint"] == a["joint"] else 0.0)
            grids.extend(list(Sc[:4]))
        valid = float("nan")
    elif "digit" not in data:
        # Any dataset satisfying factored.py's contract but not ColorMNIST's key names -- shapes3d
        # today.  `fit_probes` is hardcoded to digit/fg/bg and raised KeyError here AFTER a 36-minute
        # training run had already written a perfectly good checkpoint, which is the worst place to
        # fail.  The generic probes read data["factors"]/["names"]/["sizes"] instead.
        from lifelong_ood.evals import factor_accuracy, fit_factor_probes
        probes = fit_factor_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
        probe_val = {nm: probes[f"{nm}_val"] for nm in probes["_names"]}
        valid = float("nan")          # "is this a real slot" needs a slot decoder; not defined here
        for s in picks:
            tgt = decode_slot(np.array([int(s)]), data["sizes"])[0]
            Sc = to_data(field.sample(n=8, steps=50, seed=args.seed, clip=(-1, 1),
                                      branch=remap[int(s)])).astype(np.float32)
            a = factor_accuracy(probes, Sc, tgt)
            hits.append(a["joint"] if a["joint"] == a["joint"] else 0.0)
            grids.extend(list(Sc[:4]))
    else:
        probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
        probe_val = dict(digit=probes["digit_val_acc"], fg=probes["fg_val_acc"],
                         bg=probes["bg_val_acc"])
        pd_, pf_, pb_ = (probes["digit"].predict(S.astype(np.float32)),
                         probes["fg"].predict(S.astype(np.float32)),
                         probes["bg"].predict(S.astype(np.float32)))
        sl = np.array([slot_id(d, f, b) for d, f, b in zip(pd_, pf_, pb_)])
        valid = float(np.isin(sl, uniq).mean())
        for s in picks:
            d, f, b = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
            Sc = to_data(field.sample(n=8, steps=50, seed=args.seed, clip=(-1, 1),
                                      branch=remap[int(s)])).astype(np.float32)
            ok = ((probes["digit"].predict(Sc) == d) & (probes["fg"].predict(Sc) == f)
                  & (probes["bg"].predict(Sc) == b))
            hits.append(float(ok.mean()))
            grids.extend(list(Sc[:4]))
    save_grid(grids, os.path.join(RESULTS, f"n1_cond_{args.dataset}.png"), ncol=4,
              title=f"DDPM conditional samples, 4 per requested seen class ({args.dataset})")
    out = dict(train_minutes=mins, params=int(nparam), n_classes=len(uniq), dataset=args.dataset,
               epochs=args.epochs, n_per_slot=args.n_per_slot, img=img,
               uncond_valid_rate=valid, cond_joint_acc=float(np.mean(hits)), probe_val=probe_val)
    save_json(out, os.path.join(RESULTS, "n1_train.json"))
    print(f"[n1] unconditional valid-composition rate {valid:.3f}")
    print(f"[n1] conditional joint accuracy on requested seen slots {np.mean(hits):.3f}")
    print(f"[n1] wrote {RESULTS}/n1_train.json")


if __name__ == "__main__":
    main()
