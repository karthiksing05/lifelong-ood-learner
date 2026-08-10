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

from lifelong_ood.datasets import load_colormnist, slot_id             # noqa: E402
from lifelong_ood.evals import fit_probes, save_grid, save_json        # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet, to_data, train_ddpm  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="defaults to results/cache/ddpm_<dataset>.pt -- do NOT default this "
                        "to a fixed name, a celeba run once overwrote the colormnist backbone")
    p.add_argument("--dataset", choices=["colormnist", "celeba"], default="colormnist",
                   help="celeba is the paper's second benchmark, and the one whose attributes "
                        "OVERLAP spatially -- ColorMNIST's separability is load-bearing in several "
                        "of this repo's conclusions and CelebA is how they get tested")
    p.add_argument("--img", type=int, default=None)
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
        from lifelong_ood.celeba import IMG_CELEBA, load_celeba
        img = args.img or IMG_CELEBA
        data = load_celeba(n_per_class=args.n_per_slot, img=img, seed=args.seed,
                           min_count=25, max_rows=args.max_rows)
        data["slot"] = data["cls"]
        globals()["_IMG"] = img
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
    arch = dict(base=args.base, img=img)
    if img >= 64:
        arch["mults"] = (1, 2, 2, 2)
    model = UNet(n_classes=len(uniq), **arch)
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
        return dict(ema=ema, n_classes=len(uniq), arch=arch,
                    remap=remap, concepts={}, n_per_slot=args.n_per_slot,
                    dataset=args.dataset, img=img, epochs_done=epochs_done)

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
                     device=dev, seed=args.seed, on_epoch=on_epoch if snaps else None)
    mins = (time.time() - t0) / 60
    torch.save(package(ema, args.epochs), args.out)
    print(f"[n1] trained in {mins:.1f} min -> {args.out}", flush=True)

    # ── sanity: does it generate valid compositions? ─────────────────────────
    model.load_state_dict(ema)
    field = NeuralScoreField(model, device=dev)
    n = 16 if args.quick else 64
    S = to_data(field.sample(n=n, steps=50, seed=args.seed, clip=(-1, 1)))
    save_grid(S[:32], os.path.join(RESULTS, f"n1_uncond_{args.dataset}.png"),
              title=f"DDPM unconditional samples ({args.dataset})")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(uniq, 8 if args.quick else 16, replace=False)
    hits, grids, probe_val = [], [], {}

    if args.dataset == "celeba":
        from lifelong_ood.celeba import class_name
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
