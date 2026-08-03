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
    p.add_argument("--out", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_per_slot, args.epochs, args.base = 10, 2, 32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[n1] device {dev}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if dev == 'cuda' else ''}", flush=True)

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    seen = data["seen"]
    Xs, slots = data["X"][seen], data["slot"][seen]
    uniq = np.unique(slots)
    remap = {int(s): i for i, s in enumerate(uniq)}          # 120 seen slots -> 0..119
    y = np.array([remap[int(s)] for s in slots])
    print(f"[n1] {len(Xs)} images, {len(uniq)} seen classes, {int((~seen).sum())} held-out images "
          f"never shown", flush=True)

    model = UNet(n_classes=len(uniq), base=args.base)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[n1] UNet {nparam / 1e6:.1f}M params, {args.epochs} epochs", flush=True)
    t0 = time.time()
    ema = train_ddpm(model, Xs, y, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     device=dev, seed=args.seed)
    mins = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(ema=ema, n_classes=len(uniq), arch=dict(base=args.base),
                    remap=remap, concepts={}, n_per_slot=args.n_per_slot), args.out)
    print(f"[n1] trained in {mins:.1f} min -> {args.out}", flush=True)

    # ── sanity: does it generate valid compositions? ─────────────────────────
    model.load_state_dict(ema)
    field = NeuralScoreField(model, device=dev)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    n = 16 if args.quick else 64
    S = to_data(field.sample(n=n, steps=50, seed=args.seed, clip=(-1, 1)))
    pd_, pf_, pb_ = (probes["digit"].predict(S.astype(np.float32)),
                     probes["fg"].predict(S.astype(np.float32)),
                     probes["bg"].predict(S.astype(np.float32)))
    sl = np.array([slot_id(d, f, b) for d, f, b in zip(pd_, pf_, pb_)])
    valid = float(np.isin(sl, uniq).mean())
    save_grid(S[:32], os.path.join(RESULTS, "n1_uncond_samples.png"),
              title="DDPM unconditional samples (neural backbone)")

    # conditional: ask for specific seen slots and check the probes agree
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(uniq, 8 if args.quick else 16, replace=False)
    hits, grids = [], []
    for s in picks:
        d, f, b = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        Sc = to_data(field.sample(n=8, steps=50, seed=args.seed, clip=(-1, 1),
                                  branch=remap[int(s)])).astype(np.float32)
        ok = ((probes["digit"].predict(Sc) == d) & (probes["fg"].predict(Sc) == f)
              & (probes["bg"].predict(Sc) == b))
        hits.append(float(ok.mean()))
        grids.extend(list(Sc[:4]))
    save_grid(grids, os.path.join(RESULTS, "n1_cond_samples.png"), ncol=4,
              title="DDPM conditional samples, 4 per requested seen slot")
    out = dict(train_minutes=mins, params=int(nparam), n_classes=len(uniq),
               epochs=args.epochs, n_per_slot=args.n_per_slot,
               uncond_valid_rate=valid, cond_joint_acc=float(np.mean(hits)),
               probe_val=dict(digit=probes["digit_val_acc"], fg=probes["fg_val_acc"],
                              bg=probes["bg_val_acc"]))
    save_json(out, os.path.join(RESULTS, "n1_train.json"))
    print(f"[n1] unconditional valid-composition rate {valid:.3f}")
    print(f"[n1] conditional joint accuracy on requested seen slots {np.mean(hits):.3f}")
    print(f"[n1] wrote {RESULTS}/n1_train.json")


if __name__ == "__main__":
    main()
