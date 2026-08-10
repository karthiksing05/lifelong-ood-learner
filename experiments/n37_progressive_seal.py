"""N37-brief -- can a model built ENTIRELY from sealed blocks be trained at all?

`docs/UNIFIED.md` argues that backbone training and lifelong absorption are the same operation --
allocate, fit, seal -- at different allocation sizes, with backbone training as the t=0 case where
routing finds nothing and allocation is maximal.  Before any of that is worth pursuing, one
empirical question has to be answered: **does a model made of a chain of sealed blocks learn?**

Three arms at MATCHED parameter budget and matched total epochs:

  `full`         ordinary training, all weights unfrozen.  The ceiling.
  `progressive`  frozen base + K sealed trunk blocks of rank r, trained IN SEQUENCE: allocate block
                 k, train only it, freeze it, allocate block k+1.  This is the unified regime.
  `one-block`    frozen base + ONE block of rank K*r, trained for the whole budget.  Same parameter
                 count and same frozen base as `progressive`, but never seals.

**`one-block` is what makes this a test rather than an observation.**  If `progressive` trails
`full`, that alone does not say why: it could be that low-rank updates over a frozen base are simply
a weak function class, which would be a fact about adapters and not about sealing at all.  Holding
budget and base fixed and removing only the sealing separates the two:

    progressive ~= one-block  <  full      the cost is LOW-RANK-OVER-FROZEN, not sealing
    progressive  <  one-block ~= full      SEALING is the cost, and the unified regime pays it
    progressive ~= one-block ~= full       neither hurts; the unified regime is free

The base is frozen at its RANDOM INITIALISATION on purpose.  The question is whether a model can be
*built* out of sealed blocks, not whether blocks can adapt an already-good model -- which the rest of
this repository has answered at length.  So this is deliberately the hardest version of the question,
and a poor absolute number for every arm is expected; what is being read is the ORDERING and the
gaps, at a budget where `full` reaches something non-trivial.

Reported: final training loss (the primary comparison -- identical objective across arms), and
conditional accuracy on the SEEN slots the model was trained on, via the standard probes.

    python experiments/n37_progressive_seal.py --blocks 4 --epochs-per-block 6
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                             # noqa: E402

from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                save_json, save_grid)
from lifelong_ood.neural import NeuralScoreField, UNet, to_data, train_ddpm   # noqa: E402
from lifelong_ood.neural.adapters import inject                               # noqa: E402


def block_params(wrapped, nid):
    k = str(nid)
    return [w.A[k] for w in wrapped if k in w.A] + [w.B[k] for w in wrapped if k in w.A]


def evaluate(model, arch, data, probes, slots, remap, args, dev, tag):
    """Seen-slot conditional accuracy, on the slots the model was actually trained on."""
    model.eval()
    field = NeuralScoreField(model, device=dev)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(slots, min(args.n_eval_slots, len(slots)), replace=False)
    J = []
    shots = []
    for s in picks:
        S = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                 branch=int(remap[int(s)]), clip=(-1, 1))).astype(np.float32)
        idx = np.where(data["slot"] == s)[0]
        d, f, b = (int(data[k][idx[0]]) for k in ("digit", "fg", "bg"))
        J.append(composition_accuracy(probes, S, d, f, b)["joint"])
        shots.append(S[:4])
    if shots:
        save_grid(np.concatenate(shots)[:32],
                  os.path.join(RESULTS, f"n37_{tag}{args.tag}.png"), ncol=8,
                  title=f"n37 {tag}: seen-slot samples")
    model.train()
    return float(np.mean(J)) if J else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, default=4, help="K sealed trunk blocks")
    p.add_argument("--epochs-per-block", type=int, default=6)
    p.add_argument("--rank", type=int, default=8, help="rank r of each progressive block")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-eval-slots", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=["full", "progressive", "one-block"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.blocks, args.epochs_per_block, args.n_per_slot = 2, 1, 12
        # base stays 32: the UNet uses GroupNorm(32, C), so every channel count must be a
        # multiple of 32 and a smaller base fails at construction rather than running fast
        args.base, args.steps, args.n_eval_slots, args.n_samples = 32, 15, 2, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    total_epochs = args.blocks * args.epochs_per_block

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    seen = data["seen"]
    slots = np.unique(data["slot"][seen])
    remap = {int(s): i for i, s in enumerate(slots)}
    X = data["X"][seen]
    y = np.array([remap[int(s)] for s in data["slot"][seen]], np.int64)
    arch = dict(base=args.base, img=32, mults=(1, 2, 2, 2), cond_mode="class")
    print(f"[n37] {len(X)} images, {len(slots)} seen slots, {args.blocks} blocks x "
          f"{args.epochs_per_block} epochs = {total_epochs} total\n", flush=True)

    rows = []
    for arm in args.arms:
        torch.manual_seed(args.seed)
        model = UNet(n_classes=len(slots), **arch).to(dev)
        t0 = time.time()

        if arm == "full":
            # NOT loading the EMA weights, deliberately.  The block arms below train in K separate
            # `train_ddpm` calls, each of which starts a FRESH EMA from the current state, so there
            # is no EMA trajectory comparable to a single 32-epoch run's.  Loading EMA here and live
            # weights there compares two different things: the first version of this experiment did
            # exactly that and reported `full` as the WORST arm (0.0963 against 0.0532), which is an
            # artifact of an EMA at decay 0.999 lagging ~1000 steps behind, not a finding.  Every arm
            # is now evaluated on the weights training actually left behind.
            train_ddpm(model, X, y, epochs=total_epochs, batch=args.batch, lr=args.lr,
                       device=dev, seed=args.seed)
            n_train = sum(p_.numel() for p_ in model.parameters())

        else:
            # freeze the base at its RANDOM init: every arm below learns only in blocks
            for p_ in model.parameters():
                p_.requires_grad_(False)
            state = {"path": []}
            ranks = ([args.rank] * args.blocks if arm == "progressive"
                     else [args.rank * args.blocks])
            wrapped = inject(model, ranks[0], state, targets=("conv", "linear"))
            n_train = 0
            for k, r in enumerate(ranks):
                for w in wrapped:                     # this block's rank, for this block only
                    w.rank = r
                    w.add_node(k, dev)
                state["path"] = list(range(k + 1))    # every earlier block stays ACTIVE, frozen
                params = block_params(wrapped, k)
                for p_ in params:
                    p_.requires_grad_(True)
                n_train += sum(p_.numel() for p_ in params)
                ep = args.epochs_per_block if arm == "progressive" else total_epochs
                print(f"[n37] {arm}: block {k} (rank {r}), {ep} epochs, "
                      f"{sum(p_.numel() for p_ in params) / 1e6:.2f}M trainable", flush=True)
                train_ddpm(model, X, y, epochs=ep, batch=args.batch, lr=args.lr,
                           device=dev, seed=args.seed + k, params=params)
                for p_ in params:                     # SEAL
                    p_.requires_grad_(False)

        wall = time.time() - t0
        # final loss under the identical objective, so the arms are comparable on one number
        loss = float(train_ddpm(model, X, y, epochs=0, batch=args.batch, lr=args.lr, device=dev,
                                seed=args.seed, params=None, eval_only=True))
        acc = evaluate(model, arch, data, probes, slots, remap, args, dev, arm)
        rows.append(dict(arm=arm, final_loss=loss, seen_joint=acc, trainable_params=int(n_train),
                         wall_seconds=float(wall), epochs=total_epochs))
        print(f"[n37] {arm:12s} loss {loss:.4f}  seen_joint {acc:.3f}  "
              f"{n_train / 1e6:.2f}M trainable  {wall / 60:.1f} min\n", flush=True)
        del model
        torch.cuda.empty_cache()

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n37_progressive_seal{args.tag}.json"), seed=args.seed)

    print(f"[n37] can a model be BUILT from sealed blocks?  ({total_epochs} epochs each)")
    print(f"    {'arm':14s}{'final loss':>12s}{'seen_joint':>12s}{'trainable':>12s}")
    for r in rows:
        print(f"    {r['arm']:14s}{r['final_loss']:12.4f}{r['seen_joint']:12.3f}"
              f"{r['trainable_params'] / 1e6:11.2f}M")
    g = {r["arm"]: r for r in rows}
    if {"full", "progressive", "one-block"} <= set(g):
        pr, ob, fu = g["progressive"], g["one-block"], g["full"]
        seal_cost = pr["final_loss"] - ob["final_loss"]
        rank_cost = ob["final_loss"] - fu["final_loss"]
        print(f"\n[n37] loss attributable to SEALING          : {seal_cost:+.4f}")
        print(f"[n37] loss attributable to low-rank-over-frozen: {rank_cost:+.4f}")
        if abs(seal_cost) < 0.25 * abs(rank_cost):
            print("[n37] => sealing is nearly free; the gap is the adapter constraint, and a "
                  "chain of sealed blocks trains about as well as one unsealed block")
        elif seal_cost > 0:
            print("[n37] => sealing carries a real cost on top of the adapter constraint; the "
                  "unified regime is trainable but pays for freezing as it goes")
        else:
            print("[n37] => sealing did not hurt (and may have helped, e.g. as regularization)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
