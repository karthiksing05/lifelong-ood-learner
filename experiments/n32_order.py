"""N32 -- does the order of the stream change the result?

Order sensitivity is one of the oldest known pathologies in continual learning: present the same
tasks in a different sequence and a protection-based method converges somewhere else, because every
update is applied on top of whatever the previous ones left behind.  It is usually reported as a
variance -- "accuracy varies by X across orderings" -- and treated as a nuisance.

For an insert-only bank it should not be a variance at all.  Nothing shared is ever written to, so a
concept's parameters depend on its own query and on which ancestors existed when it arrived, and
NOT on what came after it.  The prediction is therefore stronger than "low variance": for the
concepts that land on the same path, the generation should be **bit-identical** across permutations.

That is a prediction that can be falsified, which is why it is worth running rather than asserting.
Two things are measured across `--n-orders` random permutations of one concept set:

  `exact_frac`   fraction of concepts whose fingerprint is bit-identical to the reference ordering
  `max_drift`    the largest deviation seen, over all concepts and all orderings

and the same for `finetune-stream`, the arm that writes to shared weights, as the contrast.

A caveat stated up front, because it bounds the claim: hierarchical PLACEMENT does depend on order --
the tree descends against whatever nodes exist at insertion time, so a different order can give a
concept a different parent, and a concept whose path changed is not expected to be bit-identical.
Those are reported separately (`same_path_frac`) rather than being quietly folded into the failures.

    python experiments/n32_order.py --n-concepts 6 --n-orders 4
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                              # noqa: E402

from lifelong_ood.datasets import load_colormnist                            # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                         # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,          # noqa: E402
                                 to_model)
from lifelong_ood.neural.adapters import fingerprint                         # noqa: E402


def run_stream(ckpt, dev, args, qs, names, order, arm):
    """Consolidate the concepts in `order`; return {name: (fingerprint, path)}."""
    field = load_field(ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    for i in order:
        xq = to_model(qs[i]["X"]).ravel()
        if arm == "ours":
            L.learn_concept(xq, names[i], n_pool=32, epochs=args.epochs,
                            inv_steps=args.inv_steps, seed=args.seed)
        else:
            # the contrast: same recipe, but ancestors are NOT frozen, so every consolidation may
            # move parameters an earlier concept depends on
            L.bank.freeze_ancestors = False
            L.learn_concept(xq, names[i], n_pool=32, epochs=args.epochs,
                            inv_steps=args.inv_steps, seed=args.seed)
    out = {n: (fingerprint(field, n, seed=args.seed), tuple(L.bank.concepts[n]["path"]))
           for n in names}
    del field, L
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=6)
    p.add_argument("--n-orders", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=["ours", "unfrozen-ancestors"])
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_orders, args.n_per_slot = 4, 2, 20
        args.epochs, args.steps, args.inv_steps = 3, 20, 60
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    for q in qs:
        q["X"] = data["X"][q["query"]]
    names = [f"c{q['slot']}" for q in qs]
    K = len(names)

    rng = np.random.default_rng(args.seed)
    orders = [list(range(K))] + [list(rng.permutation(K)) for _ in range(args.n_orders - 1)]
    print(f"[n32] {K} concepts x {len(orders)} orderings x {len(args.arms)} arms\n", flush=True)

    rows = []
    for arm in args.arms:
        ref = run_stream(args.ckpt, dev, args, qs, names, orders[0], arm)
        ex, dr, sp = [], [], []
        for oi, order in enumerate(orders[1:], 1):
            got = run_stream(args.ckpt, dev, args, qs, names, order, arm)
            for n in names:
                same_path = got[n][1] == ref[n][1]
                d = float(np.max(np.abs(got[n][0] - ref[n][0])))
                ex.append(d == 0.0)
                dr.append(d)
                sp.append(same_path)
            print(f"[n32] {arm}: order {oi} -> {sum(1 for x in ex[-K:] if x)}/{K} bit-identical, "
                  f"max drift {max(dr[-K:]):.3e}, {sum(sp[-K:])}/{K} same path", flush=True)
        # among concepts whose PATH was unchanged, bit-identity is the actual prediction
        ex_sp = [e for e, s in zip(ex, sp) if s]
        rows.append(dict(arm=arm, exact_frac=float(np.mean(ex)), max_drift=float(max(dr)),
                         same_path_frac=float(np.mean(sp)),
                         exact_frac_same_path=float(np.mean(ex_sp)) if ex_sp else float("nan"),
                         n_comparisons=len(ex)))

    save_json(dict(rows=rows, orders=[list(map(int, o)) for o in orders], config=vars(args)),
              os.path.join(RESULTS, f"n32_order{args.tag}.json"), seed=args.seed)

    print(f"\n[n32] stream-order sensitivity, {len(orders) - 1} permutations vs the reference order")
    print(f"    {'arm':22s}{'exact':>8s}{'max drift':>12s}{'same path':>11s}{'exact|same path':>17s}")
    for r in rows:
        print(f"    {r['arm']:22s}{r['exact_frac']:8.3f}{r['max_drift']:12.3e}"
              f"{r['same_path_frac']:11.3f}{r['exact_frac_same_path']:17.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
