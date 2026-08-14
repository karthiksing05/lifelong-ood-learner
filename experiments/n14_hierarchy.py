"""N14 — does the Cobweb-style hierarchy earn its keep, in the regime where it should?

The hierarchy is the part of this repo with the least evidence behind it.  N3 measures transfer at
+0.0013 (later concepts marginally MORE expensive, not less) and N5 has `independent` -- one private
adapter per concept, nothing shared -- tying the hierarchical arm at 0.806 vs 0.816 with fewer
parameters.  On that evidence the tree is costing parameters and buying nothing.

But N3 and N5 both stream concepts in RANDOM order over unrelated compositions, which is the
condition least favourable to sharing.  A hierarchy should pay when concepts genuinely cluster and
arrive so that a later one can reuse an earlier one's ancestor.  This gives it that:

  `clustered`  concepts arrive in families that share two of three factors, families back to back,
               so each new concept lands under a populated ancestor
  `shuffled`   the SAME concepts in random order -- the N3/N5 condition
  `flat`       no hierarchy at all: every concept private (parameter isolation)

Measured per arm: accuracy, bit-identical retention, adapter parameters, and the two numbers that
say whether sharing is real -- first-epoch distillation loss (does a related ancestor give a better
starting point?) and final loss (does it end cheaper?).  If `clustered` does not beat `shuffled` on
those, the hierarchy does not help even when handed the case designed for it.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                          # noqa: E402

from lifelong_ood.data.datasets import load_colormnist, slot_id                 # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes, save_json  # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, fingerprint, load_field,  # noqa: E402
                                 to_data, to_model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-families", type=int, default=4)
    p.add_argument("--per-family", type=int, default=3)
    p.add_argument("--n-train", type=int, default=48)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["clustered", "shuffled", "flat"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_families, args.per_family, args.n_train = 2, 2, 16
        args.n_samples, args.n_per_slot, args.epochs, args.steps = 8, 20, 4, 10
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    rng = np.random.default_rng(args.seed)

    # families: fix (fg, bg), vary digit -- concepts inside a family are genuinely close, which is
    # the structure a hierarchy is supposed to exploit
    held = [(f, b) for f in range(4) for b in range(4)]
    rng.shuffle(held)
    fams = []
    for (f, b) in held[:args.n_families]:
        digits = rng.choice(10, args.per_family, replace=False)
        fam = []
        for d in digits:
            idx = np.where((data["digit"] == d) & (data["fg"] == f) & (data["bg"] == b))[0]
            if len(idx) >= args.n_train:
                fam.append(dict(name=f"c{d}_{f}_{b}", d=int(d), f=int(f), b=int(b),
                                X=to_model(data["X"][idx[:args.n_train]])))
        if fam:
            fams.append(fam)
    clustered = [c for fam in fams for c in fam]                 # families back to back
    shuffled = list(clustered)
    rng.shuffle(shuffled)
    print(f"[n14] {len(fams)} families x ~{args.per_family} concepts = {len(clustered)} total, "
          f"device {dev}", flush=True)

    rows = []
    for arm in args.arms:
        order = shuffled if arm == "shuffled" else clustered
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev,
                                  flat=(arm == "flat"))
        marks, learned, firsts, finals, depths = {}, [], [], [], []
        t0 = time.time()
        for c in order:
            X = c["X"].reshape(len(c["X"]), -1)
            L.consolidate(X, c["name"], mu=X.mean(0), epochs=args.epochs)
            firsts.append(L.last_curve["first"]); finals.append(L.last_curve["final"])
            depths.append(L.last_curve["depth"])
            marks[c["name"]] = fingerprint(field, c["name"], seed=args.seed)
            learned.append(c)
        accs = []
        for i, c in enumerate(learned):
            S = to_data(L.sample_concept(c["name"], n=args.n_samples,
                                         seed=args.seed + 500 + i)).astype(np.float32)
            accs.append(composition_accuracy(probes, S, c["d"], c["f"], c["b"])["joint"])
        exact = float(np.mean([float(np.array_equal(fingerprint(field, c["name"], seed=args.seed),
                                                    marks[c["name"]])) for c in learned]))
        # the transfer question: do LATER concepts start and end cheaper than earlier ones?
        h = len(firsts) // 2
        early_first, late_first = float(np.mean(firsts[:h])), float(np.mean(firsts[h:]))
        early_final, late_final = float(np.mean(finals[:h])), float(np.mean(finals[h:]))
        shared = L.bank.summary().get("shared_ancestors", {})   # method is summary(), not record()
        rows.append(dict(arm=arm, acc=float(np.mean(accs)), retention_exact=exact,
                         params=L.bank.n_params(), mean_depth=float(np.mean(depths)),
                         first_early=early_first, first_late=late_first,
                         final_early=early_final, final_late=late_final,
                         transfer_first=late_first - early_first,
                         transfer_final=late_final - early_final,
                         n_shared=len([k for k, v in shared.items() if v > 1]),
                         wall_seconds=time.time() - t0))
        print(f"[n14] -> {arm}: acc {np.mean(accs):.3f}, exact {exact:.2f}, "
              f"{L.bank.n_params() / 1e6:.2f}M, depth {np.mean(depths):.1f}, "
              f"first {early_first:.4f}->{late_first:.4f}, final {early_final:.4f}->"
              f"{late_final:.4f}", flush=True)

    save_json(dict(rows=rows, config=vars(args)), os.path.join(RESULTS, "n14_hierarchy.json"),
              seed=args.seed)
    print(f"\n[n14] does the hierarchy help when concepts genuinely cluster?")
    print(f"    {'arm':12s}{'acc':>7s}{'exact':>7s}{'params':>9s}{'depth':>7s}"
          f"{'first e->l':>22s}{'final e->l':>22s}")
    for r in rows:
        print(f"    {r['arm']:12s}{r['acc']:7.3f}{r['retention_exact']:7.2f}"
              f"{r['params'] / 1e6:8.2f}M{r['mean_depth']:7.1f}"
              f"{r['first_early']:11.4f}->{r['first_late']:9.4f}"
              f"{r['final_early']:11.4f}->{r['final_late']:9.4f}")
    print("\n  transfer is late-minus-early.  NEGATIVE means later concepts were cheaper, i.e. the")
    print("  ancestor gave them something.  If clustered is not more negative than shuffled, the")
    print("  hierarchy is not exploiting the structure it was built for.")
    for r in rows:
        print(f"    {r['arm']:12s} transfer first {r['transfer_first']:+.4f}  "
              f"final {r['transfer_final']:+.4f}  shared nodes {r['n_shared']}")
    print(f"[n14] wrote {RESULTS}/n14_hierarchy.json")


if __name__ == "__main__":
    main()
