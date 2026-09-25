"""C2: does selecting concepts ITERATIVELY beat picking K from a pool grown once?

The paper's Eq. 8-9 grows a candidate pool with mode ascent and then greedily takes K.
`candidates_ascent`'s own docstring names the limitation: "submodular selection can only choose from
the pool ascent found, and a thin pool gives a bad choice", which is visible in the case studies where
selected prototypes are often unrelated to the query.  Their fix is a 20x bigger pool (Table 6 grows
1920 chains).  `compose.iterative_teacher` interleaves instead: pick one, look at what the partial
teacher still fails to explain, grow the next chains from there.

**The comparison must be budget-matched or it measures nothing.**  K rounds of n_starts/K chains cost
what one round of n_starts costs, and `iterative_teacher(match_budget=True)` splits it that way.  Any
gain is then from the DIRECTION of search, not the amount.  Four arms:

    one-shot          the paper's route: one pool of `n_starts`, greedy top-K       (the baseline)
    iter-residual     regrow from the query with explained dimensions pulled to the teacher
    iter-teacher      regrow from the partial teacher's mean
    iter-query        regrow from the query every round -- the CONTROL that separates "iterating
                      helps" from "independent restarts help", since it iterates the ascent without
                      conditioning on the selection

Scored the way every composition table in this repository is: probe accuracy of the composition the
teacher locates, plus coverage against real held-out images.  Both columns, because accuracy alone
would prefer whatever sits nearest the query (5e-quater).

    python tests/diag_iterative_select.py --n-cases 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from lifelong_ood.compose import (PAPER_POE_COLORMNIST as PAPER_POE,  # noqa: E402
                                  candidates_ascent, iterative_teacher, poe_teacher)
from lifelong_ood.data.datasets import load_colormnist                        # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,        # noqa: E402
                                precision_recall)
from lifelong_ood.neural import NeuralLifelongLearner                    # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data, to_model       # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--n-starts", type=int, default=24, help="TOTAL ascent chains, both arms")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "diag_iterative_select.json"))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    field = load_field(args.ckpt, device=dev)
    trained = set(int(s) for s in ck["remap"])
    held = [int(s) for s in np.unique(data["slot"]) if int(s) not in trained]
    rng = np.random.default_rng(args.seed)
    cases = list(rng.permutation(held)[:args.n_cases])
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    print(f"[iter] {len(cases)} held-out compositions, budget {args.n_starts} chains for EVERY arm, "
          f"K={args.K}", flush=True)

    ak = dict(iters=args.iters, probes=4)
    rows = []
    for ci, slot in enumerate(cases):
        idx = np.where(data["slot"] == int(slot))[0]
        xq = to_model(data["X"][int(idx[0])]).ravel()
        R = np.stack([to_model(data["X"][i]).ravel() for i in idx[1:1 + 24]])
        d_, f_, b_ = int(slot) // 16, (int(slot) % 16) // 4, int(slot) % 4

        arms = {}
        # baseline: one pool, greedy top-K -- the paper's route
        M, V, meta = candidates_ascent(field, xq, n_starts=args.n_starts, seed=args.seed, **ak)
        arms["one-shot"] = poe_teacher(xq, M, V, K=args.K)
        arms["one-shot"]["pool"] = len(M)
        for mode in ("residual", "teacher", "query"):
            arms[f"iter-{mode}"] = iterative_teacher(
                field, xq, K=args.K, n_starts=args.n_starts, seed=args.seed,
                match_budget=True, regrow_from=mode, **ak)

        for name, t in arms.items():
            S = to_data(L.sample_teacher(t, n=args.n_samples, seed=args.seed, **PAPER_POE))
            a = composition_accuracy(probes, S, d_, f_, b_)
            pr = precision_recall(S, R)
            rows.append(dict(slot=int(slot), arm=name, joint=a["joint"], digit=a["digit"],
                             fg=a["fg"], bg=a["bg"], recall=pr["recall"],
                             precision=pr["precision"], pool=int(t.get("pool", 0)),
                             per_round=int(t.get("per_round", 0))))
            print(f"[iter] {ci + 1}/{len(cases)} slot {slot:3d} {name:14s} "
                  f"joint {a['joint']:.3f} digit {a['digit']:.2f} fg {a['fg']:.2f} "
                  f"bg {a['bg']:.2f} | recall {pr['recall']:.3f} pool {t.get('pool', 0)}",
                  flush=True)

    print("\n[iter] " + "=" * 76)
    hdr = (f"{'arm':>16} {'joint':>8} {'digit':>7} {'fg':>7} {'bg':>7} {'recall':>8} "
           f"{'precision':>10} {'pool':>6}")
    print(hdr + "\n" + "-" * len(hdr))
    summary = []
    for name in ("one-shot", "iter-residual", "iter-teacher", "iter-query"):
        R_ = [r for r in rows if r["arm"] == name]
        if not R_:
            continue
        m = {k: float(np.mean([r[k] for r in R_])) for k in
             ("joint", "digit", "fg", "bg", "recall", "precision", "pool")}
        se = float(np.std([r["joint"] for r in R_]) / max(len(R_) ** 0.5, 1))
        summary.append(dict(arm=name, n=len(R_), se_joint=se, **m))
        print(f"{name:>16} {m['joint']:>8.3f} {m['digit']:>7.3f} {m['fg']:>7.3f} {m['bg']:>7.3f} "
              f"{m['recall']:>8.3f} {m['precision']:>10.3f} {m['pool']:>6.0f}")

    base = next((s for s in summary if s["arm"] == "one-shot"), None)
    if base:
        print(f"\n[iter] one-shot joint {base['joint']:.3f} ± {base['se_joint']:.3f} (s.e.)")
        for s in summary:
            if s["arm"] == "one-shot":
                continue
            d = s["joint"] - base["joint"]
            pooled = (s["se_joint"] ** 2 + base["se_joint"] ** 2) ** 0.5
            print(f"[iter]   {s['arm']:14s} {d:+.3f}  ({d / pooled:+.2f} pooled s.e.)"
                  + ("   <-- not distinguishable" if abs(d) < 2 * pooled else ""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(rows=rows, summary=summary, args=vars(args)),
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"[iter] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
