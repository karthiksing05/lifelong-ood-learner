"""N29 — score EVERY held-out composition, so the case-study figure can be chosen honestly.

The case-study figure showed three slots picked at random by `ood_queries(seed=0)`.  Two of the
three happened to be the same colour pair, and one of those was a poor case, which makes the figure
simultaneously unrepresentative and monotonous.  Picking nicer slots by hand would fix the look and
destroy the evidence, so the selection is made a measurement instead:

  1. run the method on **all 40** held-out compositions,
  2. report the full distribution -- that is the honest summary, and it goes in the caption,
  3. choose the figure's cases by a stated rule over that distribution.

ColorMNIST's held-out tier is exactly 4 colour pairs x 10 digits (`datasets.OOD_PAIRS`), so the rule
that spans the benchmark is **one case per colour pair**.  Within a pair the default is the
**median**-scoring digit, not the best: a figure of medians is what a reader should expect to
reproduce.  `--select best` and `--select worst` exist so the tails can be inspected too, and the
figure always says which rule produced it.

    python experiments/n29_slot_scan.py                # scan all 40, write the ranking
    python experiments/make_case_studies.py --select median-per-pair
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries                # noqa: E402

from lifelong_ood.data.datasets import (BG_NAMES, FG_NAMES, load_colormnist,       # noqa: E402
                                   slot_name)
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                       # noqa: E402


def scan(args, dev):
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(field0, C, slots, probes, data, seed=args.seed, steps=args.steps)
    print(f"[n29] gains {gains}", flush=True)

    qs = ood_queries(data, n_slots=None, seed=args.seed)          # ALL held-out slots
    print(f"[n29] scanning {len(qs)} held-out compositions", flush=True)
    rows = []
    for i, q in enumerate(qs):
        t0 = time.time()
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        L.attach_factorization(C, slot_factors(slots), gains=gains)
        xq = to_model(data["X"][q["query"]]).ravel()
        L.learn_concept_factored(xq, f"c{q['slot']}", n_pool=48, epochs=args.epochs,
                                 inv_steps=args.inv_steps, seed=args.seed,
                                 guidance=args.guidance)
        S = to_data(L.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                     seed=args.seed)).astype(np.float32)
        R = data["X"][q["rest"][:100]].astype(np.float32)
        a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
        pr = precision_recall(S, R)
        rows.append(dict(slot=int(q["slot"]), digit=int(q["d"]), fg=int(q["f"]), bg=int(q["b"]),
                         fg_name=FG_NAMES[int(q["f"])], bg_name=BG_NAMES[int(q["b"])],
                         name=slot_name(int(q["d"]), int(q["f"]), int(q["b"])),
                         joint=a["joint"], digit_acc=a["digit"], fg_acc=a["fg"], bg_acc=a["bg"],
                         recall=pr["recall"], f1=pr["f1"], vendi=vendi_score(S)))
        print(f"[n29] {i + 1}/{len(qs)} slot {q['slot']:3d} {rows[-1]['name']:34s} "
              f"joint {a['joint']:.2f} F1 {pr['f1']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        del field, L
        torch.cuda.empty_cache()
    return rows


def pick(rows, rule="median-per-pair"):
    """-> the slots the figure should show, under a stated rule."""
    pairs = {}
    for r in rows:
        pairs.setdefault((r["fg_name"], r["bg_name"]), []).append(r)
    out = []
    for k in sorted(pairs):
        g = sorted(pairs[k], key=lambda r: r["joint"])
        out.append(g[len(g) // 2] if rule == "median-per-pair" else
                   (g[-1] if rule == "best-per-pair" else g[0]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = scan(args, dev)
    j = np.array([r["joint"] for r in rows])
    f1 = np.array([r["f1"] for r in rows])
    summary = dict(n=len(rows), joint_mean=float(j.mean()), joint_median=float(np.median(j)),
                   joint_min=float(j.min()), joint_max=float(j.max()),
                   f1_mean=float(f1.mean()), f1_median=float(np.median(f1)))
    picks = {rule: [r["slot"] for r in pick(rows, rule)]
             for rule in ("median-per-pair", "best-per-pair", "worst-per-pair")}
    save_json(dict(rows=rows, summary=summary, picks=picks, config=vars(args)),
              os.path.join(RESULTS, f"n29_slot_scan{args.tag}.json"), seed=args.seed)

    print(f"\n[n29] all {len(rows)} held-out compositions")
    print(f"    joint: mean {summary['joint_mean']:.3f}  median {summary['joint_median']:.3f}  "
          f"range [{summary['joint_min']:.2f}, {summary['joint_max']:.2f}]")
    print(f"    F1   : mean {summary['f1_mean']:.3f}  median {summary['f1_median']:.3f}")
    print(f"\n    {'colour pair':28s}{'n':>4s}{'median joint':>14s}{'best':>7s}{'worst':>7s}")
    pairs = {}
    for r in rows:
        pairs.setdefault(f"{r['fg_name']} on {r['bg_name']}", []).append(r["joint"])
    for k in sorted(pairs):
        v = np.sort(pairs[k])
        print(f"    {k:28s}{len(v):4d}{np.median(v):14.3f}{v[-1]:7.2f}{v[0]:7.2f}")
    print("\n    figure slots by rule:")
    for rule, sl in picks.items():
        names = [next(r["name"] for r in rows if r["slot"] == s) for s in sl]
        print(f"      {rule:16s} {sl}")
        for n in names:
            print(f"        {n}")
    print(f"\n[n29] wrote {RESULTS}/n29_slot_scan{args.tag}.json")


if __name__ == "__main__":
    main()
