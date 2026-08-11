"""N40 -- how cheaply can a lifelong learner absorb a long stream, and what does it cost per concept?

The goal this repository is actually for is a lifelong learner whose parameter count does not blow up
as it learns. Three levers have been measured separately and never assembled:

  routing   an ARRANGEMENT extends the shared factor basis for 1024 bytes; only a genuinely new PART
            costs a block.  538x (§6c)
  rank      a block at rank 1 is 69K against rank 8's 551K, at ACC 0.840 +/- 0.009 vs 0.842 +/- 0.026
            and bit-identity 1.000 at every rank (§24d)
  placement flat costs ~20% less than hierarchical, which buys nothing downstream (§24d, §24j)

Combined they predict **75K per concept against the shipped 735K**.  This runs the assembled
configuration over a long stream and reports the growth curve against the alternatives, so the claim
is a measured curve rather than three numbers multiplied together.

Arms, all absorbing the SAME stream with the same guarantee:

  `per-instance`   one full rank-8 block per query -- the pile a kept version of the companion
                   paper's protocol produces.  The thing to beat.
  `shipped`        this repository's default: hierarchical placement, rank 8
  `flat-r8`        flat placement, rank 8            -- isolates placement
  `flat-r1`        flat placement, rank 1            -- isolates rank
  `flat-r1-routed` flat, rank 1, and arrivals the router calls ARRANGEMENTS cost 1024 bytes instead
                   of a block -- the assembled configuration

Reported after every arrival: cumulative parameters, and at the end the quality of what was absorbed,
so "cheap" cannot be bought with "broken".  Preservation is re-certified per arm: a configuration
that saves parameters by giving up bit-identity has not saved anything this project wants.

    python experiments/n40_budget.py --n-concepts 24
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries                # noqa: E402

from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.adapters import fingerprint                          # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                       # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
BASIS_BYTES = 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=24)
    p.add_argument("--arrangement-rate", type=float, default=0.5,
                   help="fraction of arrivals the router calls ARRANGEMENTS.  Default 0.5 is "
                        "deliberately conservative against n18's measured rate; the curve is "
                        "reported against this rate rather than assuming the best case.")
    p.add_argument("--n-pool", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_pool, args.n_per_slot = 4, 12, 20
        args.epochs, args.steps, args.n_samples = 3, 20, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    f0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(f0, C, slots, probes, data, seed=args.seed, steps=args.steps,
                           n=8 if args.quick else 16)
    del f0
    torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    is_arr = rng.random(len(qs)) < args.arrangement_rate      # which arrivals route free
    arms = [("per-instance", dict(flat=True, rank=8, route=False)),
            ("shipped", dict(flat=False, rank=8, route=False)),
            ("flat-r8", dict(flat=True, rank=8, route=False)),
            ("flat-r1", dict(flat=True, rank=1, route=False)),
            ("flat-r1-routed", dict(flat=True, rank=1, route=True))]

    rows, curves = [], {}
    for name, cfg in arms:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, rank=cfg["rank"], steps=args.steps, seed=args.seed,
                                  device=dev, flat=cfg["flat"])
        L.attach_factorization(C, slot_factors(slots), gains=gains)
        curve, fps, learned = [], {}, []
        extra_bytes = 0
        for i, q in enumerate(qs):
            if cfg["route"] and is_arr[i]:
                # an ARRANGEMENT: the basis is extended, no block is allocated.  Nothing to train
                # and nothing to certify beyond what is already stored.
                extra_bytes += BASIS_BYTES
            else:
                nm = f"c{q['slot']}"
                xq = to_model(data["X"][q["query"]]).ravel()
                L.learn_concept_factored(xq, nm, n_pool=args.n_pool, epochs=args.epochs,
                                         inv_steps=60 if args.quick else 300, seed=args.seed)
                learned.append((nm, q))
                fps[nm] = fingerprint(field, nm, seed=args.seed)
            curve.append(dict(i=i + 1, params=int(L.bank.n_params()),
                              bytes=int(L.bank.n_params() * 4 + extra_bytes)))
        # preservation: every concept still bit-identical at the end of the stream
        drift = max((float(np.max(np.abs(fingerprint(field, n, seed=args.seed) - f)))
                     for n, f in fps.items()), default=0.0)
        # quality: what did the last few absorbed concepts actually generate?
        accs, f1s = [], []
        for nm, q in learned[-4:]:
            S = to_data(L.sample_concept(nm, n=args.n_samples, steps=args.steps,
                                         seed=args.seed)).astype(np.float32)
            R = data["X"][q["rest"][:100]].astype(np.float32)
            accs.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"])
            f1s.append(precision_recall(S, R)["f1"])
        per = curve[-1]["bytes"] / len(qs)
        rows.append(dict(arm=name, **cfg, final_params=curve[-1]["params"],
                         final_bytes=curve[-1]["bytes"], bytes_per_arrival=per,
                         n_blocks=len(learned), retention_exact=float(drift == 0.0),
                         max_drift=drift, joint=float(np.mean(accs)) if accs else float("nan"),
                         f1=float(np.mean(f1s)) if f1s else float("nan")))
        curves[name] = curve
        print(f"[n40] {name:16s} {curve[-1]['params'] / 1e6:6.2f}M params  "
              f"{per / 1024:8.1f} KB/arrival  blocks {len(learned):2d}/{len(qs)}  "
              f"bit-identical {drift == 0.0}  joint {rows[-1]['joint']:.3f} "
              f"f1 {rows[-1]['f1']:.3f}", flush=True)
        del field, L
        torch.cuda.empty_cache()

    save_json(dict(rows=rows, curves=curves, is_arrangement=[bool(x) for x in is_arr],
                   config=vars(args)),
              os.path.join(RESULTS, f"n40_budget{args.tag}.json"), seed=args.seed)

    base = next(r for r in rows if r["arm"] == "per-instance")
    print(f"\n[n40] cost of a {len(qs)}-concept stream "
          f"(arrangement rate {args.arrangement_rate:.0%})")
    print(f"    {'arm':17s}{'params':>10s}{'KB/arrival':>12s}{'vs pile':>9s}"
          f"{'bit-ident':>11s}{'joint':>8s}{'F1':>7s}")
    for r in rows:
        print(f"    {r['arm']:17s}{r['final_params'] / 1e6:9.2f}M{r['bytes_per_arrival'] / 1024:12.1f}"
              f"{base['final_params'] / max(r['final_params'], 1):8.1f}x{r['retention_exact']:11.3f}"
              f"{r['joint']:8.3f}{r['f1']:7.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    cols = dict(zip([a for a, _ in arms],
                    ["#B03A2E", "#8A6D1F", "#7C7C7C", "#2C6FA8", "#1B7F4B"]))
    for name, cur in curves.items():
        ax.plot([c["i"] for c in cur], [c["bytes"] / 1e6 for c in cur],
                lw=2.4 if name == "flat-r1-routed" else 1.6, color=cols[name], label=name,
                zorder=3 if name == "flat-r1-routed" else 2)
    ax.set_xlabel("OOD arrivals absorbed")
    ax.set_ylabel("cumulative adapter storage (MB)")
    ax.set_title("N40 — what a lifelong stream costs.  Every arm preserves every earlier concept\n"
                 "bit-identically; the difference is only what an arrival is allowed to spend.",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG, f"f15_budget{args.tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[n40] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
