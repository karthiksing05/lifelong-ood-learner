"""N33 -- how many query images does a new concept actually need?

This repository claims single-query out-of-distribution learning everywhere and has never swept the
claim.  One image is what the companion paper's protocol provides, so it is what every table here
uses, but "works from one" and "works BEST from one" are different statements and only the first has
been measured.  If the curve is flat from m = 1 the claim is stronger than currently stated; if it
climbs steeply, several headline numbers are reported at the hardest point on their own axis and
should say so.

**Combining m queries.** `compose()` reads ONE image, so m of them have to be combined somewhere.
The combination used here is the one the framework already implies rather than a new heuristic: each
query gives a per-dimension Gaussian teacher, and m independent observations of the same concept
combine as a **product of experts** -- precisions add.

    1 / var_c = sum_i 1 / var_i          mu_c = var_c * sum_i (mu_i / var_i)

which for m = 1 is exactly the existing single-query path, so the sweep contains the current protocol
as its own first point rather than approximating it.

Arms, all at matched m:

  `ours`         combined teacher -> pool -> consolidated adapter
  `poe-guided`   the companion method, sampling the combined teacher directly (their Eq. 13/14)
  `inversion`    textual inversion on the m REAL images -- the arm that most naturally uses more
                 data, and therefore the one most likely to overtake us as m grows

    python experiments/n33_sample_efficiency.py --n-cases 3
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
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                       # noqa: E402
from lifelong_ood.neural.inversion import learn_embedding, sample_embedding   # noqa: E402

from lifelong_ood.compose import PAPER_POE_COLORMNIST as PAPER_POE


def combine_teachers(ts):
    """Product of experts over m per-dimension Gaussian teachers.  m=1 returns it unchanged."""
    if len(ts) == 1:
        return dict(ts[0])
    P = np.stack([1.0 / np.asarray(t["var"], np.float64) for t in ts])       # precisions
    M = np.stack([np.asarray(t["mu"], np.float64) for t in ts])
    var = 1.0 / P.sum(0)
    return dict(ts[0], mu=var * (P * M).sum(0), var=var)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 5, 10, 20],
                   help="number of query images the concept is learned from")
    p.add_argument("--arms", nargs="+", default=["ours", "poe-guided", "inversion"])
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.n_per_slot = 1, 8, 20
        args.epochs, args.steps, args.inv_steps = 4, 20, 60
        args.ms = [1, 5]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(field0, C, slots, probes, data, seed=args.seed, steps=args.steps,
                           n=8 if args.quick else 16)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"[n33] {len(qs)} concepts x {len(args.ms)} sample counts x {len(args.arms)} arms\n",
          flush=True)

    rows = []
    for q in qs:
        # Split the held-out slot's real images DISJOINTLY into the pool a query may be drawn from
        # and the reference precision/recall is scored against.  Without this the m-th query image
        # is also a reference image, and recall would climb with m because the method had been shown
        # the very set it is being scored against -- an artifact that would look exactly like sample
        # efficiency.  The reference is also held FIXED across m, so the metric is comparable along
        # the axis being swept.
        perm = rng.permutation(np.asarray(q["rest"]))
        n_extra = max(args.ms) - 1
        query_pool, ref_idx = perm[:n_extra], perm[n_extra:n_extra + 100]
        R = data["X"][ref_idx].astype(np.float32)
        for m in args.ms:
            take = [q["query"]] + list(query_pool[:m - 1])
            X = np.stack([to_model(data["X"][i]).ravel() for i in take[:m]])
            field = load_field(args.ckpt, device=dev)
            L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
            L.attach_factorization(C, slot_factors(slots), gains=gains)
            ts = [L.compose(x, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=8 if args.quick else 16,
                            iters=30 if args.quick else 120) for x in X]
            t = combine_teachers(ts)

            for arm in args.arms:
                if arm == "ours":
                    name = f"c{q['slot']}_m{m}"
                    pool = L.sample_teacher(t, n=48, seed=args.seed)
                    L.consolidate(pool, name, mu=t["mu"], epochs=args.epochs)
                    S = to_data(L.sample_concept(name, n=args.n_samples, steps=args.steps,
                                                 seed=args.seed)).astype(np.float32)
                elif arm == "poe-guided":
                    S = to_data(L.sample_teacher(t, n=args.n_samples, seed=args.seed,
                                                 **PAPER_POE)).astype(np.float32)
                else:
                    # textual inversion gets the m real images directly -- its natural setting
                    emb, _, _ = learn_embedding(field, X, n_steps=args.inv_steps,
                                                seed=args.seed)
                    S = to_data(sample_embedding(field, emb, n=args.n_samples, steps=args.steps,
                                                 seed=args.seed)).astype(np.float32)
                a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                pr = precision_recall(S, R)
                rows.append(dict(arm=arm, m=int(m), slot=int(q["slot"]), joint=a["joint"],
                                 digit=a["digit"], fg=a["fg"], bg=a["bg"],
                                 recall=pr["recall"], f1=pr["f1"], vendi=vendi_score(S)))
                print(f"[n33] slot {q['slot']} m={m:2d} {arm:12s} joint {a['joint']:.3f} "
                      f"recall {pr['recall']:.3f} F1 {pr['f1']:.3f}", flush=True)
            del field, L
            torch.cuda.empty_cache()

    summary = {}
    for arm in args.arms:
        for m in args.ms:
            sel = [r for r in rows if r["arm"] == arm and r["m"] == m]
            if sel:
                summary[f"{arm}@{m}"] = {k: float(np.nanmean([r[k] for r in sel]))
                                         for k in ("joint", "recall", "f1", "vendi")}
    save_json(dict(rows=rows, summary=summary, config=vars(args)),
              os.path.join(RESULTS, f"n33_sample_efficiency{args.tag}.json"), seed=args.seed)

    print(f"\n[n33] F1 against the number of query images ({len(qs)} concepts)")
    print(f"    {'arm':14s}" + "".join(f"{'m=' + str(m):>10s}" for m in args.ms))
    for arm in args.arms:
        cells = [summary.get(f"{arm}@{m}", {}).get("f1") for m in args.ms]
        print(f"    {arm:14s}" + "".join(
            f"{c:10.3f}" if c is not None else f"{'--':>10s}" for c in cells))
    ours = [summary.get(f"ours@{m}", {}).get("f1") for m in args.ms]
    if ours[0] is not None and any(c is not None for c in ours[1:]):
        best = max((c for c in ours if c is not None))
        print(f"\n[n33] ours at m=1: {ours[0]:.3f}; best over the sweep: {best:.3f} "
              f"({100 * (best - ours[0]) / max(ours[0], 1e-9):+.0f}% from {len(args.ms)} settings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
