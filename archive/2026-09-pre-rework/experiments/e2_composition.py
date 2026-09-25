"""E2 — Test-time compositional generalization from a single OOD query (poe-diffusion Section 4).

For each held-out ColorMNIST composition, one query image drives the whole pipeline: discover
prototypes, select K=3 by the submodular coverage objective, compose them into the analytic PoE
teacher, and sample it through classifier-free guidance with the variance-aware schedule.

Methods (mirroring the paper's table, plus our closed-form discovery):
  poe_tree    discovery by log-likelihood ranking over the tree's own concepts (closed form, free)
  poe_ascent  discovery by mode ascent on p_t over a grid of noise levels (the paper's route)
  query_only  single isotropic Gaussian around the query (their weakest baseline)
  top1/top3   nearest trained-prototype retrieval (their trained-class baselines)

Metrics.  Composition accuracy (do the three probes read the queried (digit, fg, bg) off the
generated samples) is the primary number, but it has a degenerate optimum -- copying the query
scores 1.0 -- so it is always reported next to the geometry triple from `evals.sample_geometry`:
novelty (distance from the query), diversity (spread among samples), and gen_rmse (distance to the
slot's OTHER held-out instances, the paper's "generalization" reference).  The claim this
experiment can support is a JOINT one: PoE reaches high composition accuracy at novelty and
diversity comparable to the reference set's own spread, where query-only reaches the same accuracy
by memorizing and top-k reaches neither.

If poe_tree ~= poe_ascent >> baselines, the paper's discovery step is real AND the hierarchy gives
it away in closed form.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, base_parser, load_all, ood_queries         # noqa: E402

from lifelong_ood.compose import cfg_weight, teacher_eps               # noqa: E402
from lifelong_ood.data.datasets import slot_name                            # noqa: E402
from lifelong_ood.evals import (composition_accuracy, sample_geometry, save_grid,  # noqa: E402
                                save_json)


def main():
    p = base_parser()
    p.add_argument("--n-slots", type=int, default=8, help="OOD slots to evaluate (40 = all)")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--ascent-slots", type=int, default=4,
                   help="slots on which to also run the expensive ascent discovery")
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_samples, args.ascent_slots = 2, 8, 1
    data, learner, probes = load_all(args)
    queries = ood_queries(data, n_slots=args.n_slots, seed=args.seed)

    rows = []
    for qi, q in enumerate(queries):
        xq = data["X"][q["query"]]
        methods = {}
        methods["poe_tree"] = learner.compose(xq, discovery="tree")
        if qi < args.ascent_slots:
            methods["poe_ascent"] = learner.compose(
                xq, discovery="ascent",
                n_starts=4 if args.quick else 8, iters=40 if args.quick else 80)
        methods["query_only"] = learner.baseline_teacher(xq, "query_only")
        methods["top1"] = learner.baseline_teacher(xq, "top1")
        methods["top3"] = learner.baseline_teacher(xq, "top3")

        ref = data["X"][q["rest"][:100]]
        row = dict(slot=q["slot"], name=slot_name(q["d"], q["f"], q["b"]))
        for name, teacher in methods.items():
            S = learner.sample_teacher(teacher, n=args.n_samples, seed=args.seed + qi)
            row[name] = dict(composition_accuracy(probes, S, q["d"], q["f"], q["b"]),
                             **sample_geometry(S, xq, ref))
            if qi < 2:
                save_grid(np.concatenate([[xq], S[:15]]),
                          os.path.join(RESULTS, f"e2_slot{q['slot']}_{name}.png"),
                          title=f"{row['name']} — {name} (first tile = query)")
        rows.append(row)
        line = "  ".join(f"{k}={row[k]['joint']:.2f}" for k in methods)
        print(f"[e2] slot {q['slot']:3d} ({row['name']}): {line}", flush=True)

    keys = ("digit", "fg", "bg", "joint", "novelty", "diversity", "gen_rmse", "ref_spread")
    summary = {}
    for k in ("poe_tree", "poe_ascent", "query_only", "top1", "top3"):
        rs = [r[k] for r in rows if k in r]
        if rs:
            summary[k] = {m: float(np.mean([r[m] for r in rs])) for m in keys if m in rs[0]}
    save_json(dict(rows=rows, summary=summary), os.path.join(RESULTS, "e2_composition.json"),
              seed=args.seed)
    print(f"\n[e2] {len(rows)} OOD slots, {args.n_samples} samples each")
    print(f"    {'method':11s} {'joint':>6s} {'digit':>6s} {'fg':>5s} {'bg':>5s} "
          f"{'novel':>7s} {'divers':>7s} {'gen':>7s}")
    for k, v in summary.items():
        print(f"    {k:11s} {v['joint']:6.3f} {v['digit']:6.2f} {v['fg']:5.2f} {v['bg']:5.2f} "
              f"{v.get('novelty', 0):7.3f} {v.get('diversity', 0):7.3f} {v.get('gen_rmse', 0):7.3f}")
    if summary and "ref_spread" in next(iter(summary.values())):
        print(f"    {'(reference set internal spread':31s} "
              f"{next(iter(summary.values()))['ref_spread']:.3f})")


if __name__ == "__main__":
    main()
