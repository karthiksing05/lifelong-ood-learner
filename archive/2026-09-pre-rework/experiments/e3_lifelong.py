"""E3 — Lifelong learning as repeated composition, WITHOUT degrading the manifold.

The loop: meet one OOD query -> compose known prototypes into an OOD prototype (PoE) -> sample the
prototype through the guided manifold -> consolidate the samples back into the manifold.  Repeat for
a whole sequence of held-out compositions.  This is the poe-diffusion LoRA-distillation step made
structural: their adapter is a gradient-trained low-rank delta on a frozen UNet; ours is a grafted
subtree on a frozen tree.  Both are additive; ours is additive BY CONSTRUCTION in the mixture, so the
interference with old knowledge is bounded analytically by the weight renormalization,
|delta loglik| <= log(1 + n_adapter / n_base), and should measure as ~zero.

Tracked after EVERY consolidation step, for each strategy:
  stability   mean log-likelihood and denoising R^2 of held-out SEEN data under the current field
              (the "does the manifold degrade over time" curve -- the point of this experiment)
  validity    unconditional sample valid-composition rate (generation quality over time)
  plasticity  composition accuracy of conditional samples from the JUST-consolidated concept
  retention   composition accuracy of conditional samples from EVERY EARLIER consolidated concept

Strategies:
  graft  frozen base + additive adapter subtree (ours)
  ifit   classic Cobweb absorption -- every statistic on the descent path mutates (the destructive
         control: this is what "just keep training" does to a generative model)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, base_parser, fresh_learner, load_all, ood_queries  # noqa: E402

from e1_manifold import valid_rate                                     # noqa: E402
from lifelong_ood.data.datasets import slot_name                            # noqa: E402
from lifelong_ood.evals import composition_accuracy, save_grid, save_json  # noqa: E402


def drift_bound(learner):
    """The analytic ceiling on how much consolidation can cost old data, from the ACTUAL weights.

    Grafting leaves every base component untouched and only adds new ones, so the merged density is
    p_new = (1-a) p_base + a p_adapters with `a` the adapter share of the mixture weight.  Since
    p_adapters >= 0, log p_new >= log p_base + log(1-a): the worst case is adapters contributing
    nothing at all on old data, and the drift is then exactly log(1-a).

    `a` must be read off the mixture weights actually in use, not from instance counts -- min_count
    pruning drops instances that sat in small children of kept internal nodes, so raw counts
    overstate the base mass and would make this bound look tighter than it is.
    """
    snap = learner.field.snap
    w = snap["W"]
    a = float(w[(snap["branch"] != "base") & snap["is_leaf"]].sum())
    return float(np.log(max(1.0 - a, 1e-12))), a


def seen_metrics(learner, Xev):
    ll = float(np.mean(learner.field.logp(Xev, u=0.0)))
    from lifelong_ood.score import cosine_abar
    u0 = 0.05
    rec = learner.field.tweedie(np.sqrt(float(cosine_abar(u0))) * Xev, u0)
    r2 = 1.0 - float(np.sum((rec - Xev) ** 2)) / float(np.sum((Xev - Xev.mean(0)) ** 2))
    return ll, r2


def main():
    p = base_parser()
    p.add_argument("--n-concepts", type=int, default=8, help="length of the OOD sequence")
    p.add_argument("--n-pool", type=int, default=48, help="samples consolidated per concept")
    p.add_argument("--n-eval", type=int, default=200, help="held-out seen images tracked")
    p.add_argument("--strategies", nargs="+", default=["graft", "ifit"])
    args = p.parse_args()
    if args.quick:
        args.n_concepts, args.n_pool, args.n_eval = 3, 12, 60
    data, learner0, probes = load_all(args)
    queries = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    seen_idx = np.where(data["seen"])[0]
    Xev = data["X"][rng.choice(seen_idx, args.n_eval, replace=False)]
    seen_slots = np.unique(data["slot"][data["seen"]])
    n_uncond = 16 if args.quick else 48

    results = {}
    for si, strat in enumerate(args.strategies):
        # each strategy gets its own untouched base (a live C++ tree can't be deep-copied)
        learner = learner0 if si == 0 and strat == "graft" and not learner0.adapters \
            else fresh_learner(args, data)
        ll0, r20 = seen_metrics(learner, Xev)
        steps = [dict(step=0, seen_loglik=ll0, seen_r2=r20,
                      valid_rate=valid_rate(probes, learner.field.sample(n=n_uncond, steps=args.steps,
                                                                         seed=args.seed), seen_slots))]
        print(f"[e3:{strat}] step 0: loglik {ll0:.1f}, R2 {r20:.3f}, valid {steps[0]['valid_rate']:.2f}")
        names = []
        for k, q in enumerate(queries, 1):
            name = f"ood_{q['slot']}"
            xq = data["X"][q["query"]]
            teacher, pool = learner.learn_concept(xq, name, n_pool=args.n_pool,
                                                  discovery="tree", strategy=strat,
                                                  seed=args.seed + k)
            names.append((name, q))
            ll, r2 = seen_metrics(learner, Xev)
            rec = dict(step=k, slot=q["slot"], name=slot_name(q["d"], q["f"], q["b"]),
                       seen_loglik=ll, seen_r2=r2, drift_loglik=ll - ll0, drift_r2=r2 - r20)
            rec["valid_rate"] = valid_rate(
                probes, learner.field.sample(n=n_uncond, steps=args.steps, seed=args.seed + 7 * k),
                seen_slots)
            # the pool is what was consolidated; the conditional samples are what comes back out.
            # Reporting both separates "the composition was bad" from "consolidation lost it", which
            # the single `new_concept` number cannot distinguish.
            rec["pool"] = composition_accuracy(probes, pool, q["d"], q["f"], q["b"])
            if strat == "graft":
                b, share = drift_bound(learner)
                rec["drift_bound"] = b
                rec["adapter_weight_share"] = share
                rec["bound_holds"] = bool(rec["drift_loglik"] >= b - 1e-6)
                S = learner.sample_concept(name, n=16 if args.quick else 32, seed=args.seed + k)
                rec["new_concept"] = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                rec["consolidation_retained"] = (rec["new_concept"]["joint"]
                                                 / max(rec["pool"]["joint"], 1e-9))
                rec["retention"] = {}
                for pname, pq in names[:-1]:
                    Sp = learner.sample_concept(pname, n=16, seed=args.seed + 100 + k)
                    rec["retention"][pname] = composition_accuracy(
                        probes, Sp, pq["d"], pq["f"], pq["b"])["joint"]
                if k <= 2:
                    save_grid(np.concatenate([[xq], pool[:15]]),
                              os.path.join(RESULTS, f"e3_pool_{name}.png"),
                              title=f"consolidated pool for {rec['name']} (first tile = query)")
                    save_grid(S[:16], os.path.join(RESULTS, f"e3_cond_{name}.png"),
                              title=f"conditional samples from grafted adapter {name}")
            steps.append(rec)
            extra = (f", pool {rec['pool']['joint']:.2f} -> new {rec['new_concept']['joint']:.2f}"
                     if "new_concept" in rec else f", pool {rec['pool']['joint']:.2f}")
            bnd = (f", bound {rec['drift_bound']:+.2f} "
                   f"[{'ok' if rec['bound_holds'] else 'VIOLATED'}]") if "drift_bound" in rec else ""
            print(f"[e3:{strat}] step {k} (+{rec['name']}): drift loglik {rec['drift_loglik']:+.2f}"
                  f"{bnd}, drift R2 {rec['drift_r2']:+.4f}, valid {rec['valid_rate']:.2f}{extra}",
                  flush=True)
        results[strat] = steps

    save_json(results, os.path.join(RESULTS, "e3_lifelong.json"), seed=args.seed)

    # the degradation curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for strat, steps in results.items():
        ks = [s["step"] for s in steps]
        axes[0].plot(ks, [s["seen_loglik"] for s in steps], marker="o", label=strat)
        axes[1].plot(ks, [s["seen_r2"] for s in steps], marker="o", label=strat)
        axes[2].plot(ks, [s["valid_rate"] for s in steps], marker="o", label=strat)
    for ax, t in zip(axes, ("held-out seen log-likelihood", "held-out seen denoise R²",
                            "unconditional valid-composition rate")):
        ax.set_title(t, fontsize=9); ax.set_xlabel("consolidations"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "e3_degradation.png"), dpi=140)
    print(f"[e3] wrote {RESULTS}/e3_lifelong.json and e3_degradation.png")


if __name__ == "__main__":
    main()
