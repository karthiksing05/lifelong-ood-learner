"""N0 — the neural sampler's tuning-of-record, and specifically its DIVERSITY problem.

N2 showed the neural backbone composing very accurately (joint 0.899, above the probe ceiling) while
producing near-identical samples: diversity 0.042 against a reference set whose own internal spread
is 0.166.  High accuracy with collapsed diversity is the same degenerate optimum `query_only` falls
into on the analytic backbone, arrived at from the other direction -- there by copying the query,
here by every chain converging to the teacher mean.

The knob that actually controls it is **wmin**, and finding that took one failed sweep.  Sweeping
u_start, the teacher covariance scale and wmax moved diversity from 0.016 to at most 0.042 -- never
close to 0.166, and no better when starting from PURE NOISE, which should have been maximally
diverse.  That ruled out the initialization and pointed at the schedule.

Classifier-free guidance mixes as `(1-w)*eps_uncond + w*eps_cond`, so `wmin=0.8` -- inherited from
the analytic backbone -- means the teacher already contributes 80% at HIGH noise, which is exactly
where the sample's identity and variety get decided.  A product of experts is a SHARP distribution
(multiplying Gaussians shrinks the covariance), so a teacher-dominated high-noise phase funnels every
chain to mu_T no matter where it started.

The analytic backbone needed that floor because its unconditional mixture actively fights the
composition (see RESULTS.md 4).  A trained UNet does not: it holds real density at unseen combinations,
so it can be trusted to supply variety at high noise and the teacher only needs to take over late.
Hence this sweep is over `wmin`, with the others kept as secondary axes.

Scored on composition accuracy AND diversity, targeting diversity NEAR the reference spread -- not
maximal.  Samples more varied than the real class are off-manifold, not better, which is the failure
the analytic backbone shows (diversity 0.348 vs 0.166).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, sample_geometry,  # noqa: E402
                                save_grid, save_json)
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_samples, args.n_per_slot = 2, 8, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    qs = ood_queries(data, n_slots=args.n_slots, seed=args.seed)
    print(f"[n0] {len(qs)} OOD slots, {args.n_samples} samples each, device {dev}", flush=True)

    teachers = [L.compose(to_model(data["X"][q["query"]]).ravel(),
                          u_grid=(0.1, 0.2, 0.3, 0.4),
                          n_starts=8 if args.quick else 16,
                          iters=40 if args.quick else 120) for q in qs]
    ref_spread = float(np.mean([sample_geometry(data["X"][q["rest"][:60]],
                                                data["X"][q["query"]])["diversity"] for q in qs]))
    print(f"[n0] reference set internal spread {ref_spread:.3f} -- the TARGET for diversity\n")

    print(f"[n0] teacher sigma: mean {float(np.mean([np.sqrt(t['var']).mean() for t in teachers])):.4f} "
          f"vs reference spread {ref_spread:.4f} -- the ratio is the diversity ceiling")
    rows = []
    grid = [(ts, wn, wm) for ts in (1.0, 10.0, 30.0, 100.0)
            for wn in (0.3, 0.8) for wm in (1.5, 2.0)]
    print(f"    {'t_scale':>7s} {'wmin':>5s} {'wmax':>5s} {'joint':>6s} {'digit':>6s} "
          f"{'fg':>5s} {'bg':>5s} {'novel':>7s} {'divers':>7s} {'div/ref':>8s} {'score':>7s}")
    for ts, wn, wm in grid:
        us = 1.0
        A, G = [], []
        for qi, (q, t0) in enumerate(zip(qs, teachers)):
            # inflate the teacher covariance: a Laplace estimate at a mode measures LOCAL curvature,
            # not how much the class actually varies, so it systematically understates the spread a
            # sample pool should have.  This sweep is what sets the correction factor.
            S = L.sample_teacher(t0, n=args.n_samples, seed=args.seed + qi, var_scale=ts,
                                 u_start=us, wmin=wn, wmax=wm)
            Sd = to_data(S).astype(np.float32)
            A.append(composition_accuracy(probes, Sd, q["d"], q["f"], q["b"]))
            G.append(sample_geometry(Sd, data["X"][q["query"]], data["X"][q["rest"][:100]]))
        m = lambda k, L_: float(np.mean([x[k] for x in L_]))            # noqa: E731
        r = dict(u_start=us, teacher_scale=ts, wmin=wn, wmax=wm,
                 **{k: m(k, A) for k in ("joint", "digit", "fg", "bg")},
                 **{k: m(k, G) for k in ("novelty", "diversity", "gen_rmse")})
        r["div_ratio"] = r["diversity"] / ref_spread
        # accuracy, penalized for diversity in EITHER direction away from the reference spread
        r["score"] = r["joint"] * float(np.exp(-abs(np.log(max(r["div_ratio"], 1e-6)))))
        rows.append(r)
        print(f"    {ts:7.1f} {wn:5.2f} {wm:5.1f} {r['joint']:6.3f} {r['digit']:6.2f} "
              f"{r['fg']:5.2f} {r['bg']:5.2f} {r['novelty']:7.3f} {r['diversity']:7.3f} "
              f"{r['div_ratio']:8.2f} {r['score']:7.3f}", flush=True)

    best = max(rows, key=lambda r: r["score"])
    save_json(dict(rows=rows, ref_spread=ref_spread, best=best),
              os.path.join(RESULTS, "n0_tune.json"), seed=args.seed)
    print(f"\n[n0] best: teacher_scale={best['teacher_scale']} wmin={best['wmin']} "
          f"wmax={best['wmax']} -> joint {best['joint']:.3f}, diversity {best['diversity']:.3f} "
          f"({best['div_ratio']:.2f}x reference)")
    # a picture of the chosen setting
    q = qs[0]
    S = to_data(L.sample_teacher(teachers[0], n=16, seed=args.seed, u_start=best["u_start"],
                                 var_scale=best["teacher_scale"],
                                 wmin=best["wmin"], wmax=best["wmax"]))
    save_grid(np.concatenate([[data["X"][q["query"]]], S[:15]]),
              os.path.join(RESULTS, "n0_best_samples.png"),
              title=f"neural PoE at the chosen setting (first tile = query)")
    print(f"[n0] wrote {RESULTS}/n0_tune.json")


if __name__ == "__main__":
    main()
