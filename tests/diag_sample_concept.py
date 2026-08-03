"""Why does sampling from a consolidated concept fail?  One concept, many sampling settings.

N3 consolidates concepts whose POOLS score up to 1.00 composition accuracy, drives distillation loss
down to ~0.04, and keeps retention exact -- yet conditional samples from the resulting adapter score
0.00.  Since the pool and the loss are both healthy, the fault is in how the adapter is SAMPLED, not
in what it learned.  This isolates that: consolidate one concept, then sweep

  guidance   1.0 = follow the adapter exactly; >1 extrapolates away from the frozen base.  If only
             w=1 works, the CFG extrapolation is the problem.
  u_start    1.0 = from pure noise; lower = start from a noised pool sample, which tests whether the
             adapter can DENOISE the concept even if it cannot generate it from scratch.
  steps      more steps = less discretization error.

It also reports the adapter's own training-objective loss on the pool at sampling time, which
separates "the adapter forgot" from "the sampler cannot reach it".
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, sample_geometry,  # noqa: E402
                                save_grid)
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model  # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402


def pool_loss(L, name, pool, seed=0, reps=4):
    """The distillation objective itself, evaluated after training -- did the adapter keep it?"""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(reps):
        u = float(rng.uniform(0.05, 0.95))
        a = float(cosine_abar(u))
        e = rng.standard_normal(pool.shape)
        xt = np.sqrt(a) * pool + np.sqrt(1 - a) * e
        tot += float(np.mean((L.field.eps(xt, u, name) - e) ** 2))
    return tot / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-pool", type=int, default=96)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=100, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=50, seed=args.seed, device=dev)

    q = ood_queries(data, n_slots=1, seed=args.seed)[0]
    xq01 = data["X"][q["query"]]
    want = (q["d"], q["f"], q["b"])
    print(f"[diag] concept: digit {want[0]}, fg {want[1]}, bg {want[2]}", flush=True)

    teacher = L.compose(to_model(xq01).ravel(), u_grid=(0.1, 0.2, 0.3, 0.4),
                        n_starts=16, iters=120)
    pool = L.sample_teacher(teacher, n=args.n_pool, seed=args.seed)
    pool01 = to_data(pool).astype(np.float32)
    pa = composition_accuracy(probes, pool01, *want)
    print(f"[diag] pool: joint {pa['joint']:.2f}, diversity "
          f"{sample_geometry(pool01, xq01)['diversity']:.3f}", flush=True)
    save_grid(np.concatenate([[xq01], pool01[:15]]), os.path.join(RESULTS, "diag_pool.png"),
              title="the pool that gets distilled")

    L.consolidate(pool, "c", mu=teacher["mu"], epochs=args.epochs, verbose=False)
    print(f"[diag] distillation loss {L.last_curve['first']:.4f} -> {L.last_curve['final']:.4f}")
    print(f"[diag] post-hoc loss on the pool, adapter active: {pool_loss(L, 'c', pool):.4f}")
    print(f"[diag] post-hoc loss on the pool, FROZEN BASE:    {pool_loss(L, None, pool):.4f}")
    print("       (adapter should be clearly lower; if not, it did not learn the pool)\n")

    print(f"    {'guidance':>8s} {'u_start':>7s} {'steps':>6s} {'joint':>6s} {'digit':>6s} "
          f"{'fg':>5s} {'bg':>5s} {'divers':>7s}")
    best = None
    for g in (1.0, 1.5, 2.0, 3.0):
        for us, st in ((1.0, 50), (1.0, 100), (0.6, 50)):
            S = to_data(L.sample_concept("c", n=args.n_samples, seed=args.seed,
                                         guidance=g, u_start=us, steps=st)).astype(np.float32)
            acc = composition_accuracy(probes, S, *want)
            div = sample_geometry(S, xq01)["diversity"]
            print(f"    {g:8.1f} {us:7.2f} {st:6d} {acc['joint']:6.2f} {acc['digit']:6.2f} "
                  f"{acc['fg']:5.2f} {acc['bg']:5.2f} {div:7.3f}", flush=True)
            if best is None or acc["joint"] > best[0]:
                best = (acc["joint"], g, us, st, S)
    print(f"\n[diag] best: joint {best[0]:.2f} at guidance={best[1]} u_start={best[2]} steps={best[3]}")
    save_grid(np.concatenate([[xq01], best[4][:15]]),
              os.path.join(RESULTS, "diag_best_concept_samples.png"),
              title=f"best conditional samples (g={best[1]}, u0={best[2]})")
    print(f"[diag] wrote {RESULTS}/diag_pool.png and diag_best_concept_samples.png")


if __name__ == "__main__":
    main()
