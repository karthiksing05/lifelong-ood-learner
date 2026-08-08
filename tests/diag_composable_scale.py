"""Pick Composable Diffusion's guidance scale, so it is reported in ITS best configuration.

This exists because of a fairness bug found by a literature audit.  `composable_eps` originally
defaulted to w_i = 1/n, which makes the total guidance 1.0 no matter how many concepts are composed
-- roughly their average.  Liu et al.'s w_i are per-concept GUIDANCE SCALES meant to add (they run
5-7.5 with GLIDE): the sum of residuals is what makes the result a product rather than a mean.  A
baseline run at 1/n is a baseline run weak, and this repository already had to correct exactly that
mistake once for `poe-guided`, where using our annealed schedule instead of the source paper's
Eq. 14 took its precision from 0.039 to 0.839.  The rule that came out of that correction is the one
applied here: every baseline is run the way its own authors run it, and the configuration is chosen
by a sweep rather than by a default.

Sweeps `scale` on ColorMNIST held-out compositions and reports the full metric set per scale, so the
number that goes in the paper is the best one this operator can produce, not the first one tried.

    python tests/diag_composable_scale.py --n-cases 3 --n-samples 16
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import CACHE, ood_queries                                         # noqa: E402
from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, density_coverage,       # noqa: E402
                                fit_probes, precision_recall)
from lifelong_ood.neural import load_field, to_data, to_model                 # noqa: E402
from lifelong_ood.neural.composable import compose_classes                    # noqa: E402
from lifelong_ood.neural.inversion import teacher_init_alpha                  # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                       # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.33, 1.0, 2.0, 3.0, 5.0, 7.5])
    p.add_argument("--mcmc-steps", type=int, default=0,
                   help="if >0, sweep the Langevin step scale at --fixed-scale instead of "
                        "sweeping the guidance scale (Du et al. 2023 corrector)")
    p.add_argument("--mcmc-lams", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.3])
    p.add_argument("--fixed-scale", type=float, default=1.0)
    args = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    field = load_field(args.ckpt, device=dev)
    # exactly n9's construction: the SEEN slots in the checkpoint's own order, null row excluded
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    null_row = ck["ema"]["cemb.weight"].numpy()[field.model.null_id]

    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 1.0])

    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    print(f"[diag] composable-diff scale sweep, {len(qs)} held-out cases, "
          f"top-{args.topk} slots, {args.n_samples} samples\n", flush=True)

    # either sweep guidance scale (no corrector) or sweep the corrector's step size at a fixed scale
    mcmc_mode = args.mcmc_steps > 0
    grid = args.mcmc_lams if mcmc_mode else args.scales
    lbl = "lam" if mcmc_mode else "scale"
    if mcmc_mode:
        print(f"[diag] MCMC corrector: {args.mcmc_steps} Langevin steps per denoising step, "
              f"guidance scale fixed at {args.fixed_scale}\n", flush=True)
    hdr = (f"    {lbl:>6s}{'joint':>8s}{'digit':>7s}{'fg':>6s}{'bg':>6s}"
           f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'dens':>7s}{'cov':>7s}")
    print(hdr)
    out = []
    for sc in grid:
        agg = []
        for qi, q in enumerate(qs):
            xq = to_model(data["X"][q["query"]]).ravel()
            # exactly the slot ranking n9 gives this arm, so only `scale` varies across the sweep
            t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120)
            a0 = teacher_init_alpha(field, t, xq, seed=args.seed)
            topk = np.argsort(-a0)[:args.topk]
            S = to_data(compose_classes(
                field, C, topk, n=args.n_samples, steps=args.steps, seed=args.seed + qi,
                null_emb=null_row,
                scale=(args.fixed_scale if mcmc_mode else float(sc)),
                mcmc=(dict(steps=args.mcmc_steps, lam=float(sc)) if mcmc_mode
                      else None))).astype(np.float32)
            R = data["X"][data["slot"] == q["slot"]].astype(np.float32)
            a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
            pr = precision_recall(S, R)
            dc = density_coverage(S, R)
            agg.append([a["joint"], a["digit"], a["fg"], a["bg"], pr["precision"],
                        pr["recall"], pr["f1"], dc["density"], dc["coverage"]])
        m = np.mean(agg, 0)
        out.append((sc, m))
        print(f"    {sc:6.2f}{m[0]:8.3f}{m[1]:7.3f}{m[2]:6.2f}{m[3]:6.2f}"
              f"{m[4]:7.3f}{m[5]:8.3f}{m[6]:7.3f}{m[7]:7.2f}{m[8]:7.2f}", flush=True)

    best = max(out, key=lambda r: (r[1][0], r[1][6]))
    print(f"\n[diag] best joint accuracy at {lbl} {best[0]:.3f} "
          f"(joint {best[1][0]:.3f}, F1 {best[1][6]:.3f})")
    if max(r[1][0] for r in out) == 0.0:
        print("[diag] joint accuracy is 0.000 at EVERY setting -- this is the operator's "
              "precondition failing, not a tuning failure: with only entangled slot conditionals "
              "there is no per-factor score to add.")


if __name__ == "__main__":
    main()
