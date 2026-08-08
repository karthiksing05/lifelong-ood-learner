"""If the sampler is not the problem, the DISCOVERY is: how spread are the modes we compose?

The sweep in `diag_poe_faithful.py` shows every sampler configuration -- including the paper's own
Eq. 14 -- producing recall 0.000 at content diversity 0.24-1.04.  So the teacher itself cannot cover
the class, and there are only two ways that happens:

  1. the selected expert MEANS are near-duplicates, so the product has nothing to spread over, or
  2. the per-expert covariances are tiny relative to the class.

The paper's teacher spreads because its modes come from ascent at several noise levels, where high
noise coarsens distinct instance modes into genuinely different class-level prototypes (their Fig. 2
and Appendix A).  Ours are grown on a `u_grid` that n9 truncates to (0.1, 0.2, 0.3, 0.4) -- all LOW
noise, where every mode sits near the query.

This measures the spread directly and sweeps the discovery grid, reporting for each configuration:
how far apart the selected means are, how wide Sigma_T ends up, and what recall the resulting teacher
can reach -- against the real class spread, which is the number all of these have to match.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, ood_queries                                   # noqa: E402

from lifelong_ood.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry)
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model  # noqa: E402

N_CASES = int(os.environ.get("DIAG_CASES", 4))
N_SAMP = int(os.environ.get("DIAG_N", 24))

GRIDS = [
    ("n9 default: low noise only", (0.1, 0.2, 0.3, 0.4), 16, 3),
    ("+ mid noise",                (0.1, 0.2, 0.4, 0.6), 16, 3),
    ("+ high noise (coarse modes)", (0.1, 0.3, 0.6, 0.8), 16, 3),
    ("wide grid, more starts",     (0.1, 0.2, 0.4, 0.6, 0.8), 32, 3),
    ("wide grid, K=8 experts",     (0.1, 0.2, 0.4, 0.6, 0.8), 32, 8),
]


def spread(X):
    """Mean pairwise per-pixel RMSE -- the same units as the real class spread."""
    X = np.asarray(X, np.float64)
    if len(X) < 2:
        return 0.0
    D = np.sqrt(((X[:, None] - X[None]) ** 2).mean(2))
    return float(D[np.triu_indices(len(X), 1)].mean())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=100, seed=0)
    probes = fit_probes(data, seed=0, epochs=30)
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    qs = ood_queries(data, n_slots=N_CASES, seed=0)

    real_spread = float(np.mean([spread(data["X"][q["rest"][:24]].astype(np.float32)) for q in qs]))
    print(f"[diag] REAL class spread (mean pairwise RMSE): {real_spread:.4f}")
    print(f"[diag] every row below has to reach that to cover the class\n", flush=True)

    print(f"  {'discovery configuration':32s}{'pool':>6s}{'sel':>5s}{'mean spread':>13s}"
          f"{'sigma_T':>10s}{'acc':>7s}{'recall':>8s}{'div':>7s}")
    for label, grid, nst, K in GRIDS:
        L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev, K=K)
        pool, sel, msp, sig, acc, rec, div = [], [], [], [], [], [], []
        for i, q in enumerate(qs):
            xq = to_model(data["X"][q["query"]]).ravel()
            t = L.compose(xq, u_grid=grid, n_starts=nst, iters=120)
            M = t["meta"] if isinstance(t.get("meta"), dict) else {}
            pool.append(M.get("n_raw", np.nan))
            sel.append(len(t["sel"]))
            # spread of the SELECTED expert means -- what the product has to work with.
            # compose() does not return the pool, so recover the selected means by re-running
            # discovery with the same seed and indexing with `sel`.
            from lifelong_ood.compose import candidates_ascent
            Mp, Vp, _ = candidates_ascent(field, xq, u_grid=grid, n_starts=nst, iters=120, seed=0)
            msp.append(spread(to_data(Mp[t["sel"]])))
            sig.append(float(np.sqrt(np.mean(t["var"]))))
            S = to_data(L.sample_teacher(t, n=N_SAMP, seed=i)).astype(np.float32)
            R = data["X"][q["rest"][:100]].astype(np.float32)
            acc.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"])
            pr = precision_recall(S, R); rec.append(pr["recall"])
            div.append(sample_geometry(S, R[0], R).get("content_ratio", np.nan))
        print(f"  {label:32s}{np.nanmean(pool):6.0f}{np.mean(sel):5.1f}"
              f"{np.mean(msp):13.4f}{np.mean(sig):10.4f}"
              f"{np.mean(acc):7.3f}{np.mean(rec):8.3f}{np.nanmean(div):7.2f}", flush=True)

    print("\n  sigma_T is the teacher's own standard deviation.  If it is far below the real class")
    print("  spread the teacher is a near-delta and NO sampler can cover the class from it --")
    print("  which would make the published method's failure here a property of the teacher we")
    print("  hand it, not of the method.")


if __name__ == "__main__":
    main()
