"""Is the PoE teacher's covariance carrying any information, or is it pinned at its clip bounds?

Eq. 14's hand-off, and the PoE's ability to say "the stroke is uncertain but the background is not",
both depend on Sigma_T being ANISOTROPIC -- large in the directions a class actually varies, small in
the directions it does not.  Our teacher's sigma came out at 0.0117 uniformly, which is suspiciously
close to `laplace_diag`'s `var_floor` of 1e-4 (sigma 0.01).  If most coordinates are simply clipped
there, the covariance is a constant, the PoE degenerates to a point mass, and no guidance schedule
can recover diversity from it -- which would explain why N6 found recall ~0 for every schedule.

Reports, for the raw Laplace output and for the composed teacher:
  the fraction of coordinates sitting at the floor and at the cap,
  the spread of the values in between,
  and whether the variance is larger on the digit stroke than on the background, which is the
  specific anisotropy the composition needs.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, ood_queries                                  # noqa: E402

from lifelong_ood.compose import candidates_ascent, poe_teacher        # noqa: E402
from lifelong_ood.data.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_model  # noqa: E402


def describe(name, V, floor=1e-4, cap=1.0):
    V = np.asarray(V).ravel()
    at_f = float(np.mean(V <= floor * 1.001))
    at_c = float(np.mean(V >= cap * 0.999))
    free = V[(V > floor * 1.001) & (V < cap * 0.999)]
    print(f"  {name:28s} n={V.size:6d}  at floor {at_f:6.1%}  at cap {at_c:6.1%}  "
          f"free {len(free) / V.size:6.1%}")
    print(f"    {'':26s}  sigma: min {np.sqrt(V.min()):.4f} med {np.sqrt(np.median(V)):.4f} "
          f"max {np.sqrt(V.max()):.4f}   ratio max/med {np.sqrt(V.max() / max(np.median(V), 1e-12)):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    ap.add_argument("--n-per-slot", type=int, default=100)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=0)
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, device=dev)
    q = ood_queries(data, n_slots=1, seed=0)[0]
    xq = to_model(data["X"][q["query"]]).ravel()

    print("raw Laplace output from mode ascent (what candidates_ascent returns):")
    M, V, meta = candidates_ascent(field, xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120,
                                   seed=0)
    describe("all candidate variances", V)

    t = poe_teacher(xq, M, V, K=3, tau=0.05)
    describe("composed teacher Sigma_T", t["var"])

    # the anisotropy the composition actually needs: is the stroke less certain than the background?
    xq01 = data["X"][q["query"]].reshape(32, 32, 3)
    bg = np.median(xq01.reshape(-1, 3), 0)
    stroke = (np.abs(xq01 - bg).sum(-1) > 0.25).ravel()
    stroke3 = np.repeat(stroke, 3)
    vt = np.asarray(t["var"]).ravel()
    print(f"\n  stroke pixels: {stroke.mean():.1%} of the image")
    print(f"  sigma on STROKE     {np.sqrt(vt[stroke3]).mean():.4f}")
    print(f"  sigma on BACKGROUND {np.sqrt(vt[~stroke3]).mean():.4f}")
    ratio = np.sqrt(vt[stroke3]).mean() / max(np.sqrt(vt[~stroke3]).mean(), 1e-9)
    print(f"  stroke/background ratio {ratio:.2f}  "
          f"({'anisotropic — usable' if ratio > 1.3 else 'FLAT — the covariance says nothing'})")

    # what the REAL class variance looks like, for comparison
    R = data["X"][q["rest"][:100]]
    real = R.var(0)
    print(f"\n  real class sigma: stroke {np.sqrt(real[stroke3]).mean():.4f}  "
          f"background {np.sqrt(real[~stroke3]).mean():.4f}  "
          f"ratio {np.sqrt(real[stroke3]).mean() / max(np.sqrt(real[~stroke3]).mean(), 1e-9):.2f}")
    print(f"  => the teacher understates the class sigma by "
          f"{np.sqrt(real.mean()) / max(np.sqrt(vt.mean()), 1e-9):.1f}x on average")


if __name__ == "__main__" and "--sweep" not in sys.argv:
    main()


def sweep():
    """Two candidate fixes for a covariance that is 99% clipped at its floor.

    (a) DELTA.  The Hutchinson finite difference uses delta=5e-3, which probes the score over a
        distance ~40x smaller than the class actually varies (sigma ~0.2).  At that scale it measures
        the UNet's local roughness, not the curvature of log p_t, and roughness reads as enormous
        negative curvature -> variance below the floor.  Larger delta should recover real curvature.

    (b) BETWEEN-EXPERT SPREAD.  A product of experts is DESIGNED to be sharper than its factors, so
        even a perfect within-mode Laplace term understates how far the concept extends.  The
        discovered prototypes are themselves samples of the concept's neighbourhood, so their spread
        is a second, model-free variance term: Var_total = E[within] + Var[between].
    """
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    ap.add_argument("--n-per-slot", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=a.n_per_slot, seed=0)
    field = load_field(a.ckpt, device=dev)
    q = ood_queries(data, n_slots=1, seed=0)[0]
    xq = to_model(data["X"][q["query"]]).ravel()
    real = data["X"][q["rest"][:100]].var(0)
    print(f"\nreal class sigma (target): {np.sqrt(real.mean()):.4f}\n")
    print(f"  {'delta':>8s}{'at floor':>10s}{'med sigma':>11s}{'mean sigma':>11s}")
    for d in (5e-3, 2e-2, 5e-2, 1e-1, 2e-1):
        xt, x0 = field.mode_ascent(xq, 0.2, n_starts=8, iters=120, seed=0)
        V = field.laplace_diag(xt, 0.2, probes=4, delta=d, seed=0)
        print(f"  {d:8.3f}{np.mean(V <= 1e-4 * 1.001):10.1%}"
              f"{np.sqrt(np.median(V)):11.4f}{np.sqrt(V.mean()):11.4f}")
    # (b) between-expert spread of the discovered prototypes
    M, V, _ = candidates_ascent(field, xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120, seed=0)
    t = poe_teacher(xq, M, V, K=3, tau=0.05)
    between = M[t["sel"]].var(0)
    print(f"\n  within-expert  sigma (current Sigma_T) {np.sqrt(np.asarray(t['var']).mean()):.4f}")
    print(f"  between-expert sigma (spread of the K selected means) "
          f"{np.sqrt(between.mean()):.4f}")
    print(f"  within + between                       "
          f"{np.sqrt((np.asarray(t['var']) + between).mean()):.4f}")
    print(f"  real class                             {np.sqrt(real.mean()):.4f}")


if __name__ == "__main__" and "--sweep" in sys.argv:
    sys.argv.remove("--sweep")
    sweep()
