"""Adam vs plain ascent on ColorMNIST, judged by the objective ascent exists to reduce.

The settings must be the SAME on both benchmarks -- a per-dataset algorithmic choice is how a
comparison stops being one -- so the choice needs an arm-neutral criterion measured on both.  On
CelebA that was done: plain ascent left 0.0% of the mode out of gamut against Adam's 11.1%, and
reached a lower eps error (0.0046 vs 0.0113) than Adam did.

Only the GAMUT half was ever measured on ColorMNIST.  The eps error -- the quantity ascent is
actually minimizing, and one that does not care which arm benefits -- was not, and it is the half
that decides a global default.  This measures both.

Plain gradient ascent replaced Adam because on the PRETRAINED CelebA backbone, Adam left 11.1% of
each returned mode outside [-1,1] and those coordinates clip to flat saturated colour.  That change
is global, and on ColorMNIST it moves the n9 table: ours improves (F1 0.605 -> 0.633) while
`poe-guided` LOSES accuracy (joint 0.839 -> 0.739) and gains nothing (recall 0.001 either way).

A default that helps our arm and handicaps the baseline needs a reason that is not "it helps our
arm".  The reason on offer is the gamut, so the question is simply whether ColorMNIST has that
problem at all.  If it does not, the fix does not belong there and Adam -- the paper's own Table 6
setting -- should stand on this dataset.

    python tests/diag_ascent_colormnist.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.neural import load_field, to_model                           # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev, batch=64)
    data = load_colormnist(n_per_slot=4, seed=0)
    xq = to_model(data["X"][:1]).ravel()
    print("[cm-gamut] ColorMNIST backbone; CelebA measured 11.1% out of range under Adam\n")
    from lifelong_ood.score import cosine_abar

    def eps_err(x0, u, seed=1):
        r = np.random.default_rng(seed)
        a = float(cosine_abar(u))
        e = r.standard_normal(np.atleast_2d(x0).shape)
        xt = np.sqrt(a) * np.atleast_2d(x0) + np.sqrt(1 - a) * e
        return float(np.mean((field.eps(xt, u) - e) ** 2))

    print(f"  {'u':>6s} {'optimizer':>10s} {'frac |x0|>1':>12s} {'max |x0|':>10s} "
          f"{'eps error':>11s} {'(query)':>9s}")
    for u in (0.05, 0.1, 0.2, 0.3, 0.4):
        q = eps_err(np.clip(xq, -1, 1), u)
        for lab, kw in (("Adam", dict(adam=True, clamp=False)),
                        ("plain", dict(adam=False, clamp=False)),
                        ("Adam+x0proj", dict(adam=True, clamp="x0"))):
            _, x0 = field.mode_ascent(xq, u, n_starts=8, iters=150, lr=0.1, seed=0,
                                      init="hybrid", tweedie=False, **kw)
            x0 = np.asarray(x0)
            print(f"  {u:6.2f} {lab:>10s} {float((np.abs(x0) > 1).mean()):12.4f} "
                  f"{float(np.abs(x0).max()):10.3f} {eps_err(np.clip(x0, -1, 1), u):11.4f} "
                  f"{q:9.4f}")
    print("\n[cm-gamut] a near-zero out-of-range fraction under Adam means ColorMNIST never had the\n"
          "           problem plain ascent fixes, and the paper's Adam should stand here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
