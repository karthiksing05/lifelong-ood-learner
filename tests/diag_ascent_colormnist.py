"""Does ColorMNIST have the out-of-gamut problem that plain ascent was introduced to fix?

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
    print(f"  {'u':>6s} {'optimizer':>10s} {'frac |x0|>1':>12s} {'max |x0|':>10s}")
    for u in (0.05, 0.1, 0.2, 0.3, 0.4):
        for lab, adam in (("Adam", True), ("plain", False)):
            _, x0 = field.mode_ascent(xq, u, n_starts=8, iters=150, lr=0.1, seed=0,
                                      init="hybrid", clamp=False, tweedie=True, adam=adam)
            x0 = np.asarray(x0)
            print(f"  {u:6.2f} {lab:>10s} {float((np.abs(x0) > 1).mean()):12.4f} "
                  f"{float(np.abs(x0).max()):10.3f}")
    print("\n[cm-gamut] a near-zero out-of-range fraction under Adam means ColorMNIST never had the\n"
          "           problem plain ascent fixes, and the paper's Adam should stand here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
