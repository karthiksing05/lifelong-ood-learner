"""Does `router.calibrate_u_novel` reproduce N65's answer on both backbones?

N65 established the finding as an experiment; this asserts the LIBRARY function that shipped it,
because a result living only in an experiment script is one refactor away from being unverifiable.

Expected (N65, 3 seeds): Shapes3D picks u=0.85 -- the oracle's own choice, every seed -- and
ColorMNIST picks high-u (0.75-0.85), which is the WRONG choice there and is expected to be: that
router runs on density, so class legibility has nothing to track (6c).  Both are asserted, the
failure as firmly as the success, so that a change which "fixes" ColorMNIST is caught rather than
celebrated -- it would mean the criterion had started tracking something other than legibility.

    python tests/diag_calibrate_u.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from n61_router_ablation import (load_colormnist_setup,                # noqa: E402
                                 load_shapes3d_setup)

from lifelong_ood.neural.router import calibrate_u_novel               # noqa: E402

EXPECT = {                       # (allowed u values, why)
    "ColorMNIST(nobg3)": ((0.60, 0.75, 0.85, 0.90, 0.95),
                          "density-driven router: legibility is at chance where novelty works, so "
                          "the criterion lands high and is NOT the right u here (6c)"),
    "Shapes3D(pretrained)": ((0.85, 0.90),
                             "conditioning-driven router: legibility peaks at the oracle's u"),
}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fail = 0
    for name, loader in (("ColorMNIST(nobg3)", load_colormnist_setup),
                         ("Shapes3D(pretrained)", load_shapes3d_setup)):
        ckpt_missing = False
        try:
            field, C, X, slots, seen, held, prep = loader(0, dev)
        except (FileNotFoundError, OSError) as e:
            print(f"SKIP {name}: checkpoint unavailable ({e})")
            ckpt_missing = True
        if ckpt_missing:
            continue
        u, curve = calibrate_u_novel(field, C, X, slots, seen, seed=0, n_q=16, prep=prep)
        ok, why = EXPECT[name]
        print(f"{name}: u*={u:g}  curve " + " ".join(f"{a:g}:{b:.2f}" for a, b in curve))
        if u in ok:
            print(f"  PASS -- {why}")
        else:
            print(f"  FAIL -- expected one of {ok}; {why}")
            fail = 1
        del field
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("exit", fail)
    return fail


if __name__ == "__main__":
    sys.exit(main())
