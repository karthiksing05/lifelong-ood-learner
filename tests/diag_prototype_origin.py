"""Why is prototype #1 always a clean instance of the composition, and #2/#3 noise?

Reading `figures/f9_case_colormnist.png`, the first of the three selected prototypes looks like a
correct in-distribution instance of the held-out composition in every block, while the other two look
like noise.  The natural worry is that the composition was not held out.

It was.  At the data level the four compositions in that figure have **0 training rows and no class
embedding**, verified directly against the checkpoint's remap.  And `diag_holdout_leak.py` shows all
24 selected prototypes sit nearer the TRAINING set than the held-out set, including the ones from
chains that never see the query.

What remains is to explain the appearance, and the explanation is now checkable rather than
plausible, because `candidates_ascent` records each candidate's provenance (`meta["src"]`).  The
prediction is:

    pick #0  came from a QUERY-seeded chain  -> starts at a real instance of the composition and
                                                walks to a nearby mode, so it still looks like one
    pick #1,#2  came from RANDOM chains      -> start in unrelated basins, so they look unrelated

If that holds, the figure is showing the initialization, not the training set.  This prints the
provenance and the noise level of every selected prototype for the exact compositions in f9, plus how
far each moved from its own starting point -- because a prototype that has not moved is the query
re-displayed, and one that has moved a class-width is a mode.

    python tests/diag_prototype_origin.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.compose import PAPER_COLORMNIST, candidates_ascent, poe_teacher  # noqa: E402
from lifelong_ood.datasets import BG_NAMES, FG_NAMES, load_colormnist          # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402

# the four blocks rendered in f9_case_colormnist.png
SHOWN = [(9, "cyan", "deep_red"), (4, "green", "navy"),
         (7, "pink", "dark_purple"), (1, "yellow", "dark_brown")]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=100, seed=0)
    ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu",
                    weights_only=False)
    remap = ck["remap"]
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    rng = np.random.default_rng(0)

    print("[origin] the compositions rendered in f9, checked against the checkpoint itself\n")
    for d, f, b in SHOWN:
        fi, bi = FG_NAMES.index(f), BG_NAMES.index(b)
        m = (data["digit"] == d) & (data["fg"] == fi) & (data["bg"] == bi)
        slot = int(data["slot"][m][0])
        idx = np.where(m)[0]
        q = int(idx[rng.integers(len(idx))])
        xq = to_model(data["X"][q]).ravel()
        rest = data["X"][idx[idx != q]]
        spread = float(np.median(np.sqrt(((rest[:40, None] - rest[None, 40:80]) ** 2).mean(2))))
        TRAIN = data["X"][data["seen"]]

        M, V, meta = candidates_ascent(field, xq, seed=0, **PAPER_COLORMNIST)
        t = poe_teacher(xq, M, V, K=3)
        src, us = np.asarray(meta["src"]), np.asarray(meta["u"])
        P = to_data(np.asarray(M)[t["sel"]]).astype(np.float32)

        print(f"  digit {d}, {f} on {b}  (slot {slot}: {int((m & data['seen']).sum())} training "
              f"rows, embedding row = {slot in remap})")
        print(f"    {'pick':>4s} {'from':>7s} {'u':>6s} {'d(query)':>9s} {'d(train)':>9s} "
              f"{'/ class spread':>14s}")
        for k, s in enumerate(t["sel"]):
            dq = float(np.sqrt(np.mean((P[k] - data["X"][q]) ** 2)))
            dt = float(np.sqrt(((P[k] - TRAIN) ** 2).mean(1)).min())
            print(f"    {k:>4d} {src[s]:>7s} {us[s]:6.3f} {dq:9.4f} {dt:9.4f} "
                  f"{dq / max(spread, 1e-9):14.2f}")
        print()

    print("[origin] `from`=query on pick 0 with d(query) around one class-spread means the chain\n"
          "         started at a real instance of the composition and walked to a nearby mode: the\n"
          "         resemblance is the INITIALIZATION, and the slot still has 0 training rows.\n"
          "         `from`=random picks explore other basins, which is what makes them look\n"
          "         unrelated -- and RESULTS 5e-quater shows the composition survives on those\n"
          "         alone, so the clean-looking one is not what carries it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
