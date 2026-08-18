"""Are the celeba_grid case-study prototypes the query handed back, or the model's own modes?

The pretrained CelebA and Shapes3D backbones return the query at every noise level, because they saw
every combination during pretraining and the query is therefore a high-density point.  This backbone
was trained from scratch on the seen cells only, so a held-out query should NOT be a mode -- and the
figure draws query-agnostic chains besides.  Both claims are checked here rather than eyeballed.

Scale bars: distance from the query to REAL images of its own cell (what "as far as a genuine other
image of this class" means), and the spread among the prototypes themselves (collapse detector).

    python tests/diag_grid_protos.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                              # noqa: E402

from lifelong_ood.data.celeba import load_celeba_grid                 # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet                # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = os.path.join(CACHE, "ddpm_celeba_grid.pt")
    if not os.path.exists(p):
        print(f"SKIP: no checkpoint at {p}")
        return 0
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})
    d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                         n_per_cell=int(sp.get("n_per_slot", 3000)), n_holdout=3,
                         max_rows=sp.get("max_rows"))
    X, slot = np.asarray(d["X"], np.float32), np.asarray(d["slot"])
    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    def pw(A):
        return float(np.mean([np.linalg.norm(A[i] - A[j])
                              for i in range(len(A)) for j in range(i + 1, len(A))]))

    print(f"{'cell':>6} {'init':>7} {'d(proto,query)':>15} {'proto spread':>13} "
          f"{'d(real,query)':>14}")
    for h in d["held"]:
        idx = np.where(slot == int(h))[0]
        if len(idx) < 6:
            continue
        xq = X[int(idx[0])]
        R = X[idx[1:7]]
        d_real = float(np.mean(np.linalg.norm(R - xq[None], axis=1)))
        for init in ("query", "random"):
            _, x0 = field.mode_ascent(xq, 0.30, n_starts=4, iters=120, lr=0.05, seed=0, init=init)
            M = np.asarray(x0)
            dq = float(np.mean(np.linalg.norm(M - xq[None], axis=1)))
            print(f"{int(h):>6} {init:>7} {dq:>15.2f} {pw(M):>13.2f} {d_real:>14.2f}")
    print("\n[gp] A prototype that IS the query sits near 0 and far below d(real,query).")
    print("[gp] A COLLAPSED row has proto spread << the spread among real images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
