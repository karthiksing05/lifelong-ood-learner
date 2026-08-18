"""Re-score the trained celeba_grid checkpoint through the CORRECT layout.

`n1_train.py` trained with `--layout hwc` and then built its eval field without passing the layout,
so sampling used the `permuted` default.  The samples came out as tiled mosaics and the run reported
conditional joint accuracy 0.111 -- exactly 1/9, chance over its nine seen classes -- while the
checkpoint itself was never examined.

This re-samples the SAME checkpoint through a field built with the layout it was trained in, and
scores both ways, so the difference is attributable to the reader rather than to the model.  Nothing
is retrained.

    python tests/diag_grid_layout.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.celeba import load_celeba_grid                  # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,    # noqa: E402
                                save_grid)
from lifelong_ood.neural import NeuralScoreField, UNet                 # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = os.path.join(CACHE, "ddpm_celeba_grid.pt")
    if not os.path.exists(p):
        print(f"SKIP: no checkpoint at {p}")
        return 0
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})
    print(f"[gl] checkpoint layout={ck.get('layout')!r}, {ck['n_classes']} seen classes", flush=True)
    d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                         n_per_cell=int(sp.get("n_per_slot", 3000)), n_holdout=3,
                         max_rows=sp.get("max_rows"))
    probes = fit_factor_probes(d, seed=0)
    names, sizes = d["names"], d["sizes"]
    remap = ck["remap"]
    uniq = np.array(sorted(int(k) for k in remap))
    F, slot = np.asarray(d["factors"]), np.asarray(d["slot"])

    for lay in ("permuted", "hwc"):
        field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                                 layout=lay)
        field.model.load_state_dict(ck["ema"])
        field.model.eval()
        hits = []
        for s in uniq[:6]:
            tgt = tuple(int(v) for v in F[slot == int(s)][0])
            S = np.asarray(field.sample(n=8, steps=100, seed=0, branch=remap[int(s)]), np.float32)
            a = factor_accuracy(probes, S, tgt)
            hits.append(a["joint"] if a["joint"] == a["joint"] else 0.0)
        m = float(np.mean(hits))
        tag = "  <- what the checkpoint records" if lay == ck.get("layout") else ""
        print(f"[gl] sampled as {lay:9s}: conditional joint {m:.3f}{tag}", flush=True)
        if lay == ck.get("layout"):
            S = np.asarray(field.sample(n=16, steps=100, seed=1, branch=None), np.float32)
            o = os.path.join(RESULTS, "n1_uncond_celeba_grid_fixed.png")
            save_grid(S, o, title="celeba_grid unconditional, sampled in the TRAINED layout")
            print(f"[gl] wrote {o}", flush=True)
    print("\n[gl] chance for this grid is 1/9 = 0.111 over the seen classes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
