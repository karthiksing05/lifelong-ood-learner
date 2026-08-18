"""N73 -- the ColorMNIST test, on a CelebA backbone with a DENSITY holdout.

Every previous CelebA result used a backbone fine-tuned from `google/ddpm-celebahq-256`, so a
"held-out" combination was held out from the CONDITIONING while its images were already in the
pretrained density.  That is why mode ascent handed the query back at every noise level there, and
why `nearest-class` scores 0.390 on the (also pretrained) Shapes3D backbone against 0.000 on
ColorMNIST.  `ddpm_celeba_grid.pt` is trained FROM SCRATCH on the nine seen cells only: the three
held-out combinations have zero training images, so they are absent from the model.

The grid is hair(3) x Male(2) x Smiling(2) = 12 cells, its size fixed by measurement rather than
preference -- only four attributes are readable at 64px and two of them are mutually exclusive hair
colours (see `load_celeba_grid`).  Held out: Blond-Male-notSmiling, Brown-Male-Smiling,
other-Female-Smiling.

Arms are the ones that make the claim falsifiable, matching N72:

  address        compose(fac, TRUE factors) and sample the model's conditional
  nearest-class  the closest SEEN class embedding -- if this matches ADDRESS the model already had it
  unconditional  the floor

Read every number against the PROBE CEILING, not against 1.0: the factor probes score
hair 0.568 / Male 0.849 / Smiling 0.859 on REAL images, so a perfect generator caps near 0.414 joint
and chance is 1/9 = 0.111.  Reporting raw joint without that ceiling would understate a working
model and overstate a broken one.

    python experiments/n73_celeba_grid_compose.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.celeba import load_celeba_grid                  # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,    # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import NeuralScoreField, UNet, sample_embedding  # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize          # noqa: E402

ARMS = ("address", "nearest-class", "unconditional")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid.pt"))
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})
    d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                         n_per_cell=int(sp.get("n_per_slot", 3000)), n_holdout=3,
                         max_rows=sp.get("max_rows"))
    X, F, slot = (np.asarray(d["X"], np.float32), np.asarray(d["factors"]),
                  np.asarray(d["slot"]))
    names, sizes = d["names"], d["sizes"]
    probes = fit_factor_probes(d, seed=args.seed)
    ceil = float(np.prod([probes[f"{n}_val"] for n in names if f"{n}_val" in probes]))
    print(f"[n73] probe ceiling on REAL images ~ {ceil:.3f}; chance = "
          f"{1.0 / len(ck['remap']):.3f}", flush=True)

    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    lab = np.stack([F[slot == int(s)][0] for s in seen])
    fac = factorize(C, lab)
    print(f"[n73] ideal-words basis R^2 {fac['r2']:.3f} over {len(seen)} seen cells", flush=True)

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    rows = []
    for ci, h in enumerate(d["held"]):
        idx = np.where(slot == int(h))[0]
        if not len(idx):
            continue
        tgt = tuple(int(v) for v in F[idx[0]])
        R = X[idx[:60]]
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        got = {
            "address": sample_embedding(field, c_new, n=args.n_samples, steps=args.steps,
                                        seed=args.seed + ci),
            "nearest-class": field.sample(n=args.n_samples, steps=args.steps,
                                          seed=args.seed + ci, branch=remap[int(seen[j_near])]),
            "unconditional": field.sample(n=args.n_samples, steps=args.steps,
                                          seed=args.seed + 37 * (ci + 1), branch=None),
        }
        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, R)
            rows.append(dict(arm=arm, cell=int(h), target=list(tgt),
                             joint=float(a["joint"]) if a["joint"] == a["joint"] else 0.0,
                             recall=pr["recall"], precision=pr["precision"],
                             vendi=float(vendi_score(S))))
        r = {x["arm"]: x for x in rows if x["cell"] == int(h)}
        print(f"[n73] cell {h} {tgt}: address {r['address']['joint']:.3f} | "
              f"nearest-class {r['nearest-class']['joint']:.3f} | "
              f"uncond {r['unconditional']['joint']:.3f} | "
              f"address recall {r['address']['recall']:.3f}", flush=True)

    def m(arm, k):
        v = [x[k] for x in rows if x["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n[n73] {'=' * 66}")
    print(f"{'arm':16s}{'joint':>9s}{'/ceiling':>10s}{'recall':>9s}{'vendi':>8s}")
    for arm in ARMS:
        print(f"{arm:16s}{m(arm, 'joint'):>9.3f}{m(arm, 'joint') / max(ceil, 1e-9):>10.2f}"
              f"{m(arm, 'recall'):>9.3f}{m(arm, 'vendi'):>8.2f}")
    print("[n73] VERDICT")
    a_, n_, u_ = m("address", "joint"), m("nearest-class", "joint"), m("unconditional", "joint")
    if a_ > max(n_, u_) + 0.10:
        print("[n73]   COMPOSES A COMBINATION THE MODEL NEVER SAW. The held-out cells have zero")
        print("[n73]   training images, so this is a DENSITY holdout -- the first CelebA result")
        print("[n73]   here that is not merely a missing embedding row.")
    elif a_ > u_ + 0.10:
        print("[n73]   Beats the floor but not nearest-class: the composition is close enough to a")
        print("[n73]   seen cell that retrieval explains it. Report as such.")
    else:
        print("[n73]   Does not clear the floor. With 9 seen cells and a 0.414 probe ceiling this")
        print("[n73]   backbone may simply be too weak -- judge against the ceiling, not 1.0.")
    save_json(dict(rows=rows, ceiling=ceil, basis_r2=float(fac["r2"]), held=d["held"],
                   config=vars(args)),
              os.path.join(RESULTS, "n73_celeba_grid_compose.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
