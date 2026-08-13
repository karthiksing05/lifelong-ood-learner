"""Is the variational-bound codelength MEASURABLE, or is `mdl_route` reading its own noise?

`elbo_bits` is the principled accounting unit for the router -- the eps-matching part of the
diffusion variational bound, in bits -- and it replaced the simple loss for a good reason (the simple
loss is that bound with its per-timestep weights divided out, which under-counts evidence by orders
of magnitude).  But the weights are not benign:

    50% of the bound's weight lies below u = 0.636, and the five highest-weighted grid points are
    u = 0.98, 0.97, 0.02, 0.96, 0.03 -- w(0.02)/w(0.45) = 10x.

The mass sits at u -> 1, where x_t is very nearly pure noise, eps-prediction is close to the identity,
and the conditioning barely enters.  So the bound concentrates its weight exactly where the decision
has no signal, and multiplies whatever sampling noise is there by a large number.  That is a specific,
checkable failure mode, and this checks it before anything is concluded from N*.

Symptom already visible in the sweep: on several queries the bound and the simple loss disagree in
SIGN about whether ADDRESS codes better than the best trained conditioning (slot 15: simple loss says
ADDRESS is worse, 0.06945 against 0.06303, while the bound says it is better and N*_addr = 2.2).  One
of them is wrong.

Measures, for the same query and the same pair of conditionings:
  * the paired bound difference across independent SEEDS -- if its sd is comparable to its mean, N*
    is noise and the sweep means nothing
  * the same restricted to an informative band of u, which is the fallback if the full bound fails
  * both at n_noise 2 and 8, to separate "needs more draws" from "cannot be measured this way"

    python tests/diag_elbo_stability.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import backbone_gains                                       # noqa: E402

from lifelong_ood.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.evals import fit_probes                               # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_model               # noqa: E402
from lifelong_ood.neural.idealwords import factorize, slot_factors      # noqa: E402
from lifelong_ood.neural.router import (basis_assignments, batch_losses,  # noqa: E402
                                        elbo_bits)

CACHE = os.path.join(HERE, "results", "cache")
BANDS = {"full 0.02-0.98": (0.02, 0.98), "band 0.10-0.70": (0.10, 0.70),
         "band 0.15-0.50": (0.15, 0.50)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist_nobg3.pt"))
    p.add_argument("--held-bg", type=int, default=3)
    p.add_argument("--n-per-slot", type=int, default=40)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "diag_elbo_stability.json"))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_trained = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_trained]]
    field = load_field(args.ckpt, device=dev)
    fac = factorize(C, slot_factors(slots_trained))
    gains = backbone_gains(field, C, slots_trained, probes, data, seed=args.seed, steps=50, n=16)
    bc, bl = basis_assignments(fac, gains=gains, sizes=fac["sizes"])
    trained = set(int(s) for s in slots_trained)
    untr = np.array([(int(r[0]) * 16 + int(r[1]) * 4 + int(r[2])) not in trained for r in bl])

    # one query per population, chosen the same way the sweep does
    picks = [("in-distribution", 12), ("new-combination", 104), ("new-vocabulary", 83)]
    out = []
    for name, slot in picks:
        idx = np.where(data["slot"] == slot)[0]
        if not len(idx):
            print(f"[elbo] slot {slot} absent, skipping")
            continue
        X = [to_model(data["X"][int(idx[0])]).ravel()]
        Lc = batch_losses(field, X, C, u_grid=(0.15, 0.30, 0.45), n_noise=4, seed=args.seed)
        c_none = C[int(Lc.argmin())]
        Lb = batch_losses(field, X, bc[untr], u_grid=(0.15, 0.30, 0.45), n_noise=4, seed=args.seed)
        c_addr = bc[untr][int(Lb.argmin())]
        d_simple = float(Lc.min()) - float(Lb.min())     # >0 means ADDRESS codes better
        print(f"\n[elbo] {name} slot {slot}: simple-loss m_none {Lc.min():.5f} "
              f"m_addr {Lb.min():.5f}  (ADDRESS better by {d_simple:+.5f})", flush=True)

        for bname, (lo, hi) in BANDS.items():
            for nn in (2, 8):
                ds = []
                for s in range(args.seeds):
                    a = elbo_bits(field, X, c_none, n_t=96, n_noise=nn, seed=1000 + 37 * s,
                                  u_lo=lo, u_hi=hi)
                    b = elbo_bits(field, X, c_addr, n_t=96, n_noise=nn, seed=1000 + 37 * s,
                                  u_lo=lo, u_hi=hi)
                    ds.append(a - b)                      # >0 means ADDRESS codes better, in bits
                ds = np.asarray(ds, float)
                mean, sd = float(ds.mean()), float(ds.std(ddof=1))
                # the quantity that decides everything: is the sign stable, and is |mean| >> sd?
                agree = float(np.mean(np.sign(ds) == np.sign(d_simple)))
                out.append(dict(pop=name, slot=slot, band=bname, n_noise=nn, mean=mean, sd=sd,
                                snr=abs(mean) / sd if sd > 0 else float("inf"),
                                sign_agrees_with_simple=agree, d_simple=d_simple))
                print(f"[elbo]   {bname:16s} n_noise {nn}: delta = {mean:+9.1f} +/- {sd:7.1f} bits"
                      f"   |mean|/sd {abs(mean)/sd if sd else float('inf'):6.2f}"
                      f"   sign agrees with simple loss on {agree * 100:3.0f}% of seeds", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(rows=out, args=vars(args)), open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\n[elbo] wrote {args.out}")

    print("\n[elbo] VERDICT")
    for bname in BANDS:
        r = [x for x in out if x["band"] == bname and x["n_noise"] == 8]
        if r:
            print(f"[elbo]   {bname:16s} median |mean|/sd {np.median([x['snr'] for x in r]):5.2f}"
                  f"   sign agreement {np.mean([x['sign_agrees_with_simple'] for x in r]):.2f}")
    print("[elbo]   a band is usable only if |mean|/sd is comfortably above 1 -- otherwise N* is "
          "reading sampling noise and no conclusion from it survives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
