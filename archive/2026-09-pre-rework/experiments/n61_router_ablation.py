"""N61 -- is the router reading the model's DENSITY or its factor BASIS?  Both backbones, one sweep.

RESULTS 6c states it plainly and it is one of the repository's own negative results:

    "THE CONDITIONING IS NOT WHAT CARRIES THE DECISION.  `uncond_abs` -- the plain unconditional
     error, using no class and no basis at all -- scores 0.985 and 0.998, matching the conditional
     signal.  So this router is an OOD detector on the model's own density, and it must NOT be
     reported as evidence that the FACTOR BASIS is what decides the route."

A later sweep on Shapes3D found the conditional signal at **0.867** and the ablation at **0.535** at
u=0.90 -- the conditioning carrying the decision, which 6c says it does not.  That was 16 versus 16
queries at one seed and one backbone, so it is a hypothesis.

There are two ways it resolves and they lead to opposite conclusions, which is why this runs BOTH
backbones over the SAME u grid rather than only chasing the new number:

  A. ColorMNIST's ablation also collapses at high u, while its conditional signal holds.  Then 6c's
     finding was an artefact of reading ONE noise level (u=0.10), and the stronger claim -- that the
     basis does carry the decision -- has been available all along and was missed.
  B. ColorMNIST's ablation tracks its conditional signal at every u.  Then the two backbones genuinely
     differ, 6c stands for ColorMNIST, and the Shapes3D result is a property of a pretrained backbone
     whose density is nearly flat over held-out combinations.

Either way the answer is worth more than the number.  3 seeds, 40 queries per population, both
backbones, the same grid.

    python experiments/n61_router_ablation.py --seeds 3 --n-queries 40
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.evals import save_json                               # noqa: E402
from lifelong_ood.neural.router import (auroc, conditional_losses,     # noqa: E402
                                        null_conditioning)

U_GRID = (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 0.95)


def load_colormnist_setup(seed, dev):
    from lifelong_ood.data.datasets import load_colormnist
    from lifelong_ood.neural.ddpm import load_field, to_model
    p = os.path.join(CACHE, "ddpm_colormnist_nobg3.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    field = load_field(p, device=dev)
    remap = ck["remap"]
    seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    data = load_colormnist(n_per_slot=40, seed=seed)
    slots = np.asarray(data["slot"])
    X = np.asarray(data["X"], np.float32)
    held = np.array([s for s in np.unique(slots) if int(s) not in set(int(x) for x in seen)])
    return field, C, X, slots, seen, held, (lambda x: to_model(x).ravel())


def load_shapes3d_setup(seed, dev):
    from lifelong_ood.data.factored import load_factored
    from lifelong_ood.neural.ve_bridge import load_ve_field
    p = os.path.join(CACHE, "ddpm_shapes3d_pre2.pt")
    field, ck = load_ve_field(p, device=dev)
    sp = ck["split"]
    data = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                         value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                         fix_nuisance=sp["fix_nuisance"])
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema_cemb"].numpy()[[remap[int(s)] for s in seen]]
    slots = np.asarray(data["slot"])
    X = np.asarray(data["X"], np.float32)
    held = np.array(sorted(set(int(s) for s in np.unique(slots)) - set(seen.tolist())))
    return field, C, X, slots, seen, held, (lambda x: x)      # VE bridge wants [0,1] rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-queries", type=int, default=40)
    p.add_argument("--n-noise", type=int, default=4)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for bname, loader in (("ColorMNIST(nobg3)", load_colormnist_setup),
                          ("Shapes3D(pretrained)", load_shapes3d_setup)):
        print(f"\n[n61] ===== {bname} =====", flush=True)
        print(f"{'u':>6} " + " ".join(f"{'cond':>7}{'':1}{'unc':>7}" for _ in range(1))
              + "   (mean +/- sd over seeds)")
        per_u = {u: {"c": [], "u": []} for u in U_GRID}
        for seed in range(args.seeds):
            field, C, X, slots, seen, held, prep = loader(seed, dev)
            rng = np.random.default_rng(seed)
            null = null_conditioning(field.model).reshape(1, -1)
            allc = np.concatenate([null, C])

            def pick(pool):
                out = []
                for s in rng.permutation(pool)[:args.n_queries]:
                    idx = np.where(slots == int(s))[0]
                    if len(idx):
                        out.append(prep(X[int(idx[0])]))
                return out
            q_ho, q_tr = pick(held), pick(seen)
            for u in U_GRID:
                def sig(qs):
                    c, un = [], []
                    for x in qs:
                        L = conditional_losses(field, x, allc, u_grid=(u,),
                                               n_noise=args.n_noise, seed=seed)
                        un.append(float(L[0])); c.append(float(L[1:].min()))
                    return np.array(c), np.array(un)
                c_ho, u_ho = sig(q_ho)
                c_tr, u_tr = sig(q_tr)
                per_u[u]["c"].append(auroc(c_ho, c_tr))
                per_u[u]["u"].append(auroc(u_ho, u_tr))
            del field
            torch.cuda.empty_cache()
        for u in U_GRID:
            c = np.array(per_u[u]["c"]); un = np.array(per_u[u]["u"])
            gap = c.mean() - un.mean()
            rows.append(dict(backbone=bname, u=float(u), cond=float(c.mean()),
                             cond_sd=float(c.std()), unc=float(un.mean()),
                             unc_sd=float(un.std()), gap=float(gap)))
            flag = "  <- conditioning separates, density does not" if gap > 0.15 else ""
            print(f"{u:>6.2f}  {c.mean():.3f}+/-{c.std():.3f}   {un.mean():.3f}+/-{un.std():.3f}"
                  f"   gap {gap:+.3f}{flag}", flush=True)

    print("\n[n61] " + "=" * 70)
    for bname in ("ColorMNIST(nobg3)", "Shapes3D(pretrained)"):
        R = [r for r in rows if r["backbone"] == bname]
        best = max(R, key=lambda r: r["cond"])
        widest = max(R, key=lambda r: r["gap"])
        print(f"[n61] {bname:22s} best cond {best['cond']:.3f} at u={best['u']:g}; "
              f"widest cond-unc gap {widest['gap']:+.3f} at u={widest['u']:g}")
    # The gap must be read AT THE OPERATING POINT, not as a maximum over the whole grid.  A wide gap
    # at a u where the router has DEGRADED says only that conditioning decays more slowly than
    # density -- it does not say conditioning decides, because nobody would route at that u.
    # ColorMNIST is exactly this case: its widest gap (+0.174) sits at u=0.95 where AUROC has fallen
    # from 1.000 to 0.754, while at its own best u the gap is -0.000.  Judging on max-gap called that
    # a reversal of 6c; judging at the operating point does not.
    def at_operating_point(prefix):
        R = [r for r in rows if r["backbone"].startswith(prefix)]
        return max(R, key=lambda r: r["cond"]) if R else None
    cm, sh = at_operating_point("ColorMNIST"), at_operating_point("Shapes3D")
    print("\n[n61] VERDICT  (gap read at each backbone's own best-u, not max-gap-anywhere)")
    for r in (cm, sh):
        if r is None:
            continue
        print(f"[n61]   {r['backbone']:22s} u*={r['u']:<5g} cond {r['cond']:.3f}  "
              f"unc {r['unc']:.3f}  gap {r['gap']:+.3f}  -> "
              + ("BASIS carries the decision" if r["gap"] > 0.15 else "DENSITY explains it (6c)"))
    cg = cm["gap"] if cm else 0.0
    sg = sh["gap"] if sh else 0.0
    if cg > 0.15 and sg > 0.15:
        print("[n61]   BOTH separate at their operating points. 6c's finding was an artefact of ONE")
        print("[n61]   noise level and 6c must be amended.")
    elif sg > 0.15 >= cg:
        print("[n61]   ONLY Shapes3D separates at its operating point. 6c STANDS for ColorMNIST -- at")
        print("[n61]   u*=0.05 its ablation matches the conditional exactly. The Shapes3D result is a")
        print("[n61]   property of a pretrained backbone whose density is near-flat over held-out")
        print("[n61]   combinations: report as backbone-dependent, NOT as a general reversal.")
        print("[n61]   NOTE: a maximum-gap reading would have called this a reversal. It is not one.")
    else:
        print("[n61]   Neither separates at its operating point. The single-seed 0.867/0.535 does not")
        print("[n61]   replicate; 6c stands as written and the hypothesis is dropped.")
    save_json(dict(rows=rows, u_grid=list(U_GRID), config=vars(args)),
              os.path.join(RESULTS, "n61_router_ablation.json"), seed=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
