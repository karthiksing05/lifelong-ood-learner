"""Did the Shapes3D backbone learn class conditioning AT ALL?  The control before any comparison.

N52's first run reported basis R^2 **0.424** (ColorMNIST 0.870, CelebA 0.620) and, more damningly,
**seen-slot joint 0.000** during gain calibration -- compositions the model was TRAINED on could not
be generated from their own composed conditioning.  A comparison whose ceiling sits at its floor
cannot rank anything (the lesson of RESULTS 5e-sexies), so nothing from that run is reportable until
this passes.

n1_train's end-of-training sanity report would have caught it, and did not run: it called ColorMNIST's
`fit_probes` and raised KeyError AFTER the 36-minute run had written the checkpoint.  That is fixed;
this file is the standalone version.

Three conditionings, on slots the model was TRAINED on, so every one of them SHOULD work:

    native      the slot's own trained embedding row  -> if this fails, the backbone never learned
                conditioning and the basis is irrelevant
    composed    the same slot rebuilt from the fitted factor basis -> isolates basis quality
    uncond      the null row -> the floor

    python tests/diag_shapes3d_control.py
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.evals import factor_accuracy, fit_factor_probes, probe_readable  # noqa: E402
from lifelong_ood.factored import decode_slot, load_factored               # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data                   # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize,            # noqa: E402
                                            sample_factored_pool)
from lifelong_ood.neural.router import null_conditioning                   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "results", "cache", "ddpm_shapes3d.pt"))
    p.add_argument("--n-slots", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    p.add_argument("--n-per-slot", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_factored("shapes3d", n_per_slot=args.n_per_slot, seed=args.seed)
    sizes, names = data["sizes"], data["names"]
    probes = fit_factor_probes(data, seed=args.seed, epochs=30)
    print("[ctl] probes: " + "  ".join(
        f"{nm} {probes[f'{nm}_val']:.3f}{'' if probe_readable(probes, nm) else ' UNUSABLE'}"
        for nm in names), flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_seen]]
    field = load_field(args.ckpt, device=dev)
    fac = factorize(C, decode_slot(slots_seen, sizes))
    print(f"[ctl] basis R^2 {fac['r2']:.3f}   [ColorMNIST 0.870, CelebA 0.620]", flush=True)
    print(f"[ctl] embedding norms: mean {np.linalg.norm(C, axis=1).mean():.3f} "
          f"sd {np.linalg.norm(C, axis=1).std():.3f}; "
          f"mean pairwise cos {float(np.mean((C / np.linalg.norm(C, axis=1, keepdims=True)) @ (C / np.linalg.norm(C, axis=1, keepdims=True)).T)):.3f}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    picks = rng.permutation(slots_seen)[:args.n_slots]
    null = null_conditioning(field.model)

    print(f"\n[ctl] {len(picks)} TRAINED slots -- every arm below should work; these are not held out")
    hdr = f"{'guidance':>9} {'arm':>10} {'joint':>8} " + " ".join(f"{n:>11}" for n in names)
    print(hdr + "\n" + "-" * len(hdr))
    best = None
    for g in args.guidance:
        agg = {k: [] for k in ("native", "composed", "uncond")}
        for s in picks:
            tgt = decode_slot(np.array([int(s)]), sizes)[0]
            S = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                     clip=(-1, 1), branch=remap[int(s)]))
            agg["native"].append(factor_accuracy(probes, S, tgt))
            c = compose(fac, tuple(int(v) for v in tgt))
            S = to_data(sample_factored_pool(
                field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                n=args.n_samples, steps=args.steps, seed=args.seed, guidance=g))
            agg["composed"].append(factor_accuracy(probes, S, tgt))
            S = to_data(sample_factored_pool(
                field, torch.as_tensor(null, dtype=torch.float32, device=dev),
                n=args.n_samples, steps=args.steps, seed=args.seed, guidance=g))
            agg["uncond"].append(factor_accuracy(probes, S, tgt))
        for arm in ("native", "composed", "uncond"):
            m = {k: float(np.nanmean([a.get(k, np.nan) for a in agg[arm]]))
                 for k in ["joint"] + names}
            print(f"{g:>9.1f} {arm:>10} {m['joint']:>8.3f} "
                  + " ".join(f"{m[n]:>11.3f}" for n in names), flush=True)
            if arm == "native" and (best is None or m["joint"] > best[1]):
                best = (g, m["joint"])

    print(f"\n[ctl] VERDICT")
    print(f"[ctl]   if `native` is near zero, the backbone never learned class conditioning and the")
    print(f"[ctl]   basis, the gains and every composition number on this checkpoint are moot.")
    print(f"[ctl]   best native joint {best[1]:.3f} at guidance {best[0]:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
