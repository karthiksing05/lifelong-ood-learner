"""N52 -- the whole claim on a THIRD dataset, using the contract instead of a port.

`factored.py` exists so that adding a dataset is a loader rather than a refactor.  Shapes3D is the
test of that assertion: this file runs the two things the architecture actually claims -- that a
factor basis exists in the backbone's conditioning space, and that a held-out COMBINATION can be
served from it at ADDRESS cost -- without touching `compose.py`, `idealwords.py` or the router.

Nothing here is ColorMNIST-shaped.  Factor values come from `data["factors"]`, composition ids from
`factored.decode_slot`, and probes from `evals.fit_factor_probes`, all of which are generic.

Three arms on the held-out compositions:

    ours (ADDRESS)   compose the target's factor values through the fitted basis -> 72 bytes
    nearest-class    the single closest TRAINED class embedding -- the retrieval floor, which cannot
                     hold a combination that was never trained
    unconditional    no class signal at all -- the floor that says whether conditioning did anything

plus the router signal `s_novel` on held-out vs trained queries, the same statistic 6c uses.

**The probes are checked before anything is scored with them.**  RESULTS 5e-sexies is a retracted
negative that was entirely a probe artifact; `probe_readable` filters factors whose probe cannot read
its own factor, and the count is printed.

    python experiments/n52_shapes3d.py --n-cases 24
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,  # noqa: E402
                                precision_recall, probe_readable, save_json)
from lifelong_ood.factored import decode_slot, load_factored          # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data, to_model     # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize,        # noqa: E402
                                            sample_factored_pool)
from lifelong_ood.neural.router import auroc, route_signals            # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")
RESULTS = os.path.join(HERE, "results")


def calibrate_gains(field, fac, sizes, probes, seed=0, steps=50, n=6,
                    grid=(1.0, 1.5, 2.0, 2.5), slots_seen=None):
    """Per-factor gain chosen on SEEN compositions only.  -> list[float].

    The backbone under-responds to some factors (5q: ColorMNIST background vectors have mean norm
    0.55 against 0.79 for digit), so composing at unit gain reads that factor weakly.  The gain is
    calibrated by asking which value makes SEEN compositions score best -- seen labels are training
    data, so no held-out label is touched.
    """
    gains = [1.0] * len(sizes)
    rng = np.random.default_rng(seed)
    probe_slots = rng.permutation(slots_seen)[:n]
    for f in range(len(sizes)):
        best, best_g = -1.0, 1.0
        for g in grid:
            trial = list(gains)
            trial[f] = float(g)
            acc = []
            for s in probe_slots:
                tgt = decode_slot(np.array([int(s)]), sizes)[0]
                c = compose(fac, tuple(int(v) for v in tgt), gains=trial)
                S = to_data(sample_factored_pool(
                    field, torch.as_tensor(c, dtype=torch.float32, device=field.device),
                    n=4, steps=steps, seed=seed))
                a = factor_accuracy(probes, S, tgt)
                if a["joint"] == a["joint"]:
                    acc.append(a["joint"])
            m = float(np.mean(acc)) if acc else 0.0
            if m > best:
                best, best_g = m, float(g)
        gains[f] = best_g
        print(f"[n52]   gain[{f}] ({probes['_names'][f]}) -> {best_g:g}  (seen joint {best:.3f})",
              flush=True)
    return gains


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_shapes3d.pt"))
    p.add_argument("--n-per-slot", type=int, default=40)
    p.add_argument("--n-cases", type=int, default=24)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-gains", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_factored("shapes3d", n_per_slot=args.n_per_slot, seed=args.seed)
    sizes, names = data["sizes"], data["names"]
    print(f"[n52] shapes3d: {len(np.unique(data['slot']))} compositions over {names} "
          f"sizes {sizes}; {int(data['seen'].sum())} seen / {int((~data['seen']).sum())} held-out "
          f"images", flush=True)

    probes = fit_factor_probes(data, seed=args.seed, epochs=args.epochs)
    readable = [nm for nm in names if probe_readable(probes, nm)]
    for nm in names:
        print(f"[n52]   probe {nm:12s} val {probes[f'{nm}_val']:.3f} "
              f"(base {probes[f'{nm}_base']:.3f})  "
              f"{'READABLE' if probe_readable(probes, nm) else 'unusable -> excluded'}", flush=True)
    if not readable:
        print("[n52] no readable probes; nothing can be scored. Stop.")
        return 1

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_seen]]
    field = load_field(args.ckpt, device=dev)

    fac = factorize(C, decode_slot(slots_seen, sizes))
    print(f"[n52] factor basis over {len(slots_seen)} trained compositions: "
          f"R^2 {fac['r2']:.3f}", flush=True)

    gains = ([1.0] * len(sizes) if args.skip_gains else
             calibrate_gains(field, fac, sizes, probes, seed=args.seed, steps=args.steps,
                             slots_seen=slots_seen))
    print(f"[n52] gains {gains}", flush=True)

    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"]))
                           - set(int(s) for s in slots_seen)))
    rng = np.random.default_rng(args.seed)
    cases = rng.permutation(held)[:args.n_cases]
    print(f"[n52] scoring {len(cases)} held-out compositions "
          f"({len(readable)}/{len(names)} readable probes)\n", flush=True)

    rows, sig_held, sig_seen = [], [], []
    for ci, slot in enumerate(cases):
        tgt = decode_slot(np.array([int(slot)]), sizes)[0]
        idx = np.where(data["slot"] == int(slot))[0]
        R = np.stack([to_model(data["X"][i]).ravel() for i in idx[:24]])

        c_addr = compose(fac, tuple(int(v) for v in tgt), gains=gains)
        arms = {"ours (ADDRESS)": to_data(sample_factored_pool(
            field, torch.as_tensor(c_addr, dtype=torch.float32, device=dev),
            n=args.n_samples, steps=args.steps, seed=args.seed))}
        d2 = ((C - c_addr[None]) ** 2).sum(1)
        arms["nearest-class"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1),
            branch=int(np.argmin(d2))))
        arms["unconditional"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed + 7919 * (ci + 1), clip=(-1, 1)))

        for nm, S in arms.items():
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, R)
            rows.append(dict(slot=int(slot), arm=nm, joint=a["joint"], recall=pr["recall"],
                             precision=pr["precision"],
                             **{k: v for k, v in a.items() if k in names}))
            print(f"[n52] {ci + 1:3d}/{len(cases)} slot {int(slot):4d} {nm:16s} "
                  f"joint {a['joint']:.3f} | "
                  + " ".join(f"{k} {a[k]:.2f}" for k in names if k in a)
                  + f" | recall {pr['recall']:.3f}", flush=True)

        sig_held.append(route_signals(field, R[0], C, n_noise=2, seed=args.seed)["s_novel"])

    for s in rng.permutation(slots_seen)[:len(cases)]:
        i = int(np.where(data["slot"] == int(s))[0][0])
        sig_seen.append(route_signals(field, to_model(data["X"][i]).ravel(), C,
                                      n_noise=2, seed=args.seed)["s_novel"])

    print("\n[n52] " + "=" * 72)
    summary = []
    for nm in ("ours (ADDRESS)", "nearest-class", "unconditional"):
        R_ = [r for r in rows if r["arm"] == nm]
        if not R_:
            continue
        keys = ["joint", "recall", "precision"] + [n for n in names if n in R_[0]]
        m = {k: float(np.nanmean([r[k] for r in R_])) for k in keys}
        se = float(np.nanstd([r["joint"] for r in R_]) / max(len(R_) ** 0.5, 1))
        summary.append(dict(arm=nm, n=len(R_), se_joint=se, **m))
        print(f"[n52] {nm:16s} joint {m['joint']:.3f} +/- {se:.3f} | "
              + " ".join(f"{k} {m[k]:.3f}" for k in names if k in m)
              + f" | recall {m['recall']:.3f} precision {m['precision']:.3f}")

    a = auroc(sig_held, sig_seen)
    print(f"\n[n52] router s_novel: held-out {np.mean(sig_held):.5f} vs trained "
          f"{np.mean(sig_seen):.5f}  AUROC {a:.3f}   [ColorMNIST 6c: 0.980]")

    o = os.path.join(RESULTS, f"n52_shapes3d{args.tag}.json")
    save_json(dict(rows=rows, summary=summary, basis_r2=float(fac["r2"]), gains=gains,
                   novel_auroc=float(a), readable=readable, names=names, sizes=sizes,
                   config=vars(args)), o, seed=args.seed)
    print(f"[n52] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
