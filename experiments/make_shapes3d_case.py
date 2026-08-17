"""F35 -- the Shapes3D case study: what the method produces for a held-out combination.

The ColorMNIST (f9) and CelebA (f26) analogue, on the third dataset, and built to be VERIFIED rather
than admired.  Every block is one held-out composition and every block says which one:

    real           the withheld combination itself, NAMED -- floor hue / object hue / shape -- with
                   the number of training images it had (0 by construction; shown so the split is
                   checkable at a glance rather than trusted)
    ours (ADDRESS) the composition rebuilt from the factor basis, 72 bytes
    nearest-class  the closest TRAINED composition -- the retrieval floor, which by definition
                   cannot hold a combination that was never trained
    unconditional  the null token: same weights, no label, the floor that says whether conditioning
                   did anything at all

Per-panel accuracy comes from `evals.fit_factor_probes`, and factors whose probe cannot read their
own factor are EXCLUDED and counted in the caption (the RESULTS 5e-sexies lesson -- a retracted
negative that was entirely a probe artefact).

    python experiments/make_shapes3d_case.py --ckpt results/cache/ddpm_shapes3d_cond.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                probe_readable)
from lifelong_ood.data.factored import decode_slot, load_factored          # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data, to_model    # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize,       # noqa: E402
                                            sample_factored_pool)

CACHE = os.path.join(HERE, "results", "cache")
FIG = os.path.join(HERE, "figures", "initial")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_shapes3d_cond.pt"))
    p.add_argument("--fallback", default=os.path.join(CACHE, "ddpm_shapes3d.pt"))
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-slot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.ckpt if os.path.exists(args.ckpt) else args.fallback
    print(f"[f35] backbone {os.path.basename(ckpt)}"
          + ("" if ckpt == args.ckpt else "  (conditioned fine-tune absent, using stage-1)"),
          flush=True)

    data = load_factored("shapes3d", n_per_slot=args.n_per_slot, seed=args.seed,
                         value_stride=2, n_holdout=25)
    sizes, names = data["sizes"], data["names"]
    probes = fit_factor_probes(data, seed=args.seed, epochs=30)
    readable = [n for n in names if probe_readable(probes, n)]
    print(f"[f35] probes readable: {readable} of {names}", flush=True)

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_seen]]
    field = load_field(ckpt, device=dev)
    fac = factorize(C, decode_slot(slots_seen, sizes))
    print(f"[f35] basis R^2 {fac['r2']:.3f} over {len(slots_seen)} trained compositions", flush=True)

    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"]))
                           - set(int(s) for s in slots_seen)))
    rng = np.random.default_rng(args.seed)
    cases = rng.permutation(held)[:args.n_cases]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["real", "ours (ADDRESS)", "nearest-class", "unconditional"]
    fig, axes = plt.subplots(len(cases) * len(order), args.n_show,
                             figsize=(1.5 * args.n_show + 4.2, 1.5 * len(cases) * len(order)))
    axes = np.atleast_2d(axes)
    r = 0
    for ci, slot in enumerate(cases):
        tgt = decode_slot(np.array([int(slot)]), sizes)[0]
        idx = np.where(data["slot"] == int(slot))[0]
        n_train = int(np.sum((data["slot"] == int(slot)) & data["seen"]))
        real = np.stack([to_model(data["X"][i]).ravel() for i in idx[:args.n_show]])
        c_addr = compose(fac, tuple(int(v) for v in tgt))
        d2 = ((C - c_addr[None]) ** 2).sum(1)
        got = {
            "real": real,
            "ours (ADDRESS)": to_data(sample_factored_pool(
                field, torch.as_tensor(c_addr, dtype=torch.float32, device=dev),
                n=args.n_show, steps=args.steps, seed=args.seed)),
            "nearest-class": to_data(field.sample(n=args.n_show, steps=args.steps,
                                                  seed=args.seed, clip=(-1, 1),
                                                  branch=int(np.argmin(d2)))),
            "unconditional": to_data(field.sample(n=args.n_show, steps=args.steps,
                                                  seed=args.seed + 7919 * (ci + 1), clip=(-1, 1))),
        }
        combo = ", ".join(f"{n}={int(v)}" for n, v in zip(names, tgt))
        for nm in order:
            S = np.asarray(got[nm], np.float32)
            a = factor_accuracy(probes, S, tgt)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(64, 64, 3), 0, 1))
            lab = (f"real - HELD OUT\n{combo}\nslot {int(slot)} - {n_train} training images"
                   if nm == "real" else nm)
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp in axes[r][0].spines.values():
                sp.set_visible(False)
            axes[r][args.n_show - 1].set_title(
                f"joint {a['joint']:.2f} | " + " ".join(f"{n[:5]} {a[n]:.2f}"
                                                        for n in names if n in a),
                fontsize=6.5, loc="right")
            r += 1
            print(f"[f35] slot {int(slot):4d} {nm:16s} joint {a['joint']:.3f}", flush=True)

    fig.suptitle(
        f"Shapes3D case study - one HELD-OUT composition per block, named on the `real` row so the "
        f"split is verifiable.\nbackbone {os.path.basename(ckpt)}; basis R^2 {fac['r2']:.3f}; "
        f"{len(readable)}/{len(names)} probes readable ({', '.join(readable)}).  "
        f"`nearest-class` is the retrieval floor - the closest TRAINED composition, which cannot "
        f"hold a combination never trained;\n`unconditional` is the same weights through the null "
        f"token.  Composition factors are (floor_hue, object_hue, shape); wall hue, scale and "
        f"orientation are NUISANCE and vary within every block.", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(FIG, f"f35_case_shapes3d{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[f35] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
