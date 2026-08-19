"""N78 -- the flagship: composition on a PRETRAINED backbone with a table big enough to mean something.

Everything the CelebA grid could not be, measured before it was trained (`tests/diag_shapes3d_readable.py`):

    400 classes against 25 free vectors        ratio 18.2   (the grid: 1.1)
    floor_hue 1.000, object_hue 1.000,
    shape 0.994 on REAL images                 ceiling 0.994 against chance 1/400
    12/12 held-out targets readable            (the grid: 16 of 36)

and the backbone is `eurecom-ds/scoresdeve-ema-shapes3d-64`, PRETRAINED -- so this is the regime a
VLA will actually be in, not a from-scratch density holdout that forces a small weak model.

`object_hue`'s long-standing blindness (0.239, RESULTS 5t-bis footnote) turned out to be a TABLE-SIZE
artifact: at `value_stride=2` it reads 0.236 and at stride 1 it reads 1.000. Halving the number of
values per hue was making the classes denser at the cost of making the hues unreadable, and nobody had
measured the trade because the readability check ran after training rather than before.

Arms are N73's, so the two datasets can be read side by side:

    address        compose(fac, TRUE factors) and sample the model's conditional
    nearest-class  the closest SEEN class embedding -- if this matches ADDRESS, retrieval explains it
    unconditional  the floor

    python experiments/n78_shapes3d_flagship.py --ckpt results/cache/ddpm_s3d_full.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                     # noqa: E402

from lifelong_ood.data.factored import decode_slot                    # noqa: E402
from lifelong_ood.data.shapes3d import load_shapes3d                  # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                memorization, precision_recall, save_json, vendi_score)
from lifelong_ood.instrument import check_instrument, to_data_space   # noqa: E402
from lifelong_ood.neural import sample_embedding                      # noqa: E402
from lifelong_ood.neural.idealwords import compose, cv_r2, factorize  # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field               # noqa: E402

ARMS = ("address", "nearest-class", "unconditional")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_s3d_full.pt"))
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-concepts", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, ck = load_ve_field(args.ckpt, device=dev)
    sp = ck.get("split", {})
    d = load_shapes3d(n_per_slot=int(sp.get("n_per_slot", 40)), seed=int(sp.get("seed", 0)),
                      value_stride=int(sp.get("value_stride", 1)),
                      fix_nuisance=sp.get("fix_nuisance", ["wall_hue"]), n_holdout=0)
    sizes = [int(s) for s in d["sizes"]]
    X, F, slot = (np.asarray(d["X"], np.float32), np.asarray(d["factors"], int),
                  np.asarray(d["slot"], int))

    remap = ck["remap"]
    seen_slots = np.array(sorted(int(k) for k in remap))
    # The split is DERIVED from the checkpoint and ASSERTED, never assumed -- the same discipline
    # n73 needed after a hardcoded n_holdout scored one backbone against another's held-out cells.
    d["seen"] = np.isin(slot, seen_slots)
    C = np.asarray(ck["ema_cemb"])[[remap[int(s)] for s in seen_slots]]
    lab = np.asarray(decode_slot(seen_slots, sizes), int)
    fac = factorize(C, lab)
    r2cv, n_cv = cv_r2(C, lab)
    probes = fit_factor_probes(d, seed=args.seed)
    print(f"[n78] {len(seen_slots)} seen classes of {int(np.prod(sizes))}, factors {d['names']} "
          f"{sizes}; basis R2 {fac['r2']:.3f} in-sample, {r2cv:.3f} leave-one-out", flush=True)

    held = [s for s in sorted(set(slot.tolist())) if s not in set(seen_slots.tolist())]
    held = [h for h in held if (slot == h).sum() >= 8][:args.n_concepts]
    gate = check_instrument(tag="n78", real=d, probes=probes, field=field,
                            targets={int(h): tuple(int(v) for v in F[slot == h][0]) for h in held},
                            n_arm=args.n_samples, table=C, labels=lab, seed=args.seed)
    print(gate.summary("n78"), flush=True)
    gate.require("n78")
    held = [h for h in held if gate.readable(int(h))]
    print(f"[n78] {len(held)} readable held-out combinations", flush=True)

    rows = []
    for ci, h in enumerate(held):
        idx = np.where(slot == int(h))[0]
        tgt = tuple(int(v) for v in F[idx[0]])
        R = X[idx[:60]]
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        got = {
            "address": to_data_space(field, sample_embedding(
                field, c_new, n=args.n_samples, steps=args.steps, seed=args.seed + ci)),
            "nearest-class": to_data_space(field, field.sample(
                n=args.n_samples, steps=args.steps, seed=args.seed + ci,
                branch=remap[int(seen_slots[j_near])])),
            "unconditional": to_data_space(field, field.sample(
                n=args.n_samples, steps=args.steps, seed=args.seed + 37 * (ci + 1), branch=None)),
        }
        train_ref = X[np.asarray(d["seen"], bool)][:600]
        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, R)
            j = float(a["joint"])
            rows.append(dict(arm=arm, slot=int(h), target=list(tgt),
                             joint=j if j == j else 0.0, recall=pr["recall"],
                             precision=pr["precision"], vendi=float(vendi_score(S)),
                             memorized=float(memorization(S, train_ref)["frac_memorized"])))
        r = {x["arm"]: x for x in rows if x["slot"] == int(h)}
        print(f"[n78] slot {h} {tgt}: address {r['address']['joint']:.3f} | nearest "
              f"{r['nearest-class']['joint']:.3f} | uncond {r['unconditional']['joint']:.3f} | "
              f"ceiling {gate.targets[int(h)].ceiling:.3f}", flush=True)

    def m(arm, k="joint"):
        v = [x[k] for x in rows if x["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    ceil = float(np.mean([gate.targets[int(h)].ceiling for h in held]))
    print(f"\n[n78] {'=' * 72}")
    print(f"{'arm':16s}{'joint':>9s}{'/ceiling':>10s}{'recall':>9s}{'vendi':>8s}{'memorized':>11s}")
    for arm in ARMS:
        print(f"{arm:16s}{m(arm):>9.3f}{m(arm) / max(ceil, 1e-9):>10.2f}"
              f"{m(arm, 'recall'):>9.3f}{m(arm, 'vendi'):>8.2f}{m(arm, 'memorized'):>11.2f}")
    print(f"{'-- REAL ceiling':16s}{ceil:>9.3f}")

    md = float(np.mean([g.min_detectable for g in gate.targets.values()]))
    a_, n_, u_ = m("address"), m("nearest-class"), m("unconditional")
    print(f"\n[n78] margins: over retrieval {a_ - n_:+.3f}, over the floor {a_ - u_:+.3f}; "
          f"the smallest margin n={args.n_samples} can resolve is {md:.3f}")
    print("[n78] VERDICT")
    if a_ - n_ > md and a_ - u_ > md:
        print("[n78]   COMPOSES, AND BEATS RETRIEVAL, on a PRETRAINED backbone -- both margins")
        print("[n78]   exceed what this sample count can resolve. This is what the CelebA grid")
        print("[n78]   could not test: a table at ratio 18.2 with every factor readable.")
    elif a_ - u_ > md:
        print(f"[n78]   Clears the floor ({a_ - u_:+.3f}) but not retrieval ({a_ - n_:+.3f}) by more")
        print(f"[n78]   than the {md:.3f} resolution. Same shape as 7ee, on a far better instrument.")
    else:
        print(f"[n78]   Does not clear the floor by more than the resolution ({a_ - u_:+.3f} vs "
              f"{md:.3f}).")
    save_json(dict(rows=rows, ceiling=ceil, instrument=gate.as_dict(), held=[int(h) for h in held],
                   basis_r2=float(fac["r2"]), basis_r2_cv=float(r2cv), n_cv=int(n_cv),
                   n_seen=int(len(seen_slots)), min_detectable=md, config=vars(args)),
              os.path.join(RESULTS, f"n78_shapes3d_flagship{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
