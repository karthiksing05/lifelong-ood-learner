"""N30 — controlling how far an exemplar sits from its prototype, for BOTH methods.

A composition method produces two things a user might want independently: a *prototype* (what the
combination looks like, canonically) and *exemplars* (instances of it that differ from each other).
This repository has so far reported only the far end of that axis, and the companion paper only the
near end -- which is exactly why their samples score high accuracy at recall 0.001 and ours score
lower accuracy with real coverage. Neither is a property of the method; both are a choice of where to
sit on one dial, and this experiment makes the dial explicit and measures its whole range.

**The dial.** Start the reverse process from a *partially noised prototype* rather than from pure
noise, and run from that noise level:

    x_init = sqrt(abar(s)) * prototype + sqrt(1 - abar(s)) * noise,   integrate from u = s

  * `s -> 0`  almost no noise is added: the prototype survives nearly intact, and the samples are
              near-copies of it.
  * `s -> 1`  the prototype is destroyed: the samples are unconstrained draws from the method's own
              conditional.

The same axis applies to both methods, which is what makes them comparable on it:

  `ours`         the prototype is the composed concept's teacher mean; the conditional is the
                 consolidated adapter + its learned embedding.
  `poe-guided`   the prototype is the analytic PoE mean mu_T; the conditional is CFG toward that
                 teacher, in the companion paper's own Eq. 14 configuration.

Reported at each setting: distance to the prototype, diversity, composition accuracy, and coverage --
so the trade is a measured curve rather than a claim.

    python experiments/n30_prototype_distance.py --n-cases 2
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries               # noqa: E402

from lifelong_ood.datasets import load_colormnist, rgb_img, slot_name        # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,            # noqa: E402
                                precision_recall, save_json, speckle, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,          # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                      # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
# the companion paper's own configuration for its teacher (RESULTS 5l-bis)
PAPER_POE = dict(schedule="variance", var_scale=1.0, w0=1.2, wmax=2.0, wmin=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.2, 0.4, 0.6, 0.8, 1.0],
                   help="noise level the reverse process starts from; 0 = the prototype itself, "
                        "1 = pure noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.n_per_slot = 1, 8, 20
        args.epochs, args.steps, args.inv_steps = 4, 20, 60
        args.strengths = [0.4, 1.0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(field0, C, slots, probes, data, seed=args.seed, steps=args.steps,
                           n=8 if args.quick else 16)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    print(f"[n30] {len(qs)} compositions x {len(args.strengths)} noise strengths x 2 methods",
          flush=True)

    def score(S, proto, R, q, arm, s):
        a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
        pr = precision_recall(S, R)
        sp = speckle(S, R)
        # distance to the prototype, and how far apart the samples are from EACH OTHER: a method can
        # sit close to the prototype either by being faithful or by collapsing, and only the second
        # number separates those.
        d_proto = float(np.sqrt(np.mean((S - np.asarray(proto).reshape(1, -1)) ** 2)))
        d_pair = float(np.sqrt(np.mean((S[:, None] - S[None, :]) ** 2)))
        return dict(arm=arm, strength=float(s), slot=int(q["slot"]),
                    dist_prototype=d_proto, dist_pairwise=d_pair,
                    joint=a["joint"], digit=a["digit"], fg=a["fg"], bg=a["bg"],
                    recall=pr["recall"], f1=pr["f1"], vendi=vendi_score(S),
                    tv_ratio=sp.get("tv_ratio", float("nan")))

    rows, panels = [], {}
    for q in qs:
        R = data["X"][q["rest"][:100]].astype(np.float32)
        xq = to_model(data["X"][q["query"]]).ravel()
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        L.attach_factorization(C, slot_factors(slots), gains=gains)
        t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=8 if args.quick else 16,
                      iters=30 if args.quick else 120)
        proto = to_data(np.asarray(t["mu"]).reshape(1, -1)).astype(np.float32)[0]
        name = f"c{q['slot']}"
        L.learn_concept_factored(xq, name, n_pool=48, epochs=args.epochs,
                                 inv_steps=args.inv_steps, seed=args.seed,
                                 guidance=args.guidance)
        blocks = {"prototype": np.repeat(proto[None], args.n_show, 0),
                  "real": R[:args.n_show]}
        for s in args.strengths:
            S = to_data(L.sample_concept(name, n=args.n_samples, steps=args.steps,
                                         seed=args.seed, u_start=float(s),
                                         x_init=t["mu"])).astype(np.float32)
            rows.append(score(S, proto, R, q, "ours", s))
            blocks[f"ours  s={s:g}"] = S[:args.n_show]
            T = to_data(L.sample_teacher(t, n=args.n_samples, seed=args.seed,
                                         u_start=float(s), **PAPER_POE)).astype(np.float32)
            rows.append(score(T, proto, R, q, "poe-guided", s))
            blocks[f"poe-guided  s={s:g}"] = T[:args.n_show]
            print(f"[n30] slot {q['slot']} s={s:g}: ours joint {rows[-2]['joint']:.2f} "
                  f"d_proto {rows[-2]['dist_prototype']:.3f} vendi {rows[-2]['vendi']:.2f} | "
                  f"poe joint {rows[-1]['joint']:.2f} d_proto {rows[-1]['dist_prototype']:.3f} "
                  f"vendi {rows[-1]['vendi']:.2f}", flush=True)
        panels[int(q["slot"])] = blocks
        del field, L
        torch.cuda.empty_cache()

    summary = {}
    for arm in ("ours", "poe-guided"):
        for s in args.strengths:
            sel = [r for r in rows if r["arm"] == arm and r["strength"] == s]
            if sel:
                summary[f"{arm}@{s:g}"] = {k: float(np.nanmean([r[k] for r in sel]))
                                           for k in ("dist_prototype", "dist_pairwise", "joint",
                                                     "recall", "f1", "vendi", "tv_ratio")}
    save_json(dict(rows=rows, summary=summary, config=vars(args)),
              os.path.join(RESULTS, f"n30_prototype_distance{args.tag}.json"), seed=args.seed)

    print(f"\n[n30] the prototype-distance dial, {len(qs)} compositions")
    print(f"    {'method':14s}{'s':>5s}{'d(proto)':>10s}{'d(pair)':>9s}{'joint':>8s}"
          f"{'recall':>8s}{'F1':>7s}{'vendi':>7s}")
    for arm in ("ours", "poe-guided"):
        for s in args.strengths:
            k = f"{arm}@{s:g}"
            if k in summary:
                v = summary[k]
                print(f"    {arm:14s}{s:5.1f}{v['dist_prototype']:10.3f}{v['dist_pairwise']:9.3f}"
                      f"{v['joint']:8.3f}{v['recall']:8.3f}{v['f1']:7.3f}{v['vendi']:7.2f}")
        print()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["real", "prototype"] + [f"{a}  s={s:g}" for a in ("ours", "poe-guided")
                                     for s in args.strengths]
    nrow = len(panels) * len(order)
    fig, axes = plt.subplots(nrow, args.n_show, figsize=(1.15 * args.n_show + 3.2, 1.2 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for slot, blocks in panels.items():
        for lab in order:
            S = blocks.get(lab)
            for c in range(args.n_show):
                ax = axes[r][c]
                ax.axis("off")
                if S is not None and c < len(S):
                    ax.imshow(rgb_img(S[c]))
            st = next((x for x in rows if x["slot"] == slot
                       and lab.startswith(x["arm"]) and f"s={x['strength']:g}" in lab), None)
            note = f"\njoint {st['joint']:.2f}  vendi {st['vendi']:.2f}" if st else ""
            axes[r][0].set_ylabel(f"{lab}{note}", rotation=0, ha="right", va="center", fontsize=7)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp_ in axes[r][0].spines.values():
                sp_.set_visible(False)
            r += 1
    fig.suptitle("Distance from the prototype is a DIAL, not a property of the method.\n"
                 "s = the noise level the reverse process starts from, having noised the prototype "
                 "to it: s->0 near-copies, s->1 unconstrained samples.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, f"n30_prototype_distance{args.tag}.png")
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"[n30] wrote {out}")


if __name__ == "__main__":
    main()
