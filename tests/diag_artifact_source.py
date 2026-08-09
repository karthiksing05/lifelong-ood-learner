"""Which STAGE of the pipeline introduces the visual artifacts?

`tests/diag_guidance_interval.py` tested the obvious hypothesis -- that classifier-free guidance at
low noise levels was injecting speckle, as it demonstrably does for the analytic teacher (RESULTS
5l-bis) -- and refuted it: restricting guidance to an interval changed the samples not at all
(tv_ratio 1.279 -> 1.261, and the five rows of the figure are indistinguishable).  It could not have
helped, because `sample_concept` already samples at guidance 1.0.

So the artifacts come from somewhere else, and this walks the pipeline stage by stage to find where.
Each stage is compared against real images of the same composition on the same metric:

  `real`              real held-out images -- the reference
  `base trained`      the FROZEN backbone sampling a class it was TRAINED on.  If this is already
                      textured, the artifacts are the backbone's and no change to our method can
                      remove them.
  `base @ c_new`      the frozen backbone conditioned on the composed vector, i.e. the distillation
                      POOL before any adapter exists.
  `adapter`           after consolidation -- what the case-study figure actually shows.

The stage where the metric jumps is the stage responsible.

    python tests/diag_artifact_source.py --n-cases 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import CACHE, backbone_gains, ood_queries                        # noqa: E402
from lifelong_ood.datasets import load_colormnist, rgb_img                    # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes, speckle      # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,          # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import sample_factored_pool, slot_factors  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200],
                   help="also sweep sampling steps: too few leaves residual noise")
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    base_steps = args.steps[0]
    field0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(field0, C, slots, probes, data, seed=args.seed, steps=base_steps)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)

    rows, panels = {}, {}
    def add(lab, S, R):
        sp = speckle(S, R)
        rows.setdefault(lab, []).append((sp["tv_ratio"], sp["isolated"], sp["tv"]))
        panels.setdefault(lab, S[:6])

    for q in qs:
        R = data["X"][q["rest"][:100]].astype(np.float32)
        add("real (reference)", R[:args.n_samples], R)
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=base_steps, seed=args.seed, device=dev)
        L.attach_factorization(C, slot_factors(slots), gains=gains)
        # the backbone on a class it was TRAINED on: the ceiling for anything we can do
        seen_i = int(np.argmax([0]))
        B = to_data(field.sample(n=args.n_samples, steps=base_steps, seed=args.seed,
                                 clip=(-1, 1), branch=seen_i)).astype(np.float32)
        add("base, TRAINED class", B, R)
        xq = to_model(data["X"][q["query"]]).ravel()
        c_new, _, _ = L.compose_factored(xq, inv_steps=args.inv_steps, seed=args.seed, gains=gains)
        for st in args.steps:
            P = to_data(sample_factored_pool(field, c_new, n=args.n_samples, steps=st,
                                             seed=args.seed, guidance=args.guidance)
                        ).astype(np.float32)
            add(f"base @ c_new, {st} steps", P, R)
        name = f"c{q['slot']}"
        L.learn_concept_factored(xq, name, n_pool=48, epochs=args.epochs,
                                 inv_steps=args.inv_steps, seed=args.seed, guidance=args.guidance)
        for st in args.steps:
            S = to_data(L.sample_concept(name, n=args.n_samples, steps=st,
                                         seed=args.seed)).astype(np.float32)
            add(f"adapter, {st} steps", S, R)
        del field, L
        torch.cuda.empty_cache()

    print(f"\n[diag] where the texture enters ({len(qs)} compositions)")
    print(f"    {'stage':28s}{'tv_ratio':>10s}{'tv':>9s}{'isolated':>10s}")
    for lab, v in rows.items():
        a = np.mean(v, 0)
        print(f"    {lab:28s}{a[0]:10.3f}{a[2]:9.4f}{a[1]:10.5f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(panels)
    fig, axes = plt.subplots(len(labs), 6, figsize=(1.2 * 6 + 3.4, 1.25 * len(labs)))
    axes = np.atleast_2d(axes)
    for r, lab in enumerate(labs):
        a = np.mean(rows[lab], 0)
        for c in range(6):
            ax = axes[r][c]; ax.axis("off")
            if c < len(panels[lab]):
                ax.imshow(rgb_img(panels[lab][c]))
        axes[r][0].set_ylabel(f"{lab}\ntv {a[0]:.2f}x", rotation=0, ha="right", va="center",
                              fontsize=7.5)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for sp_ in axes[r][0].spines.values():
            sp_.set_visible(False)
    fig.suptitle("Which stage introduces the texture? (tv relative to real images of the same class)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "diag_artifact_source.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
