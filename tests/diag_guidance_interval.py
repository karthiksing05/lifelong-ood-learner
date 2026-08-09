"""Does restricting guidance to an interval of noise levels remove the speckle?

Our samples carry scattered bright pixels that no metric in this repository could see: composition
accuracy, precision/recall, density/coverage and Vendi are all indifferent to a speckled digit,
because it is still the right digit in the right colours.

The cause is not mysterious and this repository has already diagnosed it once, for the analytic
teacher: **guidance applied at LOW noise levels amplifies high-frequency error** (RESULTS 5l-bis --
their Eq. 14 variance-aware schedule exists to cancel exactly this, and using our annealed schedule
instead injected speckle into the case-study figure). We were nonetheless applying a CONSTANT
classifier-free weight at every noise level when sampling the distillation pool, which is the same
mistake in our own half of the pipeline.

The standard fix is to apply guidance only inside an interval of noise levels and sample plainly
outside it (Kynkaanniemi et al., NeurIPS 2024, arXiv:2404.07724). This sweeps the lower edge, since
that is the end that matters, and reports:

  `tv_ratio` / `isolated_ratio`   the artifact, relative to REAL images of the same class (1.0 =
                                  as clean as real data)
  `joint` / `recall` / `vendi`    what it costs, if anything

    python tests/diag_guidance_interval.py --n-cases 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import CACHE, backbone_gains, ood_queries                         # noqa: E402
from lifelong_ood.datasets import load_colormnist, rgb_img                     # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, speckle, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                        # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--los", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5],
                   help="lower edge of the guidance interval; 0.0 is the current behaviour "
                        "(guidance everywhere)")
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
    field0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(field0, C, slots, probes, data, seed=args.seed, steps=args.steps)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)

    print(f"[diag] guidance {args.guidance}, sweeping the interval's lower edge; "
          f"{len(qs)} compositions x {args.n_samples} samples\n", flush=True)
    print(f"    {'interval':>14s}{'tv_ratio':>10s}{'isolated':>10s}{'joint':>8s}"
          f"{'recall':>8s}{'vendi':>7s}")
    panels, out = [], []
    for lo in args.los:
        iv = None if lo <= 0 else (lo, 1.0)
        agg, shot = [], None
        for qi, q in enumerate(qs):
            field = load_field(args.ckpt, device=dev)
            L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
            L.attach_factorization(C, slot_factors(slots), gains=gains)
            xq = to_model(data["X"][q["query"]]).ravel()
            L.learn_concept_factored(xq, f"c{q['slot']}", n_pool=48, epochs=args.epochs,
                                     inv_steps=args.inv_steps, seed=args.seed,
                                     guidance=args.guidance, interval=iv)
            S = to_data(L.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                         seed=args.seed)).astype(np.float32)
            R = data["X"][q["rest"][:100]].astype(np.float32)
            sp = speckle(S, R)
            a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
            pr = precision_recall(S, R)
            agg.append([sp["tv_ratio"], sp["isolated_ratio"], a["joint"], pr["recall"],
                        vendi_score(S)])
            if qi == 0:
                shot = S[:6]
            del field, L
            torch.cuda.empty_cache()
        m = np.mean(agg, 0)
        out.append((lo, m))
        lab = "none (current)" if iv is None else f"[{lo:.1f}, 1.0]"
        print(f"    {lab:>14s}{m[0]:10.3f}{m[1]:10.2f}{m[2]:8.3f}{m[3]:8.3f}{m[4]:7.2f}", flush=True)
        panels.append((f"{lab}\ntv {m[0]:.2f}  iso {m[1]:.1f}x\njoint {m[2]:.2f}", shot))

    base = out[0][1]
    best = min(out, key=lambda r: r[1][1])
    print(f"\n[diag] cleanest at interval lower edge {best[0]:.1f}: "
          f"isolated-pixel ratio {best[1][1]:.1f}x against {base[1]:.1f}x with guidance everywhere, "
          f"at joint {best[1][2]:.3f} against {base[2]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = len(panels), 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.2 * ncol + 3.0, 1.25 * nrow))
    axes = np.atleast_2d(axes)
    for r, (lab, S) in enumerate(panels):
        for c in range(ncol):
            ax = axes[r][c]; ax.axis("off")
            if S is not None and c < len(S):
                ax.imshow(rgb_img(S[c]))
        axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7.5)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for sp_ in axes[r][0].spines.values():
            sp_.set_visible(False)
    fig.suptitle("Guidance restricted to an interval of noise levels: does the speckle go away?",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIG, "diag_guidance_interval.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[diag] wrote {path}")


if __name__ == "__main__":
    main()
