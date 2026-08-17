"""Why do OUR ColorMNIST samples look worse than every baseline's, and which knob fixes it?

The case-study figure makes an uncomfortable point that the metrics hide: in `case_colormnist.png`
the `ours` row is the ONLY row with malformed digits and wrong background colours, while
`poe-guided`, `query-only` and `nearest-class` are all clean.  That is not a backbone limitation --
`nearest-class` samples the same backbone and comes out clean -- so the degradation is introduced
somewhere in our own pool -> distil -> sample path.

Two candidate causes, and this sweep separates them:

  `guidance`  the CFG weight used when the MANIFOLD generates the pool at c_new.  Low guidance gives
              a diverse but loosely-conditioned pool, and the adapter can only be as good as the pool
              it was distilled from.
  `bg_gain`   the per-factor gain correcting the model's measured under-response to the background
              factor.  On this checkpoint calibration selected 1.0, i.e. no correction at all, which
              is suspicious given that wrong BACKGROUNDS are precisely what the figure shows.

Reports joint accuracy, per-factor accuracy, recall and Vendi for each setting, and writes a grid so
the choice is made by looking as well as by the numbers -- which is how the layout bug and the
guidance-schedule bug were both caught.

    python tests/diag_ours_quality.py --n-cases 2 --n-samples 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import CACHE, ood_queries                                         # noqa: E402
from lifelong_ood.data.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, vendi_score)
from lifelong_ood.neural import load_field, to_data, to_model                 # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                       # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner         # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidances", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--bg-gains", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)

    print(f"[diag] {len(qs)} held-out compositions, {args.n_samples} samples, "
          f"pool {args.n_pool}, epochs {args.epochs}\n", flush=True)
    print(f"    {'guid':>5s}{'bggain':>8s}{'joint':>8s}{'digit':>7s}{'fg':>6s}{'bg':>6s}"
          f"{'recall':>8s}{'vendi':>7s}")

    panels, out = [], []
    for gd in args.guidances:
        for bg in args.bg_gains:
            agg, shots = [], []
            for qi, q in enumerate(qs):
                field = load_field(args.ckpt, device=dev)
                L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
                L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, float(bg)])
                xq = to_model(data["X"][q["query"]]).ravel()
                L.learn_concept_factored(xq, f"c{q['slot']}", n_pool=args.n_pool,
                                         epochs=args.epochs, inv_steps=args.inv_steps,
                                         seed=args.seed + qi, guidance=float(gd))
                S = to_data(L.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                             seed=args.seed)).astype(np.float32)
                R = data["X"][q["rest"][:100]].astype(np.float32)
                a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                pr = precision_recall(S, R)
                agg.append([a["joint"], a["digit"], a["fg"], a["bg"], pr["recall"],
                            vendi_score(S)])
                if qi == 0:
                    shots = S[:6]
            m = np.mean(agg, 0)
            out.append((gd, bg, m))
            panels.append((f"guid {gd:g}, bg_gain {bg:g}\njoint {m[0]:.2f} recall {m[4]:.2f}",
                           shots))
            print(f"    {gd:5.1f}{bg:8.1f}{m[0]:8.3f}{m[1]:7.3f}{m[2]:6.2f}{m[3]:6.2f}"
                  f"{m[4]:8.3f}{m[5]:7.2f}", flush=True)

    best = max(out, key=lambda r: (r[2][0], r[2][4]))
    print(f"\n[diag] best joint at guidance {best[0]:g}, bg_gain {best[1]:g} "
          f"(joint {best[2][0]:.3f}, bg {best[2][3]:.2f}, recall {best[2][4]:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelong_ood.data.datasets import rgb_img
    nrow, ncol = len(panels), 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.15 * ncol + 2.6, 1.2 * nrow))
    axes = np.atleast_2d(axes)
    for r, (lab, shots) in enumerate(panels):
        for c in range(ncol):
            ax = axes[r][c]; ax.axis("off")
            if c < len(shots):
                ax.imshow(rgb_img(shots[c]))
        axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7.5)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for sp in axes[r][0].spines.values():
            sp.set_visible(False)
    fig.suptitle("`ours` on ColorMNIST: pool guidance x background gain", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIG, "diag_ours_quality.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[diag] wrote {path}")


if __name__ == "__main__":
    main()
