"""N45 -- what does the permuted layout actually cost?  Two matched backbones, one variable.

`tests/diag_layout.py` proves by index algebra that this repository hands its UNet a pixel
PERMUTATION rather than an image: a plain `.reshape(-1, 3, H, W)` of an HWC row makes the network's
three "channels" out of horizontal image bands, its width axis out of the colour axis, and its
vertical axis a ~22-pixel jump across the picture, so a single 3x3 convolution reads real columns
3, 24 and 45 at once.  Training and sampling share the permutation, so images still DISPLAY
correctly -- which is exactly why it survived this long.

Proving the prior is destroyed is not the same as proving it MATTERS.  A UNet is a universal
approximator; it can fit a permuted dataset by brute force, and Color-MNIST is simple enough that
it plausibly does.  This measures the gap instead of arguing about it.

Two backbones, identical in every respect -- same architecture, data, seed, epochs, batch, lr,
schedule -- differing ONLY in `layout`.  Reported:

  * held-out epsilon-prediction loss (the training objective itself, so it is directly comparable:
    the same images, the same noise, only the arrangement of the input tensor differs)
  * joint attribute accuracy of conditional samples on TRAINED slots, which needs no held-out
    composition and no downstream machinery
  * speckle/TV against real images -- the roughness statistic that flagged the CelebA samples

If `hwc` wins clearly, every backbone in this repository is trained with its convolutional prior
switched off and the fix is a retrain, not a re-analysis.  If the gap is small, the permutation is
a curiosity and the CelebA quality problem lies elsewhere.

    python experiments/n45_layout.py --epochs 20
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                                      # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                               # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, save_json,    # noqa: E402
                                speckle)
from lifelong_ood.neural import NeuralScoreField, to_data                       # noqa: E402
from lifelong_ood.neural.ddpm import train_ddpm                                 # noqa: E402
from lifelong_ood.neural.unet import UNet                                       # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--n-eval", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_per_slot, args.n_eval = 2, 20, 8
        args.tag = args.tag or "_smoke"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    X, slots = data["X"], data["slot"]
    uniq = np.unique(slots)
    remap = {int(s): i for i, s in enumerate(uniq)}
    y = np.array([remap[int(s)] for s in slots], np.int64)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_hold = max(256, len(X) // 10)
    hold, tr = perm[:n_hold], perm[n_hold:]
    print(f"[n45] {len(tr)} train / {len(hold)} held out over {len(uniq)} slots, "
          f"{args.epochs} epochs each\n", flush=True)

    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    # each slot's (digit, fg, bg), read off the data rather than recomputed from the slot id
    facs = {int(s_): (int(data["digit"][i]), int(data["fg"][i]), int(data["bg"][i]))
            for i, s_ in enumerate(slots)}

    # Drawn ONCE, before the arm loop.  Drawing inside it advanced `rng` between arms, so the
    # two layouts were scored on DIFFERENT held-out compositions -- an unpaired comparison
    # dressed as a controlled one.
    pick = uniq[rng.permutation(len(uniq))[:args.n_eval]]
    show_slot = int(pick[0])
    rows, panels = [], {}
    for layout in ("permuted", "hwc"):
        torch.manual_seed(args.seed)
        model = UNet(n_classes=len(uniq), base=args.base)
        t0 = time.time()
        ema = train_ddpm(model, X[tr], y[tr], epochs=args.epochs, batch=args.batch, lr=args.lr,
                         device=dev, seed=args.seed, layout=layout)
        model.load_state_dict(ema)
        # held-out objective: same images, same noise, only the tensor arrangement differs
        held = train_ddpm(model, X[hold], y[hold], epochs=1, batch=args.batch, device=dev,
                          seed=args.seed + 1, eval_only=True, layout=layout)
        loss = float(held) if np.isscalar(held) or isinstance(held, float) else float("nan")
        field = NeuralScoreField(model, device=dev, batch=64, layout=layout)

        accs, S_show = [], None
        for s in pick:
            S = to_data(field.sample(n=8, steps=50, seed=args.seed,
                                     branch=int(remap[int(s)]), clip=(-1, 1))).astype(np.float32)
            dgt, fg, bg = facs[int(s)]
            a = composition_accuracy(probes, S, dgt, fg, bg)
            accs.append(float(a["joint"]))
            if S_show is None:
                S_show = S
        R = X[hold[:256]].astype(np.float32)
        sk = speckle(S_show, R, img=32)
        rows.append(dict(layout=layout, held_loss=loss, joint=float(np.nanmean(accs)),
                         tv_ratio=float(sk.get("tv_ratio", np.nan)),
                         minutes=(time.time() - t0) / 60,
                         # per-slot, in `pick` order, so the two arms can be compared PAIRED
                         per_slot=[float(v) for v in accs],
                         slots=[int(v) for v in pick]))
        panels[layout] = S_show
        print(f"[n45] {layout:9s} held-out loss {loss:.5f}   joint acc "
              f"{np.nanmean(accs):.3f}   tv vs real {sk.get('tv_ratio', float('nan')):.3f}   "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        del field, model
        torch.cuda.empty_cache()

    a, b = rows[0], rows[1]
    d = np.array(b["per_slot"]) - np.array(a["per_slot"])
    print(f"\n[n45] hwc minus permuted:  loss {b['held_loss'] - a['held_loss']:+.5f}   "
          f"joint {b['joint'] - a['joint']:+.3f}   tv {b['tv_ratio'] - a['tv_ratio']:+.3f}")
    print(f"[n45] PAIRED over {len(d)} identical slots: mean {d.mean():+.3f}, "
          f"hwc better on {int((d > 0).sum())}, worse on {int((d < 0).sum())}, "
          f"tied on {int((d == 0).sum())}")
    print("[n45] a large gain for `hwc` means every backbone here trained with its convolutional "
          "prior switched off, and the fix is a retrain rather than a re-analysis.")
    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n45_layout{args.tag}.json"), seed=args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real_rows = np.where(slots == show_slot)[0][:8]
    fig, axes = plt.subplots(3, 8, figsize=(11, 4.6))
    for c in range(8):
        for r, (name, S) in enumerate([(f"real (slot {show_slot})",
                                        X[real_rows].astype(np.float32)),
                                       ("permuted (current)", panels["permuted"]),
                                       ("hwc (true image)", panels["hwc"])]):
            axes[r][c].axis("off")
            if c < len(S):
                axes[r][c].imshow(np.clip(np.asarray(S[c], np.float32).reshape(32, 32, 3), 0, 1))
            if c == 0:
                axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
                axes[r][0].axis("on")
                axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
                for s_ in axes[r][0].spines.values():
                    s_.set_visible(False)
    fig.suptitle(f"Same UNet, same data, same seed, {args.epochs} epochs — only the input layout "
                 f"differs.\nheld-out loss {a['held_loss']:.4f} → {b['held_loss']:.4f};  "
                 f"joint accuracy {a['joint']:.3f} → {b['joint']:.3f}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(FIG, f"f21_layout_cost{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[n45] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
