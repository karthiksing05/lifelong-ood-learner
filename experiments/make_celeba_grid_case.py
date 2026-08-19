"""The CelebA-grid case study: figures/initial/case_celeba_grid.png.

The first CelebA figure in this repository whose held-out combinations are absent from the model's
DENSITY rather than merely missing an embedding row.  Every earlier CelebA panel came from a
backbone fine-tuned out of `google/ddpm-celebahq-256`, which had seen every combination during
pretraining -- which is why its `prototypes` row handed the query straight back, and why the same
thing happens on the (also pretrained) Shapes3D backbone.  Here the three held-out cells have ZERO
training images.

Rows, and what each is for:

  real            actual images of the held-out cell -- the target, never shown to the model
  prototypes      mode ascent, QUERY-AGNOSTIC chains only.  Selection (Eq. 8-9) scores candidates on
                  the query, so a query-seeded chain wins the first pick by construction and returns
                  the query; 5e-quater measures the composition surviving on the agnostic experts
                  alone (joint 0.83 either way) and COLLAPSING when restricted to query-seeded ones
                  (0.00, background wrong in 5 of 5), so the agnostic half is what the composition
                  actually rests on and what the figure should show.
  ours (ADDRESS)  compose(fac, factors) and sample the model's own conditional
  nearest-class   the closest SEEN cell -- the control for "the model already had it"
  unconditional   the floor

EVERY NUMBER IS SHOWN AGAINST THE PROBE CEILING, not against 1.0.  The factor probes read real
images at hair 0.568 / Male 0.849 / Smiling 0.859, so a perfect generator caps near 0.414 joint and
chance is 1/9 = 0.111.  A raw joint of 0.29 means something entirely different against a 0.414
ceiling than against 1.0, and a figure that omits the ceiling invites the wrong reading.

    python experiments/make_celeba_grid_case.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE                                              # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "initial")
HAIR = ("blond", "brown", "other")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid.pt"))
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="figure filename under figures/initial/")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lifelong_ood.data.celeba import load_celeba_grid
    from lifelong_ood.evals import factor_accuracy, fit_factor_probes, precision_recall
    from lifelong_ood.neural import NeuralScoreField, UNet, sample_embedding
    from lifelong_ood.neural.idealwords import compose, factorize

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})

    def grid(n_holdout):
        return load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                                n_per_cell=int(sp.get("n_per_slot", 3000)),
                                n_holdout=int(n_holdout), max_rows=sp.get("max_rows"))

    # Derived from the checkpoint and asserted, exactly as in n73 -- a figure drawn against the
    # wrong held-out cells looks entirely normal, which is the worst kind of wrong for a figure.
    d = grid(sp.get("n_holdout") or 3)
    want = sorted(int(k) for k in ck["remap"])
    if sorted(set(np.asarray(d["slot"])[np.asarray(d["seen"])].tolist())) != want:
        d = grid(len(d["live"]) - len(want))
    if sorted(set(np.asarray(d["slot"])[np.asarray(d["seen"])].tolist())) != want:
        raise SystemExit("[case] split mismatch against the checkpoint -- refusing to draw")
    X, F, slot = np.asarray(d["X"], np.float32), np.asarray(d["factors"]), np.asarray(d["slot"])
    names = d["names"]
    probes = fit_factor_probes(d, seed=args.seed)
    ceil = float(np.prod([probes[f"{n}_val"] for n in names if f"{n}_val" in probes]))
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    fac = factorize(C, np.stack([F[slot == int(s)][0] for s in seen]))
    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()
    img = int(ck.get("img", 64))

    order = ["real", "prototypes", "ours (ADDRESS)", "nearest-class", "unconditional"]
    cells = [h for h in d["held"] if (slot == int(h)).any()]
    fig, axes = plt.subplots(len(cells) * len(order), args.n_show,
                             figsize=(1.5 * args.n_show + 3.4, 1.5 * len(cells) * len(order)))
    axes = np.atleast_2d(axes)
    r = 0
    for h in cells:
        idx = np.where(slot == int(h))[0]
        tgt = tuple(int(v) for v in F[idx[0]])
        R = X[idx[:60]]
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        # query-agnostic prototypes: ascent from NOISE, never from the query
        xq = X[int(idx[0])]
        P = []
        for uu in (0.15, 0.30, 0.45):
            _, x0 = field.mode_ascent(xq, uu, n_starts=4, iters=120, lr=0.05,
                                      seed=args.seed + 17 * int(h), init="random")
            P.append(np.asarray(x0)[:1])
        got = {
            "real": R[:args.n_show],
            "prototypes": np.concatenate(P),
            "ours (ADDRESS)": np.asarray(sample_embedding(field, c_new, n=args.n_show,
                                                          steps=args.steps, seed=args.seed)),
            "nearest-class": np.asarray(field.sample(n=args.n_show, steps=args.steps,
                                                     seed=args.seed,
                                                     branch=remap[int(seen[j_near])])),
            "unconditional": np.asarray(field.sample(n=args.n_show, steps=args.steps,
                                                     seed=args.seed + 91, branch=None)),
        }
        combo = ", ".join(f"{n}={HAIR[v] if n == 'hair' else v}" for n, v in zip(names, tgt))
        for nm in order:
            S = np.asarray(got[nm], np.float32)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(img, img, 3), 0, 1))
            lab = (f"real - HELD OUT\n{combo}\ncell {int(h)} - 0 training images"
                   if nm == "real" else
                   "prototypes\n(query-agnostic\nmode ascent)" if nm == "prototypes" else nm)
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp_ in axes[r][0].spines.values():
                sp_.set_visible(False)
            if nm not in ("real", "prototypes"):
                a = factor_accuracy(probes, S, tgt)
                j = a["joint"] if a["joint"] == a["joint"] else 0.0
                pr = precision_recall(S, R)
                axes[r][args.n_show - 1].set_title(
                    f"joint {j:.2f} = {j / max(ceil, 1e-9):.0%} of ceiling | recall {pr['recall']:.2f}",
                    fontsize=6.5, loc="right")
            r += 1
    # EVERY number in this caption is read from the run.  They were hardcoded to the first
    # backbone's values -- 12 cells, 9 seen / 3 held out, hair 0.568, "a raw joint of 0.29" -- and a
    # caption that describes a different split than the figure draws is worse than no caption,
    # because nothing about the picture looks wrong.
    per_factor = " / ".join(f"{n} {probes[n + '_val']:.3f}"
                            for n in names if n + "_val" in probes)
    fig.suptitle(
        f"CelebA as a GRID, trained FROM SCRATCH with a DENSITY holdout: the {len(d['held'])} "
        "cells shown have ZERO training images.\n"
        f"Grid is {names} = {len(d['live'])} cells, {len(seen)} seen / {len(d['held'])} held out; "
        "every factor value survives among the seen cells, so a held-out cell is a new ARRANGEMENT "
        "of known parts.\n"
        f"READ AGAINST THE CEILING: the factor probes read REAL images at joint ~{ceil:.3f} "
        f"({per_factor}), and chance is {1 / len(seen):.3f} -- so a raw joint of "
        f"{0.5 * ceil:.2f} is half of what this instrument can register.\n"
        "Prototypes are QUERY-AGNOSTIC chains: on the pretrained CelebA and Shapes3D backbones a "
        "query-seeded chain returns the query, because those models saw every combination in "
        "pretraining. This one did not.", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIG, exist_ok=True)
    o = os.path.join(FIG, args.out or "case_celeba_grid.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[case] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
