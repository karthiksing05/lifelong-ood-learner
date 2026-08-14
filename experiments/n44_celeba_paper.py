"""N44 -- CelebA the way the companion paper ran it: pretrained backbone, paper-shaped benchmark.

§29 audited our CelebA setup against the paper's Appendices B/C/D/F and found fourteen differences,
three of which explain the image quality by themselves.  The largest is the backbone: **they
fine-tune `google/ddpm-celebahq-256` with `sample_size` overridden to 128; we train from scratch.**
They begin from a model that already renders photorealistic faces; ours has to learn faces *and*
conditioning from nothing, on ~26 GPU-hours against their ~200, over a benchmark 12x wider.

That penalty is paid by EVERY arm, which is why `poe-guided` looks as bad here as we do -- the
published method is not being reproduced badly so much as being run on a far weaker substrate.

This runs the composition benchmark on the pretrained backbone instead, which needs no training at
all: the weights are cached, §12 already certified the bank attaches to them at 0.000e+00
preservation, and `CondUNetShim` already presents the diffusers UNet through our field interface.

Two further corrections from the audit, both switchable so the effect of each is measurable rather
than assumed:

  `--paper-discovery`  the paper's Table 6 CelebA column instead of our ColorMNIST-derived settings:
                       timestep grid [50, 500, 50] (ten levels, not our four), **query-centred**
                       initialization (not our hybrid), 100 iterations, step 0.09, and **fixed
                       tau = 0.5** rather than our adaptive temperature.  Three of those are our own
                       documented deviations, justified on ColorMNIST and never re-validated here.
  `--paper-holdout`    held-out classes chosen to differ from a SEEN class by exactly ONE attribute,
                       the paper's controlled OOD design, rather than our held-out attribute PAIR.

Arms are the ones that matter for the complaint: `poe-guided` (the paper's own method), `ours`, and
`query-only` as the floor.  The question this answers is not "who wins" but "do the images stop
looking broken once the substrate matches".

    python experiments/n44_celeba_paper.py --paper-discovery --paper-holdout
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                                     # noqa: E402

from lifelong_ood.evals import save_json                                        # noqa: E402
from lifelong_ood.neural import NeuralLifelongLearner, to_data, to_model        # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained                      # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
# the paper's Table 6 CelebA column -- IMPORTED, not re-typed.  A local copy of a shared constant
# is how the w0 bug happened (six files each holding their own `PAPER_POE`), and it happened again
# here: PAPER_CELEBA's ascent step was corrected to 0.02 and this file kept its own 0.09.
from lifelong_ood.compose import PAPER_CELEBA as PAPER_DISC                    # noqa: E402
OURS_DISC = dict(u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=12, iters=80, lr=0.1,
                 init="hybrid", probes=4)
# Table 8, CelebA column: w0=0.5.  This file previously used 1.2, the Color-MNIST value.
from lifelong_ood.compose import PAPER_POE_CELEBA as PAPER_POE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--paper-discovery", action="store_true")
    p.add_argument("--paper-tau", type=float, default=0.5,
                   help="the paper's fixed softmax temperature; our adaptive setting is used when "
                        "--paper-discovery is off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_show, args.steps = 2, 4, 30
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    disc = dict(PAPER_DISC if args.paper_discovery else OURS_DISC)
    if args.quick:
        disc["iters"] = 20
        disc["n_starts"] = 4
        disc["u_grid"] = disc["u_grid"][:3]
    print(f"[n44] backbone {args.model} (PRETRAINED -- no training)\n"
          f"[n44] discovery = {'PAPER Table 6 CelebA' if args.paper_discovery else 'ours'}: "
          f"{len(disc['u_grid'])} noise levels, {disc['n_starts']} starts, {disc['iters']} iters, "
          f"init={disc['init']}\n", flush=True)

    field, img = load_pretrained(args.model, device=dev, batch=4)  # -> (field, img)
    rng = np.random.default_rng(args.seed)

    # Queries: real images from the model's own domain.  The pretrained backbone is unconditional
    # and has no attribute labels, so the "held-out composition" here is a REAL FACE the model was
    # not conditioned on -- the same object the paper's query is, minus the attribute bookkeeping.
    # §12 established this is how the pretrained track poses the task.
    from lifelong_ood.data.celeba import as_float, load_celeba_comp
    data = load_celeba_comp(img=img, max_rows=4000, min_count=20, n_holdout=8)
    sp = data["split"]          # hold_ids / combo_of / all_combos live here, not at top level
    hold = np.asarray(sp["hold_ids"])
    rows = []
    panels = {}
    for ci in range(args.n_cases):
        h = int(hold[ci % len(hold)])
        idx = np.where(sp["combo_of"] == h)[0]
        if len(idx) == 0:
            continue
        q = int(idx[rng.integers(len(idx))])
        xq = to_model(as_float(data["X8"][q:q + 1])).ravel()
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        # discovery run directly, not through `compose`, for two reasons: the modes M are needed for
        # the prototypes row and `compose` does not return them, and the paper's FIXED tau=0.5
        # requires adaptive_tau=False -- passing tau alone would still let our adaptive rescaling
        # override it, which is the deviation §29 flagged.
        from lifelong_ood.compose import candidates_ascent, poe_teacher
        M, V, meta = candidates_ascent(field, xq, seed=args.seed, **disc)
        t = (poe_teacher(xq, M, V, K=3, tau=args.paper_tau, adaptive_tau=False)
             if args.paper_discovery else poe_teacher(xq, M, V, K=3))
        # Report the NOISE LEVEL each selected prototype came from, and how far out of gamut it is.
        #
        # In the rendered panels one of the three prototypes is saturated while the other two are
        # clean faces, consistently across cases.  `candidates_ascent` pools ascent output over the
        # whole u-grid (CelebA: 0.05 ... 0.50) and selection then draws K=3 from that pool, and
        # `tests/diag_prototype_quality.py` measured the upper end of that grid returning junk.  So
        # the hypothesis is that the bad prototype is the high-u one -- which this line tests
        # directly instead of leaving it to be inferred from a picture.
        us = np.asarray(meta.get("u", np.full(len(M), np.nan)))[t["sel"]]
        oog = [float((np.abs(np.asarray(M)[s]) > 1).mean()) for s in t["sel"]]
        print(f"[n44] case {h}: {len(M)} modes discovered, K=3 selected {list(t['sel'])} "
              f"from u={np.round(us, 3).tolist()} with out-of-gamut "
              f"{[round(o, 3) for o in oog]}", flush=True)
        got = {
            # THE QUERY FIRST, then other images of the same class.  This row used to be
            # `idx[:n_show]` -- the first n_show images of the combo -- while every row beneath it
            # was driven by `xq`, drawn at a RANDOM index of the same combo.  The two coincided only
            # by chance, so the panel invited a comparison against images that had produced none of
            # it, and `tests/diag_celeba_cast.py` shows why that matters: the samples' colour tracks
            # the QUERY (spread 0.149, against 0.155 for the queries themselves) and barely moves
            # with the seed (0.015).  A cast that looked unexplained was faithful to an image the
            # figure did not show.
            "QUERY (col 1)\n+ others of its class": as_float(
                data["X8"][[q] + [int(i) for i in idx[:args.n_show - 1] if int(i) != q]]
            ).astype(np.float32),
            "prototypes (mode ascent)": to_data(np.asarray(M)[t["sel"]]).astype(np.float32),
            "poe-guided (paper Eq.14)": to_data(L.sample_teacher(
                t, n=args.n_show, seed=args.seed, **PAPER_POE)).astype(np.float32),
            "query-only": to_data(L.sample_teacher(
                L.baseline_teacher(xq, "query_only"), n=args.n_show, seed=args.seed,
                **PAPER_POE)).astype(np.float32),
        }
        panels[h] = got
        # sample roughness against real, the one quality number that needs no labels
        from lifelong_ood.evals import speckle
        R = as_float(data["X8"][idx[:64]]).astype(np.float32)
        for k, S in got.items():
            if S is None or k.startswith("QUERY"):     # the reference row, not an arm
                continue
            sk = speckle(S, R, img=img)   # NOT `sp` -- that is the split dict, and shadowing it
            # made case 2 fail with KeyError: combo_of
            rows.append(dict(case=h, arm=k, tv_ratio=float(sk.get("tv_ratio", np.nan)),
                             paper_discovery=bool(args.paper_discovery)))
            print(f"[n44] case {h} {k:26s} tv vs real {sk.get('tv_ratio', float('nan')):.3f}",
                  flush=True)
        del L
        torch.cuda.empty_cache()

    save_json(dict(rows=rows, discovery=("paper" if args.paper_discovery else "ours"),
                   config=vars(args)),
              os.path.join(RESULTS, f"n44_celeba_paper{args.tag}.json"), seed=args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelong_ood.data.datasets import rgb_img
    # Derived from the panels themselves, not re-typed.  A hardcoded copy of these row names went
    # out of sync with the dict the moment the first row was relabelled, and `got.get(name)` returned
    # None for it -- so the figure rendered a blank strip where the reference images belong, with no
    # error anywhere.  Dicts preserve insertion order, so the construction order IS the row order.
    order = list(next(iter(panels.values())).keys()) if panels else []
    nrow = len(panels) * len(order)
    fig, axes = plt.subplots(nrow, args.n_show,
                             figsize=(1.5 * args.n_show + 4.0, 1.55 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for h, got in panels.items():
        for name in order:
            S = got.get(name)
            for c in range(args.n_show):
                ax = axes[r][c]
                ax.axis("off")
                if S is not None and c < len(S):
                    ax.imshow(np.clip(np.asarray(S[c], np.float32).reshape(
                        img, img, 3), 0, 1))
            axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp_ in axes[r][0].spines.values():
                sp_.set_visible(False)
            r += 1
    fig.suptitle(
        f"CelebA on the backbone the companion paper actually used — pretrained {args.model}, "
        f"no training.\n"
        f"Discovery = {'the paper' + chr(39) + 's Table 6 CelebA column' if args.paper_discovery else 'ours'}; "
        f"guidance = their Table 8 w0=0.5; input layout corrected (RESULTS 30b).\n"
        f"Two defects are visible and neither is a rendering artifact. (1) The PROTOTYPES are off "
        f"the colour manifold — mode ascent improves the\nscore while leaving the data, the "
        f"likelihood/typicality gap. (2) Every generated row within a block collapses to nearly "
        f"one face, and\npoe-guided and query-only agree even though query-only never sees a "
        f"prototype — so within a block the output is dominated by something\nthey SHARE (the "
        f"seed and the unconditional model), not by the teacher. Which of the two drives the "
        f"colour cast is measured in\ntests/diag_celeba_cast.py, not asserted here. Discovery runs "
        f"on an UNCONDITIONAL pretrained model; the paper's Figure 3 uses their\nfine-tuned "
        f"CLASS-CONDITIONAL backbone, which is what N47 builds and N48 tests.", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(FIG, f"f18_celeba_paper{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[n44] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
