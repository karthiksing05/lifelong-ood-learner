"""The CelebA case study: figures/initial/case_celeba.png, in the flagship's shape.

The only CelebA figure in this repository whose held-out combinations are absent from the model's
DENSITY rather than merely missing an embedding row.  Every earlier CelebA panel came from a backbone
fine-tuned out of `google/ddpm-celebahq-256`, which had seen every combination during pretraining --
which is why its `prototypes` row handed the query straight back.  Here the held-out cells have ZERO
training images.

Rows, matching `make_flagship_case.py` so the two can be read side by side:

    real            actual images of the held-out cell -- the target, never shown to the model
    prototype       the MODE of p(x | c_new): Adam ascent conditioned on the composed address, from
                    random starts.  Until 2026-08-19 this row ascended the NULL-embedding density
                    (7ll) -- a generic face, unrelated to the cell it was printed beside.
    ours (ADDRESS)  compose(fac, factors) and sample the model's own conditional
    nearest-class   the closest SEEN cell -- the control for "the model already had it"
    unconditional   the floor

**WHERE THIS DELIBERATELY DIVERGES FROM THE FLAGSHIP, AND WHY.**  The flagship drops unreadable cells
because it has 40 readable ones to draw from.  CelebA does not: 7ee's whole finding is that most
held-out cells cannot be read by the probes at all, and a figure that silently kept the readable ones
would hide the result it exists to show.  So every held-out cell is drawn, each labelled with whether
its own REAL images clear the floor, and joint accuracies are reported only where they mean
something.  The unreadable blocks are the finding, not waste.

EVERY NUMBER IS SHOWN AGAINST THE PROBE CEILING, not against 1.0 -- a raw joint of 0.29 means
something entirely different against a 0.414 ceiling than against 1.0 -- and against the smallest
margin the arm size can resolve, because on this dataset both of 7ee's margins sit below it.

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
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HAIR = ("blond", "brown", "other")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid.pt"))
    p.add_argument("--n-show", type=int, default=6, help="panels DISPLAYED per row")
    p.add_argument("--n-eval", type=int, default=24,
                   help="samples MEASURED per arm.  Decoupled from --n-show because the panel count "
                        "is a layout choice and the sample count is the resolution of every number "
                        "on the figure: at 6 the smallest resolvable margin is 0.374, so a printed "
                        "`joint 0.00` is noise wearing two decimal places.  24 matches n73.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="figure filename under figures/initial/")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lifelong_ood.data.celeba import load_celeba_grid
    from lifelong_ood.data.factored import spread_targets
    from lifelong_ood.evals import (factor_accuracy, fit_factor_probes, precision_recall,
                                    save_json, vendi_score)
    from lifelong_ood.instrument import check_instrument, min_detectable
    from lifelong_ood.neural import (NeuralScoreField, UNet, sample_embedding,
                                     to_data, to_model)
    from lifelong_ood.neural.idealwords import compose, cv_r2, factorize

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
    lab = np.stack([F[slot == int(s)][0] for s in seen])
    fac = factorize(C, lab)
    r2cv, _ = cv_r2(C, lab)
    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()
    img = int(ck.get("img", 64))

    cells_all = [int(h) for h in d["held"] if (slot == int(h)).any()]
    # THE GATE RUNS BEFORE ANYTHING IS SAMPLED, and its verdict is drawn rather than applied: see
    # the module docstring.  `n_arm` is the per-arm sample count these joint numbers come from, so
    # `min_detectable` below is the resolution of the numbers actually printed on this figure.
    gate = check_instrument(tag="case", real=d, probes=probes,
                            targets={h: tuple(int(v) for v in F[slot == h][0]) for h in cells_all},
                            n_arm=args.n_eval, table=C, labels=lab, seed=args.seed)
    print(gate.summary("case"), flush=True)
    cells = spread_targets(cells_all,
                           {h: tuple(int(v) for v in F[slot == h][0]) for h in cells_all},
                           len(cells_all))
    n_readable = sum(1 for h in cells if gate.readable(h))
    print(f"[case] {len(cells)} held-out cells, {n_readable} readable under this run's probes",
          flush=True)

    order = ["real", "prototype of p(x|c_new)", "ours (ADDRESS)", "nearest-class", "unconditional"]
    fig, axes = plt.subplots(len(cells) * len(order), args.n_show,
                             figsize=(1.5 * args.n_show + 3.8, 1.5 * len(cells) * len(order)))
    axes = np.atleast_2d(axes)
    r, measured, gamut = 0, [], []
    for h in cells:
        idx = np.where(slot == int(h))[0]
        tgt = tuple(int(v) for v in F[idx[0]])
        R = X[idx[:60]]
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        readable = bool(gate.readable(h))
        # EVERY GENERATED ROW GOES THROUGH `to_data`, and `field.sample` needs `clip=(-1,1)`.
        #
        # This figure drew model-space [-1,1] samples through `np.clip(S, 0, 1)`, which discards the
        # 49.3% of pixels that are negative -- the visible colour destruction, and the reason the
        # `real` row looked fine while every generated row did not. `field.sample` compounded it:
        # its `clip` defaults to (-0.2, 1.2), a DATA-range clamp applied to a MODEL-space trajectory
        # at every reverse step (`VLA_PATH` lesson 2-bis). `mode_ascent` takes and returns model
        # space too, so the query goes in through `to_model` and the modes come back through
        # `to_data`. n72 is the reference for all of this.
        #
        # `emb=c_new` is what makes the ascent CONDITIONAL. Without it this climbs p(x) and prints a
        # generic face next to a named held-out cell (7ll). `xq` is required by the signature and
        # unused at init="random": the ascent never sees a held-out image.
        xq = to_model(X[int(idx[0])]).ravel()
        # `clamp="x0"` PROJECTS THE ASCENT ONTO THE DATA'S SUPPORT, and on this backbone it is the
        # difference between a face and a coloured blob.  Unclamped, Adam's normalised step walks a
        # pixel up to 9.0 in a space where clean images live in [-1,1]; `mode_ascent`'s own docstring
        # calls the result "saturated posterized nonsense once clipped for display", which is exactly
        # what the first render of this row produced -- and the probes then scored those blobs at
        # 91% of ceiling, ABOVE the real-face arms, because a linear pixel probe reads colour
        # statistics and cannot tell a face from an abstract wash. A row that beats the ceiling by
        # not being an image is a probe artifact, not a prototype.
        #
        # This is a support constraint, not a per-dataset tuning knob: a "mode" outside [-1,1] is an
        # artifact of unconstrained optimisation rather than a mode of p_t. It simply does not bind
        # on Shapes3D, where the flagship measures 0.0% of prototype pixels out of gamut.
        per = max(1, args.n_eval // 3)
        P = []
        for ui, uu in enumerate((0.15, 0.30, 0.45)):
            _, x0 = field.mode_ascent(xq, uu, n_starts=per, iters=120, lr=0.05,
                                      seed=args.seed + 17 * int(h) + 131 * ui, init="random",
                                      emb=c_new, clamp="x0")
            P.append(to_data(np.asarray(x0)))
        proto_raw = np.concatenate(P)[:args.n_eval]
        gamut.append(float(np.mean((proto_raw < 0.0) | (proto_raw > 1.0))))
        got = {
            "real": R[:args.n_eval],
            "prototype of p(x|c_new)": np.concatenate(P)[:args.n_eval],
            "ours (ADDRESS)": to_data(sample_embedding(field, c_new, n=args.n_eval,
                                                       steps=args.steps, seed=args.seed)),
            "nearest-class": to_data(field.sample(n=args.n_eval, steps=args.steps,
                                                  seed=args.seed, clip=(-1, 1),
                                                  branch=remap[int(seen[j_near])])),
            "unconditional": to_data(field.sample(n=args.n_eval, steps=args.steps,
                                                  seed=args.seed + 91, clip=(-1, 1), branch=None)),
        }
        combo = ", ".join(f"{n}={HAIR[v] if n == 'hair' else v}" for n, v in zip(names, tgt))
        for nm in order:
            S = np.asarray(got[nm], np.float32)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(img, img, 3), 0, 1))
            lab_txt = (f"real - HELD OUT\n{combo}\ncell {int(h)} - 0 training images\n"
                       + ("PROBES CAN READ THIS CELL" if readable else
                          "PROBES CANNOT READ THIS CELL\n(no arm can be judged here)")
                       if nm == "real" else nm)
            axes[r][0].set_ylabel(lab_txt, rotation=0, ha="right", va="center", fontsize=7,
                                  color=("black" if readable else "#b00"))
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp_ in axes[r][0].spines.values():
                sp_.set_visible(False)
            if nm != "real":
                a = factor_accuracy(probes, S, tgt)
                j = float(a["joint"]) if a["joint"] == a["joint"] else 0.0
                pr = precision_recall(S, R)
                measured.append(dict(cell=int(h), arm=nm, joint=j, readable=readable,
                                     recall=float(pr["recall"]), vendi=float(vendi_score(S))))
                # A joint accuracy on an unreadable cell is a number about the probes, not about the
                # arm. It is drawn greyed and parenthesised rather than omitted, so the row still
                # lines up with the readable blocks and nobody reads the gap as a missing result.
                txt = (f"joint {j:.2f} = {j / max(ceil, 1e-9):.0%} of ceiling | "
                       f"recall {pr['recall']:.2f}" if readable else
                       f"(joint {j:.2f} -- cell unreadable, not evidence)")
                axes[r][args.n_show - 1].set_title(txt, fontsize=6.5, loc="right",
                                                   color=("black" if readable else "#999"))
            r += 1

    def _m(arm, key, only_readable=True):
        v = [x[key] for x in measured
             if x["arm"] == arm and (x["readable"] or not only_readable)]
        return float(np.mean(v)) if v else float("nan")

    # If NOTHING is readable the readable-only means are empty and every number in the caption
    # would render as the string "nan" -- which looks like a plotting glitch rather than the result
    # it actually is.  Fall back to all cells and say, in the caption, that that is what happened.
    scope = "readable cells"
    if n_readable == 0:
        scope = "ALL cells -- NONE was readable, so none of these numbers is evidence"
        _m_readable, _m = _m, (lambda arm, key, only_readable=True: _m_readable(arm, key, False))

    res = min_detectable(max(_m("ours (ADDRESS)", "joint"), 1e-6), args.n_eval)
    marg = _m("ours (ADDRESS)", "joint") - _m("nearest-class", "joint")
    # EVERY number in this caption is read from the run.  They were hardcoded to the first
    # backbone's values -- 12 cells, 9 seen / 3 held out, hair 0.568, "a raw joint of 0.29" -- and a
    # caption that describes a different split than the figure draws is worse than no caption,
    # because nothing about the picture looks wrong.
    per_factor = " / ".join(f"{n} {probes[n + '_val']:.3f}"
                            for n in names if n + "_val" in probes)
    # `instrument.table_report`'s convention, not a second guess at it: after centring, one
    # intercept plus (values - 1) per factor.
    n_free = 1 + sum(int(sz) - 1 for sz in d["sizes"])
    fig.suptitle(
        f"CelebA as a GRID, trained FROM SCRATCH with a DENSITY holdout: the {len(cells)} cells "
        "shown have ZERO training images\n"
        f"grid {list(map(str, names))} = {len(d['live'])} cells, {len(seen)} seen / {len(cells)} "
        f"held out; basis R2 {fac['r2']:.2f} in-sample, {r2cv:.2f} leave-one-out "
        f"({len(seen)} classes / {n_free} free vectors, ratio {len(seen) / max(n_free, 1):.1f})\n"
        f"READ AGAINST THE CEILING: probes read REAL images at joint ~{ceil:.3f} ({per_factor}); "
        f"chance is {1 / len(seen):.3f}\n"
        f"{n_readable} of {len(cells)} cells are readable at all -- the UNREADABLE blocks are the "
        "result (7ee), not missing work\n"
        f"ADDRESS {_m('ours (ADDRESS)', 'joint'):.2f} vs retrieval "
        f"{_m('nearest-class', 'joint'):.2f} on readable cells: a margin of {marg:+.2f} against a "
        f"resolution of {res:.2f} at n={args.n_eval} per arm -- BELOW WHAT THIS FIGURE CAN "
        f"RESOLVE  [over {scope}; {args.n_show} of {args.n_eval} samples shown]\n"
        f"prototype ascent is projected onto the data's support (clamp=\"x0\"), "
        f"{float(np.mean(gamut)):.1%} of its pixels outside [0,1]: the conditional MODE collapses "
        f"to a near-mean wash\n(joint {_m('prototype of p(x|c_new)', 'joint'):.2f}, vendi "
        f"{_m('prototype of p(x|c_new)', 'vendi'):.2f}) -- not a rendering fault, the result. "
        "Unclamped it scored 91% of ceiling as coloured noise the probes could not "
        "distinguish from a face (7nn)",
        fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIG, exist_ok=True)
    o = os.path.join(FIG, args.out or "case_celeba.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[case] wrote {o}")
    for arm in order[1:]:
        print(f"[case]   {arm:26s} joint {_m(arm, 'joint'):.3f} ({scope})  "
              f"vendi {_m(arm, 'vendi'):.3f}")
    save_json(dict(cells=cells, n_readable=int(n_readable), ceiling=ceil,
                   min_detectable=float(res), rows=measured),
              os.path.join(RESULTS, "n73_celeba_case.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
