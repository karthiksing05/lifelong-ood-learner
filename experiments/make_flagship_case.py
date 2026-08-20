"""The flagship case study: figures/initial/case_flagship_s3d.png.

RESULTS 7jj is the strongest result in this repository -- ADDRESS 0.983 against retrieval's 0.010 on
a PRETRAINED backbone -- and it had no picture. That is exactly the gap that let a range fault live
for a day in the CelebA thread: a margin of 0.049 and a margin of 0.018 read identically as numbers,
and only the case-study image showed that half the pixels were being clipped away.

Rows, and what each is for:

    real            actual images of the held-out combination -- the target, never shown to the model
    prototype       the MODE of p(x | c_new) -- Adam ascent on log p_t conditioned on the composed
                    address, from random starts.  Built to show the IDENTITY half of THEORY 4 on its
                    own, and it DOES NOT: measured, mode ascent is a WORSE reader of the address than
                    simply sampling it, and it is not even less diverse.  The row is kept because
                    that IS the finding -- on a conditional this sharply peaked there is no
                    identity/coverage gap left to show, and the mode-seeking half of the companion
                    paper's pipeline buys nothing here.  The caption carries both numbers.
    ours (ADDRESS)  compose(fac, factors) and sample the model's own conditional -- the COVERAGE half
    nearest-class   the closest SEEN class embedding: the control for "the model already had it"
    unconditional   the floor

The prototype row is the reason `mode_ascent` grew `emb=`.  Until 2026-08-19 both implementations
ascended the NULL-embedding density, so a "prototype" panel on a conditional model showed the mode of
p(x) -- a plausible-looking room that has nothing to do with the concept, and no scalar in the figure
would have said so.  Random starts, not query starts: a prototype seeded from the held-out image
invites exactly the reading the row exists to refute.

The caption carries the numbers a reader needs to judge the pictures rather than admire them,
INCLUDING the one that does not flatter us: the composed conditional is accurate and sharply peaked
(vendi 1.14 against the unconditional arm's 2.08), so these rows are expected to look *similar to
each other within a row*. Reporting the accuracy without that would be 7dd's warning ignored.

    python experiments/make_flagship_case.py --ckpt results/cache/ddpm_s3d_full.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from common import CACHE                                              # noqa: E402

from lifelong_ood.data.factored import decode_slot, spread_targets    # noqa: E402
from lifelong_ood.data.shapes3d import load_shapes3d                  # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                save_json, vendi_score)
from lifelong_ood.instrument import check_instrument, to_data_space   # noqa: E402
from lifelong_ood.neural.idealwords import compose, cv_r2, factorize  # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field               # noqa: E402

FIG = os.path.join(HERE, "figures", "initial")
HUE = ["red", "orange", "yellow", "chartreuse", "green",
       "spring", "cyan", "azure", "blue", "purple"]
SHAPE = ["cube", "cylinder", "sphere", "capsule"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_s3d_full.pt"))
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--n-cells", type=int, default=6,
                   help="held-out combinations to show.  Chosen by `spread_targets`, so raising "
                        "this buys DIFFERENT factor values rather than more of the same corner.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="case_flagship_s3d.png")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    d["seen"] = np.isin(slot, seen_slots)
    C = np.asarray(ck["ema_cemb"])[[remap[int(s)] for s in seen_slots]]
    lab = np.asarray(decode_slot(seen_slots, sizes), int)
    fac = factorize(C, lab)
    r2cv, _ = cv_r2(C, lab)
    probes = fit_factor_probes(d, seed=args.seed)
    img = int(d.get("img", 64))

    held = [s for s in sorted(set(slot.tolist())) if s not in set(seen_slots.tolist())]
    held = [h for h in held if (slot == h).sum() >= 8]
    gate = check_instrument(tag="fig", real=d, probes=probes,
                            targets={int(h): tuple(int(v) for v in F[slot == h][0]) for h in held},
                            n_arm=args.n_show, table=C, labels=lab, seed=args.seed)
    # SPREAD, not the first N in slot order -- which is how all four cells came out floor=red in
    # every version of this figure before 2026-08-19.  A figure whose point is that composition
    # spans a design must not show one corner of it.
    readable = [h for h in held if gate.readable(int(h))]
    cells = spread_targets(readable,
                           {int(h): tuple(int(v) for v in F[slot == h][0]) for h in readable},
                           args.n_cells)
    # Say what was chosen FROM, not just what was chosen: the old line printed the truncated list,
    # so "4 readable held-out cells" read as "only 4 are readable" when the real number was far
    # larger and the other three factors were simply never reached.
    spread = [sorted({int(F[slot == h][0][f]) for h in cells}) for f in range(len(sizes))]
    print(f"[fig] {len(readable)} readable held-out cells; showing {len(cells)} "
          f"spread over {[len(v) for v in spread]} distinct values of "
          f"{[str(x) for x in d['names']]}: {cells}",
          flush=True)

    order = ["real", "prototype of p(x|c_new)", "ours (ADDRESS)", "nearest-class",
             "unconditional"]
    fig, axes = plt.subplots(len(cells) * len(order), args.n_show,
                             figsize=(1.5 * args.n_show + 3.6, 1.5 * len(cells) * len(order)))
    axes = np.atleast_2d(axes)
    r = 0
    gamut, measured = [], []
    for h in cells:
        idx = np.where(slot == int(h))[0]
        tgt = tuple(int(v) for v in F[idx[0]])
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        # u=0.30 / iters=120 / lr=0.05 are diag_ve_discovery's checked settings for THIS field --
        # the ones where modes provably improve the eps error over their own starting point.  `xq` is
        # required by the signature but unused at init="random"; the ascent never sees a held-out
        # image.
        _, proto = field.mode_ascent(X[idx[0]], 0.30, n_starts=args.n_show, iters=120, lr=0.05,
                                     seed=args.seed, init="random", emb=c_new)
        proto = np.asarray(to_data_space(field, proto), np.float32)
        gamut.append(float(np.mean((proto < 0.0) | (proto > 1.0))))
        got = {
            "real": X[idx[:args.n_show]],
            "prototype of p(x|c_new)": proto,
            "ours (ADDRESS)": to_data_space(field, field.sample(
                n=args.n_show, steps=args.steps, seed=args.seed, emb_override=c_new)),
            "nearest-class": to_data_space(field, field.sample(
                n=args.n_show, steps=args.steps, seed=args.seed,
                branch=remap[int(seen_slots[j_near])])),
            "unconditional": to_data_space(field, field.sample(
                n=args.n_show, steps=args.steps, seed=args.seed + 91, branch=ck["null_id"])),
        }
        combo = (f"floor={HUE[tgt[0]]}, object={HUE[tgt[1]]}, {SHAPE[tgt[2]]}")
        for nm in order:
            S = np.asarray(got[nm], np.float32)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(img, img, 3), 0, 1))
            lab_txt = (f"real - HELD OUT\n{combo}\nslot {int(h)} - 0 training images"
                       if nm == "real" else nm)
            axes[r][0].set_ylabel(lab_txt, rotation=0, ha="right", va="center", fontsize=7)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp_ in axes[r][0].spines.values():
                sp_.set_visible(False)
            if nm != "real":
                j = float(factor_accuracy(probes, S, tgt)["joint"])
                measured.append(dict(slot=int(h), arm=nm, joint=j,
                                     vendi=float(vendi_score(S))))
                # Vendi on the two conditional arms only, because the contrast between them IS the
                # claim: same address, one mode against a distribution.
                v = (f" | vendi {float(vendi_score(S)):.2f}"
                     if nm in ("prototype of p(x|c_new)", "ours (ADDRESS)") else "")
                axes[r][args.n_show - 1].set_title(f"joint {j:.2f}{v}", fontsize=7, loc="right")
            r += 1

    # Short lines, because a caption clipped at both margins carries nothing. The detail lives in
    # RESULTS 7jj; this says only what a reader needs to judge the pictures.
    def _m(arm, key):
        v = [d[key] for d in measured if d["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    # SHORT lines.  The docstring above says a caption clipped at both margins carries nothing, and
    # the first version of this one was clipped at both margins.
    fig.suptitle(
        "SHAPES3D, PRETRAINED backbone: every combination shown has ZERO training images\n"
        "ADDRESS 0.983 | nearest-class 0.010 | unconditional 0.003 | real-image ceiling 1.000\n"
        f"400 classes / 25 free vectors (ratio 18.2); basis R2 {fac['r2']:.2f} in-sample, "
        f"{r2cv:.2f} leave-one-out\n"
        "`nearest-class` is the closest SEEN class: wrong by exactly ONE factor in every row\n"
        f"MODE of the address {_m('prototype of p(x|c_new)', 'joint'):.2f} vs SAMPLING it "
        f"{_m('ours (ADDRESS)', 'joint'):.2f}, and no less diverse "
        f"(vendi {_m('prototype of p(x|c_new)', 'vendi'):.2f} vs "
        f"{_m('ours (ADDRESS)', 'vendi'):.2f}):\nmode-seeking is the WORSE reader here",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIG, exist_ok=True)
    o = os.path.join(FIG, args.out)
    fig.savefig(o, dpi=130)
    print(f"[fig] wrote {o}")
    for arm in order[1:]:
        print(f"[fig]   {arm:26s} joint {_m(arm, 'joint'):.3f}  vendi {_m(arm, 'vendi'):.3f}")
    save_json(dict(cells=[int(c) for c in cells], rows=measured,
                   gamut_outside_01=float(np.mean(gamut))),
              os.path.join(HERE, "results", "n78_flagship_case.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
