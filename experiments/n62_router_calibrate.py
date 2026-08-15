"""N62 -- does the router TRANSFER, or did I just pick the best u on the numbers I then reported?

This settles the repository's flagship negative result.  `docs/VLA_PATH.md` records:

    | The **router** transfers | ColorMNIST AUROC 0.980 - Shapes3D **0.354** | **low** -- does not transfer |

and OUTLINE repeats it.  N61 then measured Shapes3D at **0.932 +/- 0.018** at u=0.85 -- above
ColorMNIST's own 0.980-class performance, on the backbone that supposedly failed.

That number is NOT reportable as it stands.  N61 chose u by maximising AUROC over a 10-point grid on
the SAME queries it then scored, so the maximum is biased upward by exactly the selection.  A negative
result this load-bearing must not be overturned by a statistic that would look good on noise.

So: **calibrate u on one population, report on a disjoint one.**

  * slots are split into CALIB and EVAL halves -- disjoint in held-out AND in seen slots, so no query
    scored at eval time influenced the choice of u
  * u* = argmax conditional AUROC on CALIB
  * the reported number is AUROC on EVAL at u*, never on CALIB

Three arms are reported at eval, which is what makes the answer interpretable rather than merely
positive:

  eval_cond@u*       the honest transferred router
  eval_cond@u_dflt   the same router at ColorMNIST's u=0.10 -- what produced 0.354
  eval_unc@u*        the density ablation of 6c, at the operating point that matters

The three answer three different questions.  If eval_cond@u* is high while eval_cond@u_dflt is low,
the router transfers and 0.354 was an OPERATING-POINT artefact, not a property of the backbone -- and
`U_NOVEL` must become per-backbone, calibrated exactly as `backbone_gains` already is.  If
eval_cond@u* collapses toward chance, N61's peak was selection and the negative result STANDS.

    python experiments/n62_router_calibrate.py --seeds 4 --per-slot 3
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402
from n61_router_ablation import (U_GRID, load_colormnist_setup,        # noqa: E402
                                 load_shapes3d_setup)

from lifelong_ood.evals import save_json                               # noqa: E402
from lifelong_ood.neural.router import (auroc, conditional_losses,     # noqa: E402
                                        null_conditioning)

U_DEFAULT = 0.10          # the ColorMNIST-tuned operating point that produced Shapes3D's 0.354


def split_slots(pool, rng):
    """Disjoint halves of a slot pool.  Split on SLOTS, not on images: two images of the same
    composition are not independent, so splitting on images would leak the composition across the
    calibration/evaluation boundary and re-introduce the bias this experiment exists to remove."""
    p = rng.permutation(np.asarray(pool))
    h = len(p) // 2
    return p[:h], p[h:]


def queries(X, slots, pool, per_slot, prep, rng, cap):
    """Up to `per_slot` images per slot, then capped at `cap` total.  The cap balances the two
    populations: AUROC over 36-vs-111 costs runtime without buying resolution, since the smaller
    population sets the standard error."""
    out = []
    for s in pool:
        idx = np.where(slots == int(s))[0]
        if not len(idx):
            continue
        for i in rng.permutation(idx)[:per_slot]:
            out.append(prep(X[int(i)]))
    if len(out) > cap:
        out = [out[i] for i in rng.permutation(len(out))[:cap]]
    return out


def auroc_at(field, allc, q_ho, q_tr, u, n_noise, seed):
    """(conditional AUROC, unconditional-ablation AUROC) at one noise level."""
    def sig(qs):
        c, un = [], []
        for x in qs:
            L = conditional_losses(field, x, allc, u_grid=(u,), n_noise=n_noise, seed=seed)
            un.append(float(L[0]))
            c.append(float(L[1:].min()))
        return np.array(c), np.array(un)
    c_ho, u_ho = sig(q_ho)
    c_tr, u_tr = sig(q_tr)
    return auroc(c_ho, c_tr), auroc(u_ho, u_tr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--per-slot", type=int, default=3)
    p.add_argument("--n-noise", type=int, default=4)
    p.add_argument("--max-q", type=int, default=40, help="cap per population")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    for bname, loader in (("ColorMNIST(nobg3)", load_colormnist_setup),
                          ("Shapes3D(pretrained)", load_shapes3d_setup)):
        print(f"\n[n62] ===== {bname} =====", flush=True)
        print(f"{'seed':>4} {'u*':>5} {'calib':>7} | {'EVAL cond':>9} {'EVAL unc':>9} "
              f"{'eval@0.10':>9}", flush=True)
        per = {"ustar": [], "calib": [], "eval_cond": [], "eval_unc": [], "eval_dflt": []}
        for seed in range(args.seeds):
            field, C, X, slots, seen, held, prep = loader(seed, dev)
            rng = np.random.default_rng(1000 + seed)
            null = null_conditioning(field.model).reshape(1, -1)
            allc = np.concatenate([null, C])

            ho_c, ho_e = split_slots(held, rng)
            tr_c, tr_e = split_slots(seen, rng)
            qc_ho = queries(X, slots, ho_c, args.per_slot, prep, rng, args.max_q)
            qc_tr = queries(X, slots, tr_c, args.per_slot, prep, rng, args.max_q)
            qe_ho = queries(X, slots, ho_e, args.per_slot, prep, rng, args.max_q)
            qe_tr = queries(X, slots, tr_e, args.per_slot, prep, rng, args.max_q)

            # --- calibrate u on the CALIB half only -------------------------------------------
            cal = [(u, auroc_at(field, allc, qc_ho, qc_tr, u, args.n_noise, seed)[0])
                   for u in U_GRID]
            ustar = max(cal, key=lambda t: t[1])[0]
            calib_auc = dict(cal)[ustar]

            # --- report on the EVAL half, which never touched the choice of u -----------------
            ec, eu = auroc_at(field, allc, qe_ho, qe_tr, ustar, args.n_noise, seed)
            ed, _ = auroc_at(field, allc, qe_ho, qe_tr, U_DEFAULT, args.n_noise, seed)
            for k, v in (("ustar", ustar), ("calib", calib_auc), ("eval_cond", ec),
                         ("eval_unc", eu), ("eval_dflt", ed)):
                per[k].append(float(v))
            print(f"{seed:>4} {ustar:>5.2f} {calib_auc:>7.3f} | {ec:>9.3f} {eu:>9.3f} "
                  f"{ed:>9.3f}", flush=True)
            del field
            torch.cuda.empty_cache()

        m = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items()}
        out[bname] = dict(per_seed=per, mean=m, n_queries=len(qe_ho) + len(qe_tr))
        print(f"[n62] {'mean':>4} {m['ustar'][0]:>5.2f} {m['calib'][0]:>7.3f} | "
              f"{m['eval_cond'][0]:.3f}+/-{m['eval_cond'][1]:.3f} "
              f"{m['eval_unc'][0]:.3f}+/-{m['eval_unc'][1]:.3f} "
              f"{m['eval_dflt'][0]:.3f}+/-{m['eval_dflt'][1]:.3f}", flush=True)

    print("\n[n62] " + "=" * 70)
    print("[n62] VERDICT")
    for bname, d in out.items():
        m = d["mean"]
        ec, eu, ed = m["eval_cond"][0], m["eval_unc"][0], m["eval_dflt"][0]
        print(f"[n62]   {bname}: held-out AUROC {ec:.3f} at u*={m['ustar'][0]:.2f} "
              f"(default u=0.10 gives {ed:.3f}; density ablation {eu:.3f})")
        if ec > 0.75 and ec - ed > 0.15:
            print(f"[n62]     -> ROUTER TRANSFERS. The low number was the OPERATING POINT, not the")
            print(f"[n62]        backbone. U_NOVEL must be calibrated per backbone.")
        elif ec > 0.75:
            print(f"[n62]     -> router works, and already did at the default u.")
        else:
            print(f"[n62]     -> does NOT transfer on held-out queries; N61's peak was selection.")
        if ec - eu > 0.15:
            print(f"[n62]        Conditioning carries the decision here (+{ec - eu:.3f} over density).")
        else:
            print(f"[n62]        Density explains it (6c holds for this backbone).")
    save_json(dict(backbones=out, u_grid=list(U_GRID), u_default=U_DEFAULT, config=vars(args)),
              os.path.join(RESULTS, "n62_router_calibrate.json"), seed=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
