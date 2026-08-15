"""N65 -- can the router's operating point be chosen WITHOUT held-out labels?

N62 establishes that a good `u` exists per backbone and that the router transfers when `u` is right
(Shapes3D held-out AUROC ~0.85 at u*=0.85, against 0.354 at ColorMNIST's u=0.10).  But N62 picks
`u` by maximising separation against REAL HELD-OUT QUERIES -- and a deployed router does not have
those.  If it did, it would not need a novelty detector.

So N62 answers "does a good operating point exist" and this answers the question that decides
whether any of it is usable: **can the operating point be found from training data alone?**

The criterion is label-free in the sense `experiments/common.py:backbone_gains` established -- it
uses only images the model trained on and the labels it trained them with, never a held-out
combination:

    u* = argmax_u  AUROC( loss under the CORRECT class , loss under the best WRONG class )

over SEEN training images.  In words: the noise level at which the class label matters most.  The
hypothesis under test is that this coincides with the noise level at which NOVELTY is most visible,
which would make sense of both of N61's results at once -- a novel combination is exactly one no
class explains well, so the router can only see it where class identity is legible at all.

Reported per backbone:

  u_lf      the label-free choice
  u_oracle  the choice N62 would make with held-out labels
  auroc@u_lf     REAL held-out novelty AUROC at the label-free operating point   <- the number
  auroc@u_oracle the ceiling that choice is measured against
  auroc@0.10     the shipped ColorMNIST constant, i.e. what is deployed today

If `auroc@u_lf` lands near `auroc@u_oracle`, `U_NOVEL` becomes a calibrated per-backbone quantity
exactly like `backbone_gains`, and the router's negative result is retired.  If the label-free
criterion picks a bad u, the honest statement is that the router works only with an oracle -- which
is not a router.

    python experiments/n65_u_labelfree.py --seeds 3
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import RESULTS                                             # noqa: E402
from n61_router_ablation import (U_GRID, load_colormnist_setup,        # noqa: E402
                                 load_shapes3d_setup)

from lifelong_ood.evals import save_json                               # noqa: E402
from lifelong_ood.neural.router import (auroc, conditional_losses,     # noqa: E402
                                        null_conditioning)

U_SHIPPED = 0.10


def class_legibility(field, allc, X, slots, seen_slots, u, n_noise, seed, rng, prep, n_q=24):
    """AUROC separating the CORRECT class's loss from the best WRONG class's loss, on training data.

    Label-free with respect to the holdout: every query here is an image the model trained on, and
    its class is the label it was trained with.  No held-out combination is touched.

    `prep` is applied to every query, unconditionally.  It converts a stored row into the layout the
    backbone expects, and the two backbones disagree about that.  Deciding whether to apply it by
    comparing shapes would be silent when the conversion preserves shape -- which is the CHW/HWC
    fault that has already cost this repository a 974x loss error twice.
    """
    seen_sorted = np.asarray(sorted(int(v) for v in seen_slots))
    corr, wrong = [], []
    for s in rng.permutation(seen_slots)[:n_q]:
        idx = np.where(slots == int(s))[0]
        if not len(idx):
            continue
        L = conditional_losses(field, prep(X[int(idx[0])]), allc, u_grid=(u,),
                               n_noise=n_noise, seed=seed)
        cls = L[1:]                                    # drop the null row
        j = int(np.searchsorted(seen_sorted, int(s)))  # C's rows are in sorted-slot order
        corr.append(float(cls[j]))
        wrong.append(float(np.min(np.delete(cls, j))))
    return auroc(np.array(wrong), np.array(corr))      # wrong should score HIGHER (worse) loss


def novelty_auroc(field, allc, X, slots, held, seen_slots, u, n_noise, seed, rng, prep, n_q=24):
    """The real thing: held-out combinations vs trained ones, using held-out labels.  Reported as
    the number being predicted, never used to CHOOSE u."""
    def sig(pool):
        out = []
        for s in rng.permutation(pool)[:n_q]:
            idx = np.where(slots == int(s))[0]
            if len(idx):
                L = conditional_losses(field, prep(X[int(idx[0])]), allc, u_grid=(u,),
                                       n_noise=n_noise, seed=seed)
                out.append(float(L[1:].min()))
        return np.array(out)
    return auroc(sig(held), sig(seen_slots))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-q", type=int, default=24)
    p.add_argument("--n-noise", type=int, default=4)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    for bname, loader in (("ColorMNIST(nobg3)", load_colormnist_setup),
                          ("Shapes3D(pretrained)", load_shapes3d_setup)):
        print(f"\n[n65] ===== {bname} =====", flush=True)
        rec = {k: [] for k in ("u_lf", "u_oracle", "a_lf", "a_oracle", "a_shipped")}
        for seed in range(args.seeds):
            field, C, X, slots, seen, held, prep = loader(seed, dev)
            rng = np.random.default_rng(500 + seed)
            null = null_conditioning(field.model).reshape(1, -1)
            allc = np.concatenate([null, C])

            # --- choose u WITHOUT held-out labels -------------------------------------------
            leg = [(u, class_legibility(field, allc, X, slots, seen, u, args.n_noise, seed,
                                        np.random.default_rng(700 + seed), prep, args.n_q))
                   for u in U_GRID]
            u_lf = max(leg, key=lambda t: t[1])[0]

            # --- the oracle choice, and what each choice actually buys -----------------------
            nov = [(u, novelty_auroc(field, allc, X, slots, held, seen, u, args.n_noise, seed,
                                     np.random.default_rng(900 + seed), prep, args.n_q))
                   for u in U_GRID]
            u_or = max(nov, key=lambda t: t[1])[0]
            dn = dict(nov)
            rec["u_lf"].append(u_lf)
            rec["u_oracle"].append(u_or)
            rec["a_lf"].append(dn[u_lf])
            rec["a_oracle"].append(dn[u_or])
            rec["a_shipped"].append(dn[U_SHIPPED])
            print(f"[n65] seed {seed}: u_lf={u_lf:<5g} -> AUROC {dn[u_lf]:.3f} | "
                  f"u_oracle={u_or:<5g} -> {dn[u_or]:.3f} | shipped u=0.10 -> "
                  f"{dn[U_SHIPPED]:.3f}", flush=True)
            print(f"[n65]        legibility curve " +
                  " ".join(f"{u:g}:{a:.2f}" for u, a in leg), flush=True)
            del field
            torch.cuda.empty_cache()
        m = {k: float(np.mean(v)) for k, v in rec.items()}
        s = {k: float(np.std(v)) for k, v in rec.items()}
        out[bname] = dict(per_seed=rec, mean=m, sd=s)
        print(f"[n65] MEAN u_lf={m['u_lf']:.2f} AUROC {m['a_lf']:.3f}+/-{s['a_lf']:.3f} | "
              f"oracle u={m['u_oracle']:.2f} AUROC {m['a_oracle']:.3f} | "
              f"shipped {m['a_shipped']:.3f}", flush=True)

    print("\n[n65] " + "=" * 70)
    print("[n65] VERDICT")
    for bname, d in out.items():
        m = d["mean"]
        gap = m["a_oracle"] - m["a_lf"]
        print(f"[n65]   {bname}: label-free u={m['u_lf']:.2f} reaches {m['a_lf']:.3f} "
              f"vs oracle {m['a_oracle']:.3f} (gives up {gap:.3f}); shipped {m['a_shipped']:.3f}")
        if m["a_lf"] > 0.75 and gap < 0.10:
            print("[n65]     -> U_NOVEL IS CALIBRATABLE per backbone from training data alone.")
        elif m["a_lf"] > 0.75:
            print("[n65]     -> usable but the label-free choice leaves real AUROC on the table.")
        else:
            print("[n65]     -> label-free choice FAILS; the router needs an oracle u, which is "
                  "not a router. Report as such.")
    save_json(dict(backbones=out, u_grid=list(U_GRID), u_shipped=U_SHIPPED, config=vars(args)),
              os.path.join(RESULTS, "n65_u_labelfree.json"), seed=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
