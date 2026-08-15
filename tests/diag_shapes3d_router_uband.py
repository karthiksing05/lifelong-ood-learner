"""Is the Shapes3D router broken, or just reading the wrong noise level?

N58 reports router `s_novel` AUROC **0.354** on Shapes3D against ColorMNIST's 0.980 -- below chance,
which is worse than "no signal": it means held-out queries score LOWER than trained ones, the
opposite of the intended direction.

Before concluding the router does not transfer, there is a cheaper explanation to rule out.  The two
noise levels it reads are constants:

    U_NOVEL = 0.10     separates novel from in-distribution      (AUROC 0.985 on ColorMNIST)
    U_VOCAB = 0.45     separates new-vocabulary from new-combination

Both were CHOSEN on ColorMNIST by `diag_router_signals`, on a 32 px backbone with three factors.
Nothing says the informative band sits in the same place for a 64 px pretrained backbone whose
marginal is far easier -- and if it does not, 0.354 is a statement about a badly chosen constant, not
about the method.

This sweeps u across the schedule and reports AUROC at each level, for the conditional signal and for
the unconditional ablation.  Three outcomes, all informative:

    a band with AUROC well above 0.5   -> the signal exists and the CONSTANT was wrong
    flat at ~0.5 everywhere            -> no signal at any noise level; the router genuinely does not
                                          transfer, and 0.354 was noise around chance
    consistently BELOW 0.5             -> the signal is real but INVERTED here, which would need
                                          explaining before it could be used

    python tests/diag_shapes3d_router_uband.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.factored import decode_slot, load_factored      # noqa: E402
from lifelong_ood.neural.router import auroc, conditional_losses, null_conditioning  # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field                # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, ck = load_ve_field(os.path.join(CACHE, "ddpm_shapes3d_pre2.pt"), device=dev)
    sp = ck["split"]
    data = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                         value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                         fix_nuisance=sp["fix_nuisance"])
    remap = ck["remap"]
    slots_seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema_cemb"].numpy()[[remap[int(s)] for s in slots_seen]]
    null = null_conditioning(field.model).reshape(1, -1)
    slots = np.asarray(data["slot"])
    held = np.array(sorted(set(int(s) for s in np.unique(slots)) - set(slots_seen.tolist())))
    rng = np.random.default_rng(0)

    def queries(pool, n=16):
        out = []
        for s in rng.permutation(pool)[:n]:
            idx = np.where(slots == int(s))[0]
            if len(idx):
                out.append(np.asarray(data["X"], np.float32)[int(idx[0])])
        return out

    q_tr, q_ho = queries(slots_seen), queries(held)
    print(f"[uband] {len(q_tr)} trained queries, {len(q_ho)} held-out; "
          f"{len(C)} class conditionings", flush=True)
    print(f"\n{'u':>6} {'AUROC cond':>11} {'AUROC uncond':>13}   "
          f"{'mean s(held)':>12} {'mean s(train)':>13}")
    print("-" * 62)
    best = (None, 0.5)
    for u in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90):
        def sig(qs):
            cond, unc = [], []
            for x in qs:
                L = conditional_losses(field, x, np.concatenate([null, C]),
                                       u_grid=(u,), n_noise=4, seed=0)
                unc.append(float(L[0]))
                cond.append(float(L[1:].min()))
            return np.array(cond), np.array(unc)
        c_ho, u_ho = sig(q_ho)
        c_tr, u_tr = sig(q_tr)
        a_c = auroc(c_ho, c_tr)          # held-out should score HIGHER (more novel)
        a_u = auroc(u_ho, u_tr)
        print(f"{u:>6.2f} {a_c:>11.3f} {a_u:>13.3f}   {c_ho.mean():>12.5f} {c_tr.mean():>13.5f}",
              flush=True)
        if abs(a_c - 0.5) > abs(best[1] - 0.5):
            best = (u, a_c)

    print(f"\n[uband] most separating level: u={best[0]} at AUROC {best[1]:.3f} "
          f"[ColorMNIST uses u=0.10 and scores 0.980]")
    if best[1] > 0.75:
        print("[uband] VERDICT: the signal EXISTS -- the constant was wrong, not the method. "
              "U_NOVEL should be per-backbone, calibrated like the gains already are.")
    elif best[1] < 0.25:
        print("[uband] VERDICT: the signal is real but INVERTED on this backbone. That needs "
              "explaining before it could be used; do not simply flip the sign.")
    else:
        print("[uband] VERDICT: no separation at ANY noise level -- the router genuinely does not "
              "transfer here, and 0.354 was noise around chance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
