"""Does EXTEND BASIS add a DIRECTION where ADDRESS only adds a point?  (tests 6e-bis's prediction)

6e-bis established that ADDRESS is span-limited by construction: `c_new = mu + sum_f gain_f*V_f` is
built from least-squares fits to the SEEN embeddings, so it lands in their affine span at relative
residual 0.0000 while a genuinely TRAINED embedding for the same held-out class sits at 0.7359.  That
explains why 12 consolidated concepts could not move a readout over conditionings: no new axis.

The prediction that follows is falsifiable and this tests it.  EXTEND BASIS fits a NEW factor-value
direction with `extend_basis` -- a free 256-vector optimised against real images, norm-matched to the
factor's existing values but otherwise unconstrained.  If the span account is right, that vector
should sit OUTSIDE the span of the seen embeddings, unlike ADDRESS, and compositions built with it
should too.

`ddpm_colormnist_nobg3.pt` is the backbone for it: background value 3 was withheld ENTIRELY, so no
direction for it exists and ADDRESS provably cannot express it -- which is the definition of the
EXTEND tier.

    python tests/diag_extend_span.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_model              # noqa: E402
from lifelong_ood.neural.idealwords import (compose, extend_basis,     # noqa: E402
                                            factorize, slot_factors)

CACHE = os.path.join(HERE, "results", "cache")


def affine_residual(target, basis_rows):
    B = np.asarray(basis_rows, np.float64)
    t = np.asarray(target, np.float64).ravel()
    c = B.mean(0)
    A = (B - c).T
    sol, *_ = np.linalg.lstsq(A, t - c, rcond=None)
    return float(np.linalg.norm((t - c) - A @ sol) / (np.linalg.norm(t - c) + 1e-12))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(CACHE, "ddpm_colormnist_nobg3.pt"), map_location="cpu",
                    weights_only=False)
    remap = ck["remap"]
    seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    field = load_field(os.path.join(CACHE, "ddpm_colormnist_nobg3.pt"), device=dev)
    fac = factorize(C, slot_factors(seen))
    print(f"[ext] nobg3: {len(seen)} trained slots, basis R^2 {fac['r2']:.3f}, "
          f"factor sizes {fac['sizes']} -- NO direction exists for bg=3", flush=True)

    data = load_colormnist(n_per_slot=40, seed=0)
    tgt_slots = [s for s in np.unique(data["slot"]) if int(s) % 4 == 3][:6]
    X, labels = [], []
    for s in tgt_slots:
        idx = np.where(data["slot"] == int(s))[0][:8]
        for i in idx:
            X.append(to_model(data["X"][i]).ravel())
            labels.append(slot_factors(np.array([int(s)]))[0])
    X = np.stack(X); labels = np.stack(labels)
    print(f"[ext] fitting a NEW bg direction from {len(X)} real images of bg=3", flush=True)
    v, hist = extend_basis(field, X, labels, fac, 2, n_steps=400, lr=0.05, batch=8,
                           seed=0, norm_match=True)
    print(f"[ext] extend_basis: loss {hist['first']:.4f} -> {hist['final']:.4f}, "
          f"||v|| {hist['norm']:.3f} (target {hist['target_norm']:.3f})", flush=True)

    # scale bar and the two tiers
    loo = [affine_residual(C[i], np.delete(C, i, 0)) for i in range(0, len(C), 7)]
    addr, ext = [], []
    for s in tgt_slots:
        tgt = slot_factors(np.array([int(s)]))[0]
        # ADDRESS cannot express bg=3; the closest it can do is a bg it HAS
        c_addr = compose(fac, (int(tgt[0]), int(tgt[1]), 0))
        addr.append(affine_residual(c_addr, C))
        c_ext = np.asarray(fac["mu"], np.float64).copy()
        for f in range(3):
            c_ext = c_ext + (v if f == 2 else fac["V"][f][int(tgt[f])])
        ext.append(affine_residual(c_ext, C))

    print(f"\n[ext] {'quantity':56s} {'median':>9}")
    print("[ext] " + "-" * 68)
    print(f"[ext] {'ADDRESS composition (bg it already has)':56s} {np.median(addr):>9.4f}")
    print(f"[ext] {'EXTEND composition (new bg direction)':56s} {np.median(ext):>9.4f}")
    print(f"[ext] {'the new direction v itself':56s} "
          f"{affine_residual(v, C):>9.4f}")
    print(f"[ext] {'a seen embedding vs the other seen (scale bar)':56s} {np.median(loo):>9.4f}")
    print("\n[ext] VERDICT")
    if np.median(ext) > 5 * max(np.median(addr), 1e-9):
        print("[ext]   CONFIRMED -- EXTEND leaves the span ADDRESS is confined to. The tier that")
        print("[ext]   costs 1 KB buys a DIRECTION; the tier that costs 72 B buys a point. That is")
        print("[ext]   the axis along which representation can grow.")
    else:
        print("[ext]   NOT confirmed: EXTEND lands in the span too, so 6e-bis's prediction fails")
        print("[ext]   and representational growth must come from BLOCK or from nowhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
