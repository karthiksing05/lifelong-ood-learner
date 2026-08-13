"""Why is composition strong while the representation is not?  (RESULTS 6e vs 3/5q)

The two results look contradictory.  ADDRESS composes held-out combinations well -- F1 0.630 our way,
0.756 for a basis direction, joint 0.556 -- yet N50 finds the representation of a frozen backbone
plus 12 consolidated concepts **indistinguishable from the un-adapted backbone** (linear digit
0.271 vs 0.266), while an oracle trained on all 160 reaches 0.406.  Twelve extra conditioning columns
buy essentially nothing.

The hypothesis this tests is that the two facts are **the same fact**, not a tension.

    ADDRESS builds c_new = mu + sum_f gain_f * V_f[value], and mu and every V_f are least-squares
    fits to the 120 SEEN embeddings.  So c_new lies in the AFFINE SPAN of what the backbone already
    had.  It is a new POINT, not a new DIRECTION.

If that is right:
  * composition works, because placing a concept correctly only needs a point in the right place;
  * the representation cannot improve, because a readout over conditionings gains no new axis --
    the 12 new columns are (nearly) functions of the 120 old ones;
  * and the 72-byte cost is the SAME statement: 72 bytes is what it costs to name a point in a space
    you already have.  Cheapness and representational inertness are two readings of one property.

The oracle is the control that makes this measurable rather than rhetorical: its 40 held-out classes
were TRAINED, so if the hypothesis holds their embeddings should sit measurably OUTSIDE the span of
the 120, where ours sit inside it.

Reports, for held-out classes:
    residual of c_new against the affine span of the 120 seen embeddings   (ours -- expect ~0)
    residual of the ORACLE's trained embedding against that same span      (expect >> 0)
    the same in units of the embeddings' own scale, so "small" is defined

    python tests/diag_span_vs_representation.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.neural.idealwords import compose, factorize, slot_factors   # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")


def affine_residual(target, basis_rows):
    """Relative residual of `target` against the AFFINE span of `basis_rows`.  -> float in [0, 1].

    Affine, not linear: the basis is `mu + sum V`, so a constant offset is free and a linear-span
    test would understate how well the span covers.
    """
    B = np.asarray(basis_rows, np.float64)
    t = np.asarray(target, np.float64).ravel()
    c = B.mean(0)
    A = (B - c).T                                    # [d, N]
    sol, *_ = np.linalg.lstsq(A, t - c, rcond=None)
    resid = (t - c) - A @ sol
    denom = np.linalg.norm(t - c)
    return float(np.linalg.norm(resid) / (denom + 1e-12))


def main():
    base = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu",
                      weights_only=False)
    orac = torch.load(os.path.join(CACHE, "ddpm_colormnist_oracle.pt"), map_location="cpu",
                      weights_only=False)
    rb, ro = base["remap"], orac["remap"]
    seen = np.array(sorted(rb))
    C_seen = base["ema"]["cemb.weight"].numpy()[[rb[int(s)] for s in seen]]
    held = np.array(sorted(set(int(s) for s in ro) - set(int(s) for s in seen)))
    print(f"[span] {len(seen)} seen, {len(held)} held-out; embedding dim {C_seen.shape[1]}")

    fac = factorize(C_seen, slot_factors(seen))
    print(f"[span] basis R^2 {fac['r2']:.3f} over the seen embeddings")

    # scale bar: how far a SEEN embedding sits from the span of the OTHER seen ones
    loo = []
    for i in range(0, len(seen), 7):
        loo.append(affine_residual(C_seen[i], np.delete(C_seen, i, 0)))
    print(f"[span] scale bar -- a SEEN embedding vs the span of the other {len(seen) - 1}: "
          f"median relative residual {np.median(loo):.4f}")

    ours, orc = [], []
    for s in held:
        tgt = slot_factors(np.array([int(s)]))[0]
        c_new = compose(fac, tuple(int(v) for v in tgt))
        ours.append(affine_residual(c_new, C_seen))
        c_tr = orac["ema"]["cemb.weight"].numpy()[ro[int(s)]]
        orc.append(affine_residual(c_tr, C_seen))
    ours, orc = np.array(ours), np.array(orc)

    print(f"\n[span] {'quantity':52s} {'median':>9} {'mean':>9}")
    print("[span] " + "-" * 72)
    print(f"[span] {'ADDRESS c_new vs span of the 120 seen':52s} {np.median(ours):>9.4f} "
          f"{ours.mean():>9.4f}")
    print(f"[span] {'ORACLE trained embedding vs that same span':52s} {np.median(orc):>9.4f} "
          f"{orc.mean():>9.4f}")
    print(f"[span] {'a seen embedding vs the other seen (scale bar)':52s} "
          f"{np.median(loo):>9.4f} {np.mean(loo):>9.4f}")

    print(f"\n[span] VERDICT")
    if np.median(ours) < 0.05 and np.median(orc) > 5 * max(np.median(ours), 1e-9):
        print("[span]   CONFIRMED. ADDRESS adds a POINT inside the existing span; the oracle's")
        print("[span]   trained embeddings sit outside it. Composition needs only the point, so it")
        print("[span]   works; a readout over conditionings gains no axis, so the representation")
        print("[span]   cannot improve. 72 bytes buys a name for a point you already had.")
    else:
        print("[span]   NOT confirmed at these thresholds -- read the numbers above, the")
        print("[span]   explanation for 6e is elsewhere.")
    print(f"[span]   ours {np.median(ours):.4f} vs oracle {np.median(orc):.4f} "
          f"= {np.median(orc) / max(np.median(ours), 1e-9):.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
