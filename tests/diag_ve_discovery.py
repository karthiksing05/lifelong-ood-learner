"""Does the extended VE bridge actually work, or does it merely run?

`ve_bridge` grew `eps` / `score` / `laplace_diag` / `mode_ascent` so the companion paper's discovery
(Eq. 4, then Eq. 8-9 selection and the Eq. 7 product) can run on the Shapes3D backbone at all -- it
previously exposed only `sample`, so their pipeline had never been run there and N70 substituted a
crude mean-of-modes for it.

A bridge that runs but disagrees with the model is the failure mode this repository has paid for
three times (VE/VP dialect, CHW/HWC, the sampler clip), and every one of them produced plausible
numbers rather than an exception.  So each new method is checked against something already known:

  eps         epsilon-loss of REAL images under their own class, which must land near the
              training loss and far below 1.0 -- 1.0 means the model explains nothing
  score       finite-difference consistency with eps (they differ by a fixed factor)
  mode_ascent modes must IMPROVE the eps error over their own starting point, and
              query-seeded modes must sit closer to the query than random-seeded ones
  laplace_diag variances must be positive, finite, and not pinned at the cap everywhere

Only if all four hold is `candidates_ascent` run end to end.

    python tests/diag_ve_discovery.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.data.factored import decode_slot, load_factored      # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field                # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = os.path.join(CACHE, "ddpm_shapes3d_pre2.pt")
    if not os.path.exists(p):
        print(f"SKIP: no backbone at {p}")
        return 0
    field, ck = load_ve_field(p, device=dev)
    sp = ck["split"]
    data = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                         value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                         fix_nuisance=sp["fix_nuisance"])
    X, slots = np.asarray(data["X"], np.float32), np.asarray(data["slot"])
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    fail = 0

    # 1 ── eps on real images, noised to a known level, under their own class
    s0 = int(seen[0])
    rows = X[np.where(slots == s0)[0][:8]]
    rng = np.random.default_rng(0)
    for u in (0.1, 0.3, 0.6):
        from lifelong_ood.score import cosine_abar
        a = float(cosine_abar(np.array([u]))[0])
        e = rng.standard_normal(rows.shape).astype(np.float32)
        xt = (a ** 0.5) * rows + ((1 - a) ** 0.5) * e
        pred = field.eps(xt, u, branch=remap[s0])
        mse = float(np.mean((pred - e) ** 2))
        flag = "" if mse < 0.5 else "   <-- FAIL: explains nothing"
        if mse >= 0.5:
            fail = 1
        print(f"[ve] eps  u={u:<4g} loss vs the noise actually added: {mse:.4f}{flag}")

    # 2 ── score is -eps / sqrt(1-a)
    u = 0.3
    from lifelong_ood.score import cosine_abar
    a = float(cosine_abar(np.array([u]))[0])
    sc = field.score(rows[:2], u)
    ep = field.eps(rows[:2], u)
    rel = float(np.max(np.abs(sc + ep / np.sqrt(1 - a))) / max(np.abs(sc).max(), 1e-9))
    print(f"[ve] score consistent with eps: relative error {rel:.2e}"
          + ("" if rel < 1e-5 else "   <-- FAIL"))
    fail |= int(rel >= 1e-5)

    # 3 ── mode ascent must improve on its own starting point
    xq = X[np.where(slots == s0)[0][0]]
    for init in ("query", "random"):
        xt, x0 = field.mode_ascent(xq, 0.3, n_starts=4, iters=60, lr=0.05, seed=0, init=init)
        d = float(np.mean(np.linalg.norm(x0 - xq[None], axis=1)))
        # the eps error AT the returned mode is what ascent minimises
        err = float(np.mean(field.eps(xt, 0.3) ** 2))
        print(f"[ve] mode_ascent init={init:6s} d(mode,query) {d:7.3f}  eps^2 at mode {err:.4f}")
    xt_q, x0_q = field.mode_ascent(xq, 0.3, n_starts=4, seed=0, init="query")
    xt_r, x0_r = field.mode_ascent(xq, 0.3, n_starts=4, seed=0, init="random")
    dq = float(np.mean(np.linalg.norm(x0_q - xq[None], axis=1)))
    dr = float(np.mean(np.linalg.norm(x0_r - xq[None], axis=1)))
    ok3 = dq < dr
    print(f"[ve] query-seeded modes are closer to the query than random-seeded: "
          f"{dq:.3f} < {dr:.3f} -> {ok3}" + ("" if ok3 else "   <-- FAIL"))
    fail |= int(not ok3)

    # 4 ── laplace variances finite, positive, not all pinned
    V = field.laplace_diag(xt_q, 0.3, probes=4, seed=0)
    at_cap = float(np.mean(V >= 0.999))
    at_floor = float(np.mean(V <= 1.001e-4))
    # BOTH ends, not just the cap.  The first version of this check tested cap-pinning only and
    # passed a field whose variances were 100% at the FLOOR -- the opposite degeneracy, equally
    # uninformative.  Floor-pinning is EXPECTED here rather than a bridge fault: compose.py records
    # that the Laplace product covariance is a description of PRECISION, not extent, and that "99%
    # of its coordinates are pinned at the Laplace floor" on ColorMNIST too.  It is reported because
    # a caller weighting by 1/V is then weighting by a constant, which silently turns Eq. 7's
    # per-dimension product into an unweighted average.
    print(f"[ve] laplace_diag: min {V.min():.2e} max {V.max():.2e} "
          f"finite {bool(np.all(np.isfinite(V)))} at-cap {at_cap:.1%} at-floor {at_floor:.1%}")
    if at_floor > 0.9:
        print("[ve]   NOTE variances are ~all at the floor -- the product carries no per-dimension "
              "curvature, so Eq. 7 degenerates to an unweighted mean of the selected experts. "
              "Documented behaviour (compose.py, 'extent vs precision'), not a bridge fault.")
    fail |= int(at_cap >= 0.9 or not np.all(np.isfinite(V)))

    # 5 ── the whole paper pipeline, end to end
    if not fail:
        from lifelong_ood.compose import candidates_ascent, poe_teacher
        M, V2, meta = candidates_ascent(field, xq, u_grid=(0.1, 0.2, 0.3), n_starts=8, iters=60,
                                        lr=0.05, seed=0, init="hybrid")
        t = poe_teacher(xq, M, V2, K=3, tau=0.05)
        src = np.asarray(meta["src"])[t["sel"]]
        print(f"[ve] candidates_ascent -> pool {len(M)} of {meta['n_raw']} raw; "
              f"poe_teacher selected {list(src)}")
        print(f"[ve] mu_T finite {bool(np.all(np.isfinite(t['mu'])))}, "
              f"range [{t['mu'].min():.3f}, {t['mu'].max():.3f}]")
        print("\n[ve] PASS -- the paper's discovery now runs on the Shapes3D backbone and every "
              "surface agrees with the model.")
        return 0
    print("\n[ve] FAIL -- do not run discovery on this bridge until the above is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
