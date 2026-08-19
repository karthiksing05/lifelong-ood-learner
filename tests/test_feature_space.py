"""Which feature space can the k-NN metrics actually be trusted in? Decided by arms with known answers.

`evals._feat` is a pixel space and its own docstring records the retraction: Generalization recall
reads **0.000 in it and 0.180 in Inception's** on the same 1600 images (RESULTS 5e-bis). Fifteen
pinned claims are computed through it and four of them read exactly 0.000 -- and a k-NN metric
reading exactly zero is precisely what a wrong feature space produces.

Rather than argue about which space is right, calibrate both against constructions whose answers are
known in advance, **at the sample counts the experiments actually use** (24 samples against a 60-row
reference), because a metric can be fine at n=1000 and useless at n=24:

    arm          construction                                    correct answer
    real-split   two disjoint halves of REAL images of a class    precision AND recall high
    copy         24 copies of one real image                      recall ~0, memorized ~1
    permuted     real images, pixels shuffled by a fixed perm     precision ~0
    noise        uniform noise                                    precision ~0

`real-split` is the decisive one: it contains no generated images at all, so a space that scores it
badly is broken, full stop. That is the exact failure `_feat`'s docstring describes.

THE DECISION RULE, STATED BEFORE THE NUMBERS ARE SEEN:

  * new space passes all four and the old fails any  -> the move is a FIX; re-pin on new keys.
  * both pass                                        -> a nuisance; keep pixel, document the offset.
  * both fail at that n                              -> the metric is unusable at that sample count;
                                                        DELETE the claim rather than re-pin it in a
                                                        second wrong space (the four n25 claims are
                                                        the precedent).

This test PASSES as long as it can compute the table -- it is a calibration instrument, not a gate,
and it must be able to report that both spaces are broken.

    python tests/test_feature_space.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.evals import memorization, precision_recall        # noqa: E402

N_ARM, N_REF = 24, 60
FAIL = []


def ok(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


def colormnist_like(n=200, img=32, seed=0):
    """A stand-in with the structure the metrics care about: a class with genuine within-class
    variety (position jitter) plus colour identity. Position jitter is exactly what pixel distance
    mistakes for "different class", which is the failure being measured."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        x = np.full((img, img, 3), 0.05, np.float32)
        dy, dx = rng.integers(-6, 7, 2)
        y0, x0 = 10 + dy, 10 + dx
        x[max(y0, 0):y0 + 10, max(x0, 0):x0 + 10, 0] = 0.9        # a red blob that MOVES
        x[max(y0, 0):y0 + 10, max(x0, 0):x0 + 10, 1] = 0.2
        x += rng.normal(0, 0.02, x.shape).astype(np.float32)
        rows.append(np.clip(x, 0, 1).ravel())
    return np.asarray(rows, np.float32)


def arms(X, seed=0):
    rng = np.random.default_rng(seed)
    ref = X[:N_REF]
    perm = rng.permutation(X.shape[1])
    return {
        "real-split": X[N_REF:N_REF + N_ARM],
        "copy": np.repeat(X[N_REF:N_REF + 1], N_ARM, axis=0),
        "permuted": X[N_REF:N_REF + N_ARM][:, perm],
        "noise": rng.random((N_ARM, X.shape[1])).astype(np.float32),
    }, ref


def verdict(res):
    """Does this space give the four arms their known answers?  -> (passes, notes)."""
    bad = []
    if not (res["real-split"]["recall"] > 0.30 and res["real-split"]["precision"] > 0.30):
        bad.append("real-split scores low (real images judged unlike real images)")
    if res["copy"]["recall"] > 0.15:
        bad.append("copying is not detected by recall")
    if res["permuted"]["precision"] > 0.30:
        bad.append("scrambled pixels are certified as on-manifold")
    if res["noise"]["precision"] > 0.30:
        bad.append("noise is certified as on-manifold")
    return (not bad), bad


def main():
    from lifelong_ood import inception
    have_incep = inception.available()
    X = colormnist_like()
    A, ref = arms(X)
    spaces = ["pixel"] + (["inception"] if have_incep else [])
    print(f"[space] {N_ARM} samples against a {N_REF}-row reference; "
          f"spaces: {spaces}{'' if have_incep else '  (inception unavailable)'}")

    table = {}
    for sp in spaces:
        res = {}
        for name, S in A.items():
            pr = precision_recall(S, ref, space=sp)
            mem = memorization(S, ref, space=sp)
            res[name] = dict(precision=pr["precision"], recall=pr["recall"],
                             memorized=mem["frac_memorized"], used=pr.get("space", sp))
        table[sp] = res

    print(f"\n[space] {'arm':12s}" + "".join(f"{sp:>26s}" for sp in spaces))
    print(f"[space] {'':12s}" + "".join(f"{'prec   rec   mem':>26s}" for sp in spaces))
    for name in A:
        line = f"[space] {name:12s}"
        for sp in spaces:
            r = table[sp][name]
            line += f"{r['precision']:>10.3f}{r['recall']:>8.3f}{r['memorized']:>8.3f}"
        print(line)

    print()
    verdicts = {}
    for sp in spaces:
        good, notes = verdict(table[sp])
        verdicts[sp] = good
        print(f"[space] {sp:10s} {'PASSES all four arms' if good else 'FAILS: ' + '; '.join(notes)}")

    print(f"\n[space] {'=' * 68}")
    if not have_incep:
        print("[space] Only the pixel space could be measured, so no decision is available. The")
        print("[space] rule stands unapplied; install/enable Inception weights to run it.")
    elif verdicts.get("inception") and not verdicts.get("pixel"):
        print("[space] DECISION: the move to Inception is a FIX. Re-run the affected experiments")
        print("[space]   under a tag, pin twin claims on `*_inception` keys, and leave the pixel")
        print("[space]   claims asserting -- additive, so nothing is invalidated.")
    elif verdicts.get("inception") and verdicts.get("pixel"):
        print("[space] DECISION: both spaces give the known answers, so the difference is a")
        print("[space]   NUISANCE, not a fix. Keep pixel as the default, document the offset, do")
        print("[space]   not re-pin -- the migration would move numbers without buying correctness.")
    else:
        print("[space] DECISION: neither space gives the known answers at this sample count. The")
        print("[space]   metric is unusable at n=%d on this data; a claim resting on it should be" % N_ARM)
        print("[space]   DELETED rather than re-pinned in a second wrong space (the n25 precedent).")

    # This file reports; it does not gate. Only an inability to compute is a failure.
    ok(all(np.isfinite(table[sp][n]["precision"]) for sp in spaces for n in A),
       "every arm produced a finite number in every space")
    print(f"\n[space] {'PASS' if not FAIL else 'FAIL: ' + '; '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
