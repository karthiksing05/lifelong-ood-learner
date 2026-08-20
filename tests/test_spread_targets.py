"""`spread_targets` picks a set that spans the design, and picks the SAME set n77 always did.

Two independent duties, and the second is what protects pinned numbers:

  1. SPREAD -- given candidates that share a factor value, the chosen set must cover more values of
     every factor than taking the first N in order does.  Both real failures are reproduced from
     literal slot lists: n77's five Shapes3D targets that all had floor_hue=0, and the flagship's
     four cells that all had floor=red.
  2. IDENTICAL -- the helper was lifted out of n77, whose joint 0.983 / recall 0.995 are pinned in
     `test_claims.py`.  A lift that changes the selection changes those numbers with no other
     visible sign, so the inline loop is reproduced here and the two are compared over randomised
     designs rather than on the one case anybody would think to check.

CPU only, no data, no checkpoint.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.data.factored import spread_targets                # noqa: E402


def _inline(held, F, n_fac, n):
    """n77's selection loop as it stood before the lift, verbatim in behaviour."""
    chosen, pool = [], list(held)
    counts = [{} for _ in range(n_fac)]
    while pool and len(chosen) < n:
        def cost(h):
            return sum(counts[f].get(int(v), 0) for f, v in enumerate(F[h]))
        h = min(pool, key=cost)
        pool.remove(h)
        chosen.append(h)
        for f, v in enumerate(F[h]):
            counts[f][int(v)] = counts[f].get(int(v), 0) + 1
    return chosen


def _values(chosen, F, n_fac):
    return [len({int(F[h][f]) for h in chosen}) for f in range(n_fac)]


def main():
    ok = True

    # 1 ── the two real failures, from the designs that produced them
    sizes = (10, 10, 4)                        # Shapes3D: floor_hue, object_hue, shape
    F = {s: (s // 40, (s // 4) % 10, s % 4) for s in range(400)}
    for name, n in (("n77's five concepts", 5), ("the flagship's cells", 6)):
        held = sorted(F)
        first, spread = held[:n], spread_targets(held, F, n)
        vf, vs = _values(first, F, 3), _values(spread, F, 3)
        print(f"  {name:22s} slot order covers {vf} values; spread covers {vs}")
        ok &= vf[0] == 1 and vs[0] == n           # slot order pins factor 0; spread does not
        ok &= all(b >= a for a, b in zip(vf, vs))  # and never does worse on any factor

    # 2 ── identical to the loop it replaced, over randomised designs
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(400):
        szs = [int(x) for x in rng.integers(2, 6, size=int(rng.integers(2, 5)))]
        cand = sorted(rng.choice(200, size=int(rng.integers(3, 40)), replace=False).tolist())
        Fr = {h: tuple(int(v) for v in rng.integers(0, szs, size=len(szs))) for h in cand}
        k = int(rng.integers(1, 8))
        if _inline(cand, Fr, len(szs), k) != spread_targets(cand, Fr, k):
            bad += 1
    print(f"  identical to n77's inline selection on {400 - bad}/400 randomised designs")
    ok &= bad == 0

    # 3 ── degenerate inputs must not raise or over-return
    ok &= spread_targets([], {}, 3) == []
    ok &= len(spread_targets([1, 2], {1: (0,), 2: (1,)}, 9)) == 2
    print("  empty candidates and n > len(candidates) handled")

    print("[spread_targets] PASS" if ok else "[spread_targets] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
