"""Task x arm success table from the ledger (rollout rows of one experiment), with Wilson 95% intervals.

    python experiments/r3_table.py --experiment r3_compose [--bench libero_90]
"""
import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="r3_compose")
    ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    rows = ledger.load(a.experiment)
    cell = defaultdict(lambda: [0, 0])
    lang = {}
    arms = []
    for r in rows:
        if a.bench and r["bench"] != a.bench:
            continue
        arm = r["tag"] or r["keys"].get("mode", "?")
        if arm not in arms:
            arms.append(arm)
        for tid, t in r["metrics"]["tasks"].items():
            k = round(t["success_rate"] * t["n"])
            cell[(tid, arm)][0] += k
            cell[(tid, arm)][1] += t["n"]
            lang[tid] = t["language"]
    tasks = sorted(lang, key=int)
    print(f"{'task':>5} {'language':50} " + " ".join(f"{x:>18}" for x in arms))
    for tid in tasks:
        out = []
        for arm in arms:
            k, n = cell[(tid, arm)]
            if n == 0:
                out.append(f"{'-':>18}")
            else:
                lo, hi = wilson(k, n)
                out.append(f"{k:>3}/{n:<3} [{lo:.2f},{hi:.2f}]")
        print(f"{tid:>5} {lang[tid][:50]:50} " + " ".join(out))


if __name__ == "__main__":
    main()
