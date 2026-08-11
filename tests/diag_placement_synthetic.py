"""Does the placement rule cluster AT ALL, on data built so that it must?

N38 found the tree on ColorMNIST is a caterpillar -- 31 of 40 concepts on one path prefix, 3 shared
nodes.  Two explanations were live: the teacher means carry no cluster structure, or the rule cannot
find structure that is there.  N39 measures the first on real means; this settles the second in a way
no real dataset can, by planting the answer.

**Four well-separated Gaussian families of ten, in 64 dimensions, centres ~6 apart.**  Ancestor
purity of 1.00 is achievable by construction and 0.25 is chance.  Any rule that clusters at all
should be near the top of that range.

Also probes the limit that insert-only imposes regardless of criterion: the same data presented
GROUPED (all of family 0, then family 1, ...) versus SHUFFLED.  Cobweb repairs a bad early split with
merge and split; an insert-only tree cannot, so if grouped order scores far above shuffled, the gap
is what the missing operators would have bought -- and it is not something a better criterion can
recover.

CPU-only and instant; no checkpoint, no GPU.

    python tests/diag_placement_synthetic.py
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.neural.adapters import ConceptTree                          # noqa: E402

N_FAM, PER_FAM, DIM, SEP = 4, 10, 64, 6.0


def make(seed=0):
    rng = np.random.default_rng(seed)
    cent = rng.standard_normal((N_FAM, DIM)) * SEP
    X = np.concatenate([c + rng.standard_normal((PER_FAM, DIM)) for c in cent])
    return X, np.repeat(np.arange(N_FAM), PER_FAM)


def score(tree, X, lab):
    paths = [tree.insert(x)[0] for x in X]
    pref = collections.Counter(tuple(p[:-1]) for p in paths)
    beneath = collections.defaultdict(list)
    for i, p in enumerate(paths):
        for n in p[:-1]:
            beneath[n].append(lab[i])
    pur = [max(collections.Counter(v).values()) / len(v) for v in beneath.values() if len(v) > 1]
    return dict(nodes=tree._next, biggest_prefix=max(pref.values()), n_prefixes=len(pref),
                purity=float(np.mean(pur)) if pur else float("nan"))


def main():
    X, lab = make(0)
    n = len(X)
    print(f"[synth] {N_FAM} planted families x {PER_FAM}, dim {DIM}, centres ~{SEP} apart")
    print(f"[synth] achievable ancestor purity 1.00, chance {1/N_FAM:.2f}\n")
    print(f"    {'rule':>5s}{'bmax':>6s}{'dmax':>6s}{'nodes':>7s}{'big prefix':>12s}{'purity':>9s}")
    rows = []
    for rule, bmax, dmax in (("ll", 4, 3), ("ll", 16, 6), ("cu", 4, 3), ("cu", 16, 6)):
        r = score(ConceptTree(var=0.1, rule=rule, branch_max=bmax, max_depth=dmax), X, lab)
        rows.append(((rule, bmax, dmax), r))
        print(f"    {rule:>5s}{bmax:6d}{dmax:6d}{r['nodes']:7d}"
              f"{str(r['biggest_prefix']) + '/' + str(n):>12s}{r['purity']:9.2f}")

    print(f"\n[synth] arrival order, at the best setting of each rule "
          f"(insert-only cannot repair an early split; Cobweb's merge/split is what would)")
    order = np.concatenate([np.where(lab == f)[0] for f in range(N_FAM)])      # grouped
    rng = np.random.default_rng(0)
    shuf = rng.permutation(n)
    for rule in ("ll", "cu"):
        vals = {}
        for tag, idx in (("grouped", order), ("shuffled", shuf)):
            r = score(ConceptTree(var=0.1, rule=rule, branch_max=16, max_depth=6),
                      X[idx], lab[idx])
            vals[tag] = r["purity"]
        print(f"    {rule:>5s}: grouped {vals['grouped']:.2f}   shuffled {vals['shuffled']:.2f}"
              f"   gap {vals['grouped'] - vals['shuffled']:+.2f}")

    shipped = rows[0][1]["purity"]
    best = max(r["purity"] for _, r in rows)
    print(f"\n[synth] the SHIPPED setting (ll, bmax 4, dmax 3) scores {shipped:.2f} against chance "
          f"{1/N_FAM:.2f} on data with perfect planted structure.")
    print(f"[synth] best of the variants tried: {best:.2f}.  Neither is near the achievable 1.00, so "
          f"the criterion and the caps are both implicated -- and the order gap above bounds what\n"
          f"        insert-only alone costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
