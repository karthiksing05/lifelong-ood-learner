"""Can CelebA support a Shapes3D-shaped table?  A design search over attribute subsets.

7jj works on Shapes3D for reasons that are all about the TABLE, not the dataset: 400 classes over
25 free vectors (ratio 18.2), a frozen pretrained backbone, and the class table trained alone at 50x
LR.  The CelebA grid has 12 classes over 5 free vectors (ratio 1.8), which 7ee showed is fatal --
an additive basis with nearly as many parameters as observations fits anything, and in-sample R2
0.87 collapses to 0.47 out of sample.

So before training anything, ask the arithmetic question: is there ANY subset of CelebA's attributes
whose combinations give a table of Shapes3D's shape?  This needs only the attribute table -- no
images, no GPU, no checkpoint -- and it is the 7ee lesson applied in advance instead of in
retrospect.

Per design it reports:

    k            attributes used (the factors); n_free = 1 + k for binary factors
    combos       combinations with at least `--min-count` real images -- the classes
    ratio        combos / n_free.  Above ~4 in-sample and cross-validated R2 agree (THEORY 1)
    imgs/class   median support, which decides whether conditioning can take hold at all
                 (`shapes3d.py` records a 400-class table failing at 0.007 pairwise cosine
                 because each class was too rare against an easy marginal)
    reachable    held-out candidates one attribute-flip from a seen combination, which is what
                 makes the test COMPOSITIONAL rather than a novel-attribute test

    python tests/diag_celeba_design.py --min-count 200
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.data.celeba import find_celeba, _parquet_shards      # noqa: E402


def load_all_attributes():
    """Every boolean attribute in the parquet shards -> (names, [N, 40] uint8).

    `load_attributes` reads `list_attr_celeba.txt`, which this machine does not have, and
    `load_from_parquet` decodes every IMAGE to get at the attributes -- minutes of PIL work for a
    table that is a few megabytes.  The design question is pure arithmetic over the attribute
    columns, so read only those.  It also sees all FORTY attributes rather than the fourteen
    `ATTRS` exposes, which is the whole point: the grid that 7ee retired used three.
    """
    import pyarrow.parquet as pq
    shards = _parquet_shards(find_celeba())
    if not shards:
        raise FileNotFoundError("no parquet shards")
    names = [c for c in pq.ParquetFile(shards[0]).schema.names
             if c not in ("bytes", "path", "celeb_id", "image")]
    rows = []
    for sp in shards:
        t = pq.read_table(sp, columns=names)
        rows.append(np.stack([np.asarray(t[c].to_numpy(zero_copy_only=False), np.uint8)
                              for c in names], axis=1))
    return names, np.concatenate(rows)


def design(A, cols, min_count, ATTRS):
    """-> dict describing the table that attribute subset `cols` would produce."""
    sub = A[:, list(cols)]
    keys, inv, counts = np.unique(sub, axis=0, return_inverse=True, return_counts=True)
    keep = counts >= min_count
    combos, cnt = keys[keep], counts[keep]
    k = len(cols)
    n_free = 1 + k                      # binary factors: one intercept + one free vector each
    # a held-out class must be one attribute-flip from another kept class, or it is a
    # novel-attribute test wearing a composition label
    reach = 0
    for i in range(len(combos)):
        d = (combos[i][None] != combos).sum(1)
        if len(d) > 1 and np.sort(d)[1] == 1:
            reach += 1
    return dict(cols=tuple(cols), names=[ATTRS[c] for c in cols], k=k, n_combos=int(len(combos)),
                n_free=n_free, ratio=len(combos) / n_free,
                median_count=int(np.median(cnt)) if len(cnt) else 0,
                min_count_kept=int(cnt.min()) if len(cnt) else 0,
                covered=int(cnt.sum()), reachable=reach)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-count", type=int, default=200,
                   help="images a combination needs before it can be a class")
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    ATTRS, A = load_all_attributes()
    print(f"[design] {len(A)} CelebA rows, {A.shape[1]} attributes, "
          f"min-count {args.min_count}\n")

    for label, cols in (("the 3-attribute GRID that 7ee retired",
                         [ATTRS.index(a) for a in ("Blond_Hair", "Male", "Smiling")
                          if a in ATTRS]),):
        d = design(A, cols, args.min_count, ATTRS)
        print(f"[design] {label}: {d['n_combos']} classes / {d['n_free']} free vectors = "
              f"ratio {d['ratio']:.1f}, median {d['median_count']} imgs/class\n")

    # Exhaustive for small k, greedy beyond -- the point is the SHAPE of the frontier, and the
    # frontier is where ratio and images-per-class trade against each other.
    best = {}
    for k in range(2, min(args.kmax, len(ATTRS)) + 1):
        # Exhaustive over 40 attributes is 10^9 subsets by k=8, so: exhaustive at k=2, greedy
        # after.  Greedy is the right shape here because adding an attribute costs exactly one free
        # vector and can only add classes, so the frontier is close to monotone.
        if k == 2:
            cands = list(itertools.combinations(range(len(ATTRS)), 2))
        else:
            base = best[k - 1]["cols"]
            cands = [tuple(base) + (c,) for c in range(len(ATTRS)) if c not in base]
        rows = [design(A, c, args.min_count, ATTRS) for c in cands]
        rows = [r for r in rows if r["n_combos"] >= 4]
        if not rows:
            continue
        best[k] = max(rows, key=lambda r: (r["ratio"], r["median_count"]))

    print(f"[design] best subset per size, by ratio then support:")
    print(f"    {'k':>2} {'classes':>8} {'free':>5} {'ratio':>6} {'med imgs':>9} "
          f"{'min imgs':>9} {'reach':>6}  attributes")
    for k in sorted(best):
        r = best[k]
        flag = "  <-- Shapes3D-shaped" if r["ratio"] >= 4 and r["median_count"] >= args.min_count \
            else ""
        print(f"    {k:>2} {r['n_combos']:>8} {r['n_free']:>5} {r['ratio']:>6.1f} "
              f"{r['median_count']:>9} {r['min_count_kept']:>9} {r['reachable']:>6}  "
              f"{','.join(r['names'])}{flag}")

    ok = [k for k in best if best[k]["ratio"] >= 4.0 and best[k]["median_count"] >= args.min_count]
    print()
    if ok:
        k = max(ok, key=lambda k: best[k]["ratio"])
        r = best[k]
        print(f"[design] VERDICT: CelebA CAN carry a table above the ratio-4 bar -- "
              f"{r['n_combos']} classes / {r['n_free']} free vectors = {r['ratio']:.1f}, "
              f"median {r['median_count']} images each")
        print(f"[design]   attributes: {r['names']}")
        print(f"[design]   Shapes3D's flagship table is ratio 18.2 for comparison; the CelebA GRID "
              f"that 7ee retired is 1.8.")
    else:
        print("[design] VERDICT: no attribute subset clears ratio 4 with enough support. "
              "The table, not the backbone, is the binding constraint -- 7ee stands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
