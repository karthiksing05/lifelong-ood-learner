"""The CelebA table design, chosen under BOTH constraints at once.

`diag_celeba_design.py` maximised ratio and reached 13.9 through `Attractive`, `Pointy_Nose` and
`Oval_Face`.  `diag_celeba_attr_readability.py` then showed the readability half, and corrected its
own first answer: the +0.10-over-base bar admits 7 of 40 attributes and is arithmetically impossible
for the 16 whose base rate exceeds 0.90, while AUROC admits 36.

This picks a design under both constraints, and reports the quantity that actually decides whether a
case study will show anything:

    E[joint] = product of the per-factor balanced accuracies.

That product is why CelebA is hard and Shapes3D is not.  CelebA's factors are BINARY, so classes
only come from more factors, and every added factor multiplies the joint down again.  Shapes3D has
three factors with 10, 10 and 4 VALUES: ratio 18.2 while multiplying only three numbers, each ~0.95.

So the script also evaluates the move that buys CelebA the same shape -- MERGING mutually exclusive
binaries into one categorical factor:

    hair in {black, blond, brown, other}   ONE 4-valued factor, 3 free vectors, one probe
    vs Black_Hair + Blond_Hair + Brown_Hair as three binaries, 3 free vectors, three probes

Same free-vector cost, but the categorical version multiplies ONE accuracy into the joint instead of
three, and it cannot produce the incoherent cells (blond AND black) that eat the class budget.

    python tests/diag_celeba_table_design.py
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.data.celeba import _parquet_shards, find_celeba     # noqa: E402

HAIR = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]


def load_attr_table():
    import pyarrow.parquet as pq
    shards = _parquet_shards(find_celeba())
    names = [c for c in pq.ParquetFile(shards[0]).schema.names
             if c not in ("bytes", "path", "celeb_id", "image")]
    rows = []
    for sp in shards:
        t = pq.read_table(sp, columns=names)
        rows.append(np.stack([np.asarray(t[c].to_numpy(zero_copy_only=False), np.uint8)
                              for c in names], axis=1))
    return names, np.concatenate(rows)


def factorise(A, names, spec):
    """spec: list of ('bin', attr) or ('cat', [attrs], label) -> (values [N, F], sizes, labels)."""
    cols, sizes, labels = [], [], []
    for kind, *rest in spec:
        if kind == "bin":
            a = rest[0]
            cols.append(A[:, names.index(a)].astype(int))
            sizes.append(2)
            labels.append(a)
        else:
            attrs, lab = rest
            M = np.stack([A[:, names.index(a)] for a in attrs], 1).astype(int)
            # exactly-one rows keep their value; none/multiple collapse to the last ("other")
            v = np.full(len(A), len(attrs), int)
            one = M.sum(1) == 1
            v[one] = M[one].argmax(1)
            cols.append(v)
            sizes.append(len(attrs) + 1)
            labels.append(lab)
    return np.stack(cols, 1), sizes, labels


def table(V, sizes, min_count):
    key = np.zeros(len(V), np.int64)
    mul = 1
    for f, s in enumerate(sizes):
        key += V[:, f] * mul
        mul *= s
    cnt = np.bincount(key, minlength=mul)
    kept = cnt[cnt >= min_count]
    n_free = 1 + sum(s - 1 for s in sizes)
    return dict(n_combos=int(len(kept)), n_free=int(n_free),
                ratio=len(kept) / n_free, median=int(np.median(kept)) if len(kept) else 0,
                floor=int(kept.min()) if len(kept) else 0, total=int(kept.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-count", type=int, default=200)
    p.add_argument("--read", default="results/celeba_attr_readability.json")
    args = p.parse_args()

    names, A = load_attr_table()
    R = {x["attr"]: x for x in json.load(open(args.read))["inception"]["rows"]}
    bal = {a: R[a]["balanced"] for a in R}
    # the hair probe is one 4-way problem; estimate it as the WEAKEST of its components, which is
    # the conservative reading and avoids claiming a merge improves what it only reorganises
    bal_hair = min(bal[a] for a in HAIR[:3])

    DESIGNS = [
        ("7ee's retired GRID: hair x Male x Smiling",
         [("cat", HAIR[:3], "hair"), ("bin", "Male"), ("bin", "Smiling")], [bal_hair, "Male", "Smiling"]),
        ("hair x Male x Smiling x Bangs",
         [("cat", HAIR[:3], "hair"), ("bin", "Male"), ("bin", "Smiling"), ("bin", "Bangs")],
         [bal_hair, "Male", "Smiling", "Bangs"]),
        ("hair(4) x Male x Smiling x Bangs",
         [("cat", HAIR, "hair"), ("bin", "Male"), ("bin", "Smiling"), ("bin", "Bangs")],
         [bal_hair, "Male", "Smiling", "Bangs"]),
        ("hair x Male x Smiling x Bangs x Wearing_Lipstick",
         [("cat", HAIR[:3], "hair"), ("bin", "Male"), ("bin", "Smiling"), ("bin", "Bangs"),
          ("bin", "Wearing_Lipstick")],
         [bal_hair, "Male", "Smiling", "Bangs", "Wearing_Lipstick"]),
        ("hair x Male x Smiling x Bangs x High_Cheekbones",
         [("cat", HAIR[:3], "hair"), ("bin", "Male"), ("bin", "Smiling"), ("bin", "Bangs"),
          ("bin", "High_Cheekbones")],
         [bal_hair, "Male", "Smiling", "Bangs", "High_Cheekbones"]),
        ("ALL BINARY, 5 most readable (no merge)",
         [("bin", a) for a in ["Male", "Smiling", "Blond_Hair", "Black_Hair", "Bangs"]],
         ["Male", "Smiling", "Blond_Hair", "Black_Hair", "Bangs"]),
        ("ALL BINARY, 8 most readable (no merge)",
         [("bin", a) for a in ["Male", "Smiling", "Blond_Hair", "Black_Hair", "Bangs",
                               "High_Cheekbones", "Wavy_Hair", "Heavy_Makeup"]],
         ["Male", "Smiling", "Blond_Hair", "Black_Hair", "Bangs", "High_Cheekbones",
          "Wavy_Hair", "Heavy_Makeup"]),
    ]

    print(f"[table] {len(A)} rows, min-count {args.min_count}. "
          f"Shapes3D flagship for reference: 400 classes / 25 free = ratio 18.2, E[joint] ~0.86\n")
    print(f"    {'design':52s}{'cls':>5}{'free':>5}{'ratio':>7}{'med':>7}{'E[joint]':>10}")
    out = []
    for label, spec, balspec in DESIGNS:
        V, sizes, labels = factorise(A, names, spec)
        t = table(V, sizes, args.min_count)
        ej = float(np.prod([b if isinstance(b, float) else bal[b] for b in balspec]))
        t.update(design=label, sizes=sizes, factors=labels, e_joint=ej)
        out.append(t)
        print(f"    {label:52s}{t['n_combos']:>5}{t['n_free']:>5}{t['ratio']:>7.1f}"
              f"{t['median']:>7}{ej:>10.3f}")

    print()
    good = [t for t in out if t["ratio"] >= 4 and t["e_joint"] >= 0.30 and t["median"] >= 200]
    if good:
        b = max(good, key=lambda t: t["ratio"] * t["e_joint"])
        print(f"[table] BEST under both constraints: {b['design']}")
        print(f"[table]   {b['n_combos']} classes / {b['n_free']} free vectors = ratio "
              f"{b['ratio']:.1f}; median {b['median']} images; E[joint] {b['e_joint']:.3f}")
        print(f"[table]   factors {b['factors']} sizes {b['sizes']}")
    else:
        print("[table] nothing clears ratio>=4 AND E[joint]>=0.30 AND >=200 images/class.")
    json.dump(out, open(os.path.join("results", "celeba_table_design.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
