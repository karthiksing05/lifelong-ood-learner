"""What is the LARGEST readable CelebA grid, at 64px and at 128px?

`diag_grid_expand` returned a result that reaches back into the benchmark rather than forward:

    Blond_Hair  +0.103   readable
    Brown_Hair  -0.154   NOT readable -- worse than always answering "no"
    Black_Hair  +0.021   NOT readable
    Gray_Hair   +0.000   NOT readable

The grid in use is hair(3) = {Blond, Brown, other}, so ONE OF ITS THREE FACTOR VALUES IS UNREADABLE.
`probe_readable` guards at the level of a FACTOR -- hair's overall validation accuracy is 0.568,
which clears the 0.55 bar -- and cannot see that the accuracy is carried by two values out of three.
Every cell with hair = brown is therefore scored by a probe that cannot tell, which is why real
images of the held-out cells read at chance on the 7-seen backbone and why the brown cells scored
0.000 for every arm including `unconditional`.

So before any more CelebA training, the ceiling has to be measured rather than assumed: which
attributes a probe can read, PER VALUE, at each resolution the benchmark could plausibly use.
Resolution is the interesting axis -- 64px is cheap enough for from-scratch training with a density
holdout, and if 128px unlocks enough attributes to build a table whose classes outnumber its free
vectors, that is worth four times the training cost (see `diag_basis_cv` for why the ratio decides
whether the basis claim is testable at all).

MARGIN OVER BASE RATE, always. Attributes here are wildly imbalanced -- Gray_Hair is 3% -- so raw
accuracy certifies a probe that has learned nothing.

    python tests/diag_grid_factors.py --img 64
    python tests/diag_grid_factors.py --img 128
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.celeba import ATTRS, load_celeba_grid          # noqa: E402
from lifelong_ood.evals import SoftmaxProbe                           # noqa: E402

MIN_MARGIN = 0.10
MIN_SHARE = 0.05          # below this a value cannot fill a cell at any reasonable dataset size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--n-per-cell", type=int, default=1200)
    p.add_argument("--max-rows", type=int, default=120000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = load_celeba_grid(img=args.img, seed=args.seed, n_per_cell=args.n_per_cell,
                         n_holdout=0, max_rows=args.max_rows)
    X, A = np.asarray(d["X"], np.float32), np.asarray(d["attrs"])
    print(f"[factors] {len(X)} images at {args.img}px, probed exactly as the benchmark does "
          f"(SoftmaxProbe on raw pixels)", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    cut = int(0.9 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    print(f"\n[factors] {'attribute':22s}{'share':>8s}{'acc':>8s}{'base':>8s}"
          f"{'margin':>9s}{'usable':>8s}")
    ok = []
    for k, nm in enumerate(ATTRS):
        y = (A[:, k] == 1).astype(int)
        share = float(y.mean())
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        pr = SoftmaxProbe(2, seed=args.seed, epochs=30).fit(X[tr], y[tr])
        acc = float((pr.predict(X[te]) == y[te]).mean())
        cnt = np.bincount(y[te], minlength=2)
        base = float(cnt.max() / max(cnt.sum(), 1))
        marg = acc - base
        good = marg >= MIN_MARGIN and share >= MIN_SHARE and (1 - share) >= MIN_SHARE
        print(f"[factors] {nm:22s}{share:>8.3f}{acc:>8.3f}{base:>8.3f}{marg:>+9.3f}"
              f"{'yes' if good else 'NO':>8s}")
        if good:
            ok.append(nm)

    # Every readable attribute is BINARY, so k of them make a 2^k grid against 1 + 2k free vectors.
    # The ratio is what decides whether the ideal-words basis claim can be tested here at all.
    print(f"\n[factors] {len(ok)} readable at {args.img}px: {ok}")
    for k in range(2, max(3, len(ok) + 1)):
        cells, free = 2 ** k, 1 + 2 * k
        mark = "  <- available" if k <= len(ok) else ""
        print(f"[factors]   {k} binary factors: {cells:>3d} cells / {free:>2d} free vectors "
              f"= ratio {cells / free:.1f}{mark}")

    print(f"\n[factors] {'=' * 64}")
    best = 2 ** len(ok) / (1 + 2 * len(ok)) if ok else 0.0
    if best >= 4.0:
        print(f"[factors]   A TABLE THAT CAN CARRY THE BASIS CLAIM EXISTS AT {args.img}px: "
              f"ratio {best:.1f}")
        print("[factors]   with all-binary factors. Rebuild `load_celeba_grid` on these attributes")
        print("[factors]   and retrain -- ColorMNIST sits at 6.3 and its in-sample and")
        print("[factors]   cross-validated R^2 agree there.")
    elif ok:
        print(f"[factors]   The best table at {args.img}px is {2 ** len(ok)} cells against "
              f"{1 + 2 * len(ok)} free")
        print(f"[factors]   vectors, ratio {best:.1f}. Below ~4 the in-sample R^2 is mostly")
        print("[factors]   arithmetic (diag_basis_cv), so CelebA cannot carry the BASIS claim at")
        print("[factors]   this resolution however the factors are chosen. It can still carry the")
        print("[factors]   DENSITY-holdout claim, which needs readable factors and not a big table.")
    else:
        print(f"[factors]   NOTHING is readable at {args.img}px by the benchmark's own probe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
