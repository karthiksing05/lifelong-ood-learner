"""What is the LARGEST readable CelebA grid, at 64px and at 128px?

`diag_grid_expand` returned a result that reaches back into the benchmark rather than forward:

    Blond_Hair  +0.103   readable
    Brown_Hair  -0.154   NOT readable -- worse than always answering "no"
    Black_Hair  +0.021   NOT readable
    Gray_Hair   +0.000   NOT readable

The grid in use is hair(3) = {Blond, Brown, other}, so one of its three factor values is one the
benchmark's own probe cannot separate.  `probe_readable` guards at the level of a FACTOR -- hair's
overall validation accuracy is 0.568, which clears the 0.55 bar -- and cannot see that the accuracy
is carried by some values and not others.

*Do not read that table as "brown cells are the blind ones".*  These are BINARY one-vs-rest probes,
and the benchmark scores hair with a single 3-way softmax refit per seed and per seen set.  Measured
per cell, the 9-seen backbone's one READABLE held-out cell is the brown one while blond and `other`
read blind, and the 7-seen backbone gives the reverse.  Which cells are legible is a property of the
fitted probe, so it is measured per cell per run by n73's `real-gate` arm and cannot be fixed by
dropping a value here.  What this file settles is the different question of how BIG a readable grid
could be.

So before any more CelebA training, the ceiling is measured rather than assumed: which attributes a
probe separates, at each resolution and in each feature space the benchmark could plausibly use.
Resolution turned out to be the wrong axis to hope on -- 128px reads FEWER attributes than 64px, not
more, because the probe is linear over raw pixels and its dimensionality grows with the image while
the data does not.  Feature space is the live question (`--features inception`), and if a stronger
probe unlocks enough attributes to build a table whose classes outnumber its free vectors, the
benchmark should ADOPT that probe rather than merely cite it (see `diag_basis_cv` for why the ratio
decides whether the basis claim is testable at all).

MARGIN OVER BASE RATE, always. Attributes here are wildly imbalanced -- Gray_Hair is 3% -- so raw
accuracy certifies a probe that has learned nothing.

    python tests/diag_grid_factors.py --img 64
    python tests/diag_grid_factors.py --img 128
    python tests/diag_grid_factors.py --img 64 --features inception
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.celeba import ATTRS, load_celeba_grid          # noqa: E402
from lifelong_ood.evals import SoftmaxProbe, _probe_features          # noqa: E402

MIN_MARGIN = 0.10
MIN_SHARE = 0.05          # below this a value cannot fill a cell at any reasonable dataset size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--n-per-cell", type=int, default=1200)
    p.add_argument("--max-rows", type=int, default=120000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features", choices=("pixel", "inception"), default="pixel",
                   help="feature space for the probe. `pixel` is what `fit_factor_probes` uses "
                        "today, so it is what the benchmark can actually see; `inception` asks "
                        "whether a STRONGER probe would unlock more factors -- and if it does, the "
                        "benchmark should adopt it rather than the result being quoted from it.")
    args = p.parse_args()

    d = load_celeba_grid(img=args.img, seed=args.seed, n_per_cell=args.n_per_cell,
                         n_holdout=0, max_rows=args.max_rows)
    X, A = np.asarray(d["X"], np.float32), np.asarray(d["attrs"])
    Xf = _probe_features(X, args.features, img=args.img)
    if args.features == "inception" and Xf.shape[1:] == X.shape[1:]:
        print("[factors] inception features UNAVAILABLE -- fell back to pixels. The comparison "
              "below is not the one requested; install the weights or read this as a pixel run.")
    print(f"[factors] {len(X)} images at {args.img}px, {args.features} features "
          f"(dim {int(np.prod(Xf.shape[1:]))})", flush=True)

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
        pr = SoftmaxProbe(2, seed=args.seed, epochs=30).fit(Xf[tr], y[tr])
        acc = float((pr.predict(Xf[te]) == y[te]).mean())
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
    print(f"\n[factors] {len(ok)} readable at {args.img}px / {args.features}: {ok}")
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
