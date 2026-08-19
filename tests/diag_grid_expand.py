"""Can the CelebA grid be made BIGGER? Which extra hair colours are readable at 64px.

`diag_basis_cv` settled why this matters.  The ideal-words claim is that a class table is
approximately `mu + sum_f V_f[value_f]`, and the in-sample R^2 that supports it is only worth
something when classes heavily outnumber free vectors:

    ColorMNIST   120 classes / 19 vectors = 6.3   R^2 0.826 -> cv 0.772   (agree)
    CelebA grid    9 classes /  8 vectors = 1.1   R^2 0.872 -> cv 0.474   (do not)
    the same at 7 classes /  8 vectors = 0.9   R^2 0.883 -> cv 0.038   (meaningless)

hair(3) x Male(2) x Smiling(2) cannot support the basis claim at any holdout, because 12 cells
against 8 free vectors is not a table, it is an interpolation.  The cheapest way out is a FINER
hair factor: CelebA also labels Black_Hair and Gray_Hair, and every value added to a 3-value factor
adds ONE free vector while multiplying the cell count by (k+1)/k.

    hair(3) -> 12 cells /  8 vectors = 1.5
    hair(4) -> 16 cells /  9 vectors = 1.8
    hair(5) -> 20 cells / 10 vectors = 2.0

Still short of ColorMNIST, but 2.0 is where in-sample and cross-validated R^2 start to track each
other rather than diverge.

A value is only admissible if a probe can READ it -- an unreadable factor value makes every joint
accuracy in the benchmark blind, which is the fault this file exists to prevent and which the
`real` arm caught after the fact on the 7-seen backbone.  So this measures, per candidate hair
colour: how many images carry it, and whether a probe separates it from the rest above its own base
rate.  MARGIN OVER BASE RATE is the statistic, never raw accuracy: Gray_Hair is ~4% of CelebA, so a
probe that always says "not gray" scores 0.96 and has learned nothing.

    python tests/diag_grid_expand.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.celeba import ATTRS, load_celeba_grid           # noqa: E402
from lifelong_ood.evals import SoftmaxProbe                            # noqa: E402

CANDIDATES = ("Blond_Hair", "Brown_Hair", "Black_Hair", "Gray_Hair")
MIN_MARGIN = 0.10                                                       # the bar used elsewhere
MIN_CELL = 100


def probe_margin(X, y, seed=0):
    """(accuracy, base rate, margin) for THE PROBE THE BENCHMARK ACTUALLY USES.

    `fit_factor_probes` trains a `SoftmaxProbe` on raw pixels, so readability must be measured with
    that same probe on those same images.  Measuring it with a stronger probe would certify a factor
    the benchmark then cannot read -- which is precisely how the CelebA attribute set was mis-scoped
    once already.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.9 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
        return float("nan"), float("nan"), float("nan")
    p = SoftmaxProbe(2, seed=seed, epochs=30).fit(X[tr], y[tr])
    acc = float((p.predict(X[te]) == y[te]).mean())
    cnt = np.bincount(y[te], minlength=2)
    base = float(cnt.max() / max(cnt.sum(), 1))
    return acc, base, acc - base


def main():
    # The grid loader is reused so the images, size and preprocessing match the benchmark exactly;
    # a readability measured on differently-processed images would not transfer.
    d = load_celeba_grid(img=64, seed=0, n_per_cell=1200, n_holdout=0, max_rows=120000)
    X, A = np.asarray(d["X"], np.float32), np.asarray(d["attrs"])
    i = {n: ATTRS.index(n) for n in ATTRS}
    print(f"[expand] {len(X)} images at {d['img']}px, probed exactly as the benchmark does",
          flush=True)

    print(f"\n[expand] {'hair colour':14s}{'count':>9s}{'share':>8s}{'acc':>8s}"
          f"{'base':>8s}{'margin':>9s}{'usable':>8s}")
    usable = []
    for nm in CANDIDATES:
        y = (A[:, i[nm]] == 1).astype(int)
        acc, base, marg = probe_margin(X, y)
        ok = marg == marg and marg >= MIN_MARGIN and int(y.sum()) >= MIN_CELL
        print(f"[expand] {nm:14s}{int(y.sum()):>9d}{y.mean():>8.3f}{acc:>8.3f}"
              f"{base:>8.3f}{marg:>+9.3f}{'yes' if ok else 'NO':>8s}")
        if ok:
            usable.append(nm)

    # What a grid built from the usable colours would actually look like.  Cells matter as much as
    # readability: a hair value that is readable but leaves a near-empty (hair, Male, Smiling) cell
    # cannot be held out or scored, and `load_celeba_grid` drops it anyway.
    print(f"\n[expand] usable hair colours: {usable} (+ 'other')")
    k = len(usable) + 1
    cells, free = k * 4, 1 + k + 2 + 2
    print(f"[expand] that is a hair({k}) x Male(2) x Smiling(2) grid = {cells} cells against "
          f"{free} free vectors, ratio {cells / free:.1f} (now {12 / 8:.1f})")

    print(f"\n[expand] {'cell':34s}{'count':>9s}")
    thin = 0
    for h, nm in enumerate(list(usable) + ["other"]):
        for m_ in (0, 1):
            for s_ in (0, 1):
                sel = np.ones(len(A), bool)
                for c in usable:
                    sel &= (A[:, i[c]] == 1) if c == nm else (A[:, i[c]] != 1)
                if nm == "other":
                    sel = np.ones(len(A), bool)
                    for c in usable:
                        sel &= A[:, i[c]] != 1
                sel &= (A[:, i["Male"]] == m_) & (A[:, i["Smiling"]] == s_)
                n = int(sel.sum())
                thin += n < MIN_CELL
                print(f"[expand] {nm + ' M=' + str(m_) + ' S=' + str(s_):34s}{n:>9d}"
                      + ("   <- too thin to score" if n < MIN_CELL else ""))
    print(f"\n[expand] {'=' * 62}")
    if len(usable) > 2 and not thin:
        print(f"[expand]   THE GRID CAN GROW to {cells} cells, ratio {cells / free:.1f}. Retrain")
        print("[expand]   `celeba_grid` with the finer hair factor and re-measure -- the basis")
        print("[expand]   claim is not supportable at the current 1.5.")
    elif len(usable) > 2:
        print(f"[expand]   {len(usable)} colours read, but {thin} cell(s) fall under {MIN_CELL}")
        print("[expand]   images. A cell that cannot be scored cannot be held out; either lower")
        print("[expand]   min_cell with the loss of power that implies, or keep the coarse grid.")
    else:
        print("[expand]   THE GRID CANNOT GROW on hair colour. The basis claim stays on")
        print("[expand]   ColorMNIST, where 120 classes against 19 vectors support it; CelebA")
        print("[expand]   carries the DENSITY-holdout claim instead, which needs no large table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
