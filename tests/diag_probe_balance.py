"""Is a CelebA attribute probe actually readable, or just predicting the majority class?

n21's real-image ceiling for Gray_Hair is 0.25 -- i.e. the probe reads Gray_Hair correctly on a
quarter of REAL images that have it -- while the same probe reports 0.962 validation accuracy.  Both
can be true at once: Gray_Hair is ON in 4.3% of CelebA, so a probe that answers "no" every time
scores 95.7%.  Plain accuracy is the wrong readability criterion for a rare attribute, and
`attr_accuracy`'s 0.65 filter is applied to exactly that number.

This measures what matters instead: the TRUE POSITIVE RATE on real images carrying the attribute, and
balanced accuracy.  An attribute whose probe cannot recognise the attribute when it IS present cannot
score a generator asked to produce it -- the generator would be blamed for the probe's blind spot.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                # noqa: E402
from n19_celeba_train import build_probes                               # noqa: E402

from lifelong_ood.celeba import ATTRS, as_float, load_celeba_comp       # noqa: E402


def main():
    data = load_celeba_comp(img=64, min_count=60, n_holdout=24, seed=0, cache_dir=CACHE)
    probes = build_probes(data, 0, 30)
    A = data["attrs"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(A))[:8000]
    X = as_float(data["X8"][idx])
    A = A[idx]

    print(f"{'attribute':20s}{'base rate':>10s}{'val acc':>9s}{'TPR':>7s}{'TNR':>7s}"
          f"{'balanced':>10s}  verdict")
    rows = []
    for k, a in enumerate(ATTRS):
        y = A[:, k].astype(int)
        p = probes[a].predict(X.astype(np.float32))
        tpr = float((p[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
        tnr = float((p[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
        bal = 0.5 * (tpr + tnr)
        val = probes.get(f"{a}_val_acc", float("nan"))
        # usable as a TARGET means: it can recognise the attribute when present, and its balanced
        # accuracy is meaningfully above chance
        good = (tpr >= 0.5) and (bal >= 0.65)
        rows.append((a, float(y.mean()), val, tpr, tnr, bal, good))
        print(f"{a:20s}{y.mean():10.3f}{val:9.3f}{tpr:7.3f}{tnr:7.3f}{bal:10.3f}  "
              f"{'usable' if good else 'NOT usable as a target'}")

    good = [r[0] for r in rows if r[6]]
    passes_old = [r[0] for r in rows if r[2] >= 0.65]
    print(f"\n  by plain val-acc >= 0.65 : {len(passes_old)}/14  {', '.join(passes_old)}")
    print(f"  by TPR >= 0.5 and balanced >= 0.65 : {len(good)}/14  {', '.join(good)}")
    lost = [a for a in passes_old if a not in good]
    print(f"\n  attributes the current filter ADMITS but that cannot recognise their own attribute:")
    print(f"    {', '.join(lost) if lost else '(none)'}")
    for a in ("Smiling", "Big_Nose", "Gray_Hair"):
        r = [x for x in rows if x[0] == a][0]
        print(f"  {a:12s} base {r[1]:.3f}  val {r[2]:.3f}  TPR {r[3]:.3f}  balanced {r[5]:.3f}")


if __name__ == "__main__":
    main()
