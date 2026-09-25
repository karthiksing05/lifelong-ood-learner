"""Can ANY probe read CelebA attributes, or is the benchmark unmeasurable?

`fit_attr_probes` defaults to a LINEAR probe on raw pixels, and measured that way six of eight
candidate attributes sit at or BELOW their base rate -- the probe predicts the majority class and
nothing else:

    Male +0.141 | Smiling +0.096 | Wavy_Hair -0.000 | Black_Hair -0.039
    Brown_Hair +0.001 | Oval_Face -0.012 | Big_Lips -0.040 | No_Beard +0.002

If that is the instrument, no CelebA composition number can mean anything: "the sample carries the
combination" is unmeasurable for most of the combination.  It is the same fault as Shapes3D's
`object_hue` reading 0.239 against a 0.20 chance baseline, but across almost the whole attribute set,
and it explains the +0.048 attribute margin and the near-floor separation measured earlier.

So before a backbone is trained on any grid, the grid's factors must be READABLE.  This compares the
pixel probe against Inception-V3 features (already supported) on the same data and split.  Only
attributes a probe can actually read are eligible to be grid factors.

    python tests/diag_celeba_probe_features.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.celeba import ATTRS, load_celeba                # noqa: E402
from lifelong_ood.evals import fit_attr_probes                         # noqa: E402

CANDS = ["Male", "Smiling", "Wavy_Hair", "Black_Hair", "Brown_Hair", "Oval_Face",
         "Big_Lips", "No_Beard", "Big_Nose", "Blond_Hair"]


def main():
    d = load_celeba(n_per_class=300, img=64, seed=0, min_count=20)
    A = np.asarray(d["attrs"])
    X = np.asarray(d["X"], np.float32)
    rows = {}
    for feat in ("pixel", "inception"):
        try:
            pr = fit_attr_probes(d, seed=0, features=feat)
        except Exception as e:                                          # noqa: BLE001
            print(f"[pf] {feat}: unavailable ({type(e).__name__}: {e})")
            continue
        from lifelong_ood.evals import _probe_features
        F = _probe_features(X, feat, img=64)
        for n in CANDS:
            if n not in pr:
                continue
            y = A[:, ATTRS.index(n)]
            acc = float((pr[n].predict(F) == y).mean())
            base = float(max(y.mean(), 1 - y.mean()))
            rows.setdefault(n, {})[feat] = (acc, acc - base)
    print(f"\n{'attribute':14s}{'pixel acc':>11}{'margin':>9}{'incep acc':>11}{'margin':>9}"
          f"{'  readable?':>12}")
    good = []
    for n in CANDS:
        r = rows.get(n, {})
        p_a, p_m = r.get("pixel", (float('nan'),) * 2)
        i_a, i_m = r.get("inception", (float('nan'),) * 2)
        best = max([m for m in (p_m, i_m) if m == m], default=float("nan"))
        ok = best >= 0.10
        if ok:
            good.append(n)
        print(f"{n:14s}{p_a:>11.3f}{p_m:>+9.3f}{i_a:>11.3f}{i_m:>+9.3f}"
              f"{('  YES' if ok else '  no'):>12}")
    print(f"\n[pf] readable at margin >= 0.10: {good}")
    print("[pf] a grid factor MUST come from this list -- an unreadable factor makes the whole")
    print("[pf] benchmark untestable, which is what Shapes3D's object_hue did to its joint score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
