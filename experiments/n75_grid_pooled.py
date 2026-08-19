"""Pool every CelebA-grid measurement across backbones, restricted to cells the probes can READ.

Three backbones (9, 8 and 7 seen cells) x three sampling seeds x their held-out cells give 36
(backbone, cell, seed) measurements.  Twenty of them are on cells where REAL images of that cell
score at or below chance under that run's own probes -- the probe is a `SoftmaxProbe` on raw pixels
refit per seed, and most held-out CelebA combinations are simply not separable that way.  On those
cells every arm reads ~0 and none can be judged, so pooling them in measures the instrument.

This restricts to the readable 16 and reports the two margins separately, because they answer
different questions and only one of them comes out in our favour:

    ADDRESS - unconditional   does conditioning on a composed point do ANYTHING?
    ADDRESS - nearest-class   does it beat RETRIEVAL -- the backbone's own closest existing class?

The second is the one the donation claim needs, and on ColorMNIST it is decisive (nearest-class
0.000 against ADDRESS 0.599, RESULTS 7dd).  It is not decisive here, and the reason is visible in
the same table: `nearest-class` scores highly on CelebA, which means a sample conditioned on the
NEIGHBOURING cell often reads as the held-out one.  That is loose conditioning in the backbone, not
a defect in the composer -- and it puts a ceiling on what this dataset can discriminate at all.

    python experiments/n75_grid_pooled.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                            # noqa: E402

from lifelong_ood.evals import save_json                              # noqa: E402

GATE_MARGIN = 0.02


def collect():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "n73_celeba_grid_compose*__seed*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        base = os.path.basename(p)[:-5]
        tag = base.split("__seed")[0].replace("n73_celeba_grid_compose", "") or "_h3"
        seed = int(base.split("__seed")[1])
        chance = float(d.get("chance") or 1.0 / max(d.get("n_seen", 9), 1))
        by = {}
        for r in d["rows"]:
            by.setdefault(r["cell"], {})[r["arm"]] = r
        for c, arms in sorted(by.items()):
            if "address" not in arms:
                continue
            gate = arms.get("real-gate", {}).get("joint")
            out.append(dict(
                backbone=tag, seed=seed, cell=int(c), chance=chance,
                gate=(float(gate) if gate is not None else float("nan")),
                readable=bool(gate is not None and gate > chance + GATE_MARGIN),
                address=float(arms["address"]["joint"]),
                nearest=float(arms["nearest-class"]["joint"]),
                uncond=float(arms["unconditional"]["joint"]),
                memorized=float(arms["address"].get("memorized", float("nan"))),
                d_query=float(arms["address"].get("d_query", float("nan"))),
                real_spread=float(d.get("real_spread", float("nan"))),
                basis_r2=float(d.get("basis_r2", float("nan"))),
                basis_r2_cv=float(d.get("basis_r2_cv", float("nan")))))
    return out


def stat(v):
    v = np.asarray([x for x in v if x == x], float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    rows = collect()
    if not rows:
        print("[n75] no n73 grid results found")
        return 0
    R = [r for r in rows if r["readable"]]
    print(f"[n75] {len(rows)} (backbone, cell, seed) measurements over "
          f"{len({r['backbone'] for r in rows})} backbones; {len(R)} on READABLE cells")
    print(f"[n75] a cell is readable when REAL images of it, scored by that run's own probes, "
          f"beat chance by {GATE_MARGIN:.2f}")

    print(f"\n[n75] {'set':22s}{'address':>10s}{'nearest':>10s}{'uncond':>10s}{'n':>5s}")
    for label, S in (("all measurements", rows), ("READABLE only", R)):
        print(f"[n75] {label:22s}{stat([r['address'] for r in S])[0]:>10.3f}"
              f"{stat([r['nearest'] for r in S])[0]:>10.3f}"
              f"{stat([r['uncond'] for r in S])[0]:>10.3f}{len(S):>5d}")

    mr = [r["address"] - r["nearest"] for r in R]
    mu = [r["address"] - r["uncond"] for r in R]
    a, ase = stat(mr)
    b, bse = stat(mu)
    print(f"\n[n75] ADDRESS - nearest-class : {a:+.3f} +/- {ase:.3f}, won "
          f"{sum(1 for x in mr if x > 0)}/{len(mr)}")
    print(f"[n75] ADDRESS - unconditional : {b:+.3f} +/- {bse:.3f}, won "
          f"{sum(1 for x in mu if x > 0)}/{len(mu)}")
    mem, _ = stat([r["memorized"] for r in R])
    dq, _ = stat([r["d_query"] for r in R])
    sp, _ = stat([r["real_spread"] for r in R])
    print(f"[n75] copying: memorized {mem:.3f} | d(query) {dq:.1f} against a real-vs-real "
          f"spread of {sp:.1f}")
    r2, _ = stat([r["basis_r2"] for r in rows])
    r2cv, _ = stat([r["basis_r2_cv"] for r in rows])
    print(f"[n75] basis R^2 {r2:.3f} in-sample, {r2cv:.3f} leave-one-cell-out")

    print(f"\n[n75] {'=' * 72}")
    print("[n75] VERDICT")
    if b > 0.10 and a > 0.10:
        print("[n75]   COMPOSITION BEATS BOTH THE FLOOR AND RETRIEVAL on a density holdout.")
    elif b > 0.10:
        print(f"[n75]   CONDITIONING ON A COMPOSED POINT WORKS ({b:+.3f} over the unconditional")
        print(f"[n75]   floor, won {sum(1 for x in mu if x > 0)}/{len(mu)}) BUT DOES NOT BEAT")
        print(f"[n75]   RETRIEVAL ({a:+.3f}, won {sum(1 for x in mr if x > 0)}/{len(mr)} -- a coin")
        print("[n75]   flip). On ColorMNIST `nearest-class` scores 0.000 against ADDRESS's 0.599")
        print("[n75]   (7dd); here it scores close to ADDRESS, which means a sample conditioned on")
        print("[n75]   the NEIGHBOURING cell often reads as the held-out one. That is loose")
        print("[n75]   conditioning in the backbone, not a defect in the composer -- and it caps")
        print("[n75]   what this dataset can discriminate. The CelebA grid supports the")
        print("[n75]   DENSITY-holdout claim and NOT the donation-over-retrieval claim.")
    else:
        print(f"[n75]   Does not clear the unconditional floor ({b:+.3f}). Nothing here supports")
        print("[n75]   a composition claim on this dataset.")

    save_json(dict(rows=rows, n_readable=len(R), n_total=len(rows),
                   margin_vs_retrieval=dict(mean=a, se=ase,
                                            won=int(sum(1 for x in mr if x > 0)), n=len(mr)),
                   margin_vs_uncond=dict(mean=b, se=bse,
                                         won=int(sum(1 for x in mu if x > 0)), n=len(mu)),
                   address=stat([r["address"] for r in R])[0],
                   nearest=stat([r["nearest"] for r in R])[0],
                   uncond=stat([r["uncond"] for r in R])[0],
                   memorized=mem, d_query=dq, real_spread=sp,
                   basis_r2=r2, basis_r2_cv=r2cv),
              os.path.join(RESULTS, "n75_grid_pooled.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
