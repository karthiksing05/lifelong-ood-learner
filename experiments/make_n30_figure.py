"""F11 -- the prototype-distance dial, as a curve rather than a panel of faces.

`n30_prototype_distance.py` writes a qualitative image panel per seed.  What the claim actually
rests on is the *shape* of four curves against the dial setting `s`, pooled over seeds, so this
draws those with error bars from the seed-tagged JSONs.

The point of the figure is a contrast, not a score:

  ours          every quantity moves monotonically with `s` -- distance to the prototype, sample
                diversity and coverage all rise together.  The dial is real.
  poe-guided    every quantity is FLAT.  Guidance toward the analytic teacher re-collapses the
                trajectory onto the mode no matter where it starts, so at s=1 -- where the
                prototype has been destroyed and the process begins from essentially pure noise --
                the samples still land 0.055 from the prototype with recall 0.000.

    python experiments/make_n30_figure.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
RESULTS = os.path.join(HERE, "results")

ARMS = {"ours": ("#1B7F4B", "o", "ours"),
        "poe-guided": ("#B03A2E", "s", "poe-guided (companion, their Eq. 14)")}
PANELS = [("f1", "F1  (coverage x precision)", None),
          ("recall", "recall  (Kynkaanniemi 2019)", None),
          ("joint", "joint composition accuracy", None),
          ("dist_prototype", "RMS distance to the prototype", None)]


def main():
    runs = [json.load(open(f)) for f in
            sorted(glob.glob(os.path.join(RESULTS, "n30_prototype_distance_fine__seed*.json")))]
    if not runs:
        print("[f11] no n30 seed files -- run n30_prototype_distance.py first")
        return 1
    S = runs[0]["config"]["strengths"]

    def cell(arm, s, k):
        v = [r["summary"][f"{arm}@{s:g}"][k] for r in runs if f"{arm}@{s:g}" in r["summary"]]
        return (float(np.mean(v)),
                float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    for ax, (key, title, _) in zip(axes, PANELS):
        for arm, (col, mk, lab) in ARMS.items():
            m = np.array([cell(arm, s, key)[0] for s in S])
            e = np.array([cell(arm, s, key)[1] for s in S])
            ax.errorbar(S, m, yerr=e, color=col, marker=mk, ms=5, lw=1.9, capsize=3,
                        label=lab if key == "f1" else None)
            ax.fill_between(S, m - e, m + e, color=col, alpha=0.13, lw=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("s   (noise level the reverse process starts from)", fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
    # mark the operating point the sweep actually selects, and the one the coarse sweep suggested
    f1m = [cell("ours", s, "f1")[0] for s in S]
    best = S[int(np.argmax(f1m))]
    for ax in axes:
        ax.axvline(best, color="#1B7F4B", ls=(0, (5, 3)), lw=1.1, alpha=0.75, zorder=0)
        ax.axvline(0.8, color="#8A8A8A", ls=(0, (2, 3)), lw=1.1, alpha=0.75, zorder=0)
    # the F1 curve occupies the lower-left and upper-right of this panel, so the legend goes bottom
    # right (free: F1 > 0.59 there) and both callouts into the free wedge above the rising limb.
    axes[0].legend(fontsize=8.5, loc="lower right", frameon=False)
    axes[0].annotate(f"s={best:g}  selected", xy=(best, max(f1m)), xytext=(0.26, 0.94),
                     textcoords="axes fraction", fontsize=8.5, color="#1B7F4B", ha="left",
                     arrowprops=dict(arrowstyle="->", color="#1B7F4B", lw=1.0))
    axes[0].annotate("s=0.8 -- the coarse\nsweep's read, superseded",
                     xy=(0.8, f1m[S.index(0.8)]), xytext=(0.02, 0.62),
                     textcoords="axes fraction", fontsize=7.5, color="#6A6A6A", ha="left",
                     arrowprops=dict(arrowstyle="->", color="#8A8A8A", lw=0.9))
    fig.suptitle("Distance from the prototype is a DIAL for our method and a CONSTANT for theirs.  "
                 f"Mean +/- s.e.m. over {len(runs)} seeds, {runs[0]['config']['n_cases']} "
                 "compositions each.", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.text(0.5, 0.005,
             "s -> 0 starts from the near-intact prototype (near-copies); s -> 1 starts from "
             "essentially pure noise.  poe-guided is flat because guidance toward the analytic "
             "teacher re-collapses the trajectory onto the mode regardless of where it began.",
             ha="center", fontsize=8, color="#555555")
    out = os.path.join(FIG, "f11_prototype_dial.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[f11] wrote {out}")

    print(f"\n[f11] ours F1 peaks at s={best:g} ({max(f1m):.3f}); "
          f"at s=0.8 it is {f1m[S.index(0.8)]:.3f}")
    lo, hi = cell("poe-guided", S[0], "dist_prototype")[0], \
        cell("poe-guided", S[-1], "dist_prototype")[0]
    o_lo, o_hi = cell("ours", S[0], "dist_prototype")[0], cell("ours", S[-1], "dist_prototype")[0]
    print(f"[f11] dist-to-prototype span: ours {o_lo:.3f}->{o_hi:.3f} ({o_hi / o_lo:.2f}x), "
          f"poe-guided {lo:.3f}->{hi:.3f} ({hi / lo:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
