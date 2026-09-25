"""F31 -- the learning curves that were measured but never drawn.

`n5_classinc` already records, per arm, the full **accuracy matrix**: `acc_matrix[t][k]` is task t's
accuracy after the (t+k)-th task was learned.  Every incremental claim in this repository is a
SUMMARY of that matrix -- ACC is its final column, FGT is how much each row fell from its own first
entry -- and the matrix itself has never been plotted.  A summary cannot distinguish an arm that
degrades gracefully from one that collapses at a particular task, and the whole point of a lifelong
claim is the shape of the curve, not its endpoint.

Two panels, both straight out of the recorded matrix, no new computation:

  LEFT   RETENTION.  For each arm, each task's accuracy against the number of tasks learned SINCE it
         was absorbed.  A flat line is a task that never degrades. This is where bit-identity should
         be visible as a horizontal line rather than as a number in a table.

  RIGHT  PLASTICITY.  Mean accuracy over everything learned so far, against how much has been
         learned. A lifelong learner should hold this level as the stream grows; an arm that trades
         old for new bends down.

Reads `results/n5_classinc_full.json` (or the seeded variants, pooled).  Arms are drawn from the file
rather than named here, so a new baseline appears in the figure the next time n5 runs.

    python experiments/make_n5_curves.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures", "initial")

# the arms worth drawing: ours, the strongest structural rival, and the classic regularizers.
# Everything in the file is plotted, but these get labelled colours so the figure stays readable.
HILITE = {"ours": "#1F4E79", "independent": "#2E7D32", "c-lora": "#B03A2E",
          "der++": "#8E6C00", "finetune": "#888888"}


def load_rows():
    """-> {arm: [acc_matrix per seed]} pooled over whatever seed files exist."""
    files = [p for p in sorted(glob.glob(os.path.join(RES, "n5_classinc*.json")))
             if "smoke" not in p]
    out = {}
    for p in files:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        for r in d.get("rows", []):
            am = r.get("acc_matrix")
            if am:
                out.setdefault(r["arm"], []).append(am)
    return out


def retention(am, min_tasks=3):
    """-> (offsets, mean accuracy at that offset), averaged over a FIXED set of tasks.

    Offset k = "k tasks have been learned since this one was", so k=0 is the accuracy a task had the
    moment it was absorbed.

    The averaging has to be BALANCED, and the first version of this was not. The matrix is ragged --
    a task absorbed late has fewer later steps -- so averaging whatever tasks happen to reach offset
    k mixes a different subset into every point. That made `ours` appear to decay from 0.836 to
    0.786 across offsets while **every one of its rows is exactly constant** (within-row swing
    0.000, FGT 0.000, retention_exact 1.000): the apparent decline was entirely composition change
    between subsets, and it misrepresented the repository's central claim in the direction of
    understating it.

    So: pick the deepest offset that at least `min_tasks` tasks reach, and average only over tasks
    that reach it. Every point is then the same tasks, and a preserved arm plots as the flat line it
    is.
    """
    rows = [r for r in am if r]
    if not rows:
        return np.array([]), np.array([])
    depths = sorted((len(r) for r in rows), reverse=True)
    kmax = depths[min_tasks - 1] if len(depths) >= min_tasks else depths[0]
    keep = [r for r in rows if len(r) >= kmax]
    if not keep:
        return np.array([]), np.array([])
    M = np.array([r[:kmax] for r in keep], float)
    return np.arange(kmax), M.mean(0)


def plasticity(am):
    """-> (n learned, mean accuracy over everything learned so far)."""
    n = max(len(r) + i for i, r in enumerate(am) if r)
    xs, ys = [], []
    for step in range(n):
        vals = [r[step - i] for i, r in enumerate(am) if r and 0 <= step - i < len(r)]
        if vals:
            xs.append(step + 1)
            ys.append(float(np.mean(vals)))
    return np.array(xs), np.array(ys)


def main():
    data = load_rows()
    if not data:
        print("[f31] no n5_classinc results found")
        return 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for arm, mats in sorted(data.items()):
        col = HILITE.get(arm)
        style = dict(color=col or "#CCCCCC", lw=2.0 if col else 1.0,
                     alpha=1.0 if col else 0.55, zorder=3 if col else 1)
        R = [retention(m) for m in mats]
        k = min(len(x[0]) for x in R)
        ks, ys = R[0][0][:k], np.mean([r[1][:k] for r in R], 0)
        axes[0].plot(ks, ys, marker="o", ms=3, label=arm if col else None, **style)
        P = [plasticity(m) for m in mats]
        k2 = min(len(x[0]) for x in P)
        xs, zs = P[0][0][:k2], np.mean([p[1][:k2] for p in P], 0)
        axes[1].plot(xs, zs, marker="o", ms=3, label=arm if col else None, **style)

    axes[0].set_xlabel("tasks learned SINCE this one was absorbed", fontsize=8)
    axes[0].set_ylabel("accuracy on that task", fontsize=8)
    axes[0].set_title("Retention — same tasks at every offset; flat = never degrades", fontsize=9)
    axes[1].set_xlabel("tasks absorbed", fontsize=8)
    axes[1].set_ylabel("mean accuracy over all learned tasks", fontsize=8)
    axes[1].set_title("Plasticity — does the level hold as the stream grows?", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, framealpha=0.9)
    fig.suptitle(
        "Class-incremental ColorMNIST: the accuracy matrix `n5_classinc` already records, drawn.\n"
        "Grey lines are the remaining baselines. Retention and plasticity are the two things a "
        "summary statistic cannot separate.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIG, "06_learning_curves.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[f31] wrote {out}  ({len(data)} arms, "
          f"{sum(len(v) for v in data.values())} arm-runs pooled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
