"""Figure: the hierarchical factored library learned as a stream (E4: a demonstration whenever the composition does not beat the
policy; W1: only when neither the policy nor the composition reaches 8 of 10 in practice).  Cumulative demonstrations requested
against arrivals, per order; a hollow marker is an arrival kept without a demonstration.  Reads results/robot/{e4,w1}_state.json.
-> figures/robot/streams_demos.png (copied to results/robot/site/media).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/fig_streams.py
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
COL = {"e4": "#a35d2b", "w1": "#1d5e86"}                             # the page's baseline and ours colours
LAB = {"e4": "E4: demonstration unless the composition beats the policy", "w1": "W1: demonstration only when practice fails (< 8 of 10)"}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, o in zip(axes, ("a", "b")):
        for run in ("e4", "w1"):
            f = os.path.join(R, f"{run}_state.json")
            if not os.path.exists(f):
                continue
            st_ = json.load(open(f))
            if any(st_[x]["i"] < len(st_[x]["order"]) for x in ("a", "b")):
                continue                                                  # a stream still running is left out
            arr = [a for a in st_[o]["arrivals"] if "kept" in a]
            n, xs, ys, free = 0, [0], [0], []
            for i, a in enumerate(arr, 1):
                n += bool(a.get("demo")) and not a.get("demo_unavailable")
                xs.append(i)
                ys.append(n)
                if a.get("demo") is False:
                    free.append((i, n))
            ax.step(xs, ys, where="post", color=COL[run], lw=2, label=LAB[run])
            if free:
                ax.plot([x for x, _ in free], [y for _, y in free], "o", mfc="white", mec=COL[run], ms=6, mew=1.5)
        ax.plot([0, 30], [0, 29], ls=":", color="#97a3a9", lw=1)
        ax.set_title(f"order {o}", fontsize=11)
        ax.set_xlabel("arrivals")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("demonstrations requested")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "robot", "streams_demos.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    shutil.copy(out, os.path.join(R, "site", "media", "streams_demos.png"))
    print(out)


if __name__ == "__main__":
    main()
