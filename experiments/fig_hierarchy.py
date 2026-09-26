"""The prototype hierarchy's anatomy and where composition draws on it.

  (a) prototypes per noise level t (the modes of the policy's action marginal over the 40 training tasks; the raw demo
      chunks' own modes as the no-policy control)
  (b) sharing: the share of frames whose prototype is used by >= 3 training tasks
  (c) the level each borrowed (frame, dimension) pair comes from, for K = 1, 2, 4, 8 lenders per frame (r9 ablation), with
      the rebuilt chunks' distance to the demo
Writes figures/robot/hierarchy_anatomy.{png,pdf}.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
RAMP = {"0.2": "#86b6ef", "0.3": "#2a78d6", "0.4": "#104281"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7.5, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    z = np.load(os.path.join(ROOT, "results", "robot", "r2b_libero_40.npz"))
    task = z["task"]
    us = sorted(float(k.split("_")[-1]) for k in z.files if k.startswith("owner_policy_"))
    stats = {}
    for src in ("policy", "demo"):
        n, shared = [], []
        for u in us:
            o = z[f"owner_{src}_{u}"]
            ks, inv = np.unique(o, return_inverse=True)
            nt = np.array([len(set(task[o == k].tolist())) for k in ks])
            n.append(len(ks))
            shared.append(float(np.mean(nt[inv] >= 3)))
        stats[src] = (np.array(n), np.array(shared))
    r9 = [r for r in ledger.load("r9_library_ablation")]
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.34))
    ax = axs[0]
    style(ax)
    for src, c, lab in (("demo", MUTED, "raw demo chunks (control)"), ("policy", BLUE, "policy (pi0.5)")):
        ax.plot(us, stats[src][0], color=c, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("noise level t (0 = data, 1 = noise)", fontsize=8, color=INK2)
    ax.set_title("(a) prototypes per level", fontsize=8.5, color=INK, loc="left")
    ax.annotate(f"{stats['policy'][0][0]:,} of {len(task):,} frames", (us[0], stats["policy"][0][0]), xytext=(8, 2), textcoords="offset points",
                fontsize=6.8, color=INK2)
    leg = ax.legend(fontsize=6.8, frameon=False, loc="lower left")
    [t.set_color(INK) for t in leg.get_texts()]
    ax = axs[1]
    style(ax)
    for src, c in (("demo", MUTED), ("policy", BLUE)):
        ax.plot(us, stats[src][1], color=c, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("noise level t", fontsize=8, color=INK2)
    ax.set_title("(b) frames whose prototype >= 3 tasks share", fontsize=8.5, color=INK, loc="left")
    ax = axs[2]
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    if r9 and r9[-1]["metrics"]["summary"].get("band_K8", {}).get("levels"):
        S = r9[-1]["metrics"]["summary"]
        Ks = [1, 2, 4, 8]
        for i, k in enumerate(Ks):
            lv = S[f"band_K{k}"]["levels"]
            left = 0.0
            for u in ("0.4", "0.3", "0.2"):
                w = lv[u]
                ax.barh(i, w - 0.004 if w > 0.01 else w, left=left, height=0.56, color=RAMP[u], linewidth=0,
                        label=f"t = {u}" if i == 0 else None)
                left += w
            ax.text(1.02, i, f"{S[f'band_K{k}']['rms']:.3f} from the demo", va="center", fontsize=6.8, color=INK2, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(Ks)))
        ax.set_yticklabels([f"K = {k}" for k in Ks], fontsize=7.5, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(len(Ks) - 0.5, -0.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("share of the borrowed (frame, dimension) pairs", fontsize=8, color=INK2)
        leg = ax.legend(fontsize=6.8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.2)
        [t.set_color(INK) for t in leg.get_texts()]
        ax.set_title("(c) which level K lenders per frame borrow from", fontsize=8.5, color=INK, loc="left", pad=18)
    out = os.path.join(ROOT, "figures", "robot", "hierarchy_anatomy")
    fig.savefig(out + ".png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    print(f"[fig] {out}.png")


if __name__ == "__main__":
    main()
