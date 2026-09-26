"""M3 FINAL figures from results/robot/m3_stats.json (experiments/m3_stats.py):

  figures/robot/m3_frontier.{png,pdf}   storage vs success (tasks where the practice fails), cluster-bootstrap 95%
  figures/robot/m3_per_task.{png,pdf}   success per task and method (Wilson 95%)
  figures/robot/m3_paired.{png,pdf}     pooled paired differences with the non-inferiority margin

Palette: the dataviz reference palette's first three categorical slots (validated all-pairs in both modes) for ours /
LoRA / replay, muted gray for zero-shot; our variants share the blue hue and are told apart by marker and direct label.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures", "robot")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"
STYLE = {"zeroshot": (MUTED, "o", "zero-shot"), "practice_replay": (AQUA, "s", "demo replay"),
         "practice_lora": (ORANGE, "D", "few-shot LoRA (rank 16)"), "ours_dim": (BLUE, "o", "borrowing, every 5th frame"),
         "ours_dim1": (BLUE, "^", "borrowing, every frame"), "ours_dim1_last2": (BLUE, "*", "borrowing + 258 KB fold"),
         "ours_codes_full": (BLUE_LIGHT, "P", "PoE codes + 4.7 MB fold")}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fmt_bytes(b):
    return "0 B" if b <= 0 else (f"{b / 1e6:.1f} MB" if b >= 1e6 else (f"{b / 1e3:.1f} KB" if b >= 1e3 else f"{b:.0f} B"))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] figures/robot/{name}.png/.pdf")


def frontier(S, scope="prior_fails"):
    M = S["methods"]
    n_tasks = len(S["prior_fails"] if scope == "prior_fails" else S["tasks"])
    fig, ax = plt.subplots(figsize=(6.4, 3.9), facecolor=SURFACE)
    style(ax)
    z = M.get(f"zeroshot.{scope}")
    if z:
        ax.axhline(z["rate"], color=MUTED, linewidth=1.0, zorder=1)
        ax.text(1.15e3 if False else 0.012, z["rate"] + 0.012, f"zero-shot, 0 B stored: {z['k']}/{z['n']}", transform=ax.get_yaxis_transform(),
                color=INK2, fontsize=7.5, va="bottom")
    pts = []
    for m in ("ours_dim", "ours_dim1", "practice_replay", "ours_dim1_last2", "ours_codes_full", "practice_lora"):
        r = M.get(f"{m}.{scope}")
        if not r or not r["complete"]:
            continue
        c, mk, name = STYLE[m]
        x = max(r["bytes"], 1)
        ax.errorbar([x], [r["rate"]], yerr=[[r["rate"] - r["lo"]], [r["hi"] - r["rate"]]], fmt="none", ecolor=c, elinewidth=1.2,
                    capsize=0, zorder=2, alpha=0.9)
        ax.scatter([x], [r["rate"]], s=90 if mk == "*" else 46, marker=mk, color=c, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        pts.append((x, r["rate"], f"{name}\n{r['k']}/{r['n']}, {fmt_bytes(r['bytes'])}", c))
    ax.set_xscale("log")
    xs = [p[0] for p in pts]
    ax.set_xlim(min(xs) / 40, max(xs) * 12)
    ax.set_ylim(0, 1)
    # label placement per method: offsets in points (a hairline leader where the label leaves its point), or "top" =
    # centred above the method's interval
    place = {"borrowing, every 5th frame": (-10, -30, "right"), "demo replay": (-14, 34, "right"),
             "borrowing, every frame": (14, -30, "left"), "borrowing + 258 KB fold": "bottom",
             "PoE codes + 4.7 MB fold": "top"}
    for (x, y, lab, c), m in zip(pts, [m for m in ("ours_dim", "ours_dim1", "practice_replay", "ours_dim1_last2", "ours_codes_full",
                                                   "practice_lora") if M.get(f"{m}.{scope}") and M[f"{m}.{scope}"]["complete"]]):
        how = place.get(lab.split("\n")[0], (8, 0, "left"))
        if how in ("top", "bottom"):
            r = M[f"{m}.{scope}"]
            ax.annotate(lab, (x, r["hi"] if how == "top" else r["lo"]), xytext=(0, 5 if how == "top" else -5), textcoords="offset points",
                        fontsize=7, color=INK, ha="center", va="bottom" if how == "top" else "top", linespacing=1.15)
            continue
        dx, dy, ha = how
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK, va="center", ha=ha,
                    linespacing=1.15, arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6, shrinkA=0, shrinkB=4) if dy else None)
    ax.set_xlabel(f"bytes stored for the {n_tasks} tasks (log scale)", fontsize=8.5, color=INK2)
    ax.set_ylabel("success, pooled over tasks", fontsize=8.5, color=INK2)
    ax.set_title(f"Storage vs success on held-out LIBERO-90 tasks where pi0.5 fails ({n_tasks} tasks, 20 episodes each)",
                 fontsize=9, color=INK, loc="left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.text(0.01, -0.04, "Bars: 95% cluster bootstrap (tasks, then init states).  Every method first practises 10 episodes of the prior "
             "and keeps it if >= 50% succeed.", fontsize=6.8, color=MUTED)
    save(fig, f"m3_frontier{'' if scope == 'prior_fails' else '_all'}")


def per_task(S, methods=("zeroshot", "practice_replay", "practice_lora", "ours_dim1_last2")):
    from lifelong_ood.stats import wilson
    P = S["per_task"]
    tasks = sorted(S["tasks"], key=lambda t: (-P[t]["ours_dim1_last2"]["k"] if "ours_dim1_last2" in P[t] else 0, int(t)))
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(tasks) + 1.2), facecolor=SURFACE)
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    off = np.linspace(-0.27, 0.27, len(methods))
    for j, m in enumerate(methods):
        c, mk, name = STYLE[m]
        for i, t in enumerate(tasks):
            r = P[t].get(m)
            if not r:
                continue
            p = r["k"] / r["n"]
            lo, hi = wilson(r["k"], r["n"])
            y = i + off[j]
            ax.plot([lo, hi], [y, y], color=c, linewidth=1.1, alpha=0.8, solid_capstyle="round")
            ax.scatter([p], [y], s=70 if mk == "*" else 30, marker=mk, color=c, edgecolors=SURFACE, linewidths=1.0, zorder=3,
                       label=name if i == 0 else None)
    ax.set_yticks(range(len(tasks)))
    lab = []
    for t in tasks:
        s = S["language"][t]
        s = s if len(s) <= 46 else s[:44] + "..."
        kept = " (prior kept)" if t in S["kept_prior"] else ""
        lab.append(f"{s}{kept}")
    ax.set_yticklabels(lab, fontsize=7.2, color=INK)
    ax.set_ylim(len(tasks) - 0.5, -0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("success over 20 episodes (init states 30-49), Wilson 95%", fontsize=8.5, color=INK2)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(methods), fontsize=7, frameon=False, handletextpad=0.3,
                    columnspacing=1.0, borderaxespad=0.3)
    for tx in leg.get_texts():
        tx.set_color(INK)
    save(fig, "m3_per_task")


def paired(S, scope="prior_fails"):
    C = S["compare"]
    order = [("ours_dim1_last2", "practice_lora"), ("ours_dim", "practice_lora"), ("ours_dim1_last2", "practice_replay"),
             ("ours_dim", "practice_replay"), ("ours_dim1", "practice_replay"), ("practice_lora", "practice_replay"),
             ("ours_dim1_last2", "zeroshot")]
    rows = [(x, y, C[f"{x}-{y}.{scope}"]) for x, y in order if f"{x}-{y}.{scope}" in C]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.1), facecolor=SURFACE)
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    m = S["margin"]
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.axvspan(-1, -m, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    ax.text(-m - 0.008, len(rows) - 0.5, f"below the non-inferiority\nmargin (-{m * 100:.0f} points)", fontsize=6.5, color=MUTED,
            ha="right", va="bottom", linespacing=1.1)
    for i, (x, y, d) in enumerate(rows):
        c = BLUE if x.startswith("ours") else INK2
        ax.plot([d["lo"], d["hi"]], [i, i], color=c, linewidth=1.3, solid_capstyle="round")
        ax.plot([d["lo90"], d["hi90"]], [i, i], color=c, linewidth=3.2, solid_capstyle="round", alpha=0.55)
        ax.scatter([d["diff"]], [i], s=36, color=c, edgecolors=SURFACE, linewidths=1.0, zorder=3)
        pv = "p < 0.001" if d["sign_p"] < 0.001 else f"p = {d['sign_p']:.2f}"
        ax.annotate(f"{d['diff'] * 100:+.1f}  (sign test {pv})", (d["hi"], i), xytext=(6, 0), textcoords="offset points",
                    fontsize=6.8, color=INK2, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{STYLE[x][2]}  vs  {STYLE[y][2]}" for x, y, _ in rows], fontsize=7.2, color=INK)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    lo = min(-m - 0.08, min(d["lo"] for _, _, d in rows) - 0.05)
    hi = max(d["hi"] for _, _, d in rows) + 0.28
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f}"))
    ax.set_xlabel("difference in pooled success (percentage points); thin 95%, thick 90% (one-sided 95% bounds)", fontsize=8, color=INK2)
    n_t = len(S["prior_fails"] if scope == "prior_fails" else S["tasks"])
    ax.set_title(f"Paired differences on the same init states ({n_t} tasks where pi0.5 fails)", fontsize=9, color=INK, loc="left")
    save(fig, "m3_paired")


def main():
    sys.path.insert(0, ROOT)
    S = json.load(open(os.path.join(ROOT, "results", "robot", "m3_stats.json")))
    frontier(S)
    frontier(S, "all")
    per_task(S)
    paired(S)


if __name__ == "__main__":
    main()
