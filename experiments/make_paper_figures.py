"""Every figure in the paper, regenerated from `results/aggregate.json` and the raw result JSONs.

One command, no GPU, no model:  python experiments/make_paper_figures.py

Each figure exists to carry one step of the argument, and nothing is drawn that is not measured:

  F1  the loop            what the method is, with the measured cost of each route on the arrows
  F2  coverage vs copying THE argument -- attribute accuracy cannot distinguish composing from
                          copying, and coverage can.  Both benchmarks on one pair of axes.
  F3  composition         held-out combinations vs every baseline, both benchmarks
  F4  vocabulary          the routing rule: combinations are free, new values need real data
  F5  preservation        forgetting by arm, both benchmarks
  F6  native parity       against a backbone that was TRAINED on the held-out slot
  F7  router              the system deciding its own updates
  F8  growth              cost against experience, with the router's measured error charged in

Case-study image grids (F9 ColorMNIST, F10 CelebA) are written by n9_showdown.py and
n20_celeba_composition.py, which need the backbones; this script leaves them alone.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                              # noqa: E402

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch          # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
OURS, REAL, BASE1, BASE2, BASE3 = "#4C78A8", "#9E9E9E", "#E45756", "#F58518", "#72B7B2"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def agg():
    with open(os.path.join(RESULTS, "aggregate.json")) as fh:
        return json.load(fh)


def m(d, *path, default=np.nan):
    """Pull a pooled mean out of aggregate.json, returning NaN rather than raising."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, dict) and "mean" in cur:
        return float(cur["mean"])
    return float(cur) if isinstance(cur, (int, float)) else default


def se(d, *path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return 0.0
        cur = cur[k]
    return float(cur.get("se", 0.0)) if isinstance(cur, dict) else 0.0


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p)}")


# ── F1 ────────────────────────────────────────────────────────────────────────
def f1_loop(A):
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 106); ax.set_ylim(-4, 36); ax.axis("off")
    boxes = [(2, "observe one\nOOD query", "#EEEEEE"),
             (23, "ROUTE\n(no label, no probe)", "#DCE9F5"),
             (46, "compose\n0 parameters", "#D6ECD9"),
             (46, "weight update\nfrom REAL data", "#FBE0DE"),
             (70, "REGISTER\nflat, insert-only", "#DCE9F5"),
             (88, "CERTIFY\nbit-identical", "#E8E0F2")]
    ys = [18, 18, 26, 8, 18, 18]
    for (x, label, col), y in zip(boxes, ys):
        ax.add_patch(FancyBboxPatch((x, y - 4.2), 14, 8.4, boxstyle="round,pad=0.4",
                                    fc=col, ec="#666666", lw=0.9))
        ax.text(x + 7, y, label, ha="center", va="center", fontsize=8.5)
    def arrow(x0, y0, x1, y1, txt="", col="#444444", dy=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     lw=1.1, color=col, shrinkA=0, shrinkB=0))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", fontsize=7.4, color=col)
    arrow(16, 18, 23, 18)
    arrow(37, 20, 46, 26, "new COMBINATION", "#2E7D32")
    arrow(37, 16, 46, 8, "new VOCABULARY", "#C62828", dy=-2.6)
    arrow(60, 26, 70, 20)
    arrow(60, 8, 70, 16)
    arrow(84, 18, 88, 18)
    # feedback edge routed BELOW the row so it crosses nothing
    ax.add_patch(FancyArrowPatch((95, 13.5), (95, -1.5), arrowstyle="-", lw=1.1, color="#999999"))
    ax.add_patch(FancyArrowPatch((95, -1.5), (9, -1.5), arrowstyle="-", lw=1.1, color="#999999"))
    ax.add_patch(FancyArrowPatch((9, -1.5), (9, 13.5), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color="#999999"))
    ax.text(52, -3.2, "the registered concept is now a source for later composition",
            ha="center", fontsize=7.6, color="#888888")
    r = A.get("n18_router", {}).get("table", {})
    auroc = m(r, "new-vocabulary vs new-combination", "auroc")
    leak = m(r, "vocab_sent_to_free")
    ax.text(53, 33.0, f"router AUROC {auroc:.3f}   |   vocabulary sent down the free route: "
                      f"{leak:.3f}   |   combinations cost 1024 bytes, vocabulary a kept delta",
            ha="center", fontsize=8, color="#333333")
    fig.suptitle("Lifelong OOD learning as repeated composition: the loop, and what each route costs",
                 fontsize=10.5, y=1.02)
    save(fig, "f1_loop.png")


# ── F2 ────────────────────────────────────────────────────────────────────────
def f2_coverage_vs_copying(A):
    """The argument in one figure: accuracy cannot tell composing from copying; coverage can."""
    cm = A.get("n9_showdown", {}).get("table", {})
    cb = A.get("n20_celeba_composition_pair", {}).get("table", {})
    panels = [("ColorMNIST (disjoint factors)", cm, "joint",
               [("real", REAL), ("ours", OURS), ("poe-guided", BASE1),
                ("query-only", BASE2), ("nearest-class", BASE3)]),
              ("CelebA (overlapping factors)", cb, "pair",
               [("real", REAL), ("ours", OURS), ("poe-guided", BASE1),
                ("query-only", BASE2), ("nearest-class", BASE3)])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, tbl, acc_key, arms) in zip(axes, panels):
        for arm, col in arms:
            x, y = m(tbl, arm, acc_key), m(tbl, arm, "recall")
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            ax.scatter(x, y, s=150 if arm == "ours" else 90, color=col, zorder=3,
                       edgecolor="white", linewidth=1.3,
                       marker="*" if arm == "real" else "o")
            ax.annotate(arm, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=8)
        ax.axhspan(-0.02, 0.12, color="#E45756", alpha=0.07, zorder=0)
        ax.text(0.02, 0.045, "copying: right attributes, no coverage", fontsize=7.2,
                color="#B03A38", transform=ax.get_yaxis_transform(), va="center")
        ax.set_xlabel(f"attribute accuracy ({acc_key})")
        ax.set_ylabel("recall  (coverage of the held-out class)")
        ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0)
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("Attribute accuracy cannot separate composing from copying — coverage can.\n"
                 "The published method and the query-only baseline sit in the red band on BOTH "
                 "benchmarks: correct attributes, near-zero coverage.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "f2_coverage_vs_copying.png")


# ── F3 ────────────────────────────────────────────────────────────────────────
def f3_composition(A):
    cm = A.get("n9_showdown", {}).get("table", {})
    cb = A.get("n20_celeba_composition_pair", {}).get("table", {})
    arms = ["real", "ours", "poe-guided", "query-only", "nearest-class"]
    cols = [REAL, OURS, BASE1, BASE2, BASE3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tbl, keys, title) in zip(axes, [
            (cm, ["joint", "recall", "f1"], "ColorMNIST — held-out (digit, fg, bg) slots"),
            (cb, ["pair", "recall", "f1"], "CelebA — Smiling x Big_Nose never co-occurred")]):
        x = np.arange(len(keys)); w = 0.16
        for i, (a, c) in enumerate(zip(arms, cols)):
            v = [m(tbl, a, k) for k in keys]
            e = [se(tbl, a, k) for k in keys]
            ax.bar(x + (i - 2) * w, v, w, yerr=e, capsize=2, label=a, color=c)
        ax.set_xticks(x); ax.set_xticklabels(keys)
        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("score"); axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("Composition recovers combinations that were never observed, on both benchmarks "
                 "(3 seeds)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "f3_composition.png")


# ── F4 ────────────────────────────────────────────────────────────────────────
def f4_vocabulary(A):
    """The routing rule, measured: combinations are free, a new VALUE needs real data."""
    n11 = A.get("n11_basis_extension_nmadapt", {}).get("table", {})
    n21 = A.get("n21_celeba_vocabulary_Gray_Hair", {}).get("table", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    arms = [("real (ceiling)", REAL), ("UNSEEN combo", OURS), ("adapter (per concept)", BASE1)]
    v = [m(n11, a, "f1") for a, _ in arms]; e = [se(n11, a, "f1") for a, _ in arms]
    axes[0].bar(range(3), v, 0.55, yerr=e, capsize=3, color=[c for _, c in arms])
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["real\n(ceiling)", "basis\n1024 bytes", "adapter\n551K/concept"],
                            fontsize=8)
    axes[0].set_ylabel("F1"); axes[0].set_ylim(0, 1.0)
    axes[0].set_title("ColorMNIST: a new COMBINATION of known values\n"
                      "one shared direction beats a per-concept adapter at 2153x less",
                      fontsize=9)

    arms2 = [("real (ceiling) [UNSEEN]", REAL), ("adapter (real) [UNSEEN]", OURS),
             ("learned value [UNSEEN]", BASE2), ("untrained value [UNSEEN]", BASE1)]
    v2 = [m(n21, a, "target") for a, _ in arms2]; e2 = [se(n21, a, "target") for a, _ in arms2]
    axes[1].bar(range(4), v2, 0.55, yerr=e2, capsize=3, color=[c for _, c in arms2])
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["real\n(ceiling)", "adapter on\nREAL images",
                             "learned\nconditioning", "untrained\n(floor)"], fontsize=8)
    axes[1].set_ylabel("accuracy on the withheld attribute"); axes[1].set_ylim(0, 0.65)
    axes[1].set_title("CelebA: a new VOCABULARY value the backbone never saw\n"
                      "conditioning is indistinguishable from untrained; only real data works",
                      fontsize=9)
    fig.suptitle("The routing rule, measured: new combination -> free;  new vocabulary -> "
                 "a weight update from REAL data", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "f4_vocabulary.png")


# ── F5 ────────────────────────────────────────────────────────────────────────
def f5_preservation(A):
    n5 = A.get("n5_classinc", {}).get("table", {})
    n22 = A.get("n22_celeba_lifelong", {}).get("table", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tbl, title) in zip(axes, [(n5, "ColorMNIST (Split-ColorMNIST, 12 tasks)"),
                                       (n22, "CelebA (12 held-out combinations)")]):
        arms = [a for a in ("ours", "independent", "ttt", "ewc", "finetune", "replay")
                if a in tbl]
        if not arms:
            ax.text(0.5, 0.5, "n22 still running", ha="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9.5); continue
        f = [m(tbl, a, "FGT") for a in arms]
        er = [se(tbl, a, "FGT") for a in arms]
        cols = [OURS if a in ("ours", "independent") else BASE1 for a in arms]
        ax.bar(range(len(arms)), f, 0.55, yerr=er, capsize=3, color=cols)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=20, fontsize=8)
        ax.set_ylabel("forgetting (0 = none)")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("Arms that structurally REMOVE shared parameters forget exactly 0.000; "
                 "arms that try to PROTECT them forget.\n(blue = removes, red = protects)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "f5_preservation.png")


# ── F6/F7/F8: already-good figures, redrawn here so one command rebuilds everything ──
def f6_parity(A):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                                                     "figure_parity.py")],
                       capture_output=True, text=True)
    src = os.path.join(FIG, "n23_native_parity.png")
    if os.path.exists(src):
        os.replace(src, os.path.join(FIG, "f6_native_parity.png"))
        print("  wrote figures/f6_native_parity.png")
    else:
        print("  (parity figure unavailable)", r.stderr.strip()[:120])


def f7_f8_rename():
    for src, dst in (("n18_router.png", "f7_router.png"), ("n24_growth.png", "f8_growth.png"),
                     ("n9_showdown.png", "f9_case_colormnist.png"),
                     ("n20_celeba_composition_pair.png", "f10_case_celeba.png")):
        s = os.path.join(FIG, src)
        if os.path.exists(s):
            os.replace(s, os.path.join(FIG, dst))
            print(f"  wrote figures/{dst}")


def main():
    A = agg()
    print("building paper figures from results/aggregate.json:")
    f1_loop(A); f2_coverage_vs_copying(A); f3_composition(A); f4_vocabulary(A); f5_preservation(A)
    f6_parity(A); f7_f8_rename()
    print("done.")


if __name__ == "__main__":
    main()
