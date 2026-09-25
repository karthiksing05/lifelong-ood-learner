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
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                              # noqa: E402

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch          # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")
OURS, REAL, BASE1, BASE2, BASE3 = "#4C78A8", "#9E9E9E", "#E45756", "#F58518", "#72B7B2"
# Composable Diffusion gets its own colour because it is the arm the coverage argument now rests
# on: at its best scale it matches our ACCURACY on CelebA and loses on recall (RESULTS 9g).
BASE4 = "#54A24B"
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
    """The loop, with the THREE storage classes §25 measured rather than the old two.

    The previous version showed composition as "0 parameters" and registration as the destination
    for everything.  Both are now wrong: a concept expressible in the factor basis is stored as
    COORDINATES (72 bytes, §25), a new factor VALUE as one basis direction (1024 bytes), and only a
    genuinely new FACTOR costs a block.  Redrawn so the figure states the cost the measurements
    support instead of the one the design originally assumed.
    """
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 130); ax.set_ylim(-10, 53); ax.axis("off")
    # Four routes fanning from one point cannot also carry four arrow labels without collisions, so
    # the CONDITION goes inside its own box and the arrow carries only colour.
    boxes = [(2, 20, "observe one\nOOD query", "#EEEEEE", None),
             (25, 20, "ROUTE\nno label, no probe", "#DCE9F5", None),
             (56, 38, "already expressible", "in distribution\n\u2192 do nothing", "#E6E6E6"),
             (56, 26, "new ARRANGEMENT", "ADDRESS\ncoordinates \u2014 72 B", "#D6ECD9"),
             (56, 14, "new VALUE", "EXTEND BASIS\none direction \u2014 1 KB", "#D6ECD9"),
             (56, 2, "new FACTOR", "BLOCK\nlow-rank delta", "#FBE0DE"),
             (88, 20, "REGISTER\nflat, insert-only", "#DCE9F5", None),
             (110, 20, "CERTIFY\nbit-identical", "#E8E0F2", None)]
    for b in boxes:
        if b[4] is None:
            x, y, label, col, _ = b
            ax.add_patch(FancyBboxPatch((x, y - 4.4), 17, 8.8, boxstyle="round,pad=0.4",
                                        fc=col, ec="#666666", lw=0.9))
            ax.text(x + 8.5, y, label, ha="center", va="center", fontsize=8.4)
        else:
            x, y, cond, label, col = b
            ax.add_patch(FancyBboxPatch((x, y - 5.0), 18, 10.0, boxstyle="round,pad=0.4",
                                        fc=col, ec="#666666", lw=0.9))
            ax.text(x + 9, y + 2.9, cond, ha="center", va="center", fontsize=6.9,
                    style="italic", color="#555555")
            ax.text(x + 9, y - 1.1, label, ha="center", va="center", fontsize=8.4)

    def arrow(x0, y0, x1, y1, col="#444444"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     lw=1.1, color=col, shrinkA=0, shrinkB=0))
    arrow(19, 20, 25, 20)
    for y, col in ((38, "#8A8A8A"), (26, "#2E7D32"), (14, "#2E7D32"), (2, "#C62828")):
        arrow(42, 20, 56, y, col)
    for y in (26, 14, 2):
        arrow(74, y, 88, 20)
    arrow(105, 20, 110, 20)
    # feedback routed well below the lowest box (BLOCK spans y = -3 .. 7)
    for seg in (((118, 15.5), (118, -7.5)), ((118, -7.5), (10, -7.5))):
        ax.add_patch(FancyArrowPatch(*seg, arrowstyle="-", lw=1.1, color="#999999"))
    ax.add_patch(FancyArrowPatch((10, -7.5), (10, 15.5), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color="#999999"))
    ax.text(64, -9.3, "the registered concept becomes a source for later composition",
            ha="center", fontsize=7.6, color="#888888")
    r = A.get("n18_router", {}).get("table", {})
    auroc = m(r, "new-vocabulary vs new-combination", "auroc")
    leak = m(r, "vocab_sent_to_free")
    ax.text(65, 51.0, f"router AUROC {auroc:.3f}   |   vocabulary sent down a free route: "
                      f"{leak:.3f}   |   24 concepts stored as coordinates cost 1.7 KB "
                      f"against 55 MB of blocks (\u00a725)",
            ha="center", fontsize=8.2, color="#333333")
    ax.text(65, 47.6, "K is chosen per query by explaining it to 95% of its own variance, "
                      "not fixed (\u00a726)", ha="center", fontsize=7.8, color="#666666")
    fig.suptitle("Lifelong OOD learning as repeated composition: the loop, and what each route "
                 "actually costs", fontsize=10.5, y=1.01)
    save(fig, "02_lifelong_loop.png")


# ── F2 ────────────────────────────────────────────────────────────────────────
def f2_coverage_vs_copying(A):
    """The argument in one figure: accuracy cannot tell composing from copying; coverage can."""
    cm = A.get("n9_showdown", {}).get("table", {})
    cb = A.get("n20_celeba_composition_pair", {}).get("table", {})
    panels = [("ColorMNIST (disjoint factors)", cm, "joint",
               [("real", REAL), ("ours", OURS), ("poe-guided", BASE1),
                ("composable-diff", BASE4), ("nearest-class", BASE3)]),
              ("CelebA (overlapping factors)", cb, "pair",
               [("real", REAL), ("ours", OURS), ("poe-guided", BASE1),
                ("composable-diff", BASE4), ("nearest-class", BASE3)])]
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
        # precision annotated per point: the baselines are not merely accurate, their samples are
        # PLAUSIBLE members of the class (precision 0.84 / 0.94).  Only coverage separates them.
        for arm, _c in arms:
            x_, y_ = m(tbl, arm, acc_key), m(tbl, arm, "recall")
            pr = m(tbl, arm, "precision")
            if np.isfinite(x_) and np.isfinite(y_) and np.isfinite(pr):
                ax.annotate(f"prec {pr:.2f}", (x_, y_), textcoords="offset points",
                            xytext=(7, -9), fontsize=6.5, color="0.35")
        ax.set_xlabel(f"attribute accuracy ({acc_key})")
        ax.set_ylabel("recall  (coverage of the held-out class)")
        ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0)
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("Attribute accuracy cannot separate composing from copying — coverage can.\n"
                 "The baselines are run in their OWN published configuration and beat us on "
                 "accuracy AND precision;\nwhat they do not have is recall.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "08_coverage_vs_copying.png")


# ── F3 ────────────────────────────────────────────────────────────────────────
def f3_composition(A):
    cm = A.get("n9_showdown", {}).get("table", {})
    cb = A.get("n20_celeba_composition_pair", {}).get("table", {})
    arms = ["real", "ours", "composable-diff", "poe-guided", "nearest-class"]
    cols = [REAL, OURS, BASE4, BASE1, BASE3]
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
    """Two metrics, because on CelebA the usual one barely separates the arms and the other does.

    `forgetting` is accuracy drift, the standard continual-learning measure.  `bit-identical` is the
    fraction of earlier concepts whose epsilon output on fixed noise is UNCHANGED -- not close,
    unchanged.  On CelebA accuracy spans 0.817-0.864 across arms with `ewc` nominally highest, while
    the bit-identical fraction is 1.00 for the two arms that remove shared parameters and 0.25 for
    all three that try to protect them.  A method can look fine on the first while having overwritten
    three quarters of what it held.
    """
    n5 = A.get("n5_classinc", {}).get("table", {})
    n22 = A.get("n22_celeba_lifelong", {}).get("table", {})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col")
    order = ("ours", "independent", "ttt", "c-lora", "ewc", "si", "mas", "der++",
             "gen-distill", "replay", "finetune")
    for col, (tbl, title) in enumerate([(n5, "ColorMNIST — Split-ColorMNIST, 12 tasks"),
                                        (n22, "CelebA — 12 held-out combinations")]):
        arms = [a for a in order if a in tbl]
        if not arms:
            for r in (0, 1):
                axes[r][col].text(0.5, 0.5, "not run", ha="center", transform=axes[r][col].transAxes)
            axes[0][col].set_title(title, fontsize=9.5)
            continue
        # three categories, not two: `ttt` and `der++` buy their retention with STORED DATA -- ttt
        # re-fits each task from every instance it kept, der++ replays a permanent exemplar buffer --
        # which is the cost a continual method is supposed to avoid, and is invisible in a plot of
        # forgetting alone.  Grey marks "did not forget, but kept the data".
        removes, keeps_data = {"ours", "independent"}, {"ttt", "der++"}
        cols = [OURS if a in removes else ("#BDBDBD" if a in keeps_data else BASE1) for a in arms]
        for row, key, lab in ((0, "FGT", "forgetting  (0 = none)"),
                              (1, "retention_exact", "fraction bit-identical")):
            ax = axes[row][col]
            v = [m(tbl, a, key) for a in arms]
            e = [se(tbl, a, key) for a in arms]
            ax.bar(range(len(arms)), v, 0.6, yerr=e, capsize=3, color=cols)
            for i, val in enumerate(v):
                if np.isfinite(val):
                    ax.annotate(f"{val:.3f}", (i, val), textcoords="offset points",
                                xytext=(0, 4), ha="center", fontsize=7)
            ax.set_ylabel(lab, fontsize=8.5)
            ax.set_ylim(0, max(1.15 if row else max([x for x in v if np.isfinite(x)] + [0.05]) * 1.35,
                               0.05))
            ax.axhline(0, color="0.3", lw=0.8)
        axes[0][col].set_title(title, fontsize=9.5)
        axes[1][col].set_xticks(range(len(arms)))
        axes[1][col].set_xticklabels(arms, rotation=18, fontsize=8)
    fig.suptitle("Preservation is structural.  Blue REMOVES shared parameters; red tries to PROTECT "
                 "them; grey (ttt) keeps every instance forever.\nOn CelebA accuracy barely "
                 "separates the arms — the bit-identical fraction separates them completely.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "10_preservation.png")


# ── F6/F7/F8: already-good figures, redrawn here so one command rebuilds everything ──
def f6_parity(A):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                                                     "figure_parity.py")],
                       capture_output=True, text=True)
    src = os.path.join(FIG, "n23_native_parity.png")
    if os.path.exists(src):
        os.replace(src, os.path.join(FIG, "11_native_parity.png"))
        print("  wrote figures/11_native_parity.png")
    else:
        print("  (parity figure unavailable)", r.stderr.strip()[:120])


def f7_f8_rename():
    """Give the raw experiment figures their paper names.

    Two corrections here, both of which had silently stopped this doing its job:

    * `n9_showdown.png -> case_colormnist.png` COLLIDED with `make_case_studies.py`, which writes
      `case_colormnist.png` directly and is the actual case-study panel (real / prototypes / four
      methods, annotated with accuracy AND coverage).  Whichever ran last won, so the README's case
      study was whichever figure happened to be written most recently.  The showdown keeps its own
      name now.
    * The CelebA source is written under the run's TAG.  The canonical CelebA table is the
      `_pairfull` run (reproduce.sh aliases it), so `n20_celeba_composition_pair.png` is not what
      the 3-seed run produces and f10 stopped tracking it.  Both names are tried, newest first.
    """
    # f10_case_celeba is RETIRED (2026-08-13).  It aliased n20, which runs on
    # `ddpm_celeba_factored.pt` -- 64 px and carrying no `layout` key, so it loads as `permuted` and
    # renders with CNN grid artifacts.  **f26 already supersedes it**: same row structure (real /
    # prototypes / ours / poe-guided / query-only / unconditional, one block per held-out class) on
    # the companion paper's OWN backbone, `ddpm_celeba_paper128.pt`, which is `layout: hwc` at 128 px.
    # Nothing needed retraining -- the right backbone was already on disk and already in use by n48.
    # The old panel is in figures/archive/.
    pairs = [("n18_router.png", "f7_router.png"), ("n24_growth.png", "04_growth_per_concept.png"),
             ("n9_showdown.png", "f9b_showdown_colormnist.png")]
    for src, dst in pairs:
        cands = [src] if isinstance(src, str) else list(src)
        cands = [c for c in cands if os.path.exists(os.path.join(FIG, c))]
        if not cands:
            continue
        # copy, do NOT move: os.replace consumed the source, so a second run of this script had
        # nothing left to rename and the paper figure quietly went stale
        newest = max(cands, key=lambda c: os.path.getmtime(os.path.join(FIG, c)))
        shutil.copyfile(os.path.join(FIG, newest), os.path.join(FIG, dst))
        print(f"  wrote figures/{dst}  (from {newest})")


def main():
    A = agg()
    print("building paper figures from results/aggregate.json:")
    f1_loop(A); f2_coverage_vs_copying(A); f3_composition(A); f4_vocabulary(A); f5_preservation(A)
    f6_parity(A); f7_f8_rename()
    print("done.")


if __name__ == "__main__":
    main()
