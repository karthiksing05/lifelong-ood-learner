"""The one graphic for the reproduction: recall was a property of the protocol, not the method.

Two panels, because the finding has two halves and neither is convincing alone.

  LEFT   Recall against the number of queries pooled, in both feature spaces, against both of the
         paper's reference sets.  This is the mechanism: the SAME 1600 generated images score 0.000
         or 0.590 depending only on how many teachers produced them and what they are compared to.
         The repository's published `poe-guided` recall of 0.001 is the leftmost point of the lowest
         line -- one query, Generalization reference, pixel features -- reported as though it were
         the whole surface.

  RIGHT  Their Table 1 arms under their protocol, ours measured beside their published values.  The
         claim this panel supports is not that we beat them; it is that their method reproduces
         here, which is the precondition for any comparison to it being worth reading.

Reads `results/protocol_sweep.json` (tests/diag_paper_protocol.py) and
`results/n49_paper_protocol.json` (experiments/n49_paper_protocol.py).  Either may be absent; the
panel that has no data is drawn as a placeholder saying so rather than silently omitted, because a
figure that quietly loses a panel is how a stale artifact starts.

    python experiments/make_n49_figure.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures", "initial")

# their Table 1, ColorMNIST -- (Faithfulness, Generalization) F1
PAPER_T1 = {"top-1": (0.062, 0.065), "top-3": (0.051, 0.053), "query-only": (0.000, 0.000),
            "PoE": (0.753, 0.630), "+LoRA": (0.876, 0.763)}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def panel_sweep(ax, d):
    if not d:
        ax.text(0.5, 0.5, "results/protocol_sweep.json absent\nrun tests/diag_paper_protocol.py",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()
        return
    rows = d["rows"]
    styles = {("pixel", "Faithfulness"): ("#888888", "--", "o"),
              ("pixel", "Generalization"): ("#B03A2E", "--", "s"),
              ("inception", "Faithfulness"): ("#1F4E79", "-", "o"),
              ("inception", "Generalization"): ("#2E7D32", "-", "s")}
    for (space, ref), (c, ls, mk) in styles.items():
        pts = sorted((r["n_queries"], r["recall"]) for r in rows
                     if r["space"] == space and r["reference"] == ref)
        pts = [(x, y) for x, y in pts if not np.isnan(y)]
        if not pts:
            continue
        ax.plot(*zip(*pts), color=c, ls=ls, marker=mk, ms=4, lw=1.6,
                label=f"{space}, {ref[:7]}.")
    pf, pg = d["paper"]["PoE_faithful"]["recall"], d["paper"]["PoE_general"]["recall"]
    ax.axhline(pf, color="#1F4E79", lw=0.9, alpha=0.45)
    ax.axhline(pg, color="#2E7D32", lw=0.9, alpha=0.45)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 260)                      # headroom on the right for the reference labels
    ax.set_ylim(-0.04, 0.95)
    for y, c, lab in ((pf, "#1F4E79", "paper PoE, Faithful."),
                      (pg, "#2E7D32", "paper PoE, General.")):
        ax.text(255, y, f"{lab} {y:.2f} ", fontsize=6.5, color=c, va="bottom", ha="right")
    ax.set_xlabel("queries pooled per OOD class  (the paper runs 100)", fontsize=8)
    ax.set_ylabel("recall  (Kynkaanniemi k-NN, k=3)", fontsize=8)
    ax.set_title("One set of 1600 generated images, scored six ways", fontsize=9)
    ax.legend(fontsize=6.5, loc="center left", framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.annotate("this repository published\nTHIS cell as their method",
                xy=(1, 0.0), xytext=(1.25, 0.20), fontsize=6.5, color="#B03A2E",
                arrowprops=dict(arrowstyle="->", color="#B03A2E", lw=0.8))


def panel_table(ax, d, nq=None, with_paper=True, title=None):
    """F1 per arm at ONE query budget.  `nq=None` takes the largest available.

    Bars are labelled "measured here", never "ours" -- one of the ARMS is called `ours`, and a
    legend entry with the same word would make the figure ambiguous exactly where it is making its
    point.
    """
    if not d:
        ax.text(0.5, 0.5, "results/n49_paper_protocol.json absent\n"
                          "run experiments/n49_paper_protocol.py", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()
        return
    rows = d["rows"]
    nq = nq or max(r["n_queries"] for r in rows)
    arms = [a for a in ("top-1", "top-3", "query-only", "PoE", "+LoRA", "ours")
            if any(r["arm"] == a and r["n_queries"] == nq for r in rows)]
    x = np.arange(len(arms))
    w = 0.2 if with_paper else 0.36

    def cell(a, ref):
        v = [r["f1"] for r in rows if r["arm"] == a and r["reference"] == ref
             and r["n_queries"] == nq]
        return float(np.mean(v)) if v else np.nan

    if with_paper:
        ax.bar(x - 1.5 * w, [cell(a, "Faithful.") for a in arms], w, label="measured here, Faith.",
               color="#1F4E79")
        ax.bar(x - 0.5 * w, [PAPER_T1.get(a, (np.nan,))[0] for a in arms], w,
               label="Wang et al., Faith.", color="#1F4E79", alpha=0.4, hatch="//")
        ax.bar(x + 0.5 * w, [cell(a, "General.") for a in arms], w, label="measured here, Gen.",
               color="#2E7D32")
        ax.bar(x + 1.5 * w, [PAPER_T1.get(a, (np.nan, np.nan))[1] for a in arms], w,
               label="Wang et al., Gen.", color="#2E7D32", alpha=0.4, hatch="//")
    else:
        ax.bar(x - 0.5 * w, [cell(a, "Faithful.") for a in arms], w, label="Faithfulness",
               color="#1F4E79")
        ax.bar(x + 0.5 * w, [cell(a, "General.") for a in arms], w, label="Generalization",
               color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("F1", fontsize=8)
    ax.set_title(title or f"Their Table 1 arms under their protocol ({nq} queries)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2 if with_paper else 1, framealpha=0.9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.5)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep, tab = load("protocol_sweep.json"), load("n49_paper_protocol.json")
    # The third panel is the PAIRED budget: `ours` and `+LoRA` train one adapter per query, so they
    # are run on fewer of them, and every other arm is re-scored at that same count.  Recall rises
    # with the query count (that is the left panel), so comparing across budgets would be the very
    # error this figure documents.
    # `--distill-queries` IS the paired budget; taking min() over the arm's rows picked the
    # n_queries=1 block instead, where the Faithfulness reference is a single image and the metric
    # is undefined -- half an empty panel.  Read the budget the run was configured with.
    cfg = (tab or {}).get("config", {})
    nq_pair = cfg.get("distill_queries")
    if nq_pair is not None and not any(r["n_queries"] == nq_pair and r["arm"] == "ours"
                                       for r in (tab or {}).get("rows", [])):
        nq_pair = None
    ncol = 3 if nq_pair else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 4.3))
    panel_sweep(axes[0], sweep)
    panel_table(axes[1], tab)
    if nq_pair:
        panel_table(axes[2], tab, nq=nq_pair, with_paper=False,
                    title=f"All arms at a matched budget ({nq_pair} queries)")
    fig.suptitle(
        "Recall is a property of the evaluation protocol as much as of the method. "
        "Wang et al. pool 100 queries x 16 generations against two reference sets in Inception-V3;\n"
        "this repository measured 24 samples from ONE query against one reference set in pixel "
        "space, and published the result as their method's (recall 0.001 vs their 80.7).",
        fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    out = os.path.join(FIG, "extra", "f29_protocol.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[f29] wrote {out}"
          f"{'' if sweep else '  (LEFT PANEL IS A PLACEHOLDER)'}"
          f"{'' if tab else '  (RIGHT PANEL IS A PLACEHOLDER)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
