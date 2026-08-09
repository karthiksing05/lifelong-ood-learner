"""F0 — the architecture figure: what we instantiated, stage by stage.

Laid out to be read against Figure 1 of the companion paper (Wang et al., arXiv:2605.07078), which
runs left-to-right through `concept discovery via mode ascent` -> `PoE composition of top-K
concepts` -> `compositional sampling / distillation`.  Those four stages are reproduced in the top
band, unchanged and marked as inherited, so a reader can see exactly where this work starts: their
pipeline ends by distilling a LoRA and discarding it, and ours begins there.

Band 2 is what this work adds (route / consolidate / certify), band 3 expands the routing decision,
and band 4 opens up the object that makes it work -- the hierarchical adapter bank -- down to the
tensors, beside a step-by-step account of the operation each stage performs.

    python experiments/make_architecture_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch      # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

INHERIT, OURS, BANK = "#EAF1F8", "#E8F3EA", "#F7F2E6"
FROZEN, HOT, EDGE = "#DCDCDC", "#F6DCDC", "#5A5A5A"
GREEN, RED, BLUE = "#3E7050", "#A05555", "#3F5F80"


def panel(ax, x, y, w, h, fc, ec, lw=1.2, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def stage(ax, x, y, w, h, title, lines, fc):
    """A pipeline box: bold title near the top, body centred in the space beneath it."""
    panel(ax, x, y, w, h, fc, EDGE, z=2)
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + (h - 0.040) / 2, lines, ha="center", va="center", fontsize=8.2,
            zorder=3, linespacing=1.5)


def arrow(ax, p, q, color=EDGE, lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, zorder=4, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 13.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig.text(0.5, 0.973, "The instantiated architecture: an insert-only concept bank on a frozen "
             "diffusion model", ha="center", fontsize=15.5, fontweight="bold")
    fig.text(0.5, 0.949, "Stages 1–4 are the companion paper's, used unchanged.  Stages 5–7 are this "
             "work, and are what turns a one-shot method into a lifelong learner.",
             ha="center", fontsize=10, color="#333333")

    # ══ BANDS 1 & 2 — the pipeline ═════════════════════════════════════════
    y1, h1, w = 0.800, 0.105, 0.118
    panel(ax, 0.022, y1 - 0.026, 0.545, h1 + 0.055, "none", "#8FA9C4", 1.3, (0, (5, 3)))
    ax.text(0.030, y1 + h1 + 0.036, "INHERITED — their Fig. 1  (frozen model, one query, no labels)",
            fontsize=9.6, fontweight="bold", color=BLUE)
    panel(ax, 0.588, y1 - 0.026, 0.392, h1 + 0.055, "none", "#6FA97C", 1.3, (0, (5, 3)))
    ax.text(0.596, y1 + h1 + 0.036, "THIS WORK — what happens to the result", fontsize=9.6,
            fontweight="bold", color=GREEN)

    xs = [0.034, 0.170, 0.306, 0.442]
    stage(ax, xs[0], y1, w, h1, "1  DISCOVER",
          "mode ascent on $s_\\theta(x_t,t)$\nat several $t$\n→ candidate modes", INHERIT)
    stage(ax, xs[1], y1, w, h1, "2  SELECT",
          "greedy submodular\nchoice of $K$\nprototypes", INHERIT)
    stage(ax, xs[2], y1, w, h1, "3  COMPOSE",
          "per-dimension\nproduct of experts\n→ teacher $q_T$", INHERIT)
    stage(ax, xs[3], y1, w, h1, "4  SAMPLE",
          "the MANIFOLD at $c_{new}$,\nnot the Gaussian\n(§5e)", INHERIT)
    x5, x6, x7 = 0.600, 0.736, 0.872
    stage(ax, x5, y1, w, h1, "5  ROUTE",
          "arrangement, or\na new part?\nAUROC 0.980 / 0.995", OURS)
    stage(ax, x6, y1, w, h1, "6  CONSOLIDATE",
          "a new leaf;\nancestors sealed\nand unreachable", OURS)
    stage(ax, x7, y1, w, h1, "7  CERTIFY",
          "every earlier concept\nbit-identical\n(==, not ≈)", OURS)
    for a, b in zip([xs[0], xs[1], xs[2], xs[3], x5, x6], [xs[1], xs[2], xs[3], x5, x6, x7]):
        arrow(ax, (a + w, y1 + h1 / 2), (b, y1 + h1 / 2), lw=1.8 if a == xs[3] else 1.4)
    ax.text(0.034, y1 - 0.017, "their pipeline distils a LoRA here — and discards it",
            fontsize=8.4, style="italic", color=RED)

    # the loop back, drawn as an explicit routed path in the clear band beneath the pipeline so it
    # crosses no box and no text
    yl = y1 - 0.049
    ax.plot([x7 + w / 2, x7 + w / 2], [y1 - 0.026, yl], color="#7C7C7C", lw=1.3,
            ls=(0, (4, 3)), zorder=1)
    ax.plot([x7 + w / 2, 0.093], [yl, yl], color="#7C7C7C", lw=1.3, ls=(0, (4, 3)), zorder=1)
    arrow(ax, (0.093, yl), (0.093, y1), color="#7C7C7C", lw=1.3, ls=(0, (4, 3)))
    ax.text(0.50, yl + 0.008, "the registered concept becomes a SOURCE for later composition — "
            "the one arrow the companion paper does not have",
            ha="center", va="bottom", fontsize=8.8, style="italic", color="#5A5A5A")

    # ══ BAND 3 — routing, expanded ═════════════════════════════════════════
    yr, hr = 0.560, 0.145
    ax.text(0.022, yr + hr + 0.014, "STAGE 5 expanded — decided from the model's OWN denoising "
            "error.  No labels, no held-out probe.", fontsize=9.8, fontweight="bold")
    panel(ax, 0.034, yr, 0.150, hr, "#F4F4F4", EDGE, 1.1, z=2)
    ax.text(0.109, yr + hr - 0.020, "signals", ha="center", va="top", fontsize=10,
            fontweight="bold", zorder=3)
    ax.text(0.109, yr + 0.052, "$s_{novel}=\\min_c$ err at $u{=}0.10$\n"
            "$s_{vocab}=\\min_c$ err at $u{=}0.45$", ha="center", va="center", fontsize=8.4,
            zorder=3, linespacing=1.7)
    ax.text(0.109, yr + 0.020, "over trained classes and\nbasis compositions", ha="center",
            va="center", fontsize=7.6, color="#555", zorder=3, linespacing=1.4)

    oh, ogap = 0.040, 0.0125
    outs = [("in-distribution", "$s_{novel}\\leq t_{novel}$   →   do nothing", FROZEN, "#8A8A8A"),
            ("new ARRANGEMENT", "$s_{vocab}\\leq t_{vocab}$   →   the free route", OURS, GREEN),
            ("new VOCABULARY", "otherwise   →   a kept delta", HOT, RED)]
    for i, (ttl, bod, fc, col) in enumerate(outs):
        oy = yr + hr - oh - i * (oh + ogap)
        panel(ax, 0.216, oy, 0.176, oh, fc, EDGE, 1.0, z=2)
        ax.text(0.304, oy + oh - 0.010, ttl, ha="center", va="top", fontsize=8.8,
                fontweight="bold", zorder=3)
        ax.text(0.304, oy + 0.008, bod, ha="center", va="bottom", fontsize=7.8, zorder=3)
        arrow(ax, (0.186, yr + hr / 2), (0.214, oy + oh / 2), color=col, rad=0.07, lw=1.3)

    panel(ax, 0.424, yr, 0.143, hr, "#F4F4F4", EDGE, 1.1, z=2)
    ax.text(0.4955, yr + hr - 0.020, "what each costs", ha="center", va="top", fontsize=10,
            fontweight="bold", zorder=3)
    ax.text(0.4955, yr + 0.070, "arrangement:\nextend the factor basis\n**1024 bytes**".replace("**", ""),
            ha="center", va="center", fontsize=8.4, zorder=3, linespacing=1.5, color=GREEN)
    ax.text(0.4955, yr + 0.026, "new vocabulary:\none adapter leaf\n551K parameters",
            ha="center", va="center", fontsize=8.4, zorder=3, linespacing=1.5, color=RED)
    ax.text(0.022, yr - 0.020, "the error is ONE-SIDED: 0 vocabulary items were sent down the free "
            "route in 120 queries — cost is over-spent, never under-spent",
            fontsize=8.4, style="italic", color=RED)

    # ══ BAND 4 — the bank, and the step list ═══════════════════════════════
    yb, hb = 0.042, 0.470
    panel(ax, 0.022, yb, 0.545, hb, BANK, "#B9A87E", 1.3)
    ax.text(0.038, yb + hb - 0.022, "STAGE 6 expanded — HierarchicalAdapterBank", fontsize=10.8,
            fontweight="bold")
    ax.text(0.038, yb + hb - 0.045, "the frozen backbone is never written to; a concept IS a path "
            "through the tree", fontsize=8.6, style="italic", color="#5A5A5A")

    # -- the frozen UNet
    ux, uy, uw, uh = 0.040, yb + 0.238, 0.232, 0.168
    panel(ax, ux, uy, uw, uh, FROZEN, EDGE, 1.2, z=2)
    ax.text(ux + uw / 2, uy + uh - 0.018, "FROZEN UNet   $\\varepsilon_\\theta(x_t,t,c)$",
            ha="center", va="top", fontsize=9.8, fontweight="bold", zorder=3)
    ax.text(ux + uw / 2, uy + uh - 0.040, "requires_grad_(False) on every parameter, forever",
            ha="center", va="top", fontsize=7.4, style="italic", color="#555", zorder=3)
    for i, lab in enumerate(["PathLoRA wraps conv / linear", "PathLoRA wraps conv / linear",
                             "⋯", "PathLoRA wraps conv / linear"]):
        ry = uy + uh - 0.058 - i * 0.0215
        panel(ax, ux + 0.014, ry - 0.018, uw - 0.028, 0.018, "white", "#9A9A9A", 0.8, z=3)
        ax.text(ux + uw / 2, ry - 0.009, lab, ha="center", va="center", fontsize=7.3, zorder=4)
    ax.text(ux + uw / 2, uy + 0.014, "`targets` / `exclude` pick which layers — attention-only on a\n"
            "pretrained backbone, where 32 of its 58 Linears are time_emb_proj",
            ha="center", va="center", fontsize=7.1, color="#444", zorder=3, linespacing=1.4)

    # -- the tensor-level operation
    lx, ly, lw_, lh = 0.040, yb + 0.030, 0.232, 0.188
    panel(ax, lx, ly, lw_, lh, "white", "#9A9A9A", 1.0, z=2)
    ax.text(lx + lw_ / 2, ly + lh - 0.020, "inside one wrapped layer", ha="center", va="top",
            fontsize=9.4, fontweight="bold", zorder=3)
    ax.text(lx + lw_ / 2, ly + 0.115,
            "$\\mathrm{out}\\;=\\;W_{frozen}\\,x\\;+\\!\\!\\sum_{n\\,\\in\\,\\mathrm{path}}\\!\\!B_n(A_n x)$",
            ha="center", va="center", fontsize=12.5, zorder=3)
    ax.text(lx + lw_ / 2, ly + 0.074, "$A_n$ kaiming-init,      $B_n=\\mathbf{0}$", ha="center",
            va="center", fontsize=8.8, zorder=3)
    ax.text(lx + lw_ / 2, ly + 0.046, "a fresh node is an EXACT no-op, not a small one",
            ha="center", va="center", fontsize=7.8, style="italic", color=RED, zorder=3)
    ax.text(lx + lw_ / 2, ly + 0.022, "switching concepts = switching ONE list —\n"
            "no weight copying, no re-injection", ha="center", va="center", fontsize=7.6,
            style="italic", color="#555", zorder=3, linespacing=1.4)

    # -- the tree
    tx, tw = 0.292, 0.262
    ax.text(tx + tw / 2, yb + hb - 0.070, "the concept tree   (insert-only)", ha="center",
            fontsize=9.8, fontweight="bold")
    root = (tx + tw / 2, yb + 0.352)
    kids = [(tx + 0.062, yb + 0.262), (tx + 0.131, yb + 0.262), (tx + 0.200, yb + 0.262)]
    leaves = [(tx + 0.032, yb + 0.172), (tx + 0.092, yb + 0.172), (tx + 0.152, yb + 0.172),
              (tx + 0.212, yb + 0.172)]
    edges = [(0, 0), (0, 1), (1, 2), (2, 3)]
    for k in kids:
        ax.plot([root[0], k[0]], [root[1], k[1]], color="#A8A8A8", lw=1.0, zorder=2)
    for ki, li in edges:
        ax.plot([kids[ki][0], leaves[li][0]], [kids[ki][1], leaves[li][1]],
                color="#A8A8A8", lw=1.0, zorder=2)
    ax.plot([root[0], kids[2][0]], [root[1], kids[2][1]], color=GREEN, lw=2.8, zorder=2)
    ax.plot([kids[2][0], leaves[3][0]], [kids[2][1], leaves[3][1]], color=GREEN, lw=2.8, zorder=2)
    for c in [root] + kids[:2] + leaves[:3]:
        ax.add_patch(Circle(c, 0.0135, fc=FROZEN, ec=EDGE, lw=1.1, zorder=3))
    for c in (kids[2], leaves[3]):
        ax.add_patch(Circle(c, 0.0145, fc=OURS, ec=GREEN, lw=2.2, zorder=3))
    ax.text(leaves[3][0] + 0.026, leaves[3][1], "new\nleaf", ha="left", va="center", fontsize=7.4,
            color=GREEN, fontweight="bold", linespacing=1.3)
    ax.text(tx + tw / 2, yb + 0.140, 'state["path"] = [root, node, leaf]', ha="center",
            fontsize=8.6, color=GREEN, fontweight="bold")
    ax.text(tx + tw / 2, yb + 0.113, "grey = .frozen — sealed by an earlier insertion, absent from\n"
            "every future optimizer        green = ._open — the ONLY trainables",
            ha="center", va="center", fontsize=7.4, color="#444", linespacing=1.6)
    panel(ax, tx + 0.010, yb + 0.044, tw - 0.020, 0.046, "white", "#C0A0A0", 1.1, z=2)
    ax.text(tx + tw / 2, yb + 0.078, "params = [p for n in path if n in ._open]", ha="center",
            va="center", fontsize=8.4, color=RED, fontweight="bold", zorder=3)
    ax.text(tx + tw / 2, yb + 0.057, "the preservation guarantee, in one line:\n"
            "ancestors are not in the optimizer, so no gradient can reach them",
            ha="center", va="center", fontsize=7.2, color=RED, style="italic", zorder=3,
            linespacing=1.4)

    # -- the step list
    rx, rw = 0.588, 0.392
    panel(ax, rx, yb, rw, hb, "#FCFCFC", "#B0B0B0", 1.2)
    ax.text(rx + 0.016, yb + hb - 0.022, "Step by step — the operation each stage performs",
            fontsize=10.8, fontweight="bold")

    steps = [
        ("1–4", "COMPOSE   (inherited)", INHERIT,
         "mode-ascent on the frozen score field → submodular top-$K$ → PoE teacher $q_T$.\n"
         "Read it in the ideal-words FACTOR basis: one softmax per factor, 18 params in all,\n"
         "$c_{new}=\\mu+\\sum_f g_f (w_f\\!\\cdot\\!V_f)$.  The teacher says WHERE; the model GENERATES."),
        ("5", "ROUTE", OURS,
         "score $x_q$ under every conditional the bank can already express, at two noise levels.\n"
         "Three outcomes, one-sided error.  This is what decides the update's COST."),
        ("6", "CONSOLIDATE", OURS,
         "_seal() moves ._open → .frozen  ·  tree.insert($\\mu$) → path, leaf  ·  add_node on every\n"
         "wrapped layer ($B{=}0$, so the model is still unchanged)  ·  pin and freeze $c_{new}$  ·\n"
         "Adam over ._open ONLY, at high noise levels (§14)."),
        ("7", "CERTIFY", OURS,
         "fingerprint every earlier concept and assert bit-identity.  1.000 across 3 seeds, two\n"
         "datasets and 11 CL baselines — where every method that PROTECTS shared weights\n"
         "rather than removing them sits at 0.250."),
        ("→", "READ THE BANK", "#F4F4F4",
         "one concept: activate its path, condition on its vector.  Several: compose in SCORE\n"
         "space, $\\varepsilon=\\varepsilon_0+\\sum_i w_i(\\varepsilon_i-\\varepsilon_0)$ — never activate two paths at once, which\n"
         "sums deltas that were each trained to act ALONE (§12e)."),
        ("$K$", "BATCH ADDITION", "#F4F4F4",
         "._open holds $K$ leaves plus their shared node, and ONE optimizer sees all of them,\n"
         "so the ancestor learns from every member.  _seal() runs once for the whole batch, so\n"
         "the guarantee is identical at any $K$: 2.14× faster AND +0.052 ACC at $K{=}12$."),
    ]
    yy = yb + hb - 0.052
    for tag, head, tc, body in steps:
        panel(ax, rx + 0.014, yy - 0.026, 0.032, 0.026, tc, EDGE, 0.9, z=3)
        ax.text(rx + 0.030, yy - 0.013, tag, ha="center", va="center", fontsize=8.8,
                fontweight="bold", zorder=4)
        ax.text(rx + 0.056, yy - 0.002, head, fontsize=9.4, fontweight="bold", va="top")
        ax.text(rx + 0.056, yy - 0.022, body, fontsize=7.7, va="top", linespacing=1.55,
                color="#333333")
        yy -= 0.070

    fig.text(0.5, 0.016, "Every backbone parameter is frozen at construction and never re-enabled.  "
             "A concept is a path of low-rank deltas plus one conditioning vector — and nothing an "
             "earlier concept depends on is ever in an optimizer again.",
             ha="center", fontsize=8.8, style="italic", color="#555555")

    out = os.path.join(FIG, "f0_architecture.png")
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
