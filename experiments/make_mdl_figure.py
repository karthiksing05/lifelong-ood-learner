"""F32 -- how much does an OOD query warrant addition?  The answer is a curve, not a threshold.

Every router in this repository before `mdl_route` answered that question with a number: a novelty
statistic and two calibrated cut-points.  RESULTS 6c-bis records five label-free attempts to place
the second cut-point and why all five closed.  This figure draws the reason they had to.

Under the description-length rule the tier a query warrants is not a property of the query.  It is a
property of the query AND how many examples of it have arrived, because the fit term scales with N
and the storage cost does not:

    L(tier) = C_tier  +  N * (bits to code one example under that tier)

A threshold is a function of the query alone, so it is answering an underdetermined question -- which
is a better explanation of 6c-bis's five failures than any of the five individual causes.

    LEFT   THE PROMOTION CURVE.  For each population, the fraction routed to each tier against how
           many examples of the concept the system has seen.  The SAME concepts at every N (the
           diagnostic draws them once), so movement is evidence accumulating, not the sample
           changing.  A lifelong learner should climb the ladder here, not jump.

    RIGHT  N*, THE EXCHANGE RATE.  Per query, how many examples it would take before each tier pays
           for itself.  This is the calibration-invariant view: N and the bits-per-residual constant
           enter the rule only as a product, so mis-estimating the constant rescales every N* by the
           same factor and cannot reorder them.

Reads `results/diag_mdl_router.json` (written by `tests/diag_mdl_router.py`).

    python experiments/make_mdl_figure.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

ROUTES = ["in-distribution", "new-combination", "new-vocabulary"]
COL = {"in-distribution": "#888888", "new-combination": "#1F4E79", "new-vocabulary": "#B03A2E"}


def main():
    path = os.path.join(RES, "diag_mdl_router.json")
    if not os.path.exists(path):
        print(f"[f32] {path} not found -- run tests/diag_mdl_router.py first")
        return 1
    d = json.load(open(path, encoding="utf-8"))
    rows = list(d["rows"])
    # The deep run (N=32,64) is a SEPARATE file because those batch sizes cost ~4x.  Both runs use
    # seed 0 and consume the rng identically before drawing concepts, so the deep run's slots are a
    # prefix of the shallow run's -- merging is legitimate, but only over the slots they SHARE.
    # Restricting to the intersection is what keeps "the same concepts at every N" true, which is the
    # property the promotion curve depends on; pooling all slots would let the curve move because the
    # population changed rather than because the evidence did (the same mistake f31 had to be fixed
    # for).
    deep = os.path.join(RES, "diag_mdl_router_deep.json")
    if os.path.exists(deep):
        drows = json.load(open(deep, encoding="utf-8"))["rows"]
        common = {r["slot"] for r in rows} & {r["slot"] for r in drows}
        rows = [r for r in rows if r["slot"] in common]
        have = {(r["slot"], r["N"]) for r in rows}
        rows += [r for r in drows if r["slot"] in common and (r["slot"], r["N"]) not in have]
        print(f"[f32] merged {os.path.basename(deep)} over {len(common)} shared concepts")
    Ns = sorted({r["N"] for r in rows})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # ── LEFT: promotion curve ────────────────────────────────────────────────────────────
    # mean position on the ladder (0 = store nothing, 1 = ADDRESS, 2 = EXTEND), per population.
    rank = {r: i for i, r in enumerate(ROUTES)}
    for truth in ROUTES:
        ys, es = [], []
        for N in Ns:
            v = [rank[r["mdl"]] for r in rows if r["N"] == N and r["truth"] == truth]
            ys.append(np.mean(v) if v else np.nan)
            es.append((np.std(v) / max(len(v) ** 0.5, 1)) if v else 0.0)
        axes[0].errorbar(Ns, ys, yerr=es, marker="o", ms=4, lw=2.0, capsize=2,
                         color=COL[truth], label=f"truth: {truth}")
        # where the ORACLE would sit for this population, as a flat reference
        axes[0].axhline(rank[truth], color=COL[truth], ls=":", lw=1.0, alpha=0.5)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(Ns)
    axes[0].set_xticklabels([str(n) for n in Ns])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["store nothing\n(0 B)", "ADDRESS\n(72 B)", "EXTEND BASIS\n(1 KB)"],
                            fontsize=7)
    axes[0].set_xlabel("examples of the concept seen (N)", fontsize=8)
    axes[0].set_title("Promotion: the same concepts climb the ladder as evidence accumulates\n"
                      "(dotted line = the tier that concept actually needs)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.25, lw=0.5)

    # ── RIGHT: N* distributions ──────────────────────────────────────────────────────────
    n1 = [r for r in rows if r["N"] == Ns[0]]
    CAP = 1000.0                                  # plotting cap; inf and beyond pile up on the right
    for k, (key, name) in enumerate((("n_star_addr", "ADDRESS (72 B)"),
                                     ("n_star_ext", "EXTEND (1 KB)"))):
        for j, truth in enumerate(ROUTES):
            v = np.array([r[key] for r in n1 if r["truth"] == truth], float)
            n_inf = int(np.sum(~np.isfinite(v)))
            v = np.clip(np.nan_to_num(v, posinf=CAP), 1, CAP)
            y = k * 4 + j
            axes[1].scatter(v, np.full(len(v), y) + np.random.default_rng(j).normal(0, .07, len(v)),
                            s=18, alpha=0.75, color=COL[truth],
                            label=truth if k == 0 else None)
            axes[1].scatter([np.median(v)], [y], marker="|", s=420, lw=2.2, color="black", zorder=5)
            axes[1].text(CAP * 1.25, y, f"{name.split()[0]}  {truth}"
                         + (f"   ({n_inf}/{len(v)} never)" if n_inf else ""),
                         fontsize=6.5, va="center", color=COL[truth])
    axes[1].set_xscale("log")
    axes[1].set_xlim(1, CAP * 60)
    axes[1].set_yticks([])
    axes[1].set_ylim(-1.0, 7.2)
    axes[1].axvline(1, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("N* -- examples needed before this tier pays for its own bits "
                       "(log; capped at 1000)", fontsize=8)
    axes[1].set_title("The exchange rate, per query.  Invariant to the codelength constant:\n"
                      "N and it enter the rule only as a product, so an error rescales all of these",
                      fontsize=9)
    axes[1].legend(fontsize=7, loc="lower left", framealpha=0.95)
    axes[1].grid(alpha=0.25, lw=0.5, axis="x")

    a = d.get("n_star_auroc")
    fig.suptitle(
        "Routing by description length instead of by a calibrated novelty threshold "
        f"(ColorMNIST, bg=3 withheld entirely).\nSeparation of new-vocabulary from new-combination "
        f"on N*: AUROC {a:.3f}" if a is not None else "", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(FIG, "f32_mdl_router.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[f32] wrote {out}  ({len(rows)} query-decisions over N in {Ns})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
