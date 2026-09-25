"""F33 and F34 -- the two 'main performance' tests that had numbers but no figure.

The goal outline lists five main performance tests. Three were already drawn (incremental strength
f31, OOD strength f3/f26/f29, parameter count f16/f8). Two were measured and left in tables:

    F33  REPRESENTATIONAL STRENGTH (N50, RESULTS 6d) -- are an absorbed concept's representations as
         informative as a trained one's?  Linear probes and k-NN, against a diffusion model trained
         on every composition from the start.  **This is an honest negative and the figure must read
         as one**: `ours` sits on top of `base`, and the oracle is ahead on the factor that separates
         classes most.  Drawing it makes the size of the gap legible in a way the table does not.

    F34  BATCH-SIZE INVARIANCE (N4, RESULTS 16/16b) -- can concepts be absorbed in GROUPS rather than
         one at a time, and does anything degrade when they are?  Plotted against K, with the
         bit-identical preservation flag shown per point, because "faster" is only interesting if
         nothing broke.

Both read recorded results; neither needs a GPU.

    python experiments/make_performance_figures.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures", "initial")

BLUE, GREEN, GREY, RED = "#1F4E79", "#2E7D32", "#888888", "#B03A2E"


def _pool(pattern, key="rows"):
    """-> list of rows pooled over seed files, skipping smoke runs."""
    out = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        if "smoke" in p:
            continue
        try:
            out += json.load(open(p, encoding="utf-8")).get(key, [])
        except Exception:                                            # noqa: BLE001
            continue
    return out


def f33_representation(plt):
    rows = _pool("n50_representation*.json")
    if not rows:
        print("[f33] no n50 results")
        return False
    arms = ["base", "ours", "oracle"]
    metrics = [("lin_digit", "linear\ndigit"), ("lin_fg", "linear\nfg"), ("lin_bg", "linear\nbg"),
               ("lin_joint", "linear\njoint"), ("knn_digit", "k-NN\ndigit"),
               ("knn_fg", "k-NN\nfg"), ("knn_bg", "k-NN\nbg")]
    got = {a: {} for a in arms}
    for a in arms:
        R = [r for r in rows if r.get("arm") == a]
        for k, _ in metrics:
            v = [r[k] for r in R if k in r]
            got[a][k] = (float(np.mean(v)), float(np.std(v) / max(len(v) ** 0.5, 1))) if v else (np.nan, 0)

    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    x = np.arange(len(metrics))
    w = 0.26
    for i, (a, col, lab) in enumerate((("base", GREY, "base — 120 slots, no adaptation"),
                                       ("ours", GREEN, "ours — frozen + 12 consolidated"),
                                       ("oracle", BLUE, "oracle — trained on all 160"))):
        ys = [got[a][k][0] for k, _ in metrics]
        es = [got[a][k][1] for k, _ in metrics]
        ax.bar(x + (i - 1) * w, ys, w, yerr=es, capsize=2, color=col, label=lab, alpha=0.92)
    ax.axhline(0.1, color="black", ls=":", lw=1.0)
    ax.text(len(metrics) - 0.4, 0.105, "chance (digit)", fontsize=7, ha="right", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics], fontsize=8)
    ax.set_ylabel("readout accuracy", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.set_title("Representational strength — an HONEST NEGATIVE (N50, RESULTS §6d).\n"
                 "`ours` sits on top of `base`: absorbing a concept does not measurably improve the "
                 "representation over not adapting at all,\nand the gap to joint training stays open "
                 "on digit. `ours` gets 12 extra conditioning columns and they buy nothing.",
                 fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG, "09_representation.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[f33] wrote {out}")
    return True


def f34_batch(plt):
    """Batch invariance from the FAIR files only, and only the per-seed ones.

    Two traps here, both already paid for once in this repository:

      * `n4_batch_K*.json` is the SUPERSEDED comparison -- it ran batch and sequential under
        different discovery configs, so K=1 was not a no-op when it must be, and it shows a "cost of
        batching" that disappeared when the configs were matched (docs/README §177).  Only
        `n4_batch_fair*` is drawn.
      * the unsuffixed file duplicates one of the `__seed*` files, so pooling both double-counts that
        seed.  Only `__seed*` is read.
    """
    per = {}
    for p_ in sorted(glob.glob(os.path.join(RES, "n4_batch_fair*__seed*.json"))):
        try:
            d = json.load(open(p_, encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        K = d.get("config", {}).get("batch_size")
        if not K:
            continue
        for r in d.get("rows", []):
            per.setdefault((int(K), r.get("mode")), []).append(r)
    Ks = sorted({k for k, _ in per})
    if not Ks:
        print("[f34] no fair per-seed n4 results")
        return False

    def agg(K, mode, key):
        R = per.get((K, mode), [])
        v = [r[key] for r in R if key in r and r[key] == r[key]]
        return (float(np.mean(v)), float(np.std(v) / max(len(v) ** 0.5, 1)), len(v)) if v else (np.nan, 0, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for mode, col, lab in (("batch", GREEN, "absorbed in one batch of K"),
                           ("sequential", GREY, "absorbed one at a time (control)")):
        m = [agg(K, mode, "ACC") for K in Ks]
        axes[0].errorbar(Ks, [a for a, _, _ in m], yerr=[e for _, e, _ in m], marker="o", ms=5,
                         lw=2.0, capsize=3, color=col, label=lab)
    for K in Ks:
        ok = all(r.get("retention_exact", 0) >= 1.0
                 for mo in ("batch", "sequential") for r in per.get((K, mo), []))
        a = agg(K, "batch", "ACC")[0]
        axes[0].annotate("✓" if ok else "✗", (K, a), textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=11, color=GREEN if ok else RED)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(Ks); axes[0].set_xticklabels([str(k) for k in Ks])
    axes[0].set_xlabel("concepts absorbed per update (K)", fontsize=8)
    axes[0].set_ylabel("ACC after the stream", fontsize=8)
    axes[0].set_title("Accuracy does not depend on K — ✓ = bit-identical preservation at every point",
                      fontsize=9)
    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    axes[0].grid(alpha=0.25, lw=0.5)

    # PAIRED sequential: the control from the SAME run at the SAME K, not sequential at K=1.
    # Same seeds, same machine, same discovery config -- so the ratio isolates batching. Referencing
    # K=1 instead mixes in run-to-run wall-clock drift and gave 1.98x where the paired figure the
    # docs quote is 2.15x.
    sp = [agg(K, "sequential", "wall_seconds")[0] / agg(K, "batch", "wall_seconds")[0] for K in Ks]
    axes[1].plot(Ks, sp, marker="o", ms=5, lw=2.0, color=BLUE)
    axes[1].axhline(1.0, color="black", ls=":", lw=1.0)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(Ks); axes[1].set_xticklabels([str(k) for k in Ks])
    axes[1].set_xlabel("concepts absorbed per update (K)", fontsize=8)
    axes[1].set_ylabel("speedup vs sequential", fontsize=8)
    axes[1].set_title(f"Wall-clock: {sp[-1]:.2f}× at K={Ks[-1]} — the reason to batch",
                      fontsize=9)
    axes[1].grid(alpha=0.25, lw=0.5)

    n = agg(Ks[-1], "batch", "ACC")[2]
    fig.suptitle("Batch-size invariance (N4 fair comparison, "
                 f"{n} seeds, RESULTS §16/§16b): absorbing concepts in GROUPS costs nothing in "
                 "accuracy and nothing in preservation.\nThis is invariance in the number of "
                 "CONCEPTS per update. Invariance in the number of INSTANCES per concept is a "
                 "different axis — f32.", fontsize=8.6)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIG, "05_batch_invariance.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[f34] wrote {out}")
    return True


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [f33_representation(plt), f34_batch(plt)]
    return 0 if any(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
