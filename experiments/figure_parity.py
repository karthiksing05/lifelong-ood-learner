"""Redraw the native-parity figure from the POOLED seeds, with error bars.

Each n23 run writes its own figure, so `figures/n23_native_parity.png` shows whichever seed finished
last while the table beside it reports the mean over three.  A reader comparing the two would find
bars that do not match the numbers.  This redraws from `results/aggregate.json`, so the figure and
the table are the same measurement.

CPU only; reads JSON and draws.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                              # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")
ARMS = ["real", "native", "ours", "paper"]
LABEL = {"real": "real held-out images", "native": "native (trained on this slot)",
         "ours": "ours (consolidated, no retraining)", "paper": "paper (per-query LoRA)"}
COL = {"real": "#9E9E9E", "native": "#4C78A8", "ours": "#54A24B", "paper": "#E45756"}
KEYS = ["joint", "f1", "recall"]


def get(tbl, arm, k):
    s = tbl.get(arm, {}).get(k)
    return (float(s["mean"]), float(s.get("se", 0.0))) if s else (np.nan, 0.0)


def main():
    agg = json.load(open(os.path.join(RESULTS, "aggregate.json")))
    n23 = agg.get("n23_native_parity")
    if not n23:
        sys.exit("no n23_native_parity in aggregate.json -- run aggregate.py first")
    tbl, par, n = n23["table"], n23["parity"], n23["n_seeds"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(KEYS))
    w = 0.2
    for i, a in enumerate(ARMS):
        m = [get(tbl, a, k)[0] for k in KEYS]
        e = [get(tbl, a, k)[1] for k in KEYS]
        axes[0].bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=2.5, label=LABEL[a], color=COL[a])
    axes[0].set_xticks(x); axes[0].set_xticklabels(KEYS)
    axes[0].set_ylabel("score")
    axes[0].set_title(f"Held-out compositions ({n} seeds)", fontsize=9)
    # legend below the axes: every in-plot corner is occupied here -- `real`/`native` are near 0.9
    # on all three metrics and `recall` fills the lower right
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
                   frameon=False)

    for i, a in enumerate(("ours", "paper")):
        m = [float(par[a][k]["mean"]) for k in KEYS]
        e = [float(par[a][k].get("se", 0.0)) for k in KEYS]
        axes[1].bar(x + (i - 0.5) * 0.35, m, 0.35, yerr=e, capsize=2.5, label=a, color=COL[a])
    axes[1].axhline(1.0, ls="--", color="0.3", lw=1)
    axes[1].text(2.35, 1.02, "native", fontsize=7, color="0.3", ha="right")
    axes[1].set_xticks(x); axes[1].set_xticklabels(KEYS)
    axes[1].set_ylabel("consolidated / native")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Native-parity ratio (1.0 = as good as having trained on it)", fontsize=9)
    axes[1].legend(fontsize=8)
    for k, xi in zip(KEYS, x):
        axes[1].annotate(f"{float(par['ours'][k]['mean']):.2f}", (xi - 0.175, 0.02),
                         fontsize=7, ha="center", color="white", weight="bold")

    fig.suptitle("N23 — a consolidated concept against a backbone that was TRAINED on the slot, "
                 "at zero retraining", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "n23_native_parity.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} from {n} pooled seeds")
    for a in ("ours", "paper"):
        print(f"  {a:6s} " + "  ".join(f"{k} {float(par[a][k]['mean']):.3f}"
                                       f"+/-{float(par[a][k].get('se', 0)):.3f}" for k in KEYS))


if __name__ == "__main__":
    main()
