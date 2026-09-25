"""The class-table learning-rate ladder: figures/f_lr_ladder.png (from N57 x5 / N58 x5).

`docs/VLA_PATH.md` lesson 1 says a strong pretrained backbone starves the conditioning pathway, and
rested on a single setting -- the class table's own learning rate at 50x, which was a first guess
that worked.  This is the curve that turns it into a stated rule, and it says three things the
single setting could not:

  * 1x genuinely fails (1.8x the unconditional floor, below the composing threshold) while every
    rung from 5x up composes at 10-14x.  The pathway really is starved by default.
  * 50x is not a tuned optimum, it is the PLATEAU -- 50x and 100x are identical at 0.542.  The
    choice was robust rather than lucky, which is what matters before porting it to a policy.
  * the eps-loss gap is ~0.0001 at EVERY rung, including the one that does not compose.  Lessons 1
    and 3 are the same fact seen twice: the training loss is nearly independent of whether the
    conditioning was learned at all.

And one the ladder was not built to find: **basis R^2 does not track ADDRESS.**  R^2 is HIGHEST at
5x (0.637), where ADDRESS is 0.323 -- its second-worst rung -- and lowest at 100x (0.502), where
ADDRESS is at its best.  The additive fit quality of the embedding table is not what decides whether
composition works, so it must not be used as the health metric for a new backbone.

TWO CAVEATS, because this is one seed per rung.  ADDRESS carries an SE of ~0.11 over 12 held-out
cases, so every rung from 5x up is within about one SE of every other: what is established is
**1x fails and >=5x works**, NOT an ordering among the rungs that succeed.  And each rung is a
separate training run, so the R^2 trend includes run-to-run backbone variance, which 5p measured at
up to 0.07 F1 between two runs of the SAME command.  The R^2/ADDRESS dissociation is safe to state
as "R^2 does not predict ADDRESS" and not as a monotone relationship in either direction.

    python experiments/make_lr_ladder_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "experiments"))
from common import RESULTS                                             # noqa: E402

MULTS = (1, 5, 20, 50, 100)
FIGS = os.path.join(HERE, "figures", "initial")


def main():
    addr, floor, se, r2, gap, have = [], [], [], [], [], []
    for m in MULTS:
        p = os.path.join(RESULTS, f"n58_shapes3d_lr{m}__seed0.json")
        if not os.path.exists(p):
            print(f"[ladder] missing {p} -- skipping {m}x")
            continue
        d = json.load(open(p))
        rows = {r["arm"]: r for r in d.get("summary", [])}     # summary is a LIST of arm dicts
        a, u = rows.get("ours (ADDRESS)"), rows.get("unconditional")
        if a is None or u is None:
            print(f"[ladder] {m}x: arms missing ({list(rows)}) -- skipping")
            continue
        addr.append(float(a["joint"]))
        floor.append(float(u["joint"]))
        se.append(float(a.get("se", 0.0)))
        r2.append(float(d.get("basis_r2", np.nan)))
        gap.append(float(d.get("gap", np.nan)))
        have.append(m)
    if not have:
        print("[ladder] no results; run n58 with --tag _lr<m> first")
        return 1

    x = np.arange(len(have))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax[0].errorbar(x, addr, yerr=se, fmt="o-", lw=2, capsize=3, color="#1f77b4",
                   label="ADDRESS (held-out joint), +/-1 SE over 12 cases")
    ax[0].plot(x, floor, "s--", lw=1.5, color="#888", label="unconditional floor")
    ax[0].axhspan(0, max(floor) * 2.5, color="#d62728", alpha=.07)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{m}x" for m in have])
    ax[0].set_xlabel("class-table learning-rate multiplier")
    ax[0].set_ylabel("joint accuracy on held-out combinations")
    ax[0].set_title("1x is starved; 5x already composes; 50x is the plateau\n"
                    "(rungs above 5x are within 1 SE of each other)")
    ax[0].legend(fontsize=8)
    for i, (a, f) in enumerate(zip(addr, floor)):
        ax[0].annotate(f"{a / max(f, 1e-9):.1f}x", (i, a), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8)

    ax[1].plot(x, r2, "o-", lw=2, color="#2ca02c")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{m}x" for m in have])
    ax[1].set_xlabel("class-table learning-rate multiplier")
    ax[1].set_ylabel("ideal-words basis $R^2$")
    ax[1].set_title("$R^2$ does not track ADDRESS\n(highest at 5x, where ADDRESS is far from best)")

    ax[2].plot(x, np.array(gap) * 1e3, "o-", lw=2, color="#9467bd")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{m}x" for m in have])
    ax[2].set_xlabel("class-table learning-rate multiplier")
    ax[2].set_ylabel(r"eps-loss gap  ($\times 10^{-3}$)")
    ax[2].set_title("the training loss barely moves\neven where composition goes 1.8x -> 11.6x")

    fig.suptitle("Shapes3D: a pretrained backbone starves the conditioning pathway "
                 "(N57/N58, one seed per rung)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIGS, exist_ok=True)
    o = os.path.join(FIGS, "f_lr_ladder.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[ladder] wrote {o}")
    for m, a, f, q in zip(have, addr, floor, r2):
        print(f"  {m:>4}x  ADDRESS {a:.3f}  floor {f:.3f}  ({a / max(f, 1e-9):>4.1f}x)  R2 {q:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
