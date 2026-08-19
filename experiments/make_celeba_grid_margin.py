"""The CelebA-grid composition margin, per held-out cell, across every backbone that has one.

The single-backbone table answers "does ADDRESS beat retrieval?" with three cells and one training
run, which is not enough to rest a claim on -- and the honest n is the number of DISTINCT held-out
cells, not the number of (cell, sampling seed) pairs, since three seeds over the same three cells
replicate the sampler rather than the claim.

So this draws every `n73_celeba_grid_compose*` result it can find, one bar group per (backbone,
cell), with:

  * the PROBE CEILING as a horizontal band, because raw joint is uninterpretable without it -- the
    factor probes read REAL images at ~0.41-0.57 depending on the seed, not at 1.0;
  * CHANCE as a second line;
  * the paired ADDRESS - nearest-class margin per cell in the right-hand panel, which is the
    quantity the "donation, not retrieval" claim actually rests on.

Backbones differ in how many cells they hold out -- more held out means fewer seen, which is a real
trade-off and is why they are shown separately rather than pooled into one bar.

    python experiments/make_celeba_grid_margin.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                             # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "initial")

ARMS = ("address", "nearest-class", "unconditional")
COL = {"address": "#1f77b4", "nearest-class": "#ff7f0e", "unconditional": "#999999"}


def load_all():
    """-> {backbone_tag: [run, ...]}, pooling the sampling seeds of each backbone."""
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "n73_celeba_grid_compose*.json"))):
        base = os.path.basename(p)[:-5]
        if "__seed" not in base:
            continue                              # the untagged copy duplicates a __seed file
        tag = base.split("__seed")[0].replace("n73_celeba_grid_compose", "") or "_h3"
        with open(p) as fh:
            out.setdefault(tag, []).append(json.load(fh))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_all()
    if not runs:
        print("[margin] no n73 results found")
        return 0

    # (tag, cell) -> per-arm mean over sampling seeds, and the per-cell paired margin
    cells, marg, blind = [], [], []
    for tag in sorted(runs):
        rs = runs[tag]
        for c in sorted({r["cell"] for run in rs for r in run["rows"]}):
            vals = {a: [r["joint"] for run in rs for r in run["rows"]
                        if r["arm"] == a and r["cell"] == c] for a in ARMS + ("real-gate",)}
            if not vals["address"]:
                continue
            # DROP A BLIND CELL rather than draw it. Where REAL images of the cell score at
            # chance, that run's probes cannot read that combination and every arm reads ~0 there;
            # plotting it as a three-way tie would be drawing the instrument, not the model.
            ch = 1.0 / (rs[0].get("n_seen") or (12 - len(rs[0]["held"])))
            if vals["real-gate"] and float(np.mean(vals["real-gate"])) <= ch + 0.02:
                blind.append((tag, c))
                continue
            ceil = float(np.mean([run["ceiling"] for run in rs]))
            # Chance is 1/n_seen and n_seen DIFFERS by backbone -- holding out five cells leaves
            # seven, not nine. Every backbone here shares the same 12-cell grid, so the seen count
            # follows from the holdout.
            n_seen = rs[0].get("n_seen") or (12 - len(rs[0]["held"]))
            cells.append(dict(tag=tag, cell=c, ceil=ceil,
                              target=[r["target"] for run in rs for r in run["rows"]
                                      if r["cell"] == c][0],
                              n_seen=n_seen,
                              **{a: (float(np.mean(vals[a])), float(np.std(vals[a])))
                                 for a in ARMS}))
            paired = [ad - nc for ad, nc in zip(vals["address"], vals["nearest-class"])]
            marg.append(float(np.mean(paired)))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(max(11, 0.85 * len(cells) + 6), 4.6),
                                 gridspec_kw=dict(width_ratios=[2.3, 1]))
    x = np.arange(len(cells))
    w = 0.26
    for i, a in enumerate(ARMS):
        ax.bar(x + (i - 1) * w, [c[a][0] for c in cells], w, yerr=[c[a][1] for c in cells],
               label=a, color=COL[a], capsize=2, error_kw=dict(lw=0.8))
    # The ceiling is per-backbone, so it is drawn as a step rather than one line.
    ax.step(np.concatenate([x - 0.5, [x[-1] + 0.5]]),
            [c["ceil"] for c in cells] + [cells[-1]["ceil"]], where="post",
            color="#2ca02c", lw=1.4, ls="--", label="probe ceiling (REAL images)")
    ax.step(np.concatenate([x - 0.5, [x[-1] + 0.5]]),
            [1.0 / c["n_seen"] for c in cells] + [1.0 / cells[-1]["n_seen"]], where="post",
            color="#d62728", lw=1.0, ls=":", label="chance (1 / seen cells)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['tag'].lstrip('_')}\ncell {c['cell']}\n{c['n_seen']} seen"
                        for c in cells], fontsize=6.5)
    ax.set_ylabel("joint factor accuracy")
    ax.set_title("Held-out cell, by backbone: composition vs the model's own best existing class",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # THE HEADLINE COMES FROM N75, NOT FROM THIS FILE.  N75 gates per (cell, seed); averaging the
    # gate over seeds first -- which the left panel does, because it draws one bar group per cell --
    # keeps a different subset and would print a different margin beside the same bars.  One
    # artifact owns the number; the figure displays it.
    pooled = None
    pp = os.path.join(RESULTS, "n75_grid_pooled.json")
    if os.path.exists(pp):
        with open(pp) as fh:
            pooled = json.load(fh)

    bx.axhline(0, color="k", lw=1.0)
    bx.bar(x, marg, 0.62, color=["#1f77b4" if m > 0 else "#d62728" for m in marg])
    mm = float(np.mean(marg))
    se = float(np.std(marg, ddof=1) / np.sqrt(len(marg))) if len(marg) > 1 else 0.0
    bx.axhline(mm, color="#1f77b4", lw=1.2, ls="--")
    bx.set_xticks(x)
    bx.set_xticklabels([str(c["cell"]) for c in cells], fontsize=7)
    bx.set_xlabel("held-out cell")
    bx.set_ylabel("ADDRESS - nearest-class")
    if pooled:
        r = pooled["margin_vs_retrieval"]
        u = pooled["margin_vs_uncond"]
        bx.set_title(f"POOLED over {pooled['n_readable']} readable of {pooled['n_total']} "
                     f"measurements (N75)\n"
                     f"vs retrieval {r['mean']:+.3f} +/- {r['se']:.3f}, won {r['won']}/{r['n']}"
                     f"   |   vs floor {u['mean']:+.3f} +/- {u['se']:.3f}, won {u['won']}/{u['n']}",
                     fontsize=8)
    else:
        bx.set_title(f"paired margin: {mm:+.3f} +/- {se:.3f} over {len(marg)} cells\n"
                     f"won on {sum(1 for m in marg if m > 0)}/{len(marg)}", fontsize=9)
    bx.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "CelebA grid, DENSITY holdout: every cell shown has ZERO training images.\n"
        "`nearest-class` is the adversarial control -- the backbone's own closest existing "
        "embedding. ADDRESS clears the UNCONDITIONAL floor decisively and does NOT clear "
        "RETRIEVAL (right panel): on ColorMNIST nearest-class scores 0.000, here it scores 0.273.\n"
        "Backbones hold out different numbers of cells (more held out = fewer seen), so they are "
        "shown separately rather than pooled.\n"
        + (f"{len(blind)} cell(s) dropped: REAL images of them score at chance under that run's "
           "own probes, so no arm can be judged there." if blind else ""), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(FIG, exist_ok=True)
    o = os.path.join(FIG, "f_celeba_grid_margin.png")
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"[margin] {len(cells)} (backbone, cell) measurements over {len(runs)} backbones")
    if blind:
        print(f"[margin] {len(blind)} dropped as unreadable (real images at chance): {blind}")
    print(f"[margin] paired margin {mm:+.3f} +/- {se:.3f}, won on "
          f"{sum(1 for m in marg if m > 0)}/{len(marg)}")
    print(f"[margin] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
