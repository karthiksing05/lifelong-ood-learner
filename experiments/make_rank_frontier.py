"""F12 -- the quality-vs-parameters frontier, and whether shrinking the adapter closes our one loss.

`independent` beats us on the retention table at FEWER parameters (ACC 0.863 +/- 0.015 at 7.16M
against our 0.839 +/- 0.025 at 8.82M), and this repository has never tried making the adapter
smaller.  Rank was set to 8 once and never revisited, so the comparison has been ours-at-an-arbitrary
-capacity against a baseline at its own natural size.

This does NOT run a new experiment.  `n5_classinc.py` already takes `--rank`, so the sweep is that
same tested code at r in {1, 2, 4, 8, 16}.

**`independent` is swept too, and has to be.**  It is this same bank with flat placement, so it has
the identical rank knob, and an earlier version of this figure swept only ours -- which would have
compared our tuned capacity against their arbitrary one and reported a win we had bought by tuning a
knob for ourselves alone.  The other nine baselines are not adapter methods and keep their published
numbers from `results/aggregate.json`.

Reads `results/n5_classinc_rank<r>__seed*.json` (one job per rank per seed) and draws:

  * ACC against total parameters, ours and `independent` both as CURVES and the remaining baselines
    as fixed points, so "are we on the frontier" is answered by looking rather than asserted;
  * bit-identical retention against rank, to check that shrinking capacity does not quietly cost the
    guarantee -- it should be exactly 1.000 at every rank, since preservation is structural and has
    nothing to do with how big the delta is.

    python experiments/make_rank_frontier.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
RESULTS = os.path.join(HERE, "results")
RANKS = (1, 2, 4, 8, 16)
# baselines to place as fixed points, and how to label them
BASELINES = ("independent", "c-lora", "mas", "si", "ttt", "der++", "gen-distill", "ewc",
             "finetune", "replay")


def load_rank(r, arm="ours"):
    """-> pooled ACC / params / bit-identity for `arm` at rank r, over whatever seeds exist.

    `independent` is swept too, and must be: it is this same bank with flat placement, so it has the
    IDENTICAL rank knob.  Comparing our swept rank against its fixed rank-8 point would be the
    "baseline at a configuration nobody chose for it" unfairness this repository forbids elsewhere.
    """
    fs = sorted(glob.glob(os.path.join(RESULTS, f"n5_classinc_rank{r}__seed*.json")))
    if not fs:
        f = os.path.join(RESULTS, f"n5_classinc_rank{r}.json")
        fs = [f] if os.path.exists(f) else []
    if not fs:
        return None
    acc, par, ex = [], [], []
    for f in fs:
        rows = json.load(open(f)).get("rows", [])
        row = next((x for x in rows if x["arm"] == arm), None)
        if row is None:
            continue
        acc.append(row.get("ACC"))
        par.append(row.get("params"))
        ex.append(row.get("retention_exact"))
    if not acc:
        return None
    acc = [a for a in acc if a is not None]
    return dict(rank=r, arm=arm, acc=float(np.mean(acc)),
                sem=float(np.std(acc, ddof=1) / np.sqrt(len(acc))) if len(acc) > 1 else 0.0,
                params=float(np.mean(par)), exact=float(np.mean(ex)), n_seeds=len(acc))


def main():
    pts = [p for p in (load_rank(r) for r in RANKS) if p]
    ind = [p for p in (load_rank(r, "independent") for r in RANKS) if p]
    if not pts:
        print("[f12] no rank-sweep results yet -- run n5_classinc.py --arms ours --rank R "
              "--tag _rankR for R in", RANKS)
        return 1
    A = json.load(open(os.path.join(RESULTS, "aggregate.json")))
    base = A.get("n5_classinc", {}).get("table", {})

    def cell(arm, k):
        c = base.get(arm, {}).get(k)
        return None if not c else c.get("mean")

    print(f"[f12] ours across rank ({pts[0]['n_seeds']} seed(s) each)")
    print(f"    {'rank':>6s}{'params':>12s}{'ACC':>16s}{'bit-identical':>15s}")
    for p in pts:
        print(f"    {p['rank']:6d}{p['params'] / 1e6:11.2f}M{p['acc']:10.3f}+/-{p['sem']:.3f}"
              f"{p['exact']:15.3f}")
    if ind:
        print(f"\n[f12] independent across rank")
        print(f"    {'rank':>6s}{'params':>12s}{'ACC':>16s}")
        for p in ind:
            print(f"    {p['rank']:6d}{p['params'] / 1e6:11.2f}M{p['acc']:10.3f}+/-{p['sem']:.3f}")
        # the fair question: at MATCHED rank, who wins, and does the answer depend on rank?
        print(f"\n[f12] head to head at matched rank")
        print(f"    {'rank':>6s}{'ours ACC':>12s}{'indep ACC':>12s}{'delta':>9s}"
              f"{'ours par':>11s}{'indep par':>11s}")
        for a in pts:
            b = next((x for x in ind if x["rank"] == a["rank"]), None)
            if not b:
                continue
            print(f"    {a['rank']:6d}{a['acc']:12.3f}{b['acc']:12.3f}{a['acc'] - b['acc']:+9.3f}"
                  f"{a['params'] / 1e6:10.2f}M{b['params'] / 1e6:10.2f}M")
        wins = [a["rank"] for a in pts
                if (b := next((x for x in ind if x["rank"] == a["rank"]), None))
                and a["acc"] >= b["acc"]]
        print(f"\n[f12] ranks where ours >= independent at MATCHED rank: {wins or 'NONE'}")
    if any(p["exact"] != 1.0 for p in pts):
        print("[f12] WARNING: bit-identity is not 1.000 at every rank -- preservation is supposed "
              "to be independent of capacity, so this would be a real defect")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    ax = axes[0]
    x = [p["params"] / 1e6 for p in pts]
    y = [p["acc"] for p in pts]
    e = [p["sem"] for p in pts]
    ax.errorbar(x, y, yerr=e, color="#1B7F4B", marker="o", ms=6, lw=2, capsize=3,
                label="ours, by rank", zorder=3)
    for p in pts:
        ax.annotate(f"r={p['rank']}", (p["params"] / 1e6, p["acc"]), textcoords="offset points",
                    xytext=(6, -11), fontsize=8, color="#1B7F4B")
    if ind:
        ax.errorbar([p["params"] / 1e6 for p in ind], [p["acc"] for p in ind],
                    yerr=[p["sem"] for p in ind], color="#B03A2E", marker="s", ms=5, lw=2,
                    capsize=3, label="independent, by rank", zorder=3)
    for arm in [b for b in BASELINES if b != "independent"]:
        a, q = cell(arm, "ACC"), cell(arm, "params")
        if a is None or q is None:
            continue
        star = False
        ax.scatter([q / 1e6], [a], s=64 if star else 34,
                   marker="*" if star else "s",
                   color="#B03A2E" if star else "#7C7C7C", zorder=4 if star else 2)
        ax.annotate(arm, (q / 1e6, a), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#B03A2E" if star else "#6A6A6A",
                    fontweight="bold" if star else "normal")
    ax.set_xlabel("total parameters (M)")
    ax.set_ylabel("ACC")
    ax.set_title("Quality vs parameters: is our curve on the frontier?", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    ax = axes[1]
    ax.plot([p["rank"] for p in pts], [p["exact"] for p in pts], color="#1B7F4B",
            marker="o", ms=6, lw=2)
    ax.axhline(1.0, color="#8A8A8A", ls=(0, (4, 3)), lw=1)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(RANKS))
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("fraction bit-identical")
    ax.set_title("Preservation is structural: capacity must not affect it", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("Shrinking the adapter: does it close the parameter loss to `independent`?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "extra", "f12_rank_frontier.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[f12] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
