"""Aggregate results across seeds into mean +/- standard error, and render the tables.

Every experiment writes `results/<name>.json` and takes `--seed`.  Running the same experiment under
several seeds with `--tag` writes `results/<name>__seed<k>.json`; this script collects those and
prints (and saves) the pooled table.

    # one job per seed, then:
    python experiments/aggregate.py e2_composition e3_lifelong n3_lifelong

Single-seed numbers are reported as-is with the count shown, so a table never silently looks like a
3-seed result when it is not -- which is the failure mode this script exists to prevent.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def load_seeds(name):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, f"{name}__seed*.json"))):
        with open(p) as fh:
            out.append((p, json.load(fh)))
    base = os.path.join(RESULTS, f"{name}.json")
    if not out and os.path.exists(base):
        with open(base) as fh:
            out.append((base, json.load(fh)))
    return out


def stat(vals):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return None
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                n=int(len(v)))


def fmt(s, prec=3):
    if s is None:
        return "  --  "
    return f"{s['mean']:.{prec}f}" + (f"±{s['se']:.{prec}f}" if s["n"] > 1 else "")


def agg_e2(runs):
    """E2 / N2: method x metric summary table."""
    methods, keys = [], ("joint", "digit", "fg", "bg", "novelty", "diversity", "gen_rmse")
    for _, r in runs:
        methods += [m for m in r.get("summary", {}) if m not in methods]
    table = {m: {k: stat([r["summary"][m][k] for _, r in runs
                          if m in r.get("summary", {}) and k in r["summary"][m]])
                 for k in keys} for m in methods}
    ref = stat([r["summary"][methods[0]].get("ref_spread") for _, r in runs if methods])
    print(f"    {'method':12s}" + "".join(f"{k:>11s}" for k in keys))
    for m, row in table.items():
        print(f"    {m:12s}" + "".join(f"{fmt(row[k]):>11s}" for k in keys))
    if ref:
        print(f"    (reference set internal spread {fmt(ref)})")
    return dict(table=table, ref_spread=ref, n_seeds=len(runs))


def agg_e3(runs):
    """E3: per-strategy drift at the final step, plus the worst drift seen along the way."""
    strategies = sorted({s for _, r in runs for s in r})
    print(f"    {'strategy':10s}{'final drift ll':>16s}{'worst drift ll':>16s}"
          f"{'final drift R2':>16s}{'valid rate':>13s}")
    out = {}
    for st in strategies:
        fin_ll, worst_ll, fin_r2, valid = [], [], [], []
        for _, r in runs:
            steps = r.get(st, [])
            d = [s.get("drift_loglik") for s in steps if s.get("drift_loglik") is not None]
            if not d:
                continue
            fin_ll.append(d[-1]); worst_ll.append(min(d))
            r2 = [s.get("drift_r2") for s in steps if s.get("drift_r2") is not None]
            fin_r2.append(r2[-1] if r2 else None)
            v = [s.get("valid_rate") for s in steps if s.get("valid_rate") is not None]
            valid.append(v[-1] if v else None)
        out[st] = dict(final_drift_loglik=stat(fin_ll), worst_drift_loglik=stat(worst_ll),
                       final_drift_r2=stat(fin_r2), final_valid=stat(valid))
        print(f"    {st:10s}{fmt(out[st]['final_drift_loglik']):>16s}"
              f"{fmt(out[st]['worst_drift_loglik']):>16s}"
              f"{fmt(out[st]['final_drift_r2'], 5):>16s}{fmt(out[st]['final_valid']):>13s}")
    return dict(table=out, n_seeds=len(runs))


def agg_n3(runs):
    """N3: per-arm retention / plasticity / transfer."""
    arms = sorted({a for _, r in runs for a in r})
    print(f"    {'arm':10s}{'ret_exact':>12s}{'ret_acc':>12s}{'new_acc':>12s}"
          f"{'base_drift':>13s}{'transfer':>12s}")
    out = {}
    for a in arms:
        re_, ra, na, bd, tr = [], [], [], [], []
        for _, r in runs:
            steps = r.get(a, {}).get("steps", [])
            if not steps:
                continue
            re_.append(steps[-1].get("retention_exact"))
            ra.append(steps[-1].get("retention_acc"))
            na.append(float(np.mean([s["new_acc"]["joint"] for s in steps if "new_acc" in s])))
            bd.append(steps[-1].get("base_drift"))
            t = r[a].get("transfer")
            tr.append(t["late_final"] - t["early_final"] if t else None)
        out[a] = dict(retention_exact=stat(re_), retention_acc=stat(ra), new_acc=stat(na),
                      base_drift=stat(bd), transfer=stat(tr))
        print(f"    {a:10s}{fmt(out[a]['retention_exact']):>12s}{fmt(out[a]['retention_acc']):>12s}"
              f"{fmt(out[a]['new_acc']):>12s}{fmt(out[a]['base_drift'], 5):>13s}"
              f"{fmt(out[a]['transfer'], 4):>12s}")
    return dict(table=out, n_seeds=len(runs))


AGG = {"e2_composition": agg_e2, "n2_composition": agg_e2,
       "e3_lifelong": agg_e3, "n3_lifelong": agg_n3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", default=list(AGG))
    args = p.parse_args()
    summary = {}
    for name in args.names:
        runs = load_seeds(name)
        if not runs:
            print(f"\n[{name}] no results found")
            continue
        print(f"\n[{name}] {len(runs)} run(s): {', '.join(os.path.basename(p) for p, _ in runs)}")
        if len(runs) == 1:
            print("    NOTE: single seed -- no error bars.  Re-run with several --seed values.")
        fn = AGG.get(name)
        if fn is None:
            print(f"    no aggregator for {name}")
            continue
        try:
            summary[name] = fn(runs)
        except Exception as e:                                  # noqa: BLE001
            print(f"    could not aggregate: {type(e).__name__}: {e}")
    if summary:
        out = os.path.join(RESULTS, "aggregate.json")
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
