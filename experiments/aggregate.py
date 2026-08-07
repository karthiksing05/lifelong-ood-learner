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
          f"{'base_drift':>13s}{'transfer':>12s}{'coverage':>12s}")
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
        cov = []
        for _, r in runs:
            steps = r.get(a, {}).get("steps", [])
            v = [s_["coverage"]["recall"] for s_ in steps
                 if isinstance(s_.get("coverage"), dict) and s_["coverage"].get("recall") == s_["coverage"].get("recall")]
            if v:
                cov.append(float(np.mean(v)))
        out[a] = dict(retention_exact=stat(re_), retention_acc=stat(ra), new_acc=stat(na),
                      base_drift=stat(bd), transfer=stat(tr), coverage_recall=stat(cov))
        print(f"    {a:10s}{fmt(out[a]['retention_exact']):>12s}{fmt(out[a]['retention_acc']):>12s}"
              f"{fmt(out[a]['new_acc']):>12s}{fmt(out[a]['base_drift'], 5):>13s}"
              f"{fmt(out[a]['transfer'], 4):>12s}{fmt(out[a]['coverage_recall']):>12s}")
    return dict(table=out, n_seeds=len(runs))


def agg_n7(runs):
    """N7: composition arms scored with the paper's coverage metrics."""
    arms, keys = [], ("joint", "digit", "fg", "bg", "precision", "recall", "f1", "content_ratio")
    for _, r in runs:
        arms += [a for a in r.get("summary", {}) if a not in arms]
    table = {a: {k: stat([r["summary"][a][k] for _, r in runs
                          if a in r.get("summary", {}) and k in r["summary"][a]])
                 for k in keys} for a in arms}
    ceil = runs[0][1].get("ceiling_model", {})
    print(f"    {'arm':22s}" + "".join(f"{k:>13s}" for k in keys))
    for a, row in table.items():
        print(f"    {a:22s}" + "".join(f"{fmt(row[k]):>13s}" for k in keys))
    if ceil:
        print(f"    {'-- ceiling (base, seen)':22s}"
              + "".join(f"{ceil.get(k, float('nan')):13.2f}" for k in keys))
    return dict(table=table, ceiling=ceil, n_seeds=len(runs))


def agg_n5(runs):
    """N5 / N9: one row per method, the standard CL metrics plus what each one costs."""
    arms, keys = [], ("ACC", "BWT", "FGT", "retention_exact", "params", "wall_seconds")
    for _, r in runs:
        arms += [x["arm"] for x in r.get("rows", []) if x["arm"] not in arms]
    table = {}
    for a in arms:
        vals = {k: [] for k in keys}
        for _, r in runs:
            row = next((x for x in r.get("rows", []) if x["arm"] == a), None)
            if row is None:
                continue
            for k in keys:
                if k in row:
                    vals[k].append(row[k])
        table[a] = {k: stat(v) for k, v in vals.items()}
    print(f"    {'arm':13s}{'ACC':>13s}{'BWT':>14s}{'FGT':>13s}{'bit-identical':>15s}"
          f"{'params':>12s}{'wall(s)':>11s}")
    for a, row in table.items():
        pm = f"{row['params']['mean'] / 1e6:.2f}M" if row['params'] else "--"
        ws = f"{row['wall_seconds']['mean']:.0f}" if row['wall_seconds'] else "--"
        print(f"    {a:13s}{fmt(row['ACC']):>13s}{fmt(row['BWT']):>14s}{fmt(row['FGT']):>13s}"
              f"{fmt(row['retention_exact']):>15s}{pm:>12s}{ws:>11s}")
    return dict(table=table, n_seeds=len(runs))


def agg_n9(runs):
    """N9 showdown: method x composition-quality metrics."""
    arms, keys = [], ("joint", "digit", "fg", "bg", "precision", "recall", "f1", "content_ratio")
    for _, r in runs:
        arms += [a for a in r.get("summary", {}) if a not in arms]
    table = {a: {k: stat([r["summary"][a][k] for _, r in runs
                          if a in r.get("summary", {}) and k in r["summary"][a]])
                 for k in keys} for a in arms}
    print(f"    {'method':18s}" + "".join(f"{k:>13s}" for k in keys))
    for a, row in table.items():
        print(f"    {a:18s}" + "".join(f"{fmt(row[k]):>13s}" for k in keys))
    return dict(table=table, n_seeds=len(runs))


def agg_n18(runs):
    """N18 router: the two AUROCs, their unconditional ablation, and routing accuracy."""
    keys = ["novel vs in-distribution", "new-vocabulary vs new-combination"]
    out = dict(accuracy=stat([r.get("accuracy") for _, r in runs]))
    print(f"    {'decision':38s}{'AUROC':>12s}{'uncond ablation':>18s}")
    for k in keys:
        a = stat([r.get("auroc", {}).get(k) for _, r in runs])
        b = stat([r.get("auroc_uncond_ablation", {}).get(k) for _, r in runs])
        out[k] = dict(auroc=a, uncond=b)
        print(f"    {k:38s}{fmt(a):>12s}{fmt(b):>18s}")
    print(f"    {'self-routing accuracy':38s}{fmt(out['accuracy']):>12s}")
    for c in ("vocab_sent_to_free", "combination_sent_to_weights",
              "vocab_free_route_accuracy", "combination_free_route_accuracy"):
        s = stat([r.get("cost", {}).get(c) for _, r in runs])
        out[c] = s
        if s:
            print(f"    {c:38s}{fmt(s):>12s}")
    return dict(table=out, n_seeds=len(runs))


def agg_n20(runs):
    """N20 / N21: CelebA composition arms.  `target` is the discriminative column."""
    arms, keys = [], ("target", "joint", "mean_attr", "precision", "recall", "f1", "content_ratio")
    for _, r in runs:
        arms += [a for a in r.get("summary", {}) if a not in arms]
    table = {a: {k: stat([r["summary"][a][k] for _, r in runs
                          if a in r.get("summary", {}) and k in r["summary"][a]])
                 for k in keys} for a in arms}
    print(f"    {'arm':30s}" + "".join(f"{k:>12s}" for k in keys))
    for a, row in table.items():
        print(f"    {a:30s}" + "".join(f"{fmt(row[k]):>12s}" for k in keys))
    return dict(table=table, n_seeds=len(runs))


def agg_n22(runs):
    """N22: CelebA class-incremental, CL metrics on both the mean-attribute and target scores."""
    arms = []
    keys = ("ACC", "FGT", "BWT", "ACC_target", "FGT_target", "retention_exact", "params")
    for _, r in runs:
        arms += [x["arm"] for x in r.get("rows", []) if x["arm"] not in arms]
    table = {}
    for a in arms:
        vals = {k: [] for k in keys}
        for _, r in runs:
            row = next((x for x in r.get("rows", []) if x["arm"] == a), None)
            if row is None:
                continue
            for k in keys:
                if k in row:
                    vals[k].append(row[k])
        table[a] = {k: stat(v) for k, v in vals.items()}
    print(f"    {'arm':13s}{'ACC':>13s}{'FGT':>13s}{'tgtACC':>13s}{'tgtFGT':>13s}"
          f"{'bit-identical':>15s}{'params':>11s}")
    for a, row in table.items():
        pm = f"{row['params']['mean'] / 1e6:.2f}M" if row["params"] else "--"
        print(f"    {a:13s}{fmt(row['ACC']):>13s}{fmt(row['FGT']):>13s}"
              f"{fmt(row['ACC_target']):>13s}{fmt(row['FGT_target']):>13s}"
              f"{fmt(row['retention_exact']):>15s}{pm:>11s}")
    return dict(table=table, n_seeds=len(runs))


def agg_n23(runs):
    """N23: native parity — the ratio is the number this experiment exists to produce."""
    arms, keys = ["real", "native", "ours", "paper"], ("joint", "f1", "recall", "content_ratio")
    table = {a: {k: stat([r["summary"][a][k] for _, r in runs
                          if a in r.get("summary", {}) and k in r["summary"][a]])
                 for k in keys} for a in arms}
    print(f"    {'arm':10s}" + "".join(f"{k:>14s}" for k in keys))
    for a in arms:
        print(f"    {a:10s}" + "".join(f"{fmt(table[a][k]):>14s}" for k in keys))
    par = {}
    print(f"\n    parity (consolidated / native):")
    for a in ("ours", "paper"):
        par[a] = {k: stat([r["parity"][a][k] for _, r in runs
                           if a in r.get("parity", {}) and k in r["parity"][a]])
                  for k in ("joint", "f1", "recall")}
        print(f"    {a:10s}" + "".join(f"{k} {fmt(par[a][k])}   " for k in par[a]))
    drag = {k: stat([r.get("drag", {}).get(k) for _, r in runs])
            for k in ("deployed (before)", "ours (after)", "oracle")}
    print(f"\n    seen-slot accuracy (drag): "
          + "  ".join(f"{k} {fmt(v)}" for k, v in drag.items() if v))
    return dict(table=table, parity=par, drag=drag, n_seeds=len(runs))


AGG = {"e2_composition": agg_e2, "n2_composition": agg_e2,
       "e3_lifelong": agg_e3, "n3_lifelong": agg_n3, "n7_inversion": agg_n7,
       "n5_classinc": agg_n5, "n9_showdown": agg_n9, "n8_idealwords": agg_n7,
       "n18_router": agg_n18, "n20_celeba_composition": agg_n20,
       "n22_celeba_lifelong": agg_n22, "n23_native_parity": agg_n23,
       "n21_celeba_vocabulary_Gray_Hair": agg_n20}


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
