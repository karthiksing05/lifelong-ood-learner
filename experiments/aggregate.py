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
    arms, keys = [], ("joint", "digit", "fg", "bg", "precision", "recall", "f1", "content_ratio",
                      "vendi", "memorized", "density", "coverage")
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


def _backfill(row, run):
    """Compute A / REM / SSS for runs recorded before those columns existed.

    Everything they need was already saved -- the full accuracy matrix, the retained-image count and
    the stream size -- so re-running hours of GPU time only to add three derived numbers would be
    waste.  MS is NOT backfilled: it needs the parameter count at every step, which older runs did
    not record, and inventing it from the endpoints would misreport the one axis where our method
    does worst.
    """
    if "A" in row or "acc_matrix" not in row:
        return row
    acc = row["acc_matrix"]
    if not acc:
        return row
    out = dict(row)
    tri = [v for r in acc for v in r]
    out["A"] = float(np.mean(tri)) if tri else float("nan")
    K = len(acc)
    bwt = float(np.mean([acc[i][-1] - acc[i][0] for i in range(K - 1)])) if K > 1 else 0.0
    out["REM"] = float(1.0 - abs(min(bwt, 0.0)))
    # CelebA's target column was blanked by a nan-poisoned mean; recompute it here so the seeds
    # already spent do not have to be spent again (see n4_batch.cl_metrics).
    tgt = row.get("target_matrix")
    if tgt:
        fin = [r[-1] for r in tgt if len(r)]
        if fin and not np.isfinite(row.get("ACC_target", float("nan"))):
            with np.errstate(invalid="ignore"):
                out["ACC_target"] = float(np.nanmean(fin))
                fg = [np.nanmax(r) - r[-1] for r in tgt[:-1] if len(r)]
                out["FGT_target"] = float(np.nanmean(fg)) if fg else 0.0
    cfg = run.get("config", {})
    n_stream = float(cfg.get("n_tasks", K)) * float(cfg.get("n_train", 64))
    if "stored_images" in row and n_stream > 0:
        out["SSS"] = float(1.0 - min(1.0, float(row["stored_images"]) / n_stream))
    return out


def agg_n5(runs):
    """N5 / N9: one row per method, the standard CL metrics plus what each one costs."""
    arms, keys = [], ("ACC", "BWT", "FGT", "retention_exact", "params", "wall_seconds",
                      "qAFQ", "qAIQ", "qFR", "stored_images", "A", "REM", "MS", "SSS",
                      "ACC_target", "FGT_target")
    for _, r in runs:
        arms += [x["arm"] for x in r.get("rows", []) if x["arm"] not in arms]
    table = {}
    for a in arms:
        vals = {k: [] for k in keys}
        for _, r in runs:
            row = next((x for x in r.get("rows", []) if x["arm"] == a), None)
            if row is None:
                continue
            row = _backfill(row, r)
            for k in keys:
                if k in row:
                    vals[k].append(row[k])
        table[a] = {k: stat(v) for k, v in vals.items()}
    print(f"    {'arm':13s}{'ACC':>13s}{'A':>13s}{'FGT':>13s}{'bit-identical':>15s}"
          f"{'REM':>13s}{'SSS':>13s}{'MS':>13s}{'stored':>9s}{'params':>12s}")
    for a, row in table.items():
        pm = f"{row['params']['mean'] / 1e6:.2f}M" if row['params'] else "--"
        ws = f"{row['wall_seconds']['mean']:.0f}" if row['wall_seconds'] else "--"
        st = f"{row['stored_images']['mean']:.0f}" if row.get('stored_images') else "--"
        print(f"    {a:13s}{fmt(row['ACC']):>13s}{fmt(row.get('A')):>13s}"
              f"{fmt(row['FGT']):>13s}{fmt(row['retention_exact']):>15s}"
              f"{fmt(row.get('REM')):>13s}{fmt(row.get('SSS')):>13s}{fmt(row.get('MS')):>13s}"
              f"{st:>9s}{pm:>12s}")
    return dict(table=table, n_seeds=len(runs))


def agg_n9(runs):
    """N9 showdown: method x composition-quality metrics."""
    arms, keys = [], ("joint", "digit", "fg", "bg", "precision", "recall", "f1", "content_ratio",
                      "vendi", "memorized", "density", "coverage")
    for _, r in runs:
        arms += [a for a in r.get("summary", {}) if a not in arms]
    table = {a: {k: stat([r["summary"][a][k] for _, r in runs
                          if a in r.get("summary", {}) and k in r["summary"][a]])
                 for k in keys} for a in arms}
    print(f"    {'method':18s}" + "".join(f"{k:>13s}" for k in keys))
    for a, row in table.items():
        print(f"    {a:18s}" + "".join(f"{fmt(row[k]):>13s}" for k in keys))
    return dict(table=table, n_seeds=len(runs))


def agg_n71(runs):
    """N71: the loop WITH storage -- address (ceiling) vs the consolidated concept, plus what
    preservation and the stream did.  `stored` is the claim; `address` is the ceiling it is read
    against, so both are kept as arms rather than collapsed into one headline."""
    out = {}
    for arm in ("address", "stored"):
        out[arm] = {k: stat([r[k] for _, run in runs for r in run["rows"] if r["arm"] == arm])
                    for k in ("joint", "recall", "precision", "vendi")}
    out["stream"] = dict(forgetting=stat([run.get("forgetting") for _, run in runs]),
                         frozen=stat([float(bool(run.get("frozen"))) for _, run in runs]))
    print(f"    {'arm':10s}{'joint':>10s}{'recall':>10s}{'vendi':>9s}")
    for arm in ("address", "stored"):
        print(f"    {arm:10s}{fmt(out[arm]['joint']):>10s}{fmt(out[arm]['recall']):>10s}"
              f"{fmt(out[arm]['vendi']):>9s}")
    print(f"    forgetting {fmt(out['stream']['forgetting'])} | base frozen "
          f"{fmt(out['stream']['frozen'])}")
    return dict(table=out, n_seeds=len(runs))


def agg_n72(runs):
    """N72: the four donation conditions, as arms.  `nearest-class` is the adversarial control --
    the model's own best existing embedding -- so its joint is the number that says whether the
    capability was already there."""
    arms = sorted({r["arm"] for _, run in runs for r in run["rows"]})
    out = {a: {k: stat([r[k] for _, run in runs for r in run["rows"] if r["arm"] == a])
               for k in ("joint", "recall", "precision", "vendi", "memorized", "d_query")}
           for a in arms}
    cond = {k: stat([float(bool(run["conditions"][k])) for _, run in runs])
            for k in ("model_lacked_it", "new_point", "not_copies", "covers")}
    out["conditions"] = cond
    print(f"    {'arm':16s}{'joint':>10s}{'recall':>10s}{'memorized':>12s}{'d(query)':>11s}")
    for a in arms:
        print(f"    {a:16s}{fmt(out[a]['joint']):>10s}{fmt(out[a]['recall']):>10s}"
              f"{fmt(out[a]['memorized']):>12s}{fmt(out[a]['d_query']):>11s}")
    print("    conditions: " + " ".join(f"{k}={fmt(v)}" for k, v in cond.items()))
    return dict(table=out, n_seeds=len(runs))


def agg_n64(runs):
    """N64 multi-object: ADDRESS on held-out PAIRS, the basis fit, and the two controls.

    The two controls are aggregated as arms in their own right because each one licenses the
    headline rather than decorating it: `detector` is the instrument (a weak one poisons every
    number downstream) and `self_cond` is the model conditioned on its OWN trained row (without it,
    a low ADDRESS score cannot be told apart from a model that never learned to condition).
    """
    def g(r, *path, default=None):
        for k in path:
            r = (r or {}).get(k) if isinstance(r, dict) else None
        return r if r is not None else default
    out = {
        "address": dict(
            exact=stat([g(r, "summary", "exact") for _, r in runs]),
            recall=stat([g(r, "summary", "recall") for _, r in runs])),
        "floor": dict(
            exact=stat([g(r, "summary", "floor_exact") for _, r in runs]),
            recall=stat([g(r, "summary", "floor_recall") for _, r in runs])),
        "basis": dict(
            r2=stat([r.get("r2") for _, r in runs]),
            r2_cv=stat([r.get("r2_cv") for _, r in runs]),
            perm_null=stat([r.get("perm_null_mean") for _, r in runs])),
        "controls": dict(
            detector=stat([g(r, "detector", "top2_exact") for _, r in runs]),
            self_cond=stat([r.get("self_cond") for _, r in runs])),
    }
    print(f"    {'arm':12s}{'exact':>10s}{'recall':>10s}")
    for k in ("address", "floor"):
        print(f"    {k:12s}{fmt(out[k]['exact']):>10s}{fmt(out[k]['recall']):>10s}")
    print(f"    basis R2 {fmt(out['basis']['r2'])} | CV {fmt(out['basis']['r2_cv'])} | "
          f"null {fmt(out['basis']['perm_null'])}")
    print(f"    controls: detector {fmt(out['controls']['detector'])} | "
          f"self-cond {fmt(out['controls']['self_cond'])}")
    return dict(table=out, n_seeds=len(runs))


def agg_n62(runs):
    """N62/N65 router operating point: held-out AUROC per backbone, and how it was chosen.

    Reported per backbone rather than pooled, because the whole finding is that the two backbones
    differ: ColorMNIST routes on DENSITY (its ablation matches its conditional) and Shapes3D routes
    on CONDITIONING.  A mean over the two would erase exactly the thing being claimed.
    """
    out = {}
    print(f"    {'backbone':24s}{'held-out AUROC':>16s}{'ablation':>12s}{'at u=0.10':>12s}")
    for _, r in runs:
        for name, d in (r.get("backbones") or {}).items():
            m = d.get("mean", {})
            def g(k):
                v = m.get(k)
                return float(v[0]) if isinstance(v, (list, tuple)) else (
                    float(v) if v is not None else None)
            row = dict(u=g("ustar") if "ustar" in m else g("u_lf"),
                       auroc=g("eval_cond") if "eval_cond" in m else g("a_lf"),
                       ablation=g("eval_unc"),
                       shipped=g("eval_dflt") if "eval_dflt" in m else g("a_shipped"),
                       oracle=g("a_oracle"))
            # `table[arm][metric]` with a `mean`, because that is the shape tests/test_claims.py
            # navigates.  A flat dict aggregates fine and then every claim against it reports
            # MISSING rather than mismatching -- which reads as "no claim made" instead of "claim
            # not checked".
            out[name] = {k: stat([v]) for k, v in row.items() if v is not None}

            def show(v):
                return "-" if v is None else f"{v:.3f}"
            print(f"    {name:24s}{show(row['auroc']):>16}{show(row['ablation']):>12}"
                  f"{show(row['shipped']):>12}")
    return dict(table=out, n_seeds=len(runs))


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
    arms, keys = [], ("pair", "target", "joint", "mean_attr", "precision", "recall", "f1", "vendi",
                      "content_ratio")
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
            row = _backfill(row, r)
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


def _agg_rows(runs, label_key, keys, width=24):
    """Generic per-`label_key` aggregator for experiments that write a flat `rows` list.

    Used by the exploration experiments (n25-n28), whose rows are one dict per arm/rule rather than
    the accuracy-matrix shape the older aggregators expect.  Keys absent from a run are simply
    skipped, so an experiment can gain a column without breaking the seeds already collected.
    """
    labels = []
    for _, r in runs:
        labels += [x[label_key] for x in r.get("rows", []) if x[label_key] not in labels]
    table = {}
    for a in labels:
        vals = {k: [] for k in keys}
        for _, r in runs:
            row = next((x for x in r.get("rows", []) if x[label_key] == a), None)
            if row is None:
                continue
            for k in keys:
                if k in row and row[k] is not None:
                    vals[k].append(row[k])
        table[a] = {k: stat(v) for k, v in vals.items()}
    print(f"    {label_key:{width}s}" + "".join(f"{k:>14s}" for k in keys))
    for a, row in table.items():
        print(f"    {a:{width}s}" + "".join(f"{fmt(row[k]):>14s}" for k in keys))
    return dict(table=table, n_seeds=len(runs))


def agg_n25(runs):
    """N25: the pretrained case study -- per-arm attribute rates and diversity."""
    ks = sorted({k for _, r in runs for x in r.get("rows", []) for k in x
                 if k not in ("arm",)})
    return _agg_rows(runs, "arm", tuple(ks))


def agg_n26(runs):
    """N26: batch vs sequential on a pretrained backbone."""
    return _agg_rows(runs, "arm", ("wall_seconds", "mean_attr", "vendi", "retention_exact",
                                   "n_nodes", "shared_nodes", "params"), width=16)


def agg_n27(runs):
    """N27: what a composition inherits from its conditionals."""
    return _agg_rows(runs, "arm", ("joint", "digit", "fg", "bg", "precision", "recall", "f1",
                                   "vendi"))


def agg_n28(runs):
    """N28: which update rule composes best, and how far it moves the frozen field."""
    return _agg_rows(runs, "rule", ("drift", "alone_fg", "alone_bg", "alone_digit", "joint",
                                    "recall", "f1", "vendi", "wall_seconds"), width=16)


def agg_n30(runs):
    """N30: the prototype-distance dial, aggregated per (arm, strength).

    Unlike n25-n28 a row here is identified by TWO fields, so the generic aggregator cannot be
    reused: it would collapse all seven strengths of an arm into one cell.  Rows are relabelled
    `arm@s` -- the same key the experiment's own `summary` uses -- and then averaged over the
    compositions within a seed before averaging over seeds, so a seed counts once regardless of how
    many compositions it drew.
    """
    merged = []
    for path, r in runs:
        by = {}
        for x in r.get("rows", []):
            by.setdefault(f"{x['arm']}@{x['strength']:g}", []).append(x)
        merged.append((path, dict(rows=[
            dict(key=k, **{m: float(np.nanmean([v[m] for v in xs if m in v]))
                           for m in ("dist_prototype", "dist_pairwise", "joint", "digit", "fg",
                                     "bg", "recall", "f1", "vendi", "tv_ratio")
                           if any(m in v for v in xs)})
            for k, xs in by.items()])))
    return _agg_rows(merged, "key", ("dist_prototype", "dist_pairwise", "joint", "recall", "f1",
                                     "vendi"), width=18)


def agg_n73(runs):
    """N73: composition on the CelebA grid -- the only CelebA backbone with a DENSITY holdout.

    Two things make this aggregation different from `agg_n72`.  First, `joint` must be read against
    the `ceiling` each run carries: attribute probes on REAL images score 0.414 here, so an ADDRESS
    joint of 0.194 is 47 % of what the instrument can register, not 19 % of the truth.  Second, the
    number the claim rests on is the PAIRED per-cell margin of `address` over `nearest-class` --
    the backbone's own best existing embedding -- because a pooled difference of means hides which
    cells carry it, and at three cells one bad cell is a third of the evidence.
    """
    arms = ("address", "nearest-class", "unconditional")
    rows = [r for _, run in runs for r in run["rows"]]
    out = {a: {k: stat([r[k] for r in rows if r["arm"] == a])
               for k in ("joint", "recall", "precision", "vendi")} for a in arms}
    ceil = stat([run["ceiling"] for _, run in runs])
    out["ceiling"] = ceil
    out["basis_r2"] = stat([run["basis_r2"] for _, run in runs])

    # NORMALISE WITHIN THE RUN before pooling.  The probe ceiling is refit per seed and moves a
    # long way -- 0.414 to 0.570 across three seeds here -- so a pooled raw joint mixes runs whose
    # instruments have different dynamic range.  `joint_frac` is each run's joint divided by ITS
    # OWN ceiling, which is the quantity the claim is actually about.
    for a in arms:
        fr = [r["joint"] / max(run["ceiling"], 1e-9)
              for _, run in runs for r in run["rows"] if r["arm"] == a]
        out[a]["joint_frac"] = stat(fr)

    # PAIRED, per cell.  Cells differ enormously in support (518 to 15349 images here), so an
    # unpaired comparison is dominated by which cells a run happened to hold out.
    per = {}
    for _, run in runs:
        by = {(r["arm"], r["cell"]): r["joint"] for r in run["rows"]}
        for c in sorted({r["cell"] for r in run["rows"]}):
            if ("address", c) in by and ("nearest-class", c) in by:
                per.setdefault(c, []).append(by[("address", c)] - by[("nearest-class", c)])
    # REPORT BOTH, and label them, because the two n's mean different things and only one of them
    # is the evidence.  Pooling every (cell, seed) pair gives n=9 from three sampling seeds over the
    # SAME three cells -- that is replication of the sampler, not of the claim, and quoting its
    # error bar as though it were three times the evidence would be wrong.  The per-cell mean over
    # DISTINCT held-out cells is the honest n; more of it means a backbone with a bigger holdout.
    marg = [v for xs in per.values() for v in xs]
    by_cell = [float(np.mean(xs)) for xs in per.values()]
    out["margin_vs_retrieval"] = stat(by_cell)
    out["margin_pooled_cell_seed"] = stat(marg)
    out["cells_won"] = stat([float(v > 0) for v in by_cell])
    out["n_cells"] = len(by_cell)
    out["n_cell_seed_pairs"] = len(marg)

    print(f"    {'arm':16s}{'joint':>14s}{'% ceil':>8s}{'recall':>14s}{'vendi':>14s}")
    m = ceil["mean"] if isinstance(ceil, dict) else ceil
    for a in arms:
        j = out[a]["joint"]
        jm = j["mean"] if isinstance(j, dict) else j
        print(f"    {a:16s}{fmt(j):>14s}{100 * (out[a]['joint_frac']['mean'] if isinstance(out[a]['joint_frac'], dict) else out[a]['joint_frac']):>7.0f}%"
              f"{fmt(out[a]['recall']):>14s}{fmt(out[a]['vendi']):>14s}")
    print(f"    probe ceiling {fmt(ceil)} | basis R2 {fmt(out['basis_r2'])}")
    print(f"    ADDRESS - nearest-class, paired over {out['n_cells']} DISTINCT cells: "
          f"{fmt(out['margin_vs_retrieval'])}, won on {fmt(out['cells_won'])}")
    print(f"      (over all {out['n_cell_seed_pairs']} cell x seed pairs: "
          f"{fmt(out['margin_pooled_cell_seed'])} -- sampler replication, not more cells)")
    return dict(table=out, n_seeds=len(runs))


AGG = {"n71_absorb_and_generate": agg_n71, "n72_donation": agg_n72,
       "n73_celeba_grid_compose": agg_n73,
       "n64_multiobject_address": agg_n64, "n62_router_calibrate": agg_n62, "n65_u_labelfree": agg_n62,
       "e2_composition": agg_e2, "n2_composition": agg_e2,
       "e3_lifelong": agg_e3, "n3_lifelong": agg_n3, "n7_inversion": agg_n7,
       "n5_classinc": agg_n5, "n9_showdown": agg_n9, "n8_idealwords": agg_n7,
       "n18_router": agg_n18, "n20_celeba_composition": agg_n20,
       "n22_celeba_lifelong": agg_n22, "n23_native_parity": agg_n23,
       "n21_celeba_vocabulary_Gray_Hair": agg_n20,
       "n20_celeba_composition_pair": agg_n20,
       "n11_basis_extension_nmadapt": agg_n7,
       "n25_pretrained_case": agg_n25, "n26_batch_pretrained": agg_n26,
       "n27_factor_conditionals": agg_n27, "n28_update_rules": agg_n28,
       "n30_prototype_distance_fine": agg_n30}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", default=list(AGG))
    p.add_argument("--as", dest="as_name", default=None,
                   help="store the result under this canonical key instead of the run name, so a "
                        "tagged re-run (n9_showdown_cal) feeds the figures and tables that ask for "
                        "the experiment (n9_showdown).  Only valid with a single name.")
    p.add_argument("--force", action="store_true",
                   help="overwrite a key that was last written by a DIFFERENT source (see below)")
    args = p.parse_args()
    if args.as_name and len(args.names) != 1:
        p.error("--as takes exactly one name")
    summary, claimed = {}, {}
    for name in args.names:
        runs = load_seeds(name)
        if not runs:
            print(f"\n[{name}] no results found")
            continue
        print(f"\n[{name}] {len(runs)} run(s): {', '.join(os.path.basename(p) for p, _ in runs)}")
        if len(runs) == 1:
            print("    NOTE: single seed -- no error bars.  Re-run with several --seed values.")
        # tagged runs (`n5_classinc_full`, `n20_celeba_composition_pairfull`) use the aggregator of
        # the experiment they came from.  Longest-prefix, so `..._pair` wins over `..._composition`.
        fn = AGG.get(name)
        if fn is None:
            k = max((q for q in AGG if name.startswith(q)), key=len, default=None)
            fn = AGG.get(k) if k else None
        if fn is None:
            print(f"    no aggregator for {name}")
            continue
        try:
            key = args.as_name or name
            summary[key] = fn(runs)
            claimed[key] = name
        except Exception as e:                                  # noqa: BLE001
            print(f"    could not aggregate: {type(e).__name__}: {e}")
    if summary:
        out = os.path.join(RESULTS, "aggregate.json")
        # MERGE rather than replace.  Running this with a subset of names used to overwrite the
        # whole file with just those, silently dropping every other experiment's pooled table --
        # and the figures are built from this file, so they came out empty rather than wrong.
        prev = {}
        if os.path.exists(out):
            try:
                with open(out) as fh:
                    prev = json.load(fh)
            except (ValueError, OSError):
                prev = {}

        # DO NOT let a bare re-run silently demote an ALIASED canonical entry.
        #
        # `reproduce.sh` populates the canonical keys from tagged runs -- e.g.
        #     aggregate.py n9_showdown_cal --as n9_showdown
        # because the canonical Color-MNIST composition table is the `_cal` configuration.  But a
        # stale, partial `results/n9_showdown.json` from an old smoke run also exists, and running
        # `aggregate.py` with NO arguments re-derives every default key from same-named files -- so
        # it quietly replaced the 3-seed canonical numbers with that 7 KB smoke run.  Nothing failed;
        # `tests/test_claims.py` simply went from 33/33 to 23/33 and the documents looked wrong
        # instead of the aggregate.  This makes that overwrite refuse rather than happen.
        prov = prev.get("_provenance", {}) if isinstance(prev.get("_provenance"), dict) else {}
        blocked = []
        for k in list(summary):
            was, now = prov.get(k), claimed[k]
            if was and was != now and not args.force:
                blocked.append((k, was, now))
                del summary[k]
        for k, was, now in blocked:
            print(f"\n  REFUSED to overwrite `{k}`: it was aggregated from `{was}`, and this run "
                  f"would replace it with `{now}`.\n"
                  f"    That is the aliasing `reproduce.sh` sets up, so a bare re-run is almost "
                  f"always a mistake.\n"
                  f"    Intended?  `aggregate.py {now}"
                  f"{'' if k == now else f' --as {k}'} --force`")
        if not summary:
            print("\nnothing written")
            return
        prov.update({k: claimed[k] for k in summary})
        prev.update(summary)
        prev["_provenance"] = prov
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(prev, fh, indent=1, default=float)
        os.replace(tmp, out)
        n_keys = len([k for k in prev if not k.startswith("_")])
        print(f"\nwrote {out} ({len(summary)} updated, {n_keys} total"
              f"{f', {len(blocked)} REFUSED' if blocked else ''})")


if __name__ == "__main__":
    main()
