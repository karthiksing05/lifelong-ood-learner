"""Does perception transfer across tasks by object TYPE?  (offline; after experiments/p6_reader.py --synth-only --synth-dir p6_synth_rj)

The per-task reader learns a task's objects from that task's own rendered scenes (0.13-0.14 cm).  Here a reader per object TYPE
(lifelong_ood/policy/phase.py type_of) is trained on the rendered scenes of every OTHER task in the pool that contains that type
(the scene, layout and neighbours differ), leave-one-task-out, and tested on the held-out task's first frames of init states 20-29
(the cached test features of p6_reader.py).  Features: the spatial ResNet-18 maps both readers use (p6_feats caches); standardized
ridge (dual form), the penalty chosen by cross-validation grouped by source task.  Types with several instances in one scene are
skipped (a type reader cannot say which instance).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/p6_type_reader.py [--record]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
LAMS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def type_of(n):
    return re.sub(r"_\d+$", "", str(n))


def ridge(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    sx = X.std(0) + 1e-8
    Xc, Yc = (X - mx) / sx, Y - my
    W = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(X)), Yc)
    return W / sx[:, None], mx, my


def load():
    """{task: dict(names, Xr, Yr [n, k, 3], Xt, Yt [m, k, 3])} for tasks with both rendered and test caches."""
    out = {}
    for f in glob.glob(f"{R}/p6_feats/p6_synth_rj_*_spatial4.npz"):
        t = int(re.search(r"p6_synth_rj_(\d+)_spatial4", f).group(1))
        tf = f"{R}/p6_feats/{t}_20_spatial4.npz"
        if not os.path.exists(tf):
            continue
        r, c = np.load(f), np.load(tf, allow_pickle=True)
        names = [str(n) for n in c["names"]]
        sel = (c["trial"] >= 20) & (c["step"] == 0)
        out[t] = dict(names=names, Xr=r["X"].astype(np.float32), Yr=r["Y"].reshape(len(r["Y"]), -1, 3),
                      Xt=c["X"][sel].astype(np.float32), Yt=c["Y"][sel].reshape(int(sel.sum()), -1, 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=150, help="rendered scenes per source task (memory)")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    D = load()
    print(f"[type] {len(D)} tasks with rendered and test features", flush=True)
    unique = {t: {type_of(n): j for j, n in enumerate(d["names"]) if sum(type_of(m) == type_of(n) for m in d["names"]) == 1}
              for t, d in D.items()}
    res = dict(per_object={}, lams=list(LAMS), per_task_scenes=a.per_task)
    for t in sorted(D):
        for T, j in unique[t].items():
            src = [s for s in D if s != t and T in unique[s]]
            key = f"{t}:{D[t]['names'][j]}"
            if not src:
                res["per_object"][key] = dict(type=T, sources=0)
                continue
            X = np.concatenate([D[s]["Xr"][:a.per_task] for s in src]).astype(np.float64)
            Y = np.concatenate([D[s]["Yr"][:a.per_task, unique[s][T]] for s in src])
            g = np.concatenate([np.full(min(a.per_task, len(D[s]["Xr"])), s) for s in src])
            if len(src) >= 2:
                cv = {}
                for lam in LAMS:
                    e = []
                    for s in src:
                        W, mx, my = ridge(X[g != s], Y[g != s], lam)
                        e.append(np.linalg.norm((X[g == s] - mx) @ W + my - Y[g == s], axis=-1).mean())
                    cv[lam] = float(np.mean(e))
                lam = min(cv, key=cv.get)
            else:
                lam, cv = 1e3, {}
            W, mx, my = ridge(X, Y, lam)
            err = np.linalg.norm((D[t]["Xt"].astype(np.float64) - mx) @ W + my - D[t]["Yt"][:, j], axis=-1) * 100
            res["per_object"][key] = dict(type=T, sources=len(src), source_tasks=src, lam=lam, cv_cm=cv.get(lam, None) and cv[lam] * 100,
                                          err_cm_median=float(np.median(err)), err_cm_max=float(err.max()))
            print(f"[type] {key} ({T}; {len(src)} source tasks, lam {lam:g}): {np.median(err):.2f} cm (max {err.max():.2f})", flush=True)
    got = [v["err_cm_median"] for v in res["per_object"].values() if v.get("sources")]
    res["median_cm"] = float(np.median(got))
    res["n_objects"] = len(got)
    res["unseen_types"] = sorted({v["type"] for v in res["per_object"].values() if not v.get("sources")})
    print(f"[type] median over {len(got)} objects with sources: {res['median_cm']:.2f} cm; types with no other task: {res['unseen_types']}")
    json.dump(res, open(f"{R}/p6_type_reader.json", "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p6_reader", bench="libero_90", tag="type_transfer", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p6_type_reader.py", command=" ".join(sys.argv), artifacts=[f"{R}/p6_type_reader.json"])


if __name__ == "__main__":
    main()
