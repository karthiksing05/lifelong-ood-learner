"""How small can a task's pose reader be? (the storage side of archive/2026-10-program/PLAN_TRANSFER.md P6)

The deployed reader (experiments/p6_reader.py --features spatial --pool 4 --synth-only --synth-dir p6_synth_rj) is a ridge map from
16,384 ResNet-18 map features to the objects' positions: 16,384 x 3 x (objects) weights, about 0.4 MB per task in fp32 -- the bulk
of a task's bytes once its phases take 2-3 KB.  Here every task's head is refit on a SHARED k-dimensional code of the features:
- pca:    the principal directions of the OTHER half of the tasks' rendered scenes (cross-fitted: a task's own scenes never shape its
          basis; stored once, k x 16,384 fp16, for all tasks);
- random: a seeded Gaussian projection (0 stored bytes: the seed);
and the per-task head is ridge on the k codes (penalty by 5-fold CV grouped as in p6_reader).  Offline read = the median error (cm)
on the first frames of init states 20-29, the deployed reader's test set, against the full reader refit here the same way.

    python experiments/p6_compress.py [--ks 64,128,256,512] [--record]      (CPU, ~2 GB)
    python experiments/p6_compress.py --export pca512      -> results/robot/p6_reader_pca512_rj_<t>.npz for every task (the basis
        cross-fitted on the other half, folded into W so the phasebind server loads it unchanged; its bytes = the head's, fp16,
        plus the shared basis once: results/robot/p6_compress_export_<variant>.json)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
F = os.path.join(R, "p6_feats")
LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def ridge(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    sx = X.std(0) + 1e-8
    Xc, Yc = (X - mx) / sx, Y - my
    if X.shape[1] > X.shape[0]:
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(X.shape[0]), Yc)
    else:
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    return W / sx[:, None], mx, my


def cv_lam(X, Y, folds=5, seed=0):
    """Rendered scenes are independent placements: 5 random folds."""
    idx = np.random.default_rng(seed).permutation(len(X))
    err = {}
    for lam in LAMS:
        e = []
        for f in range(folds):
            te = np.zeros(len(X), bool)
            te[idx[f::folds]] = True
            W, mx, my = ridge(X[~te], Y[~te], lam)
            e.append(np.linalg.norm(((X[te] - mx) @ W + my - Y[te]).reshape(int(te.sum()), -1, 3), axis=-1).mean())
        err[lam] = float(np.mean(e))
    return min(err, key=err.get)


def load(t):
    s = np.load(f"{F}/p6_synth_rj_{t}_spatial4.npz")
    c = np.load(f"{F}/{t}_20_spatial4.npz", allow_pickle=True)
    sel = (c["trial"] >= 20) & (c["trial"] < 1000) & (c["step"] == 0)
    return (s["X"].astype(np.float32), s["Y"].astype(np.float64), c["X"][sel].astype(np.float32), c["Y"][sel].astype(np.float64),
            [str(n) for n in c["names"]])


def pca_basis(Xs, k, seed=0, iters=3):
    """Top-k principal directions of the stacked features (randomized range finder, float32)."""
    X = np.concatenate(Xs)
    mu = X.mean(0)
    X = X - mu
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(X @ rng.standard_normal((X.shape[1], k + 16)).astype(np.float32))[0]
    for _ in range(iters):
        Q = np.linalg.qr(X @ (X.T @ Q))[0]
    _, _, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return mu, Vt[:k]


def err_cm(pred, Y):
    return np.linalg.norm((pred - Y).reshape(len(Y), -1, 3), axis=-1) * 100


def export(variant):
    """Compressed readers for the phasebind server (reader=<variant>_rj): pred = (f - mx) @ W + my with W = basis^T @ head."""
    kind, k = variant[:-len(variant.lstrip("pcarnd"))], int(variant.lstrip("pcarnd"))
    tasks = sorted(int(f.rsplit("_", 2)[-2]) for f in glob.glob(f"{F}/p6_synth_rj_*_spatial4.npz"))
    tasks = [t for t in tasks if os.path.exists(f"{F}/{t}_20_spatial4.npz")]
    halves = (tasks[0::2], tasks[1::2])
    meta = dict(variant=variant, tasks=tasks, halves=[list(h) for h in halves], head_bytes={}, basis_bytes=0)
    for h, other in ((0, 1), (1, 0)):
        if kind == "pca":
            mu, V = pca_basis([load(t)[0] for t in halves[other]], k)
            meta["basis_bytes"] = 2 * k * (V.shape[1] + 1)
        for t in halves[h]:
            Xs, Ys, _, _, names = load(t)
            if kind == "pca":
                Zs, B, m0 = (Xs - mu) @ V.T, V.T, mu
            else:
                B = np.random.default_rng(1234).standard_normal((Xs.shape[1], k)).astype(np.float32) / np.sqrt(k)
                Zs, m0 = Xs @ B, np.zeros(Xs.shape[1])
            Zs = Zs.astype(np.float64)
            lam = cv_lam(Zs, Ys)
            Wh, mz, my = ridge(Zs, Ys, lam)
            Wh, mz, my = (x.astype(np.float16).astype(np.float64) for x in (Wh, mz, my))      # the stored head is fp16
            np.savez_compressed(f"{R}/p6_reader_{variant}_rj_{t}.npz", W=B.astype(np.float64) @ Wh, mx=np.asarray(m0, np.float64),
                                my=my - mz @ Wh, names=np.array(names), features="spatial", pool=4, lam=lam)
            meta["head_bytes"][str(t)] = 2 * (Wh.size + mz.size + my.size)
            print(f"[export] {variant} task {t}: lam {lam:g}, head {meta['head_bytes'][str(t)]} B", flush=True)
    json.dump(meta, open(f"{R}/p6_compress_export_{variant}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="64,128,256,512")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--export", default=None, help="pca<k> or rand<k>: write the compressed readers")
    a = ap.parse_args()
    if a.export:
        return export(a.export)
    ks = [int(k) for k in a.ks.split(",")]
    tasks = sorted(int(f.rsplit("_", 2)[-2]) for f in glob.glob(f"{F}/p6_synth_rj_*_spatial4.npz"))
    tasks = [t for t in tasks if os.path.exists(f"{F}/{t}_20_spatial4.npz")]
    data = {t: load(t) for t in tasks}
    halves = (tasks[0::2], tasks[1::2])
    res = dict(tasks=tasks, ks=ks, per_task={}, bytes={})
    bases = {}
    for h, other in ((0, 1), (1, 0)):
        for k in ks:
            bases[(h, k)] = pca_basis([data[t][0] for t in halves[other]], k)
            print(f"[compress] basis for half {h} (fit on half {other}), k={k}", flush=True)
    D = data[tasks[0]][0].shape[1]
    for h in (0, 1):
        for t in halves[h]:
            Xs, Ys, Xt, Yt, names = data[t]
            row = {}
            lam = cv_lam(Xs.astype(np.float64), Ys)
            W, mx, my = ridge(Xs.astype(np.float64), Ys, lam)
            row["full"] = dict(err=float(np.median(err_cm((Xt - mx) @ W + my, Yt))), bytes=4 * (W.size + mx.size + my.size))
            W16 = W.astype(np.float16).astype(np.float64)
            row["full_fp16"] = dict(err=float(np.median(err_cm((Xt - mx) @ W16 + my, Yt))), bytes=2 * (W.size + mx.size + my.size))
            for k in ks:
                mu, V = bases[(h, k)]
                Zs, Zt = (Xs - mu) @ V.T, (Xt - mu) @ V.T
                lam = cv_lam(Zs.astype(np.float64), Ys)
                W, mx, my = ridge(Zs.astype(np.float64), Ys, lam)
                row[f"pca{k}"] = dict(err=float(np.median(err_cm((Zt - mx) @ W + my, Yt))), bytes=2 * (W.size + mx.size + my.size))
                P = np.random.default_rng(1234).standard_normal((D, k)).astype(np.float32) / np.sqrt(k)
                Zs, Zt = Xs @ P, Xt @ P
                lam = cv_lam(Zs.astype(np.float64), Ys)
                W, mx, my = ridge(Zs.astype(np.float64), Ys, lam)
                row[f"rand{k}"] = dict(err=float(np.median(err_cm((Zt - mx) @ W + my, Yt))), bytes=2 * (W.size + mx.size + my.size))
            row["n_objects"] = len(names)
            res["per_task"][str(t)] = row
            print(f"[compress] task {t}: " + ", ".join(f"{m} {v['err']:.2f} cm/{v['bytes'] / 1e3:.1f} KB" for m, v in row.items()
                                                       if isinstance(v, dict)), flush=True)
    methods = [m for m in res["per_task"][str(tasks[0])] if m != "n_objects"]
    res["summary"] = {m: dict(median_err_cm=float(np.median([r[m]["err"] for r in res["per_task"].values()])),
                              p90_err_cm=float(np.percentile([r[m]["err"] for r in res["per_task"].values()], 90)),
                              mean_bytes=float(np.mean([r[m]["bytes"] for r in res["per_task"].values()]))) for m in methods}
    res["bytes"]["shared_basis_fp16"] = {str(k): 2 * k * (D + 1) for k in ks}
    for m, v in res["summary"].items():
        print(f"[compress] {m:10s} median {v['median_err_cm']:.3f} cm (p90 {v['p90_err_cm']:.3f}) at {v['mean_bytes'] / 1e3:.1f} KB per task")
    out = f"{R}/p6_compress.json" if a.ks == "64,128,256,512" else f"{R}/p6_compress_k{a.ks.replace(',', '_')}.json"
    json.dump(res, open(out, "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p6_reader", bench="libero_90", tag="compress", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p6_compress.py", command=" ".join(sys.argv), artifacts=[out])


if __name__ == "__main__":
    main()
