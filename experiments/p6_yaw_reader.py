"""The objects' YAW for the pose reader, where it varies (books: ~13 deg across placements; everything else fixed within ~4 deg).

Per task: for each object whose yaw varies across the task's rendered placements (std > 3 deg; experiments/p6_synth.py --robot-init,
results/robot/p6_synth_rj/<t>.npz obj_rot), a standardized ridge (dual form, penalty by grouped CV) from the same cached spatial
features as the position reader (p6_feats/p6_synth_rj_<t>_spatial4.npz) to (cos, sin) of its yaw; trained on the rendered scenes
only.  Test: the first frames of init states 20-29 rendered as the client sends them (p6_synth_rj/<t>_init.npz).  Saved for the
phasebind server as results/robot/p6_yaw_<t>.npz (W, mx, my, names, varying mask, the rendered mean yaw).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/p6_yaw_reader.py [--tasks 73+74] [--record]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
spec = importlib.util.spec_from_file_location("p6r", os.path.join(ROOT, "experiments", "p6_reader.py"))
P6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P6)


def cv_lam(X, Y, folds=5):
    """The penalty with the lowest held-out angle error, 5 random folds of rendered scenes."""
    idx = np.random.default_rng(0).permutation(len(X))
    err = {}
    for lam in P6.LAMS:
        e = []
        for f in range(folds):
            te = np.zeros(len(X), bool)
            te[idx[f::folds]] = True
            W, mx, my = P6.ridge(X[~te], Y[~te], lam)
            P = (X[te] - mx) @ W + my
            k = Y.shape[1] // 2
            e.append(np.abs(np.angle(np.exp(1j * (np.arctan2(P[:, k:], P[:, :k]) - np.arctan2(Y[te][:, k:], Y[te][:, :k]))))).mean())
        err[lam] = float(np.mean(e))
    return min(err, key=err.get), err


def yaw(Rm):
    return np.arctan2(Rm[..., 1, 0], Rm[..., 0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--min-std", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    tasks = [int(x) for x in a.tasks.split("+")] if a.tasks else sorted(int(t) for t in pool)
    enc = None
    res = dict(min_std_deg=a.min_std, per_task={})
    for t in tasks:
        z = np.load(os.path.join(R, "p6_synth_rj", f"{t}.npz"), allow_pickle=True)
        names = [str(n) for n in z["names"]]
        yw = yaw(z["obj_rot"])                                                   # [n, k]
        std = np.degrees(np.std(np.unwrap(yw, axis=0), axis=0))
        vary = std > a.min_std
        mean_yaw = np.arctan2(np.sin(yw).mean(0), np.cos(yw).mean(0))
        if not vary.any():
            np.savez_compressed(os.path.join(R, f"p6_yaw_{t}.npz"), names=np.array(names), vary=vary, mean_yaw=mean_yaw)
            res["per_task"][str(t)] = dict(varying=[])
            continue
        cache = np.load(os.path.join(R, "p6_feats", f"p6_synth_rj_{t}_spatial4.npz"))
        X = cache["X"].astype(np.float64)
        Y = np.concatenate([np.cos(yw[:, vary]), np.sin(yw[:, vary])], 1)
        lam, _ = cv_lam(X, Y)
        W, mx, my = P6.ridge(X, Y, lam)
        zi = np.load(os.path.join(R, "p6_synth_rj", f"{t}_init.npz"), allow_pickle=True)
        sel = zi["trial"] >= 20
        if enc is None:
            from lifelong_ood.policy.baselines import VinnPolicy
            enc = VinnPolicy(np.zeros((1, 7), np.float32), np.zeros((1, 224, 224, 3), np.uint8), np.zeros((1, 224, 224, 3), np.uint8), device=a.device)
        Xt = P6.embed(enc, zi["image"][sel], zi["wrist"][sel], "spatial", 4).astype(np.float64)
        order = [[str(n) for n in zi["names"]].index(n) for n in names]
        yt = yaw(zi["obj_rot"][sel][:, order])[:, vary]
        P = (Xt - mx) @ W + my
        k = int(vary.sum())
        yp = np.arctan2(P[:, k:], P[:, :k])
        err = np.degrees(np.abs(np.angle(np.exp(1j * (yp - yt)))))
        fixed = np.degrees(np.abs(np.angle(np.exp(1j * (mean_yaw[vary][None] - yt)))))
        np.savez_compressed(os.path.join(R, f"p6_yaw_{t}.npz"), W=W, mx=mx, my=my, names=np.array(names), vary=vary, mean_yaw=mean_yaw)
        vn = [n for n, v in zip(names, vary) if v]
        res["per_task"][str(t)] = dict(varying=vn, lam=lam, err_deg_median={n: float(np.median(err[:, j])) for j, n in enumerate(vn)},
                                       fixed_deg_median={n: float(np.median(fixed[:, j])) for j, n in enumerate(vn)})
        print(f"[yaw] task {t}: " + ", ".join(f"{n} {np.median(err[:, j]):.1f} deg (fixed yaw {np.median(fixed[:, j]):.1f})" for j, n in enumerate(vn)), flush=True)
    e = [v for r in res["per_task"].values() for v in r.get("err_deg_median", {}).values()]
    f = [v for r in res["per_task"].values() for v in r.get("fixed_deg_median", {}).values()]
    res.update(median_deg=float(np.median(e)) if e else None, fixed_median_deg=float(np.median(f)) if f else None)
    print(f"[yaw] median {res['median_deg']:.1f} deg vs {res['fixed_median_deg']:.1f} deg for the fixed yaw")
    json.dump(res, open(os.path.join(R, "p6_yaw_reader.json"), "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p6_reader", bench="libero_90", tag="yaw", seed=0, metrics=res, bytes=dict(experts=0, tree=0, codes=0, samples=0,
                      teacher_copies=0, total=0), config=vars(a), script="experiments/p6_yaw_reader.py", command=" ".join(sys.argv),
                      artifacts=[os.path.join(R, "p6_yaw_reader.json")])


if __name__ == "__main__":
    main()
