"""P6 (archive/2026-10-program/PLAN_TRANSFER.md): a LEARNED object-pose reader in place of the simulator oracle.

Per task, a ridge regression from frozen ImageNet ResNet-18 features of both cameras (the encoder of the VINN baseline and of the
instance memory, lifelong_ood/policy/baselines.VinnPolicy._embed; features standardized) to the 3-D positions of the task's objects
of interest.  Training frames: the first 20 steps of every recorded practice episode (m5_practice, m6_round1, m6_round2: init
states 0-29; the objects are static before the first contact), with the positions from the bit-exact replays
(experiments/p0_replay.py).  The penalty is chosen by 5-fold cross-validation grouped by init state on states 0-19; the offline
check tests on the first frame of states 20-29.  The final reader (all of 0-29, the same penalty) is saved for the phasebind server
(results/robot/p6_reader_<t>.npz: W with the scale folded in, feature mean, target mean, the objects' names).  Features are cached
in results/robot/p6_feats/.

    /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/p6_reader.py [--record]
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
TASKS = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
EXPS = ("m5_practice", "m6_round1", "m6_round2")


LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def ridge(X, Y, lam):
    """Standardized ridge; returns (W, mx, my) with pred = (x - mx) @ W + my (the scale folded into W)."""
    mx, my = X.mean(0), Y.mean(0)
    sx = X.std(0) + 1e-8
    Xc, Yc = (X - mx) / sx, Y - my
    if X.shape[1] > X.shape[0]:                       # dual form: more features than frames
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(X.shape[0]), Yc)
    else:
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    return W / sx[:, None], mx, my


def cv_lam(X, Y, trial, folds=5):
    """The penalty with the lowest held-out error (all frames of the held-out states), folds grouped by init state."""
    ks = np.unique(trial)
    err = {}
    for lam in LAMS:
        e = []
        for f in range(folds):
            te = np.isin(trial, ks[f::folds])
            W, mx, my = ridge(X[~te], Y[~te], lam)
            e.append(np.linalg.norm(((X[te] - mx) @ W + my - Y[te]).reshape(int(te.sum()), -1, 3), axis=-1).mean())
        err[lam] = float(np.mean(e))
    return min(err, key=err.get), err


def embed(enc, images, wrist, kind="pooled", pool=4):
    """The reader's features: the global-pooled ResNet-18 of both cameras, or their pooled feature maps (kind spatial)."""
    F = enc._embed(images, wrist) if kind == "pooled" else enc._embed_spatial(images, wrist, pool=pool)
    return F.float().cpu().numpy().astype(np.float64 if kind == "pooled" else np.float32)


def synth_features(t, enc, names, kind="pooled", pool=4, sdir="p6_synth"):
    """The simulator-rendered first frames of task t (experiments/p6_synth.py) and their objects' positions, in `names` order."""
    cache = f"{R}/p6_feats/synth_{t}_{kind}{pool}.npz" if sdir == "p6_synth" else f"{R}/p6_feats/{sdir}_{t}_{kind}{pool}.npz"
    if os.path.exists(cache):
        c = np.load(cache)
        return c["X"].astype(np.float64), c["Y"]
    z = np.load(f"{R}/{sdir}/{t}.npz", allow_pickle=True)
    order = [[str(n) for n in z["names"]].index(n) for n in names]
    X = np.concatenate([embed(enc, z["image"][i:i + 200], z["wrist"][i:i + 200], kind, pool) for i in range(0, len(z["image"]), 200)])
    Y = z["obj_pos"][:, order].reshape(len(X), -1).astype(np.float64)
    np.savez_compressed(cache, X=X, Y=Y)
    return X.astype(np.float64), Y


def features(t, enc, steps, kind="pooled", pool=4):
    """(X, Y, trial, step, names) for task t, cached."""
    cache = f"{R}/p6_feats/{t}_{steps}.npz" if kind == "pooled" else f"{R}/p6_feats/{t}_{steps}_{kind}{pool}.npz"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        return c["X"], c["Y"], c["trial"], c["step"], [str(n) for n in c["names"]]
    X, Y, trial, step = [], [], [], []
    names = None
    init = f"{R}/p6_synth_rj/{t}_init.npz"
    if not glob.glob(f"{R}/m*_{t}_*_rec/task{t}_trial*.npz") and os.path.exists(init):
        # a task never practised: the first frames of init states 0-29 rendered exactly as the client sends them (p6_synth.py
        # --init-states), step 0 only
        z = np.load(init, allow_pickle=True)
        X = embed(enc, z["image"], z["wrist"], kind, pool)
        Y = z["obj_pos"].reshape(len(X), -1).astype(np.float64)
        trial, step, names = np.asarray(z["trial"]), np.zeros(len(X), int), [str(n) for n in z["names"]]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=X, Y=Y, trial=trial, step=step, names=np.array(names))
        return X, Y, trial, step, names
    for e in EXPS:
        for f in sorted(glob.glob(f"{R}/{e}_{t}_*_rec/task{t}_trial*.npz")):
            k = int(f.rsplit("trial", 1)[1][:-4])
            rp = f"{R}/p0_replay/{e}_{t}_trial{k}.npz"
            if not os.path.exists(rp):
                continue
            z = np.load(rp, allow_pickle=True)
            if not bool(z["exact"]):
                continue
            names = [str(n) for n in z["names"]]
            rec = np.load(f)
            n = min(steps, len(rec["image"]))
            F = embed(enc, rec["image"][:n], rec["wrist"][:n], kind, pool)
            X.append(F); Y.append(np.repeat(z["obj_pos"][0].reshape(1, -1), n, 0)); trial += [k] * n; step += list(range(n))
        print(f"[p6]   task {t} {e}: {len(X)} episodes embedded", flush=True)
    X, Y, trial, step = np.concatenate(X), np.concatenate(Y), np.asarray(trial), np.asarray(step)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, X=X, Y=Y, trial=trial, step=step, names=np.array(names))
    return X, Y, trial, step, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", default=None, help="+-separated subset")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tag", default="offline", help="ledger tag (a second task set gets its own tag)")
    ap.add_argument("--features", default="pooled", choices=("pooled", "spatial"))
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--synth", action="store_true", help="also train on the simulator-rendered scenes (experiments/p6_synth.py)")
    ap.add_argument("--synth-dir", default="p6_synth", help="p6_synth_rj: scenes rendered with --robot-init")
    ap.add_argument("--synth-only", action="store_true", help="train on the rendered scenes ONLY (no practice recordings: a task "
                    "composed from other tasks' demos has none of its own); the practice first frames of states 20-29 stay the test set")
    a = ap.parse_args()
    a.synth = a.synth or a.synth_only
    from lifelong_ood.policy.baselines import VinnPolicy
    enc = VinnPolicy(np.zeros((1, 7), np.float32), np.zeros((1, 224, 224, 3), np.uint8), np.zeros((1, 224, 224, 3), np.uint8), device=a.device)
    res = dict(steps=a.steps, lams=list(LAMS), features=a.features, pool=a.pool, synth=a.synth, synth_only=a.synth_only, per_task={})
    for t in ([int(x) for x in a.tasks.split("+")] if a.tasks else TASKS):
        X, Y, trial, step, names = features(t, enc, a.steps, a.features, a.pool)
        X = X.astype(np.float64)
        if a.synth:                                    # rendered scenes: groups 1000+, step 0; never tested on
            Xs, Ys = synth_features(t, enc, names, a.features, a.pool, a.synth_dir)
            X, Y = np.concatenate([X, Xs]), np.concatenate([Y, Ys])
            trial = np.concatenate([trial, 1000 + np.arange(len(Xs))])
            step = np.concatenate([step, np.zeros(len(Xs), int)])
        tr, first = ((trial >= 1000) if a.synth_only else ((trial < 20) | (trial >= 1000))), step == 0
        lam, cverr = cv_lam(X[tr], Y[tr], trial[tr])
        W, mx, my = ridge(X[tr], Y[tr], lam)
        sel = (trial >= 20) & (trial < 1000) & first
        pred = (X[sel] - mx) @ W + my
        err = np.linalg.norm((pred - Y[sel]).reshape(len(pred), -1, 3), axis=-1) * 100            # cm per object
        base = np.linalg.norm((Y[tr].mean(0) - Y[sel]).reshape(len(pred), -1, 3), axis=-1) * 100   # the mean placement
        fin = (trial >= 1000) if a.synth_only else np.ones(len(X), bool)
        W2, mx2, my2 = ridge(X[fin], Y[fin], lam)
        sfx = (("" if a.features == "pooled" else f"_{a.features}{a.pool}") + ("_synthonly" if a.synth_only else "_synth" if a.synth else "")
               + ("" if a.synth_dir == "p6_synth" else "_rj"))
        np.savez_compressed(f"{R}/p6_reader{sfx}_{t}.npz", W=W2, mx=mx2, my=my2, names=np.array(names), features=a.features, pool=a.pool)
        res["per_task"][str(t)] = dict(names=names, lam=lam, cv_err_m={str(k): v for k, v in cverr.items()}, test_frames=int(sel.sum()),
                                       train_frames=int(tr.sum()),
                                       err_cm_median={n: float(np.median(err[:, j])) for j, n in enumerate(names)},
                                       err_cm_max={n: float(err[:, j].max()) for j, n in enumerate(names)},
                                       mean_placement_err_cm_median={n: float(np.median(base[:, j])) for j, n in enumerate(names)})
        print(f"[p6] task {t} (lam {lam:g}): " + ", ".join(f"{n} {np.median(err[:, j]):.1f} cm (max {err[:, j].max():.1f}; mean placement "
                                                         f"{np.median(base[:, j]):.1f})" for j, n in enumerate(names)), flush=True)
    allm = [v for r in res["per_task"].values() for v in r["err_cm_median"].values()]
    res["median_of_medians_cm"] = float(np.median(allm))
    res["mean_placement_median_of_medians_cm"] = float(np.median([v for r in res["per_task"].values()
                                                                  for v in r["mean_placement_err_cm_median"].values()]))
    print(f"[p6] median error {res['median_of_medians_cm']:.2f} cm (mean placement {res['mean_placement_median_of_medians_cm']:.2f} cm)")
    out = f"{R}/p6_reader{'' if a.tag == 'offline' else '_' + a.tag}.json"
    json.dump(res, open(out, "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p6_reader", bench="libero_90", tag=a.tag, seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p6_reader.py", command=" ".join(sys.argv), artifacts=[out])


if __name__ == "__main__":
    main()
