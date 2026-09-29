"""P6 follow-up (offline): does re-reading the grasped object when the hand is NEAR it beat the first-frame reader?

The first-frame reader (experiments/p6_reader.py) misses the thin bottles by about a centimetre, and the loss of P6 sits there.
Near the object the wrist camera sees it large, so the object's position relative to the hand should be easier to read.  Per task:
the anchor object of the demo's approach (lifelong_ood/policy/phase.py rules v2); training frames = the pre-contact frames (the
object has not moved 2 mm yet) of the recorded practice episodes of init states 0-19 with the hand 2-20 cm from its grasp point
(where it first closes);
target = the object's position in the HAND's frame (from the bit-exact replays, experiments/p0_replay.py); features = frozen
ResNet-18 of both cameras (VinnPolicy._embed), standardized ridge with the penalty by grouped cross-validation (p6_reader.py).
Test = init states 20-29 at the first frame the hand comes within `--near` of its grasp point: absolute error of hand + R rel against
the replay; the first-frame reader's error on the same episodes for comparison (trained on states 0-19, as p6_reader's offline).

    /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/p6_near_reader.py [--tasks 52+62] [--record]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import phase as PH                               # noqa: E402

R = os.path.join(ROOT, "results", "robot")
P1_TASKS = [42, 32, 1, 35, 40, 36, 52, 62, 41]
EXPS = ("m5_practice", "m6_round1", "m6_round2")
spec = importlib.util.spec_from_file_location("p6r", os.path.join(ROOT, "experiments", "p6_reader.py"))
P6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P6)


def episodes(t, j, enc, lo, hi, cache):
    """Pre-contact frames with the hand lo..hi m from its grasp point (where it first closes before object j moves): (features, rel target, hand pos, hand rot, object pos, trial, dist)."""
    if os.path.exists(cache):
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    out = {k: [] for k in ("X", "rel", "p", "Rm", "obj", "trial", "d")}
    for e in EXPS:
        for f in sorted(glob.glob(f"{R}/{e}_{t}_*_rec/task{t}_trial*.npz")):
            k = int(f.rsplit("trial", 1)[1][:-4])
            rp = f"{R}/p0_replay/{e}_{t}_trial{k}.npz"
            if not os.path.exists(rp):
                continue
            z = np.load(rp, allow_pickle=True)
            if not bool(z["exact"]):
                continue
            o0 = z["obj_pos"][0, j].astype(np.float64)
            moved = np.where(np.linalg.norm(z["obj_pos"][:, j] - o0, axis=-1) > 0.002)[0]
            m = int(moved[0]) if len(moved) else len(z["obj_pos"])
            closes = np.where(np.asarray(z["action"])[:m, 6] > 0)[0]
            gp = z["eef_pos"][int(closes[0]) if len(closes) else max(m - 1, 0)]      # where the hand closes (a handle, a knob)
            d = np.linalg.norm(z["eef_pos"][:m] - gp, axis=-1)
            sel = np.where((d >= lo) & (d <= hi))[0]
            if not len(sel):
                continue
            rec = np.load(f)
            F = enc._embed(rec["image"][sel], rec["wrist"][sel]).float().cpu().numpy().astype(np.float64)
            for i, s in enumerate(sel):
                Rm = PH.q2m(z["eef_quat"][s])
                out["X"].append(F[i]); out["rel"].append(Rm.T @ (o0 - z["eef_pos"][s])); out["p"].append(z["eef_pos"][s])
                out["Rm"].append(Rm); out["obj"].append(o0); out["trial"].append(k); out["d"].append(d[s])
    out = {k: np.asarray(v) for k, v in out.items()}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="+".join(map(str, P1_TASKS)))
    ap.add_argument("--near", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    from lifelong_ood.policy.baselines import VinnPolicy
    enc = VinnPolicy(np.zeros((1, 7), np.float32), np.zeros((1, 224, 224, 3), np.uint8), np.zeros((1, 224, 224, 3), np.uint8), device=a.device)
    res = dict(near_m=a.near, per_task={})
    for t in [int(x) for x in a.tasks.split("+")]:
        z = PH.at_rest(PH.load_demo(R, t))
        j = PH.cut(z)[0][6]
        name = str(z["names"][j])
        E = episodes(t, j, enc, 0.02, 0.20, f"{R}/p6_feats/near2_{t}.npz")
        tr = E["trial"] < 20
        lam, _ = P6.cv_lam(E["X"][tr], E["rel"][tr], E["trial"][tr])
        W, mx, my = P6.ridge(E["X"][tr], E["rel"][tr], lam)
        errs, trials = [], []
        for k in sorted(set(E["trial"][~tr].tolist())):
            idx = np.where((E["trial"] == k) & (E["d"] <= a.near))[0]
            if not len(idx):
                continue
            i = idx[0]                                         # the first frame the hand is within `near` (frames are in time order)
            rel = (E["X"][i] - mx) @ W + my
            errs.append(np.linalg.norm(E["p"][i] + E["Rm"][i] @ rel - E["obj"][i]) * 100)
            trials.append(k)
        # the first-frame reader on the same episodes (trained on states 0-19)
        X1, Y1, tr1, st1, names1 = P6.features(t, enc, 20)
        ok = tr1 < 20
        lam1, _ = P6.cv_lam(X1[ok], Y1[ok], tr1[ok])
        W1, mx1, my1 = P6.ridge(X1[ok], Y1[ok], lam1)
        jj = names1.index(name)
        first = []
        for k in trials:
            s = np.where((tr1 == k) & (st1 == 0))[0][0]
            first.append(np.linalg.norm(((X1[s] - mx1) @ W1 + my1).reshape(-1, 3)[jj] - Y1[s].reshape(-1, 3)[jj]) * 100)
        r = dict(object=name, lam=lam, train_frames=int(tr.sum()), test_episodes=len(errs),
                 near_err_cm_median=float(np.median(errs)) if errs else None, near_err_cm_max=float(np.max(errs)) if errs else None,
                 first_err_cm_median=float(np.median(first)) if first else None, first_err_cm_max=float(np.max(first)) if first else None)
        res["per_task"][str(t)] = r
        if not errs:
            print(f"[p6-near] task {t} {name}: the hand never comes within {a.near} m of its grasp point (no test episodes)", flush=True)
            continue
        print(f"[p6-near] task {t} {name}: near read {r['near_err_cm_median']:.2f} cm (max {r['near_err_cm_max']:.2f}) vs first frame "
              f"{r['first_err_cm_median']:.2f} cm (max {r['first_err_cm_max']:.2f}) on {len(errs)} test episodes; {r['train_frames']} "
              f"training frames, lam {lam:g}", flush=True)
    v = [r for r in res["per_task"].values() if r["near_err_cm_median"] is not None]
    res["median_near_cm"] = float(np.median([r["near_err_cm_median"] for r in v]))
    res["median_first_cm"] = float(np.median([r["first_err_cm_median"] for r in v]))
    print(f"[p6-near] median over tasks: near {res['median_near_cm']:.2f} cm vs first frame {res['median_first_cm']:.2f} cm")
    json.dump(res, open(f"{R}/p6_near_reader.json", "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p6_reader", bench="libero_90", tag="near_offline", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p6_near_reader.py", command=" ".join(sys.argv), artifacts=[f"{R}/p6_near_reader.json"])


if __name__ == "__main__":
    main()
