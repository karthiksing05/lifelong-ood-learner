"""PHASE OUTCOMES: how far each recorded episode got along its task's grammar (docs/PLAN_TRANSFER.md, the reach / grasp / carry /
place phases of rules v2) -- a readable failure attribution per arm.

Episodes recorded with experiments/libero_client.py --record --light --objects (states, actions, object poses; no images).  Stages,
each read from the simulator's object poses (analysis only; the policy never sees them):
- reach: the hand within 6 cm of the object's GRASP POINT -- the hand's position in the object's frame at the close in the task's own
  demo (a pan is held by its handle, a bowl by its rim), carried along with the object's current pose;
- lift: the object 3 cm above its starting height;
- bring: the object within 6 cm of its goal -- the object's final pose in the receptacle's frame in the task's own demo
  (results/robot/p0_demo/<t>.npz), carried along with the receptacle's current pose;
- place: the task's success.
An episode's failure stage = the first stage it did not reach.  Tasks without a manipulated object (fixtures) report success only.

    python experiments/phase_outcomes.py results/robot/dev_handoff_*_streamh_*.json [--json out.json]
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import phase as PH                              # noqa: E402
from lifelong_ood.policy.phase import q2m                               # noqa: E402

R = os.path.join(ROOT, "results", "robot")
STAGES = ("reach", "lift", "bring", "place")
REACH, LIFT, BRING = 0.06, 0.03, 0.06


def goal_rel(t, obj, rec):
    """The object's final position in the receptacle's frame in task t's own demo (None if either is missing)."""
    f = os.path.join(R, "p0_demo", f"{t}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    n = [str(x) for x in d["names"]]
    if obj not in n or rec not in n:
        return None
    i, j = n.index(obj), n.index(rec)
    P, Q = np.asarray(d["demo_obj_pos"], np.float64), np.asarray(d["demo_obj_quat"], np.float64)
    return q2m(Q[-1, j]).T @ (P[-1, i] - P[-1, j])


def grasp_point(t, obj):
    """The hand's position in the object's frame at the end of the approach (the grasp binding) in task t's own demo."""
    f = os.path.join(R, "p0_demo", f"{t}_match.npz")
    if not os.path.exists(f) or "actions" not in np.load(f, allow_pickle=True).files:
        return None
    z = PH.at_rest(PH.load_demo(R, t))
    parts = PH.cut(z)
    n = [str(x) for x in z["names"]]
    if parts[0][5] != "obj" or n[parts[0][6]] != obj:
        return None
    e, j = parts[0][4] - 1, parts[0][6]
    return q2m(z["obj_quat"][e, j]).T @ (np.asarray(z["eef_pos"][e]) - np.asarray(z["obj_pos"][e, j]))


def episode(z, obj, rec, rel, gp=None):
    """-> the stages reached (a tuple of booleans in STAGES order)."""
    n = [str(x) for x in z["obj_names"]]
    if obj not in n:
        return None
    i = n.index(obj)
    P = np.asarray(z["obj_pos"], np.float64)
    eef = np.asarray(z["state"], np.float64)[:, :3]
    if gp is not None:
        Qo = np.asarray(z["obj_quat"], np.float64)
        target = np.stack([P[s, i] + q2m(Qo[s, i]) @ gp for s in range(len(P))])
    else:
        target = P[:, i]
    reach = bool((np.linalg.norm(eef - target, axis=-1) < REACH).any())
    lift = bool((P[:, i, 2] - P[0, i, 2] > LIFT).any())
    bring = False
    if rel is not None and rec in n:
        j = n.index(rec)
        Q = np.asarray(z["obj_quat"], np.float64)
        goal = np.stack([P[s, j] + q2m(Q[s, j]) @ rel for s in range(len(P))])
        bring = bool((np.linalg.norm(P[:, i] - goal, axis=-1) < BRING).any())
    succ = bool(z["success"])
    return (reach or succ, lift or succ, bring or succ, succ)


def outcomes(result_json, pool):
    t = int(re.match(r".*?_(\d+)_[a-z]", os.path.basename(result_json)).group(1))
    v = pool.get(str(t), {})
    obj, rec = v.get("object"), v.get("receptacle")
    rdir = result_json[:-5] + "_rec"
    rows = []
    for f in sorted(glob.glob(os.path.join(rdir, f"task{t}_trial*.npz"))):
        z = np.load(f, allow_pickle=True)
        if obj is None or "obj_names" not in z.files:
            rows.append((None, None, None, bool(z["success"])))
            continue
        rows.append(episode(z, obj, rec, goal_rel(t, obj, rec), grasp_point(t, obj)))
    return t, rows


def stage_of(r):
    return next((s for s, ok in zip(STAGES, r) if ok is False), "success")


def main():
    files = [f for a in sys.argv[1:] if not a.startswith("--") for f in sorted(glob.glob(a))]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    res, tot = {}, {s: 0 for s in STAGES + ("success",)}
    for f in files:
        if f.endswith("_rec") or not f.endswith(".json"):
            continue
        t, rows = outcomes(f, pool)
        if not rows:
            continue
        c = {s: 0 for s in STAGES + ("success",)}
        for r in rows:
            c[stage_of(r)] += 1
        for k in c:
            tot[k] += c[k]
        res[os.path.basename(f)[:-5]] = dict(task=t, n=len(rows), failed_at=c)
        print(f"{os.path.basename(f)[:-5]:44s} n={len(rows):2d}  " + "  ".join(f"{s} {c[s]}" for s in STAGES + ("success",)))
    print("total: " + "  ".join(f"failed at {s} {tot[s]}" for s in STAGES) + f"  success {tot['success']}")
    if out:
        json.dump(dict(per_run=res, total=tot), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
