"""The overhaul's online binding read: when the instruction names object X, which object does the policy move?

Reads light recordings with every object's pose (launch_pack --record --light --all-objects; results/robot/
<experiment>_<task>_<tag>_rec/task<t>_trial<k>.npz).  An episode BINDS object j if j is the first movable object displaced
more than 3 cm from its start while the policy's gripper command is closed (action[6] > 0).  Binding accuracy BA = share of
episodes that bind the named object; habit = share that bind the object the scene's trained / readable sibling task moves.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/f1_binding.py --experiment f1_bind [--record]
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
from lifelong_ood.policy import factors as F                                   # noqa: E402

R = os.path.join(ROOT, "results", "robot")
MOVE = 0.03


def bound(z, types):
    """-> the type of the first object displaced > MOVE while the gripper command is closed (None if none)."""
    P = np.asarray(z["obj_pos"], np.float64)                 # [T, N, 3]
    a = np.asarray(z["action"], np.float64)
    T = min(len(P), len(a))
    closed = a[:T, 6] > 0
    disp = np.linalg.norm(P[:T] - P[0][None], axis=-1)       # [T, N]
    for t in range(T):
        if closed[t]:
            moved = [j for j in range(P.shape[1]) if disp[t, j] > MOVE]
            if moved:
                j = max(moved, key=lambda j: disp[t, j])
                return types[j]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="f1_bind")
    ap.add_argument("--suite", default="libero_90")
    a, _ = ap.parse_known_args()
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    res = {}
    for d in sorted(glob.glob(os.path.join(R, f"{a.experiment}_*_rec"))):
        m = re.match(rf"{a.experiment}_(?:{a.suite}_)?(\d+)_(.+)_rec$", os.path.basename(d))
        if not m:
            continue
        tid, tag = int(m.group(1)), m.group(2)
        t = by[(a.suite, tid)]
        named = tag.split("promptsub_", 1)[1] if tag.startswith("promptsub_") else (t.moved[0] if t.moved else None)
        out = []
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            names = [str(x) for x in z["obj_names"]]
            types = [t.objects.get(n, n) for n in names]
            b = bound(z, types)
            out.append(dict(trial=int(z["trial"]), success=bool(z["success"]), bound=b, named=named))
        if out:
            res[f"{tid}:{tag}"] = dict(task=tid, tag=tag, named=named, n=len(out),
                                       BA=float(np.mean([o["bound"] == named for o in out])),
                                       none=float(np.mean([o["bound"] is None for o in out])),
                                       bound_counts={k: sum(o["bound"] == k for o in out) for k in sorted({str(o["bound"]) for o in out})},
                                       success=float(np.mean([o["success"] for o in out])), episodes=out)
            v = res[f"{tid}:{tag}"]
            print(f"[bind] {tid:>3} {tag:32s} named {named:20s} BA {v['BA']:.2f} none {v['none']:.2f} success {v['success']:.2f} "
                  f"{v['bound_counts']}", flush=True)
    json.dump(res, open(os.path.join(R, f"{a.experiment}_binding.json"), "w"), indent=1)
    if "--record" in sys.argv and res:
        from lifelong_ood import ledger
        print(ledger.record("f1_binding", bench=a.suite, tag=a.experiment, seed=0, metrics=dict(move_m=MOVE, rows=res),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                            script="experiments/f1_binding.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
