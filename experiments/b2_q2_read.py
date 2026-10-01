"""The Q2-mini read (ledger m3_prereg tags b_bind_q2 and b_bind_q2_amend1): binding accuracy of the attention arms (bind@...),
the role arms (role, roleplacebo) and the frozen policy (zs) on practice states, from the client's bind tracking
(results/robot/<experiment>_<task>_<arm>.json, episodes[].bound = the first movable object displaced > 3 cm while the
commanded gripper is closed).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/b2_q2_read.py [--experiment b_q2mini] [--record]
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import bind_hooks as BH          # noqa: E402
from lifelong_ood.policy import factors as F              # noqa: E402
from lifelong_ood.policy import roles as RO               # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def main():
    exp = sys.argv[sys.argv.index("--experiment") + 1] if "--experiment" in sys.argv else "b_q2mini"
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    zs = {}
    for f in glob.glob(os.path.join(ROOT, "ledger", "zeroshot90", "*.json")):
        zs.update({int(k): v["success_rate"] for k, v in json.load(open(f))["metrics"]["tasks"].items()})
    bind0 = json.load(open(os.path.join(R, "f1_bind_libero_90_binding.json")))
    rows = {}
    for f in sorted(glob.glob(os.path.join(R, f"{exp}_*.json"))):
        m = re.match(rf"{exp}_(\d+)_(.+)\.json$", os.path.basename(f))
        if not m:
            continue
        tid, arm = int(m.group(1)), m.group(2)
        t = by[("libero_90", tid)]
        named = BH.referents(t)["open"][0]
        eps = [e for v in json.load(open(f))["tasks"].values() for e in v["episodes"]]
        if not eps or "bound" not in eps[0]:
            continue
        placebo = None
        if arm.startswith("roleplacebo"):
            placebo = RO.placebo_code(t, RO.role_code(t, tasks, zs, bind0))["placebo_object"]
        elif arm.startswith("roleshape"):
            placebo = RO.placebo_code(t, RO.role_code(t, tasks, zs, bind0), shape_different=True)["placebo_object"]
        bound = [e["bound"] for e in eps]
        rows[f"{tid}:{arm}"] = dict(
            task=tid, arm=arm, named=named, n=len(eps), BA=float(np.mean([b == named for b in bound])),
            NONE=float(np.mean([b is None for b in bound])), success=float(np.mean([e["success"] for e in eps])),
            bound_counts={str(k): int(sum(b == k for b in bound)) for k in sorted({str(b) for b in bound})},
            **({"placebo_object": placebo, "BA_placebo": float(np.mean([b == placebo for b in bound]))} if placebo else {}))
        v = rows[f"{tid}:{arm}"]
        print(f"[q2] {tid:>3} {arm:24s} named {named:22s} BA {v['BA']:.2f} NONE {v['NONE']:.2f} success {v['success']:.2f} "
              f"{v['bound_counts']}" + (f" | placebo {placebo} {v['BA_placebo']:.2f}" if placebo else ""), flush=True)
    dec = {}
    arms = sorted({v["arm"] for v in rows.values()})
    for arm in arms:
        vs = [v for v in rows.values() if v["arm"] == arm]
        dec[arm] = dict(tasks=len(vs), BA_mean=float(np.mean([v["BA"] for v in vs])),
                        tasks_BA_ge_0_5=int(sum(v["BA"] >= 0.5 for v in vs)),
                        NONE_max=float(max(v["NONE"] for v in vs)), success_mean=float(np.mean([v["success"] for v in vs])),
                        **({"BA_placebo_pooled": float(np.mean([v["BA_placebo"] for v in vs]))} if arm.startswith(("roleplacebo", "roleshape")) else {}))
        print(f"[q2] ARM {arm:24s} {dec[arm]}", flush=True)
    res = dict(experiment=exp, rows=rows, arms=dec)
    json.dump(res, open(os.path.join(R, f"{exp}_read.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("b2_q2", bench="libero_90", tag=exp, seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/b2_q2_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
