"""Plan E, screen E0 (EXPLORATORY; docs/PLAN_FACTORED.md): the OBJECT primitive.  D1 left the whole gap to the per-task LoRA on
the approach (the policy grasps the wrong object: the milk for the orange juice, the alphabet soup for the ketchup).  The object
primitive is the approach's counterpart of the goal primitive: a rank-4 LoRA fitted only on demonstrations' frames BEFORE their
first grasp, keyed by the step's object (verb|object, the instance qualifier left out; primitives.object_key), consolidated over
every LIBERO-90 demonstration with that object, and switched on only until the grasp; the goal primitive takes over from the grasp
to the release (PrimHooks approach_paths; arm fac@<object>.<goal>).

Arms on the 30 stream tasks, stored init states 0-9, noise seed 11 (the D1 acceptance runs give zero-shot, goal, goal + binding and
the per-task LoRA on the same states and noise):
    loo   object and goal primitives fitted on the OTHER tasks' demonstrations (none of this task's)
    all   both fitted on every demonstration with the key, this task's one demonstration included (the library after the stream)
    allo  the all-in object primitive alone (the policy carries)

    ~/flash/miniconda3/envs/chunks/bin/python experiments/e0_object.py --plan
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e0_object.py --fit
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e0_object.py --launch
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e0_object.py --read [--record]
"""
import importlib.util
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "e0_plan.json")
SEED, RANK = 11, 4
GOAL_NAME = {"on top of|the cabinet": "cab", "on|the plate": "plate", "in|the basket": "basket", "in|the tray": "tray"}


def texts90():
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    return r3p.libero90_texts()[0]


def plan():
    texts = texts90()
    d1 = json.load(open(os.path.join(R, "d1_plan.json")))
    by_obj, by_goal = {}, {}
    for t, x in texts.items():
        for c in P.steps_of(x):
            by_obj.setdefault(P.object_key(c), []).append(int(t))
            by_goal.setdefault(P.goal_key(c), []).append(int(t))
    objs = sorted({P.object_key(P.steps_of(texts[int(t)])[0]) for t in d1})
    out = {}
    for t, v in d1.items():
        c = P.steps_of(texts[int(t)])[0]
        ok, gk = P.object_key(c), P.goal_key(c)
        assert gk == v["key"], (t, gk, v["key"])
        out[t] = dict(object_key=ok, goal_key=gk, proto=v["proto"], lora=v["lora"],
                      loo_obj=sorted(set(by_obj[ok]) - {int(t)}), all_obj=sorted(set(by_obj[ok])),
                      loo_goal=v["goal"], all_goal=f"Gall{GOAL_NAME[gk]}", all_goal_tasks=sorted(set(by_goal[gk])),
                      obj_loo_name=f"Oloo{t}", obj_all_name=f"Oall{objs.index(ok)}")
    json.dump(out, open(PLAN, "w"), indent=1)
    print(f"[e0] {len(out)} tasks, {len(objs)} object keys; LOO object donors missing for "
          f"{[t for t, v in out.items() if not v['loo_obj']]}", flush=True)
    return out


def sbatch(name, tasks, key, flag):
    if os.path.exists(os.path.join(R, f"prim_{name}_0.pt")):
        return None
    out = subprocess.run(["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short",
                          f"--export=ALL,ENV=openpi,SCRIPT=experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, tasks))} --key "
                          f"{key.replace(' ', '_')} --name {name} --rank {RANK} {flag}", "slurm/jobs/robot_gpu.sbatch"],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def fit():
    pl = json.load(open(PLAN))
    todo = {}
    for t, v in pl.items():
        if v["loo_obj"]:
            todo[v["obj_loo_name"]] = (v["loo_obj"], v["object_key"], "--approach")
        todo[v["obj_all_name"]] = (v["all_obj"], v["object_key"], "--approach")
        todo[v["all_goal"]] = (v["all_goal_tasks"], v["goal_key"], "--goal")
    for name, (tasks, key, flag) in todo.items():
        print(name, key, tasks, sbatch(name, tasks, key, flag), flush=True)


def arms(v):
    a = {"all": f"fac@{v['obj_all_name']}.{v['all_goal']}", "allo": f"fac@{v['obj_all_name']}.x"}
    if v["loo_obj"]:
        a["loo"] = f"fac@{v['obj_loo_name']}.{v['loo_goal']}"
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch():
    for t, v in json.load(open(PLAN)).items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", "e0_object",
               "--first", "0", "--trials", "10", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-160:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return [e for v in json.load(open(f))["tasks"].values() for e in v["episodes"]]


def read(record):
    pl = json.load(open(PLAN))
    d1 = {t: {"zs": "zs", "goal": f"prims@{v['loo_goal']}.0_goal", "goal_bind": f"prims@{v['loo_goal']}.0_goal_bind", "lora": "lora"}
          for t, v in pl.items()}
    per, missing, grasp = {}, [], {}
    for t, v in pl.items():
        row = {}
        for k, a in arms(v).items():
            e = load("e0_object", t, a)
            if e is None:
                missing.append(f"{t}:{k}")
                continue
            row[k] = sum(bool(x["success"]) for x in e)
            grasp[f"{t}:{k}"] = [x.get("bound") for x in e]
        for k, a in d1[t].items():
            e = load("d1_accept", t, a)
            if e is not None:
                row[k] = sum(bool(x["success"]) for x in e)
                grasp[f"{t}:{k}"] = [x.get("bound") for x in e]
        per[t] = row
    keys = ("zs", "goal", "goal_bind", "lora", "loo", "all", "allo")
    tot = {k: sum(r[k] for r in per.values() if k in r) for k in keys}
    cov = {k: sum(k in r for r in per.values()) for k in keys}
    on = {k: {b: sum(r[b] for r in per.values() if k in r and b in r) for b in ("zs", "lora", k)} for k in ("loo", "all", "allo")}
    by_goal = {}
    for t, r in per.items():
        g = GOAL_NAME[pl[t]["goal_key"]]
        for k, x in r.items():
            by_goal.setdefault(g, {}).setdefault(k, 0)
            by_goal[g][k] += x
    res = dict(per_task=per, totals=tot, tasks=cov, paired=on, by_goal=by_goal, missing=missing, grasp=grasp)
    print("[e0] totals", tot, "tasks", cov, flush=True)
    print("[e0] on each arm's tasks", on, flush=True)
    for g, r in by_goal.items():
        print("[e0]  ", g, r, flush=True)
    if missing:
        print("[e0] missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "e0_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("e0_object", bench="libero_90", tag="screen", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/e0_object.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
