"""Plan E2, screen (EXPLORATORY; docs/PLAN_FACTORED.md): INSTANCE children of an object concept.  E1 left 45 of its 58 missing
successes on four tasks that name an instance of the black bowl ("the middle black bowl", "the black bowl at the back"): the
type prototype, pooled over 14 demonstrations of black bowls, does not pick the named one.  Cobweb's hierarchy: a type concept has
children, and an instance is served by the most specific concept that matches it.  A step whose object phrase names an instance
(primitives.instance_key: the object key and its qualifier) gets the CHILD prototype, consolidated over every LIBERO-90
demonstration with that object AND that qualifier; a child starts from its parent's prototype and specialises on its own
demonstrations (c0_fit --init).  Control: the child fitted from scratch.

Arms on the stream's instance tasks, stored init states 0-9, noise seed 12 (paired with E1's acceptance runs: zero-shot, the type
library, the per-task LoRA):
    child    fac@<child from the parent>.<goal>
    scratch  fac@<child from scratch>.<goal>

    ~/flash/miniconda3/envs/chunks/bin/python experiments/e2_instance.py --plan --fit
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e2_instance.py --launch
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e2_instance.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "e2_plan.json")
SEED, RANK = 12, 4


def plan():
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts = r3p.libero90_texts()[0]
    e1 = json.load(open(os.path.join(R, "e1_plan.json")))
    by_inst = {}
    for t, x in texts.items():
        for c in P.steps_of(x):
            k = P.instance_key(c)
            if k:
                by_inst.setdefault(k, []).append(int(t))
    kids = sorted({P.instance_key(P.steps_of(texts[int(t)])[0]) for t in e1} - {None})
    out = {}
    for t, v in e1.items():
        k = P.instance_key(P.steps_of(texts[int(t)])[0])
        if k is None:
            continue
        i = kids.index(k)
        out[t] = dict(instance_key=k, object_key=v["object_key"], parent=v["obj_name"], goal_name=v["goal_name"],
                      child_tasks=sorted(set(by_inst[k])), child=f"Ich{i}", scratch=f"Sch{i}")
    json.dump(out, open(PLAN, "w"), indent=1)
    for t, v in out.items():
        print(f"[e2] task {t}: {v['instance_key']} <- {v['child_tasks']} (parent {v['parent']})", flush=True)


def sb(script):
    out = subprocess.run(["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short",
                          f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def fit():
    done = set()
    for v in json.load(open(PLAN)).values():
        for name, init in ((v["child"], f" --init prim_{v['parent']}_0.pt"), (v["scratch"], "")):
            if name in done or os.path.exists(os.path.join(R, f"prim_{name}_0.pt")):
                continue
            done.add(name)
            print(name, v["instance_key"], sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, v['child_tasks']))} --key "
                                              f"{v['object_key'].replace(' ', '_')} --name {name} --rank {RANK} --approach{init}"), flush=True)


def arms(v):
    return {"child": f"fac@{v['child']}.{v['goal_name']}", "scratch": f"fac@{v['scratch']}.{v['goal_name']}"}


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch():
    for t, v in json.load(open(PLAN)).items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", "e2_instance",
               "--first", "0", "--trials", "10", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-160:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return [e for v in json.load(open(f))["tasks"].values() for e in v["episodes"]]


def read(record):
    pl = json.load(open(PLAN))
    e1 = json.load(open(os.path.join(R, "e1_plan.json")))
    per, missing = {}, []
    for t, v in pl.items():
        row = {}
        for k, a in list(arms(v).items()) + [("zs", "zs"), ("type", f"fac@{e1[t]['obj_name']}.{e1[t]['goal_name']}"), ("lora", "lorasc")]:
            e = load("e2_instance" if k in ("child", "scratch") else "e1_accept", t, a)
            if e is None:
                missing.append(f"{t}:{k}")
                continue
            row[k] = sum(bool(x["success"]) for x in e)
        per[t] = row
    tot = {k: sum(r.get(k, 0) for r in per.values()) for k in ("zs", "type", "child", "scratch", "lora")}
    res = dict(per_task=per, totals=tot, missing=missing, plan=pl, episodes_per_task=10)
    print("[e2] totals", tot, ("missing " + str(missing)) if missing else "", flush=True)
    for t, r in per.items():
        print(f"[e2]   {t} {pl[t]['instance_key']}: {r}", flush=True)
    json.dump(res, open(os.path.join(R, "e2_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("e2_instance", bench="libero_90", tag="screen", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/e2_instance.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
