"""Plan E, read E5 (ledger m3_prereg tag e5_replicate): the hierarchical factored library REPLICATED on new LIBERO-90 tasks.

The 37 single-step pick-and-place tasks of LIBERO-90 outside the stream (with a demonstration of their own scene), none of them
used by any design decision.  Each is composed from its object's prototype (its instance child when the instruction names one) and
its goal's prototype, each consolidated over every LIBERO-90 demonstration of its concept (E3's prototypes where the concept exists;
new ones fitted the same way: rank 4, c0_fit --joint --approach / --goal).  Arms: zs, hfac, lora (per-task one-demo LoRA on the
task's own scene).  Acceptance on stored states 0-9, 10 episodes, noise seed 14; test on fresh start states (LIBERO's sampler at
seeds 200000-200019; results/robot/fresh_states_b), 20 per task, seed 14.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/e5_read.py --plan
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e5_read.py --fit        # missing prototypes, LoRAs, fresh states
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e5_read.py --accept --test
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e5_read.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P             # noqa: E402

spec_ = importlib.util.spec_from_file_location("e1", os.path.join(ROOT, "experiments", "e1_read.py"))
E1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(E1)

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "e5_plan.json")
SEED, RANK = 14, 4
ORDER = ("zs", "hfac")


def plan():
    from lifelong_ood.policy import data as D
    texts = E1.texts90()
    d90 = D.Demos90()
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    stream = {int(t) for t in e3}
    known = {}                                                   # concept -> E3's prototype
    for v in e3.values():
        known["O:" + v["object_key"]] = v["type_proto"]
        known["G:" + v["goal_key"]] = v["goal_proto"]
        if v["instance_key"]:
            known["I:" + v["instance_key"]] = v["object_proto"]
    members = {}
    for t, x in texts.items():
        if not d90.episodes_of_task(t):
            continue
        for c in P.steps_of(x):
            ik = P.instance_key(c)
            for k in ("O:" + P.object_key(c), "G:" + P.goal_key(c)) + (("I:" + ik,) if ik else ()):
                members.setdefault(k, set()).add(int(t))
    tasks = []
    for t, x in texts.items():
        st = P.steps_of(x)
        if t in stream or len(st) != 1 or not P.object_key(st[0]).startswith("put|") or not d90.episodes_of_task(t):
            continue
        tasks.append(t)
    new, out = {}, {}
    for t in tasks:
        c = P.steps_of(texts[t])[0]
        ik = P.instance_key(c)
        row = dict(object_key=P.object_key(c), goal_key=P.goal_key(c), instance_key=ik)
        for role, k in (("type", "O:" + row["object_key"]), ("goal", "G:" + row["goal_key"])) + ((("child", "I:" + ik),) if ik else ()):
            if k not in known and k not in new:
                new[k] = dict(name=f"X{k[0]}{len(new)}", tasks=sorted(members[k]))
            row[role] = known.get(k) or new[k]["name"]
        row["object_proto"] = row.get("child") or row["type"]
        out[str(t)] = row
    json.dump(dict(tasks=out, new=new), open(PLAN, "w"), indent=1)
    print(f"[e5] {len(out)} tasks; {len(new)} new prototypes: {sorted(new)}", flush=True)


def fit():
    from lifelong_ood import ledger
    pl = json.load(open(PLAN))
    for k, e in pl["new"].items():
        if os.path.exists(os.path.join(R, f"prim_{e['name']}_0.pt")):
            continue
        key = k[2:] if k[0] != "I" else k[2:].rsplit("|", 1)[0]
        flag = "--goal" if k[0] == "G" else "--approach"
        print(e["name"], k, e["tasks"], E1.sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, e['tasks']))} --key "
                                             f"{key.replace(' ', '_')} --name {e['name']} --rank {RANK} {flag}"), flush=True)
    wrong = set(ledger.load("l90_scenes", tag="map")[-1]["metrics"]["wrong_scene_tasks"])
    for t in pl["tasks"]:
        f = f"r3_lora_{t}_sc.pt" if int(t) in wrong else f"r3_lora_{t}.pt"
        if not os.path.exists(os.path.join(R, f)):
            print("lora", t, E1.sb(f"experiments/r3_fewshot.py --task {t}" + (" --tag _sc" if int(t) in wrong else "")), flush=True)
    todo = [t for t in pl["tasks"] if not os.path.exists(os.path.join(R, "fresh_states_b", f"libero_90_{t}.npy"))]
    for t in todo:
        out = subprocess.run(["sbatch", "--parsable", "--time=02:00:00", "--exclude=hal,tars,t1000,vicki,johnny5,c3po",
                              f"--export=ALL,TASKS={t},N=20,BASE=200000,FRESH_DIR=fresh_states_b", "experiments/fresh_states.sbatch"],
                             capture_output=True, text=True, cwd=ROOT)
        print("fresh", t, out.stdout.strip() or out.stderr.strip(), flush=True)


def arms(v):
    return {"zs": "zs", "hfac": f"fac@{v['object_proto']}.{v['goal']}", "lora": "lorasc"}


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, trials, fresh):
    for t, v in json.load(open(PLAN))["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-140:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    pl = json.load(open(PLAN))["tasks"]
    acc, kept, kept_lora, test, missing = {}, {}, {}, {}, []
    for t, v in pl.items():
        s = {k: load("e5_accept", t, x) for k, x in arms(v).items()}
        x_ = {k: load("e5_main", t, x) for k, x in arms(v).items()}
        if None in s.values() or None in x_.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(z.values())) for k, z in s.items()}
        kept[t] = max(ORDER, key=lambda k: (acc[t][k], -ORDER.index(k)))
        kept_lora[t] = "lora" if acc[t]["lora"] > acc[t]["zs"] else "zs"
        test[t] = x_
    C = E1.compare
    h1 = C({t: {"x": x[kept[t]], "y": x["zs"]} for t, x in test.items()})
    h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
    h2 = C({t: {"x": x[kept[t]], "y": x[kept_lora[t]]} for t, x in test.items()})
    h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
    # H3: pooled with E3's 29 tasks (the same library, the same comparison; E3's tasks keep their own seed 13 runs)
    s3 = importlib.util.spec_from_file_location("e3", os.path.join(ROOT, "experiments", "e3_read.py"))
    E3 = importlib.util.module_from_spec(s3)
    s3.loader.exec_module(E3)
    e3r = json.load(open(os.path.join(R, "e3_read.json")))
    e3p = json.load(open(E3.PLAN))
    pool = {}
    for t in e3r["kept_lora"]:
        x = {k: E3.load("e3_main", t, a) for k, a in E3.arms(e3p[t], True).items()}
        pool["e3_" + t] = {"x": x[e3r["kept"][t]], "y": x[e3r["kept_lora"][t]]}
    for t, x in test.items():
        pool["e5_" + t] = {"x": x[kept[t]], "y": x[kept_lora[t]]}
    h3 = C(pool)
    h3["noninferior"] = bool(h3["diff"]["lo"] > -0.05)
    desc = {"hfac_vs_lora": C({t: {"x": x["hfac"], "y": x["lora"]} for t, x in test.items()}),
            "lora_acc_vs_zs": C({t: {"x": x[kept_lora[t]], "y": x["zs"]} for t, x in test.items()})}
    res = dict(missing=missing, acceptance=acc, kept=kept, kept_lora=kept_lora, hypotheses={"H1": h1, "H2": h2, "H3": h3},
               descriptive=desc, per_task={t: {k: int(sum(z.values())) for k, z in x.items()} | {"kept": kept[t]} for t, x in test.items()})
    for h, v in res["hypotheses"].items():
        print(f"[e5] {h}: {v['x']} vs {v['y']} of {v['n']} ({v['tasks']} tasks), diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f}, "
              f"{v['diff']['hi']:+.3f}], sign p {v['sign_p']:.3g}", flush=True)
    print("[e5] kept", {k: sum(v == k for v in kept.values()) for k in ORDER}, ("missing " + str(missing)) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "e5_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("e5_replicate", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config={}, script="experiments/e5_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("e5_accept", 10, False)
    if "--test" in sys.argv:
        launch("e5_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
