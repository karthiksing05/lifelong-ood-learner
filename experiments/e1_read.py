"""Plan E's main read (ledger m3_prereg tag e1_factored; docs/PLAN_FACTORED.md): the FACTORED LIBRARY against the per-task LoRA.

Every task is the composition of two consolidated prototypes, both fitted on each task's OWN scene's demonstration
(data.episodes_of_task; ledger l90_scenes): its OBJECT primitive (rank-4 LoRA on the approach frames of every LIBERO-90
demonstration with its object key, until the policy's grasp) and its GOAL primitive (rank-4 LoRA on the transport frames of every
demonstration with its goal key, from the grasp to the release).  The 30 stream tasks need 15 object and 4 goal prototypes.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/e1_read.py --plan
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e1_read.py --fit        # library + scene-correct LoRAs (sbatch)
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e1_read.py --accept     # stored init states 0-9, seed 12
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e1_read.py --test       # fresh start states, seed 12
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e1_read.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402
from lifelong_ood.policy import primitives as P             # noqa: E402

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "e1_plan.json")
SEED, RANK = 12, 4
ORDER = ("zs", "fac")
GOAL_NAME = {"on top of|the cabinet": "cab", "on|the plate": "plate", "in|the basket": "basket", "in|the tray": "tray"}
PRIM_BYTES, LORA_BYTES = 4677632, 18808599


def texts90():
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    return r3p.libero90_texts()[0]


def plan():
    texts = texts90()
    stream = json.load(open(os.path.join(R, "d1_plan.json")))
    by_obj, by_goal = {}, {}
    for t, x in texts.items():
        for c in P.steps_of(x):
            by_obj.setdefault(P.object_key(c), []).append(int(t))
            by_goal.setdefault(P.goal_key(c), []).append(int(t))
    objs = sorted({P.object_key(P.steps_of(texts[int(t)])[0]) for t in stream})
    out = {}
    for t, v in stream.items():
        c = P.steps_of(texts[int(t)])[0]
        ok, gk = P.object_key(c), P.goal_key(c)
        out[t] = dict(object_key=ok, goal_key=gk, lora=v["lora"], obj_name=f"Osc{objs.index(ok)}", obj_tasks=sorted(set(by_obj[ok])),
                      goal_name=f"Gsc{GOAL_NAME[gk]}", goal_tasks=sorted(set(by_goal[gk])))
    json.dump(out, open(PLAN, "w"), indent=1)
    print(f"[e1] {len(out)} tasks, {len({v['obj_name'] for v in out.values()})} object and {len({v['goal_name'] for v in out.values()})} "
          f"goal prototypes", flush=True)


def sb(script):
    out = subprocess.run(["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short",
                          f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def fit():
    from lifelong_ood import ledger
    pl = json.load(open(PLAN))
    todo = {}
    for v in pl.values():
        todo[v["obj_name"]] = (v["obj_tasks"], v["object_key"], "--approach")
        todo[v["goal_name"]] = (v["goal_tasks"], v["goal_key"], "--goal")
    for name, (tasks, key, flag) in todo.items():
        if not os.path.exists(os.path.join(R, f"prim_{name}_0.pt")):
            print(name, key, tasks, sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, tasks))} --key {key.replace(' ', '_')} "
                                       f"--name {name} --rank {RANK} {flag}"), flush=True)
    wrong = set(ledger.load("l90_scenes", tag="map")[-1]["metrics"]["wrong_scene_tasks"])
    for t, v in pl.items():
        if v["lora"] and int(t) in wrong and not os.path.exists(os.path.join(R, f"r3_lora_{t}_sc.pt")):
            print("lora", t, sb(f"experiments/r3_fewshot.py --task {t} --tag _sc"), flush=True)


def arms(v):
    a = {"zs": "zs", "fac": f"fac@{v['obj_name']}.{v['goal_name']}"}
    if v["lora"]:
        a["lora"] = "lorasc"
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, trials, fresh):
    for t, v in json.load(open(PLAN)).items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        print(t, subprocess.run(cmd + (["--fresh"] if fresh else []), capture_output=True, text=True, cwd=ROOT).stdout.strip()[-160:],
              flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def compare(tab):
    pairs = stats.paired(tab, "x", "y", list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    return dict(x=int(sum(sum(v[0]) for v in pairs.values())), y=int(sum(sum(v[1]) for v in pairs.values())),
                n=int(sum(len(v[0]) for v in pairs.values())), tasks=len(pairs), diff=d, discordant=[int(b), int(c)],
                sign_p=float(stats.sign_test(b, c)))


def read(record):
    pl = json.load(open(PLAN))
    acc, kept, kept_lora, test, missing = {}, {}, {}, {}, []
    for t, v in pl.items():
        a = arms(v)
        s = {k: load("e1_accept", t, x) for k, x in a.items()}
        x_ = {k: load("e1_main", t, x) for k, x in a.items()}
        if None in s.values() or None in x_.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(z.values())) for k, z in s.items()}
        kept[t] = max(ORDER, key=lambda k: (acc[t][k], -ORDER.index(k)))
        if "lora" in a:
            kept_lora[t] = "lora" if acc[t]["lora"] > acc[t]["zs"] else "zs"
        test[t] = x_
    if missing:
        print("[e1] missing", missing, flush=True)
    h1 = compare({t: {"x": x[kept[t]], "y": x["zs"]} for t, x in test.items()})
    h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
    h2 = compare({t: {"x": x[kept[t]], "y": x[kept_lora[t]]} for t, x in test.items() if t in kept_lora})
    h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
    desc = {"fac_vs_zs": compare({t: {"x": x["fac"], "y": x["zs"]} for t, x in test.items()}),
            "lora_vs_zs": compare({t: {"x": x["lora"], "y": x["zs"]} for t, x in test.items() if "lora" in x}),
            "fac_vs_lora": compare({t: {"x": x["fac"], "y": x["lora"]} for t, x in test.items() if "lora" in x}),
            "lora_acc_vs_zs": compare({t: {"x": x[kept_lora[t]], "y": x["zs"]} for t, x in test.items() if t in kept_lora})}
    by_goal = {}
    for t, x in test.items():
        g = by_goal.setdefault(GOAL_NAME[pl[t]["goal_key"]], {})
        for k, z in list(x.items()) + [("kept", x[kept[t]])] + ([("lora_acc", x[kept_lora[t]])] if t in kept_lora else []):
            g[k] = g.get(k, 0) + int(sum(z.values()))
    lib = {v["obj_name"] for v in pl.values()} | {v["goal_name"] for v in pl.values()}
    n_lora = sum(bool(v["lora"]) for v in pl.values())
    res = dict(missing=missing, acceptance=acc, kept=kept, kept_lora=kept_lora, hypotheses={"H1": h1, "H2": h2}, descriptive=desc,
               by_goal=by_goal, per_task={t: {k: int(sum(z.values())) for k, z in x.items()} | {"kept": kept[t]} for t, x in test.items()},
               bytes=dict(library=len(lib) * PRIM_BYTES, prototypes=len(lib), per_task_lora=n_lora * LORA_BYTES, loras=n_lora))
    print(f"[e1] H1 kept vs zs: {h1['x']} vs {h1['y']} of {h1['n']}, discordant {h1['discordant']}, sign p {h1['sign_p']:.3g} -> "
          f"passes {h1['passes']}", flush=True)
    print(f"[e1] H2 kept vs LoRA+acc: {h2['x']} vs {h2['y']} of {h2['n']} ({h2['tasks']} tasks), diff {h2['diff']['diff']:+.3f} "
          f"[{h2['diff']['lo']:+.3f}, {h2['diff']['hi']:+.3f}] -> non-inferior {h2['noninferior']}", flush=True)
    for k, v in desc.items():
        print(f"[e1]   {k}: {v['x']} vs {v['y']} of {v['n']}, diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}]",
              flush=True)
    print("[e1] kept", {k: sum(v == k for v in kept.values()) for k in ORDER}, "| by goal", by_goal, flush=True)
    json.dump(res, open(os.path.join(R, "e1_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("e1_factored", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=res["bytes"]["library"], tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=res["bytes"]["library"]),
                            config={}, script="experiments/e1_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("e1_accept", 10, False)
    if "--test" in sys.argv:
        launch("e1_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
