"""Plan E's confirmatory read (ledger m3_prereg tag e3_hier; docs/PLAN_FACTORED.md): the HIERARCHICAL factored library against the
per-task LoRA, on a SECOND set of fresh start states that no design decision has seen (results/robot/fresh_states_b, LIBERO's
placement sampler at seeds 200000-200019).

The library is E1's (one object prototype per object key, one goal prototype per goal key, each consolidated over every
LIBERO-90 demonstration of its own scene) with two changes found since: a step that names an INSTANCE of its object
(primitives.instance_key) uses the instance's child prototype, consolidated over the demonstrations with that object and that
qualifier (E2's from-scratch children: Sch0 back, Sch1 front, Sch2 middle); and the tray goal prototype is refit with task 60,
whose goal the parser had misread (Ghtray).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/e3_read.py --plan --fit
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e3_read.py --accept     # stored init states 0-9, seed 13
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e3_read.py --test       # fresh start states (set b), seed 13
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e3_read.py --read [--record]
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
PLAN = os.path.join(R, "e3_plan.json")
SEED, RANK = 13, 4
ORDER = ("zs", "hfac")
TRAY = ("in|the tray", "Ghtray")


def plan():
    texts = E1.texts90()
    e1 = json.load(open(os.path.join(R, "e1_plan.json")))
    e2 = json.load(open(os.path.join(R, "e2_plan.json")))
    tray = sorted({int(t) for t, x in texts.items() for c in P.steps_of(x) if P.goal_key(c) == TRAY[0]})
    out = {}
    for t, v in e1.items():
        c = P.steps_of(texts[int(t)])[0]
        assert P.object_key(c) == v["object_key"] and P.goal_key(c) == v["goal_key"], t
        ik = P.instance_key(c)
        assert (ik is None) == (t not in e2), t
        obj = e2[t]["scratch"] if ik else v["obj_name"]
        goal = TRAY[1] if v["goal_key"] == TRAY[0] else v["goal_name"]
        out[t] = dict(v, instance_key=ik, object_proto=obj, goal_proto=goal,
                      object_tasks=e2[t]["child_tasks"] if ik else v["obj_tasks"],
                      goal_tasks=tray if goal == TRAY[1] else v["goal_tasks"], type_proto=v["obj_name"])
    json.dump(out, open(PLAN, "w"), indent=1)
    protos = {v["object_proto"] for v in out.values()} | {v["goal_proto"] for v in out.values()}
    print(f"[e3] {len(out)} tasks, {len(protos)} prototypes; tray goal from {tray}", flush=True)


def fit():
    if not os.path.exists(os.path.join(R, f"prim_{TRAY[1]}_0.pt")):
        tasks = next(v["goal_tasks"] for v in json.load(open(PLAN)).values() if v["goal_proto"] == TRAY[1])
        print(TRAY[1], tasks, E1.sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, tasks))} --key "
                                    f"{TRAY[0].replace(' ', '_')} --name {TRAY[1]} --rank {RANK} --goal"), flush=True)


def arms(v, test=False):
    a = {"zs": "zs", "hfac": f"fac@{v['object_proto']}.{v['goal_proto']}"}
    if v["lora"]:
        a["lora"] = "lorasc"
    if test and v["instance_key"]:                  # descriptive: the type prototype on the instance tasks
        a["type"] = f"fac@{v['type_proto']}.{v['goal_proto']}"
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, trials, fresh):
    for t, v in json.load(open(PLAN)).items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v, fresh).values()), "--experiment", exp,
               "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-160:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    pl = json.load(open(PLAN))
    acc, kept, kept_lora, test, missing = {}, {}, {}, {}, []
    for t, v in pl.items():
        s = {k: load("e3_accept", t, x) for k, x in arms(v).items()}
        x_ = {k: load("e3_main", t, x) for k, x in arms(v, True).items()}
        if None in s.values() or None in x_.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(z.values())) for k, z in s.items()}
        kept[t] = max(ORDER, key=lambda k: (acc[t][k], -ORDER.index(k)))
        if "lora" in s:
            kept_lora[t] = "lora" if acc[t]["lora"] > acc[t]["zs"] else "zs"
        test[t] = x_
    if missing:
        print("[e3] missing", missing, flush=True)
    C = E1.compare
    h1 = C({t: {"x": x[kept[t]], "y": x["zs"]} for t, x in test.items()})
    h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
    h2 = C({t: {"x": x[kept[t]], "y": x[kept_lora[t]]} for t, x in test.items() if t in kept_lora})
    h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
    desc = {"hfac_vs_zs": C({t: {"x": x["hfac"], "y": x["zs"]} for t, x in test.items()}),
            "lora_vs_zs": C({t: {"x": x["lora"], "y": x["zs"]} for t, x in test.items() if "lora" in x}),
            "hfac_vs_lora": C({t: {"x": x["hfac"], "y": x["lora"]} for t, x in test.items() if "lora" in x}),
            "lora_acc_vs_zs": C({t: {"x": x[kept_lora[t]], "y": x["zs"]} for t, x in test.items() if t in kept_lora}),
            "instance_child_vs_type": C({t: {"x": x["hfac"], "y": x["type"]} for t, x in test.items() if "type" in x}),
            "instance_child_vs_lora": C({t: {"x": x["hfac"], "y": x["lora"]} for t, x in test.items() if "type" in x})}
    by_goal = {}
    for t, x in test.items():
        g = by_goal.setdefault(E1.GOAL_NAME[pl[t]["goal_key"]], {})
        for k, z in list(x.items()) + [("kept", x[kept[t]])] + ([("lora_acc", x[kept_lora[t]])] if t in kept_lora else []):
            g[k] = g.get(k, 0) + int(sum(z.values()))
    protos = {v["object_proto"] for v in pl.values()} | {v["goal_proto"] for v in pl.values()}
    n_lora = sum(bool(v["lora"]) for v in pl.values())
    res = dict(missing=missing, acceptance=acc, kept=kept, kept_lora=kept_lora, hypotheses={"H1": h1, "H2": h2}, descriptive=desc,
               by_goal=by_goal, per_task={t: {k: int(sum(z.values())) for k, z in x.items()} | {"kept": kept[t]} for t, x in test.items()},
               bytes=dict(library=len(protos) * E1.PRIM_BYTES, prototypes=len(protos), per_task_lora=n_lora * E1.LORA_BYTES, loras=n_lora))
    print(f"[e3] H1 kept vs zs: {h1['x']} vs {h1['y']} of {h1['n']}, discordant {h1['discordant']}, sign p {h1['sign_p']:.3g} -> "
          f"passes {h1['passes']}", flush=True)
    print(f"[e3] H2 kept vs LoRA+acc: {h2['x']} vs {h2['y']} of {h2['n']} ({h2['tasks']} tasks), diff {h2['diff']['diff']:+.3f} "
          f"[{h2['diff']['lo']:+.3f}, {h2['diff']['hi']:+.3f}] -> non-inferior {h2['noninferior']}", flush=True)
    for k, v in desc.items():
        print(f"[e3]   {k}: {v['x']} vs {v['y']} of {v['n']}, diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}]",
              flush=True)
    print("[e3] kept", {k: sum(v == k for v in kept.values()) for k in ORDER}, "| by goal", by_goal, flush=True)
    json.dump(res, open(os.path.join(R, "e3_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("e3_hier", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=res["bytes"]["library"], tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=res["bytes"]["library"]),
                            config={}, script="experiments/e3_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("e3_accept", 10, False)
    if "--test" in sys.argv:
        launch("e3_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
