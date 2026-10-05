"""Plan E, read E4 (ledger m3_prereg tag e4_stream; docs/PLAN_FACTORED.md): the hierarchical factored library LEARNED AS A STREAM.

The library holds three kinds of concepts, each a rank-4 primitive consolidated over every demonstration it has seen: OBJECT
(type) concepts O:<verb|object> on approach frames, INSTANCE children I:<verb|object|qualifier> on approach frames, GOAL concepts
G:<prep|target> on transport frames (primitives.object_key / instance_key / goal_key).  It starts from the demonstrations of the
LIBERO-90 tasks OUTSIDE the stream (the prior library).  The stream's 30 tasks then arrive one at a time in a seeded order:
  1. compose from the current library: the instance child if the step names an instance and the child exists, else the type
     concept; the goal concept; a missing factor is left to the policy (fac@x.<goal>, fac@<object>.x); with neither, no composition;
  2. the acceptance test (stored init states 0-9, 10 episodes, noise seed 13; zero-shot from experiment e3_accept): keep the
     composition only if it beats the policy -- then NO demonstration is requested;
  3. otherwise request the task's ONE demonstration (its own scene's); it updates every concept on its path (Cobweb): the instance
     child (if named), the type concept and the goal concept are refit on all their demonstrations so far; acceptance again.
At the end each task is SERVED by the final library's composition if its arrival kept the library, else by the policy; served arms
and arrival-time compositions are tested on E3's fresh start states (set b, seed 13), paired with E3's zero-shot and LoRA runs.
Two orders (a, b) run in parallel, each with its own library.  Resumable: state in results/robot/e4_state.json.

    nohup ~/flash/miniconda3/envs/chunks/bin/python experiments/e4_stream.py --run > results/robot/e4_driver.log 2>&1 &
    ~/flash/miniconda3/envs/chunks/bin/python experiments/e4_stream.py --read [--record]
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P              # noqa: E402

spec_ = importlib.util.spec_from_file_location("e1", os.path.join(ROOT, "experiments", "e1_read.py"))
E1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(E1)

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
STATE = os.path.join(R, "e4_state.json")
SEED, RANK = 13, 4
ORDERS = {"a": 4, "b": 5}
KIND = {"O": "--approach", "I": "--approach", "G": "--goal"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout


def jobs_of(out):
    return re.findall(r"job (\d+)", out) or re.findall(r"^(\d+)\s*$", out, flags=re.M)


def running(jobs):
    if not jobs:
        return False
    out = subprocess.run(["squeue", "-h", "-j", ",".join(jobs), "-o", "%i"], capture_output=True, text=True)
    if out.returncode != 0:
        return "Invalid job id" not in out.stderr
    return bool(out.stdout.strip())


def keys_of(texts, t):
    c = P.steps_of(texts[int(t)])[0]
    ik = P.instance_key(c)
    return dict(O=f"O:{P.object_key(c)}", I=f"I:{ik}" if ik else None, G=f"G:{P.goal_key(c)}")


def init():
    from lifelong_ood.policy import data as D
    texts = E1.texts90()
    d90 = D.Demos90()
    stream = sorted(int(t) for t in json.load(open(os.path.join(R, "e1_plan.json"))))
    has = {t for t in texts if d90.episodes_of_task(t)}
    need = sorted({k for t in stream for k in keys_of(texts, t).values() if k})
    owners = {}
    for t, x in texts.items():
        for c in P.steps_of(x):
            ik = P.instance_key(c)
            for k in (f"O:{P.object_key(c)}", f"I:{ik}" if ik else None, f"G:{P.goal_key(c)}"):
                if k in need:
                    owners.setdefault(k, set()).add(int(t))
    st = {"_meta": dict(stream=stream, keys={str(t): keys_of(texts, t) for t in stream}, need=need)}
    for o, seed in ORDERS.items():
        lib = {k: dict(idx=i, tasks=sorted(t for t in owners.get(k, ()) if t not in stream and t in has), name=None, v=0, jobs=[])
               for i, k in enumerate(need)}
        st[o] = dict(order=[int(x) for x in np.random.default_rng(seed).permutation(stream)], i=0, phase="prior", jobs=[],
                     library=lib, arrivals=[], demos=0)
    return st


def name(o, k, e):
    return f"E4{o}{k[0]}{e['idx']}v{e['v']}"


def fit(o, k, e):
    e["v"] += 1
    nm = name(o, k, e)
    key = k[2:] if k[0] != "I" else k[2:].rsplit("|", 1)[0]          # an instance child fits on its object key's steps
    out = sh(["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short",
              f"--export=ALL,ENV=openpi,SCRIPT=experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, e['tasks']))} --key "
              f"{key.replace(' ', '_')} --name {nm} --rank {RANK} {KIND[k[0]]}", "slurm/jobs/robot_gpu.sbatch"])
    e["pending"] = nm
    return jobs_of(out)


def compose(s, ks):
    lib = s["library"]
    obj = None
    if ks["I"] and lib[ks["I"]]["name"]:
        obj = lib[ks["I"]]["name"]
    elif lib[ks["O"]]["name"]:
        obj = lib[ks["O"]]["name"]
    goal = lib[ks["G"]]["name"]
    if obj is None and goal is None:
        return None
    return f"fac@{obj or 'x'}.{goal or 'x'}"


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def succ(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return int(sum(bool(e["success"]) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]))


def pack(t, arms, exp, trials, fresh=False):
    cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(arms), "--experiment", exp, "--first", "0",
           "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
    return jobs_of(sh(cmd + (["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else [])))


def request(o, s, a, ks, has_demo):
    """The arrival's ONE demonstration joins every concept on its path; those concepts are refit."""
    a["demo"] = True
    if not has_demo:
        a["demo_unavailable"] = True
        return []
    s["demos"] += 1
    js = []
    for k in [ks["I"], ks["O"], ks["G"]]:
        if k:
            e = s["library"][k]
            e["tasks"] = sorted(set(e["tasks"]) | {a["task"]})
            js += fit(o, k, e)
    return js


def finish(s, a, comp, acc):
    a["kept"] = comp if comp and acc is not None and acc > a["zs"] else "zs"
    a["kept_names"] = comp if a["kept"] != "zs" else None
    s["i"], s["phase"] = s["i"] + 1, "start"


def step(o, s, meta, has):
    if s["i"] >= len(s["order"]):
        return True
    if running(s["jobs"]):
        return False
    lib = s["library"]
    if s["phase"] == "prior":
        if not any(e["tasks"] for e in lib.values()) or all(e["name"] or not e["tasks"] for e in lib.values()):
            s["phase"] = "start"
            return False
        js = []
        for k, e in lib.items():
            if e["tasks"] and not e["name"]:
                nm = e.get("pending")
                if nm and os.path.exists(os.path.join(R, f"prim_{nm}_0.pt")):
                    e["name"], e["pending"] = nm, None
                else:
                    e["v"] -= 1 if nm else 0
                    js += fit(o, k, e)
        s["jobs"] = js
        log(o, "prior library", {k: e["tasks"] for k, e in lib.items() if e["tasks"]}, js)
        return False
    t = s["order"][s["i"]]
    ks = meta["keys"][str(t)]
    zs = succ("e3_accept", t, "zs")
    if s["phase"] == "start":
        comp = compose(s, ks)
        a = dict(task=t, zs=zs, comp1=comp, library_before={k: e["name"] for k, e in lib.items()})
        s["arrivals"].append(a)
        if comp:
            s["jobs"], s["phase"] = pack(t, [comp], "e4_accept", 10), "accept1"
            log(o, "arrival", t, "try", comp, s["jobs"])
        else:
            s["jobs"], s["phase"] = request(o, s, a, ks, t in has), "fit"
            log(o, "arrival", t, "nothing to compose; demonstration", s["jobs"])
        return False
    a = s["arrivals"][-1]
    if s["phase"] == "accept1":
        a["acc1"] = succ("e4_accept", t, a["comp1"])
        if a["acc1"] is None:
            s["jobs"] = pack(t, [a["comp1"]], "e4_accept", 10)
            return False
        if a["acc1"] > zs:
            a["demo"] = False
            finish(s, a, a["comp1"], a["acc1"])
            log(o, "arrival", t, "kept", a["comp1"], a["acc1"], "vs", zs, "-- no demonstration")
            return False
        s["jobs"], s["phase"] = request(o, s, a, ks, t in has), "fit"
        log(o, "arrival", t, a["acc1"], "vs", zs, "-> demonstration", s["jobs"])
        return False
    if s["phase"] == "fit":
        if a.get("demo_unavailable"):
            finish(s, a, a.get("comp1"), a.get("acc1"))
            log(o, "arrival", t, "no demonstration exists; kept", a["kept"])
            return False
        js = []
        for k in [ks["I"], ks["O"], ks["G"]]:
            if not k or not lib[k].get("pending"):
                continue
            nm = lib[k]["pending"]
            if os.path.exists(os.path.join(R, f"prim_{nm}_0.pt")):
                lib[k]["name"], lib[k]["pending"] = nm, None
            else:
                lib[k]["v"] -= 1
                js += fit(o, k, lib[k])
        if js:
            s["jobs"] = js
            log(o, "arrival", t, "refit missing", js)
            return False
        a["comp2"] = compose(s, ks)
        s["jobs"], s["phase"] = pack(t, [a["comp2"]], "e4_accept", 10), "accept2"
        log(o, "arrival", t, "accept refit", a["comp2"], s["jobs"])
        return False
    if s["phase"] == "accept2":
        a["acc2"] = succ("e4_accept", t, a["comp2"])
        if a["acc2"] is None:
            s["jobs"] = pack(t, [a["comp2"]], "e4_accept", 10)
            return False
        finish(s, a, a["comp2"], a["acc2"])
        log(o, "arrival", t, "kept", a["kept"], a["acc2"], "vs", zs, "(demonstration)")
        return False
    return False


def served(st):
    """-> {order: {task: (served arm, arrival-time arm)}}: the final library's composition where the arrival kept the library."""
    out = {}
    for o in ORDERS:
        s = st[o]
        out[o] = {}
        for a in s["arrivals"]:
            if a["kept"] == "zs":
                out[o][str(a["task"])] = ("zs", "zs")
            else:
                out[o][str(a["task"])] = (compose(s, st["_meta"]["keys"][str(a["task"])]), a["kept"])
    return out


def run():
    from lifelong_ood.policy import data as D
    st = json.load(open(STATE)) if os.path.exists(STATE) else init()
    json.dump(st, open(STATE, "w"), indent=1)
    d90 = D.Demos90()
    has = {t for t in st["_meta"]["stream"] if d90.episodes_of_task(t)}
    while True:
        done = [step(o, st[o], st["_meta"], has) for o in ORDERS]
        json.dump(st, open(STATE, "w"), indent=1)
        if all(done):
            break
        time.sleep(90)
    test = st.setdefault("_test", {})
    if not test.get("jobs"):
        js = []
        for o, sv in served(st).items():
            for t, (fin, arr) in sv.items():
                arms = sorted({fin, arr} - {"zs"})
                if arms:
                    js += pack(t, arms, "e4_main", 20, fresh=True)
        test["jobs"] = js
        json.dump(st, open(STATE, "w"), indent=1)
        log("test packs", len(js))
    while running(test["jobs"]):
        time.sleep(120)
    log("stream done")


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record, state=None, mains=("e4_main",), ledger_name="e4_stream", out="e4_read.json"):
    """-> per order: H1 served vs zero-shot, damage (final vs arrival-time), served vs E3's LoRA + acceptance, demonstrations.
    `mains`: the experiments a served arm's test may live in (W1 reuses E4's runs of an identical arm)."""
    st = json.load(open(state or STATE))

    def test_of(t, arm):
        for m in mains:
            z = load(m, t, arm)
            if z is not None:
                return z
        return None
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    C = E1.compare
    res, missing = {}, []
    for o, sv in served(st).items():
        x = {}
        for t, (fin, arr) in sv.items():
            z = {"zs": load("e3_main", t, "zs"), "fin": load("e3_main", t, "zs") if fin == "zs" else test_of(t, fin),
                 "arr": load("e3_main", t, "zs") if arr == "zs" else test_of(t, arr)}
            if t in e3["kept_lora"]:
                z["lora_acc"] = load("e3_main", t, "zs" if e3["kept_lora"][t] == "zs" else "lorasc")
            if None in z.values():
                missing.append(f"{o}:{t}")
                continue
            x[t] = z
        s = st[o]
        h1 = C({t: {"x": z["fin"], "y": z["zs"]} for t, z in x.items()})
        h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
        kept_lib = [t for t, (fin, arr) in sv.items() if arr != "zs" and t in x]
        dmg = C({t: {"x": x[t]["fin"], "y": x[t]["arr"]} for t in kept_lib if sv[t][0] != sv[t][1]}) if any(
            sv[t][0] != sv[t][1] for t in kept_lib) else None
        if dmg:
            dmg["noninferior"] = bool(dmg["diff"]["lo"] > -0.05)
        vs_lora = C({t: {"x": z["fin"], "y": z["lora_acc"]} for t, z in x.items() if "lora_acc" in z})
        n_demo = sum(bool(a.get("demo")) and not a.get("demo_unavailable") for a in s["arrivals"])
        res[o] = dict(H1=h1, damage=dmg, vs_lora_acc=vs_lora, demos=n_demo, arrivals=len(s["arrivals"]),
                      unavailable=sum(bool(a.get("demo_unavailable")) for a in s["arrivals"]),
                      kept_without_demo=[a["task"] for a in s["arrivals"] if a.get("demo") is False],
                      kept_library=sum(a["kept"] != "zs" for a in s["arrivals"]),
                      prototypes=sum(bool(e["name"]) for e in s["library"].values()),
                      changed_since_arrival=sum(sv[t][0] != sv[t][1] for t in kept_lib),
                      per_task={t: {k: int(sum(v.values())) for k, v in z.items()} for t, z in x.items()},
                      arrivals_log=s["arrivals"])
        print(f"[e4] order {o}: served {h1['x']} vs zs {h1['y']} of {h1['n']} (sign p {h1['sign_p']:.2g}); vs LoRA+acc {vs_lora['x']} vs "
              f"{vs_lora['y']} of {vs_lora['n']}, diff {vs_lora['diff']['diff']:+.3f} [{vs_lora['diff']['lo']:+.3f}, "
              f"{vs_lora['diff']['hi']:+.3f}]; demos {n_demo} of {len(s['arrivals'])}; kept without demo {res[o]['kept_without_demo']}",
              flush=True)
        if dmg:
            print(f"[e4]   damage (final vs arrival-time, {dmg['tasks']} tasks): {dmg['x']} vs {dmg['y']}, diff {dmg['diff']['diff']:+.3f} "
                  f"[{dmg['diff']['lo']:+.3f}, {dmg['diff']['hi']:+.3f}]", flush=True)
    res["missing"] = missing
    json.dump(res, open(os.path.join(R, out), "w"), indent=1)
    if missing:
        print("[e4] missing", missing, flush=True)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record(ledger_name, bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=int(max(res[o]["prototypes"] for o in ORDERS) * E1.PRIM_BYTES), tree=0, codes=0, samples=0,
                                       teacher_copies=0, total=int(max(res[o]["prototypes"] for o in ORDERS) * E1.PRIM_BYTES)),
                            config={}, script=f"experiments/{ledger_name.split('_')[0]}_stream.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--init" in sys.argv:
        st = init()
        json.dump(st, open(STATE, "w"), indent=1)
        for o in ORDERS:
            print(o, st[o]["order"], {k: e["tasks"] for k, e in st[o]["library"].items() if e["tasks"]})
    if "--run" in sys.argv:
        run()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
