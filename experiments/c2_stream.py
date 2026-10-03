"""Plan C, read C2 (ledger m3_prereg tag c2_stream): the STREAM with motion primitives.  Per step key with >= 6 single-step
LIBERO-90 tasks, the tasks arrive one at a time in a seeded order.  An arrival first tries the key's current primitive against
the policy on init states 0-9 (the acceptance test, 10 episodes each); only if the primitive is not kept (or none exists) does it
request its ONE demonstration, which joins the key's library, and the key's primitive is REFIT on every demonstration requested so
far (consolidation; c0_fit --joint, rank 4) and tried again.  At the end, init states 30-49 score each arrival's kept arm, zero-shot,
and the key's FINAL primitive (does consolidation damage earlier arrivals?).  Noise seed 8.  Keys run in parallel; each key's
arrivals run in order.  Resumable: all state in results/robot/c2_state.json.

    nohup ~/flash/miniconda3/envs/chunks/bin/python experiments/c2_stream.py --run > results/robot/c2_driver.log 2>&1 &
    ~/flash/miniconda3/envs/chunks/bin/python experiments/c2_stream.py --read [--record]
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
STATE = os.path.join(R, "c2_state.json")
SEED, RANK = 8, 4
KEYS = {"cab": ("put|on top of|the cabinet", [10, 15, 25, 31, 41, 43]),
        "plate": ("put|on|the plate", [9, 12, 13, 14, 30, 36, 71, 72]),
        "basket": ("put|in|the basket", [46, 47, 48, 49, 50, 51, 52, 53, 54]),
        "tray": ("put|in|the tray", [55, 56, 57, 58, 59, 61, 62])}


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
        if "Invalid job id" in out.stderr:
            return False                                       # finished long enough ago that the scheduler forgot them
        return True                                            # the scheduler did not answer: assume still running
    return bool(out.stdout.strip())


def pack(task, arms, exp, first, trials):
    out = sh([PY, "experiments/launch_pack.py", "--tasks", str(task), "--arms", ",".join(arms), "--experiment", exp, "--first",
              str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"])
    return jobs_of(out)


def fit(name, tasks, key):
    out = sh(["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short",
              f"--export=ALL,ENV=openpi,SCRIPT=experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, tasks))} --key "
              f"{key.replace(' ', '_')} --name {name} --rank {RANK}", "slurm/jobs/robot_gpu.sbatch"])
    return jobs_of(out)


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def succ(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return int(sum(bool(e["success"]) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]))


def order(tasks, k):
    return [int(x) for x in np.random.default_rng(sum(map(ord, k))).permutation(tasks)]


def init():
    st = {}
    for k, (key, tasks) in KEYS.items():
        st[k] = dict(key=key, order=order(tasks, k), library=[], prim=None, n_fit=0, i=0, phase="start", jobs=[], arrivals=[])
    return st


def step(k, s):
    """Advance key k's chain by one phase if its jobs are done.  -> True when the chain is finished."""
    if s["i"] >= len(s["order"]):
        return True
    if running(s["jobs"]):
        return False
    t = s["order"][s["i"]]
    a = s["arrivals"][s["i"]] if len(s["arrivals"]) > s["i"] else None
    if s["phase"] == "start":
        s["arrivals"].append(dict(task=t, library_before=list(s["library"]), prim_before=s["prim"]))
        arms = ["zs"] + ([f"prims@{s['prim']}.0"] if s["prim"] else [])
        s["jobs"], s["phase"] = pack(t, arms, "c2_accept", 0, 10), "accept1"
        log(k, "arrival", t, "accept", arms, s["jobs"])
        return False
    if s["phase"] == "accept1":
        a["zs"] = succ("c2_accept", t, "zs")
        a["prims1"] = succ("c2_accept", t, f"prims@{s['prim']}.0") if s["prim"] else None
        if a["zs"] is None or (s["prim"] and a["prims1"] is None):
            log(k, "arrival", t, "acceptance results missing; relaunching")
            s["phase"] = "start"
            s["arrivals"].pop()
            return False
        if s["prim"] and a["prims1"] > a["zs"]:
            a.update(demo=False, kept=f"prims@{s['prim']}.0")
            log(k, "arrival", t, "kept", s["prim"], a["prims1"], "vs", a["zs"], "-- no demonstration")
            s["i"], s["phase"] = s["i"] + 1, "start"
            return False
        s["library"].append(t)                                     # request the one demonstration; it joins the key's library
        s["n_fit"] += 1
        name = f"C2{k}{s['n_fit']}"
        s["jobs"], s["phase"], a["demo"], a["refit"] = fit(name, s["library"], s["key"]), "fit", True, name
        log(k, "arrival", t, "demonstration requested; refit", name, "on", s["library"], s["jobs"])
        return False
    if s["phase"] == "fit":
        if not os.path.exists(os.path.join(R, f"prim_{a['refit']}_0.pt")):
            log(k, "fit", a["refit"], "missing; refitting")
            s["jobs"] = fit(a["refit"], s["library"], s["key"])
            return False
        s["prim"] = a["refit"]
        s["jobs"], s["phase"] = pack(t, [f"prims@{s['prim']}.0"], "c2_accept", 0, 10), "accept2"
        log(k, "arrival", t, "accept refit", s["prim"], s["jobs"])
        return False
    if s["phase"] == "accept2":
        a["prims2"] = succ("c2_accept", t, f"prims@{s['prim']}.0")
        if a["prims2"] is None:
            s["jobs"] = pack(t, [f"prims@{s['prim']}.0"], "c2_accept", 0, 10)
            return False
        a["kept"] = f"prims@{s['prim']}.0" if a["prims2"] > a["zs"] else "zs"
        log(k, "arrival", t, "kept", a["kept"], a["prims2"], "vs", a["zs"])
        s["i"], s["phase"] = s["i"] + 1, "start"
        return False
    return False


def run():
    st = json.load(open(STATE)) if os.path.exists(STATE) else init()
    while True:
        done = [step(k, s) for k, s in st.items() if not k.startswith("_")]
        json.dump(st, open(STATE, "w"), indent=1)
        if all(done):
            break
        time.sleep(60)
    test = st.setdefault("_test", {})
    if not test.get("jobs"):
        js = []
        for k, s in st.items():
            if k.startswith("_"):
                continue
            for a in s["arrivals"]:
                arms = sorted({"zs", a["kept"]} | ({f"prims@{s['prim']}.0"} if s["prim"] else set()))
                a["test_arms"] = arms
                js += pack(a["task"], arms, "c2_main", 30, 20)
        test["jobs"] = js
        json.dump(st, open(STATE, "w"), indent=1)
        log("test packs", len(js))
    while running(test["jobs"]):
        time.sleep(120)
    log("stream done")


def read(record):
    from lifelong_ood import stats
    st = json.load(open(STATE))
    tab_k, tab_f, per, missing = {}, {}, {}, []
    for k, s in st.items():
        if k.startswith("_"):
            continue
        final = f"prims@{s['prim']}.0" if s["prim"] else "zs"
        for a in s["arrivals"]:
            t = str(a["task"])
            def ep(arm):
                f = os.path.join(R, f"c2_main_{t}_{tag(arm)}.json")
                if not os.path.exists(f):
                    return None
                return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}
            z, kp, fn = ep("zs"), ep(a["kept"]), ep(final)
            if None in (z, kp, fn):
                missing.append(t)
                continue
            tab_k[t] = {"x": kp, "zs": z}
            tab_f[t] = {"x": fn, "zs": z}
            per[t] = dict(key=k, demo=a["demo"], kept=a["kept"], kept_s=sum(kp.values()), final_s=sum(fn.values()), zs=sum(z.values()))
    hyp = {}
    for h, tab in (("H1", tab_k), ("H2", tab_f)):
        pairs = stats.paired(tab, "x", "zs", list(tab))
        d = stats.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    order_ = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run_ = 0.0
    for r_, h in enumerate(order_):
        run_ = max(run_, min(1.0, (len(order_) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run_
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run_ < 0.05)
    demos = {k: len(s["library"]) for k, s in st.items() if not k.startswith("_")}
    arr = {k: len(s["arrivals"]) for k, s in st.items() if not k.startswith("_")}
    res = dict(missing=missing, hypotheses=hyp, per_task=per, demos=demos, arrivals=arr,
               primitives={k: s["prim"] for k, s in st.items() if not k.startswith("_")},
               bytes=int(4677632 * sum(1 for k, s in st.items() if not k.startswith("_") and s["prim"])))
    for h, v in hyp.items():
        print(f"[c2] {h}: {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}, Holm "
              f"{v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("[c2] demonstrations", demos, "of arrivals", arr, flush=True)
    json.dump(res, open(os.path.join(R, "c2_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("c2_stream", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=res["bytes"], tree=0, codes=0, samples=0, teacher_copies=0, total=res["bytes"]),
                            config={}, script="experiments/c2_stream.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        print(json.dumps({k: v["order"] for k, v in init().items()}))
    if "--run" in sys.argv:
        run()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
