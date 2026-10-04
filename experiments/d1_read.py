"""Plan D's main read (ledger m3_prereg tag d1_factored).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/d1_read.py --accept     # stored init states 0-9, seed 11
    ~/flash/miniconda3/envs/chunks/bin/python experiments/d1_read.py --test       # fresh start states, seed 11, all arms
    ~/flash/miniconda3/envs/chunks/bin/python experiments/d1_read.py --read [--record]
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED = 11
ORDER = ("zs", "goal", "goal_bind")


def arms():
    plan = json.load(open(os.path.join(R, "d1_plan.json")))
    if os.path.exists(os.path.join(R, "q4_object_protos.npz")):
        have = set(np.load(os.path.join(R, "q4_object_protos.npz")).files)
        for t, v in plan.items():
            v["proto"] = t in have
    out = {}
    for t, v in plan.items():
        a = {"zs": "zs", "goal": f"prims@{v['goal']}.0_goal"}
        if v["proto"]:
            a["goal_bind"] = f"prims@{v['goal']}.0_goal_bind"
        if v["lora"]:
            a["lora"] = "lora"
        out[t] = a
    return out


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, first, trials, fresh):
    for t, a in arms().items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(a.values()), "--experiment", exp, "--first",
               str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        print(subprocess.run(cmd + (["--fresh"] if fresh else []), capture_output=True, text=True, cwd=ROOT).stdout.strip()[-200:],
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
    A = arms()
    acc, kept, kept_lora, test, missing = {}, {}, {}, {}, []
    for t, a in A.items():
        s = {k: load("d1_accept", t, v) for k, v in a.items()}
        x = {k: load("d1_main", t, v) for k, v in a.items()}
        if None in s.values() or None in x.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(v.values())) for k, v in s.items()}
        opts = [k for k in ORDER if k in a]
        kept[t] = max(opts, key=lambda k: (acc[t][k], -opts.index(k)))
        if "lora" in a:
            kept_lora[t] = "lora" if acc[t]["lora"] > acc[t]["zs"] else "zs"
        test[t] = x
    h1 = compare({t: {"x": x[kept[t]], "y": x["zs"]} for t, x in test.items()})
    h1["holm_p"] = h1["sign_p"]
    h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
    h2 = compare({t: {"x": x[kept[t]], "y": x[kept_lora[t]]} for t, x in test.items() if t in kept_lora})
    h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
    desc = {n: compare({t: {"x": x[a_], "y": x[b_]} for t, x in test.items() if a_ in x and b_ in x})
            for n, (a_, b_) in {"goal_vs_zs": ("goal", "zs"), "goal_bind_vs_zs": ("goal_bind", "zs"), "lora_vs_zs": ("lora", "zs")}.items()}
    desc["lora_acc_vs_zs"] = compare({t: {"x": x[kept_lora[t]], "y": x["zs"]} for t, x in test.items() if t in kept_lora})
    res = dict(missing=missing, acceptance=acc, kept=kept, kept_lora=kept_lora, hypotheses={"H1": h1, "H2": h2}, descriptive=desc,
               per_task={t: {k: int(sum(v.values())) for k, v in x.items()} | {"kept": kept[t]} for t, x in test.items()})
    print(f"[d1] H1 kept vs zs: {h1['x']} vs {h1['y']} of {h1['n']}, discordant {h1['discordant']}, sign p {h1['sign_p']:.3g} -> "
          f"passes {h1['passes']}", flush=True)
    print(f"[d1] H2 kept vs LoRA+acc: {h2['x']} vs {h2['y']} of {h2['n']} ({h2['tasks']} tasks), diff {h2['diff']['diff']:+.3f} "
          f"[{h2['diff']['lo']:+.3f}, {h2['diff']['hi']:+.3f}] -> non-inferior {h2['noninferior']}", flush=True)
    for k, v in desc.items():
        print(f"[d1]   {k}: {v['x']} vs {v['y']} of {v['n']}", flush=True)
    print("[d1] kept", {k: sum(v == k for v in kept.values()) for k in ORDER}, ("missing " + str(missing)) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "d1_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("d1_factored", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=int(4 * 4677632), tree=0, codes=0, samples=0, teacher_copies=0, total=int(4 * 4677632)),
                            config={}, script="experiments/d1_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--accept" in sys.argv:
        launch("d1_accept", 0, 10, False)
    if "--test" in sys.argv:
        launch("d1_main", 0, 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
