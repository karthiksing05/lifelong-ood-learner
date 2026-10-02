"""The self-prototype read (ledger m3_prereg tag q18_self): NO demonstration.  The 48 failing tasks with a moved object each practise
zero-shot on init states 0-9 (recorded); every task with a success gets a 4 KB object prototype from its own successful grasps
(experiments/q18_selfproto.py); the acceptance test on init states 10-19 keeps the binding form with that prototype or the policy;
init states 30-49 score the kept arm against zero-shot.  Noise seed 4 throughout.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q18_read.py --explore          # recorded zero-shot practice, states 0-9
    sbatch ... experiments/q18_selfproto.py --rec q18_explore --tag zs_seed4                 # the self-prototypes
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q18_read.py --accept           # states 10-19
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q18_read.py --select [--launch]
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q18_read.py --read [--record]
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
SEED = 4
FORM = "bindw4@rho0.7_approach_self"
KEPT = os.path.join(R, "q18_kept.json")


def tasks():
    return sorted(int(t) for t in json.load(open(os.path.join(R, "q17_plan.json")))["info"])   # the same 48 as q17_stream


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, first, trials, per_task_arms, record=False):
    out = []
    for t, arms in per_task_arms.items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(arms), "--experiment", exp, "--first",
               str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        out.append(subprocess.run(cmd + (["--record"] if record else []), capture_output=True, text=True, cwd=ROOT).stdout.strip()[-300:])
        print(out[-1], flush=True)
    return out


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def with_proto():
    f = os.path.join(R, "q18_self_protos.npz")
    return sorted(int(k) for k in np.load(f).files) if os.path.exists(f) else []


def select(do_launch):
    kept, acc, missing = {}, {}, []
    have = set(with_proto())
    for t in tasks():
        if t not in have:
            kept[str(t)] = "zs"                                  # no success in practice: nothing to build, the policy stays
            continue
        r = {a: load("q18_accept", t, a) for a in ("zs", FORM)}
        if any(v is None for v in r.values()):
            missing.append(t)
            continue
        acc[str(t)] = {a: int(sum(v.values())) for a, v in r.items()}
        kept[str(t)] = FORM if acc[str(t)][FORM] > acc[str(t)]["zs"] else "zs"     # ties to the policy
    print("[q18] self-prototypes", len(have), "forms kept", sum(a != "zs" for a in kept.values()),
          ("missing " + str(missing)) if missing else "", flush=True)
    if missing:
        return
    json.dump(dict(kept=kept, acceptance=acc), open(KEPT, "w"), indent=1)
    if do_launch:
        launch("q18_main", 30, 20, {int(t): sorted({"zs", a}) for t, a in kept.items()})


def read(record):
    k = json.load(open(KEPT))
    proto = json.load(open(os.path.join(R, "q18_self_protos.json")))
    tab, missing = {}, []
    for t, a in k["kept"].items():
        x, z = load("q18_main", t, a), load("q18_main", t, "zs")
        if x is None or z is None:
            missing.append(t)
            continue
        tab[t] = {"kept": x, "zs": z}
    pairs = stats.paired(tab, "kept", "zs", list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    h1 = dict(x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
              n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    h1["passes"] = bool(h1["x"] > h1["control"] and h1["sign_p"] < 0.05)
    pts = [v["pointing"] for v in proto["per_task"].values() if "pointing" in v]
    res = dict(missing=missing, kept=k["kept"], acceptance=k["acceptance"], hypotheses={"H1": h1},
               self_protos=int(proto["n_protos"]), pointing_mean=float(np.mean(pts)) if pts else None,
               forms_kept=int(sum(a != "zs" for a in k["kept"].values())),
               per_task={t: [int(sum(v["kept"].values())), int(sum(v["zs"].values()))] for t, v in tab.items()},
               bytes=int(4096 * sum(a != "zs" for a in k["kept"].values())))
    print(f"[q18] H1: {h1['x']} vs {h1['control']} of {h1['n']}, discordant {h1['discordant']}, sign p {h1['sign_p']:.3g} -> "
          f"passes {h1['passes']}; self-prototypes {res['self_protos']} (pointing {res['pointing_mean']}), forms kept "
          f"{res['forms_kept']}", flush=True)
    print("[q18] kept forms per task", {t: v for t, v in res["per_task"].items() if k["kept"][t] != "zs"}, flush=True)
    json.dump(res, open(os.path.join(R, "q18_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q18_self", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=res["bytes"], samples=0, teacher_copies=0, total=res["bytes"]),
                            config={}, script="experiments/q18_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--explore" in sys.argv:
        launch("q18_explore", 0, 10, {t: ["zs"] for t in tasks()}, record=True)
    if "--accept" in sys.argv:
        launch("q18_accept", 10, 10, {t: ["zs", FORM] for t in with_proto()})
    if "--select" in sys.argv:
        select("--launch" in sys.argv)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
