"""The library read (ledger m3_prereg tag q16_library): no demonstration of the new task; candidate binding forms use object
prototypes already in the library (other tasks' demos, same type), and the acceptance test on init states 0-9 keeps one of them or
the policy alone.  Noise seed 2.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q16_read.py --select [--launch]   # acceptance -> kept arm (-> test packs)
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q16_read.py --read [--record]      # kept vs zero-shot (H1), vs own demo (H2)
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
CAND = {12: [13, 14, 15, 16, 17, 2], 2: [12, 13, 14, 15, 16, 17], 13: [12, 14, 15, 16, 17, 2], 14: [12, 13, 15, 16, 17, 2],
        15: [12, 13, 14, 16, 17, 2], 79: [77, 78, 80, 73, 74, 75], 70: [69, 5], 43: [36, 37]}
KEPT = os.path.join(R, "q16_kept.json")


def arm(k):
    return "zs" if k is None else f"bindw4@rho0.7_approach_src{k}"


def tag(a):
    return a.replace("@", "_") + "_seed2"


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def select(launch):
    acc, kept, missing = {}, {}, []
    for t, cs in CAND.items():
        opts = [None] + cs                                       # the order breaks ties: the policy, then the earlier candidate
        r = {k: load("q16_accept", t, arm(k)) for k in opts}
        missing += [f"{t}:{arm(k)}" for k, v in r.items() if v is None]
        if any(v is None for v in r.values()):
            continue
        acc[str(t)] = {("zs" if k is None else str(k)): int(sum(v.values())) for k, v in r.items()}
        best = max(opts, key=lambda k: (sum(r[k].values()), -opts.index(k)))
        kept[str(t)] = arm(best)
    print("[q16] acceptance", acc, flush=True)
    print("[q16] kept", kept, ("missing " + str(missing)) if missing else "", flush=True)
    if missing:
        return
    json.dump(dict(acceptance=acc, kept=kept), open(KEPT, "w"), indent=1)
    if launch:
        for t, a in kept.items():
            arms = ",".join(dict.fromkeys([a, "zs", "bindw4@rho0.7_approach"]))
            cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", arms, "--experiment", "q16_main", "--first", "30",
                   "--trials", "20", "--seed", "2", "--bind", "track", "--partition", "overcap", "--per", "2"]
            print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-400:], flush=True)


def read(record):
    k = json.load(open(KEPT))
    tab, missing = {}, []
    for t, a in k["kept"].items():
        r = {"kept": load("q16_main", t, a), "zs": load("q16_main", t, "zs"), "own": load("q16_main", t, "bindw4@rho0.7_approach")}
        missing += [f"{t}:{x}" for x, v in r.items() if v is None]
        if all(v is not None for v in r.values()):
            tab[t] = r
    res = dict(missing=missing, acceptance=k["acceptance"], kept=k["kept"],
               per_task={t: {x: int(sum(v.values())) for x, v in r.items()} for t, r in tab.items()})
    hyp = {}
    for h, x, y in (("H1", "kept", "zs"), ("H2", "kept", "own")):
        pairs = stats.paired(tab, x, y, list(tab))
        if not pairs:
            continue
        d = stats.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    if "H1" in hyp:
        hyp["H1"]["passes"] = bool(hyp["H1"]["x"] > hyp["H1"]["control"] and hyp["H1"]["sign_p"] < 0.05)
    res["hypotheses"] = hyp
    res["kept_forms"] = int(sum(a != "zs" for a in k["kept"].values()))
    for h, v in hyp.items():
        print(f"[q16] {h}: {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}"
              + (f" -> passes {v['passes']}" if "passes" in v else ""), flush=True)
    print("[q16] per task", res["per_task"], ("missing " + str(missing)) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "q16_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q16_library", bench="libero_90", tag="read", seed=2, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config={}, script="experiments/q16_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--select" in sys.argv:
        select("--launch" in sys.argv)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
