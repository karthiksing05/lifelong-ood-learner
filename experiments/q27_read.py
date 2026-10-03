"""The consolidated-primitive read (ledger m3_prereg tag q27_joint).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q27_read.py --accept            # init states 0-9, seed 7
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q27_read.py --select [--launch] # kept arm -> init states 30-49
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q27_read.py --read [--record]
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
SEED = 7
KEPT = os.path.join(R, "q27_kept.json")


def arms():
    plan = json.load(open(os.path.join(R, "q27_plan.json")))
    names = json.load(open(os.path.join(R, "q27_names.json")))
    return {t: "prims@" + "+".join(f"{names[f'{t}:{p['step']}']}.0" if p["donors"] else "x" for p in v["steps"])
            for t, v in plan.items()}


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, first, trials, per):
    out = []
    for t, a in per.items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(a), "--experiment", exp, "--first",
               str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        out.append(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-200:])
        print(out[-1], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def select(do_launch):
    kept, acc, missing = {}, {}, []
    for t, a in arms().items():
        x, z = load("q27_accept", t, a), load("q27_accept", t, "zs")
        if x is None or z is None:
            missing.append(t)
            continue
        acc[t] = dict(prims=int(sum(x.values())), zs=int(sum(z.values())))
        kept[t] = a if acc[t]["prims"] > acc[t]["zs"] else "zs"
    print("[q27] acceptance", acc, "\n[q27] kept", {t: k for t, k in kept.items() if k != "zs"}, ("missing " + str(missing)) if missing else "",
          flush=True)
    if missing:
        return
    json.dump(dict(kept=kept, acceptance=acc), open(KEPT, "w"), indent=1)
    if do_launch:
        launch("q27_main", 30, 20, {t: ["zs", a] for t, a in arms().items()})


def read(record):
    k = json.load(open(KEPT))
    A = arms()
    tabs, missing = {"kept": {}, "prims": {}}, []
    for t, a in A.items():
        x, z = load("q27_main", t, a), load("q27_main", t, "zs")
        if x is None or z is None:
            missing.append(t)
            continue
        tabs["prims"][t] = {"x": x, "zs": z}
        tabs["kept"][t] = {"x": x if k["kept"][t] != "zs" else z, "zs": z}
    hyp = {}
    for h, m in (("H1", "kept"), ("H2", "prims")):
        pairs = stats.paired(tabs[m], "x", "zs", list(tabs[m]))
        d = stats.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(arm=m, x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    order = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run < 0.05)
    per = {t: dict(prims=int(sum(v["x"].values())), zs=int(sum(v["zs"].values())), kept=k["kept"][t] != "zs") for t, v in tabs["prims"].items()}
    res = dict(missing=missing, kept=k["kept"], acceptance=k["acceptance"], hypotheses=hyp, per_task=per,
               keys=sorted({p["key"] for v in json.load(open(os.path.join(R, "q27_plan.json"))).values() for p in v["steps"] if p["donors"]}))
    for h, v in hyp.items():
        print(f"[q27] {h} ({v['arm']}): {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}, "
              f"Holm {v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("[q27] per task", per, flush=True)
    json.dump(res, open(os.path.join(R, "q27_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q27_joint", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=int(18.7e6 * len(res["keys"])), tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=int(18.7e6 * len(res["keys"]))),
                            config={}, script="experiments/q27_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--accept" in sys.argv:
        launch("q27_accept", 0, 10, {t: ["zs", a] for t, a in arms().items()})
    if "--select" in sys.argv:
        select("--launch" in sys.argv)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
