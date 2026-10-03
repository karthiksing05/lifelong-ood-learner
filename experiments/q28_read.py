"""The binding-aware primitive read (ledger m3_prereg tag q28_bindprim).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q28_read.py --accept      # stored init states 0-9, seed 10
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q28_read.py --test        # fresh start states, seed 10, all arms
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q28_read.py --read [--record]
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
SEED = 10


def arms():
    plan = json.load(open(os.path.join(R, "q28_plan.json")))
    return {t: dict(zs="zs", plain=f"prims@{v['plain']}.0", bind=f"prims@{v['bind']}.0_bind", key=v["key"]) for t, v in plan.items()}


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(exp, first, trials, fresh):
    for t, v in arms().items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join((v["zs"], v["plain"], v["bind"])), "--experiment", exp,
               "--first", str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        print(subprocess.run(cmd + (["--fresh"] if fresh else []), capture_output=True, text=True, cwd=ROOT).stdout.strip()[-200:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    A = arms()
    kept, acc, tab, missing = {}, {}, {"H1": {}, "H2": {}}, []
    for t, v in A.items():
        s = {k: load("q28_accept", t, v[k]) for k in ("zs", "plain", "bind")}
        x = {k: load("q28_main", t, v[k]) for k in ("zs", "plain", "bind")}
        if None in s.values() or None in x.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(z.values())) for k, z in s.items()}
        kept[t] = max(("zs", "plain", "bind"), key=lambda k: (acc[t][k], -("zs", "plain", "bind").index(k)))
        tab["H1"][t] = {"x": x[kept[t]], "y": x["zs"]}
        tab["H2"][t] = {"x": x["bind"], "y": x["plain"]}
    hyp = {}
    for h in ("H1", "H2"):
        pairs = stats.paired(tab[h], "x", "y", list(tab[h]))
        if not pairs:
            continue
        d = stats.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(x=int(sum(sum(v[0]) for v in pairs.values())), y=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    order = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["y"] and run < 0.05)
    per = {t: {k: int(sum(load("q28_main", t, A[t][k]).values())) for k in ("zs", "plain", "bind")} | {"kept": kept[t]}
           for t in kept}
    res = dict(missing=missing, acceptance=acc, kept=kept, hypotheses=hyp, per_task=per,
               totals={k: sum(v[k] for v in per.values()) for k in ("zs", "plain", "bind")})
    for h, v in hyp.items():
        print(f"[q28] {h}: {v['x']} vs {v['y']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}, Holm "
              f"{v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("[q28] totals", res["totals"], "kept", {t: k for t, k in kept.items() if k != "zs"}, ("missing " + str(missing)) if missing else "",
          flush=True)
    json.dump(res, open(os.path.join(R, "q28_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q28_bindprim", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config={}, script="experiments/q28_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--accept" in sys.argv:
        launch("q28_accept", 0, 10, False)
    if "--test" in sys.argv:
        launch("q28_main", 0, 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
