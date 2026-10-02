"""The scheme's pre-registered main read (ledger m3_prereg tag b_scheme_main): on init states 30-49 of the 14 role-coded tasks,
zero-shot pi0.5 against the practice-chosen composition form (oracle) and against the folded token alone (own instruction +
18.4 KB, nothing else), and the transfer of task 59's tomato-sauce token to task 49.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/b5_scheme_read.py [--record]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TASKS = [18, 21, 26, 48, 49, 52, 53, 58, 59, 60, 62, 65, 66, 71]
FORM = {52: "role_open", 53: "role_open", 60: "role_open", 59: "bind_rho0.7_all_approach"}


def episodes(t, tag):
    f = os.path.join(R, f"b_scheme_{t}_{tag}.json")
    if not os.path.exists(f):
        return None
    eps = [e for v in json.load(open(f))["tasks"].values() for e in v["episodes"]]
    return {int(e["trial"]): int(bool(e["success"])) for e in eps}, {int(e["trial"]): e.get("bound") for e in eps}


def compare(tab, x, y, tasks):
    pairs = stats.paired(tab, x, y, [str(t) for t in tasks])
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    kx = sum(int(sum(v[0])) for v in pairs.values())
    ky = sum(int(sum(v[1])) for v in pairs.values())
    n = sum(len(v[0]) for v in pairs.values())
    return dict(x=kx, control=ky, n=int(n), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))


def main():
    tab, bound, per_task, missing = {}, {}, {}, []
    tokens = [t for t in FORM if os.path.exists(os.path.join(R, f"ftoken_{t}_s2.pt"))]
    for t in TASKS:
        row, brow = {}, {}
        arms = {"zs": "zs"}
        if t in FORM:
            arms["form"] = FORM[t]
        if t in tokens:
            arms["token"] = "ftoken_self_s2"
        if t == 49 and 59 in tokens:
            arms["transfer"] = "ftoken_59_s2"
        for k, tag in arms.items():
            e = episodes(t, tag)
            if e is None:
                missing.append(f"{t}:{tag}")
                continue
            row[k], brow[k] = e
        tab[str(t)], bound[str(t)] = row, brow
        per_task[str(t)] = {k: int(sum(v.values())) for k, v in row.items()}
    if missing:
        print("missing:", missing)
    res = dict(tokens=tokens, per_task=per_task, missing=missing)
    with_token = [t for t in tokens]
    with_form = list(FORM)
    hyp = {}
    if with_token:
        hyp["H1"] = dict(comparison="token vs zero-shot, tasks with a token", tasks=with_token, **compare(tab, "token", "zs", with_token))
    hyp["H2"] = dict(comparison="form vs zero-shot, tasks with a form", tasks=with_form, **compare(tab, "form", "zs", with_form))
    if "transfer" in tab.get("49", {}):
        hyp["H4"] = dict(comparison="task 59's token on task 49 vs zero-shot", tasks=[49], **compare(tab, "transfer", "zs", [49]))
    order = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run < 0.05)
    res["hypotheses"] = hyp
    # population: the scheme (token where it exists, else the frozen policy) against zero-shot over all 14 tasks
    sch = {t: (tab[t].get("token") or tab[t]["zs"]) for t in tab if "zs" in tab[t]}
    zs = {t: tab[t]["zs"] for t in sch}
    res["population"] = dict(scheme=int(sum(sum(v.values()) for v in sch.values())), zero_shot=int(sum(sum(v.values()) for v in zs.values())),
                             episodes=int(sum(len(v) for v in zs.values())), tasks=len(zs))
    # binding of the named object (first object moved) per arm, pooled over the tasks with a token
    res["binding"] = {}
    for k in ("zs", "form", "token"):
        vals = []
        for t in with_token:
            from lifelong_ood.policy import bind_hooks as BH
            from lifelong_ood.policy import factors as F
            named = BH.referents(next(x for x in F.load() if x.suite == "libero_90" and x.index == t))["open"][0]
            if k in bound[str(t)]:
                vals += [b == named for b in bound[str(t)][k].values()]
        if vals:
            res["binding"][k] = float(np.mean(vals))
    for h, v in hyp.items():
        print(f"{h} {v['comparison']}: {v['x']} vs {v['control']} of {v['n']}, diff {v['diff']['diff']:+.3f} "
              f"[{v['diff']['lo']:+.3f},{v['diff']['hi']:+.3f}], discordant {v['discordant']}, sign p {v['sign_p']:.3g}, "
              f"Holm {v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("population", res["population"], "binding", res["binding"], flush=True)
    print("per task", per_task, flush=True)
    json.dump(res, open(os.path.join(R, "b5_scheme_read.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        tb = sum(18432 for _ in with_token)
        print(ledger.record("b5_scheme", bench="libero_90", tag="main", seed=0, metrics=res,
                            bytes=dict(experts=tb, tree=0, codes=0, samples=0, teacher_copies=0, total=tb), config={},
                            script="experiments/b5_scheme_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
