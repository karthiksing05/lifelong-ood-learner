"""The stream (ledger m3_prereg tag q17_stream): the 48 failing LIBERO-90 tasks with a moved object arrive one at a time, in a
seeded order and its mirror.  An arrival whose object type the library lacks asks for ONE demonstration, used once: its first grasp
gives a 4 KB object prototype that joins the library.  An arrival whose type is known asks for nothing: its candidate forms use up
to K library prototypes of that type (same scene first, then other scenes, each in arrival order).  The acceptance test on init
states 0-9 keeps a candidate or the policy alone.  Reference condition: a demonstration for EVERY task (candidates: its own
prototype).  Every arm runs at noise seed 3; the library only grows when a demonstration is requested, so every arrival's
candidates follow from the order and the acceptance runs are shared between the orders.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q17_stream.py --plan
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q17_stream.py --launch-accept
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q17_stream.py --select [--launch]
    ~/flash/miniconda3/envs/chunks/bin/python experiments/q17_stream.py --read [--record]
"""
import glob
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
K = 4
SEED = 3
PLAN = os.path.join(R, "q17_plan.json")
KEPT = os.path.join(R, "q17_kept.json")
PROTO_BYTES = 4096


def pool():
    from lifelong_ood.policy import bind_hooks as BH
    from lifelong_ood.policy import factors as F
    d = {t.index: t for t in F.load() if t.suite == "libero_90"}
    rows = json.load(open(glob.glob(os.path.join(ROOT, "ledger", "b2_q2", "*q9_w4screen*"))[0]))["metrics"]["rows"]
    tasks = sorted({v["task"] for v in rows.values()})
    info = {}
    for t in tasks:
        mv = BH.referents(d[t])["open"]
        info[t] = dict(type=d[t].objects.get(mv[0], mv[0]), scene=d[t].scene, language=d[t].language)
    return tasks, info


def arm(k):
    return "zs" if k is None else f"bindw4@rho0.7_approach_src{k}"


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def plan():
    tasks, info = pool()
    order_a = [int(x) for x in np.random.default_rng(0).permutation(tasks)]
    orders = {"A": order_a, "B": order_a[::-1]}
    out = dict(K=K, seed=SEED, info={str(t): v for t, v in info.items()}, orders=orders, arrivals={})
    for name, order in orders.items():
        lib, arr = [], []
        for t in order:
            ty, sc = info[t]["type"], info[t]["scene"]
            same = [k for k in lib if info[k]["type"] == ty]
            if not same:                                      # a new type: ask for one demonstration; its prototype joins the library
                arr.append(dict(task=t, demo=True, candidates=[t]))
                lib.append(t)
            else:
                cands = ([k for k in same if info[k]["scene"] == sc] + [k for k in same if info[k]["scene"] != sc])[:K]
                arr.append(dict(task=t, demo=False, candidates=cands))
        out["arrivals"][name] = arr
    pairs = sorted({(a["task"], c) for arr in out["arrivals"].values() for a in arr for c in a["candidates"]} |
                   {(t, t) for t in tasks})
    out["accept_pairs"] = [list(p) for p in pairs]
    out["demos"] = {n: sum(a["demo"] for a in arr) for n, arr in out["arrivals"].items()}
    json.dump(out, open(PLAN, "w"), indent=1)
    print(f"[q17] {len(tasks)} tasks, demos requested {out['demos']}, acceptance arms {len(pairs)} + zero-shot {len(tasks)} "
          f"-> {10 * (len(pairs) + len(tasks))} episodes", flush=True)
    return out


def launch(exp, first, trials, per_task_arms):
    for t, arms in per_task_arms.items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(arms), "--experiment", exp, "--first",
               str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-300:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def keep(t, cands):
    opts = [None] + list(cands)                               # ties: the policy, then the earlier candidate
    r = {k: load("q17_accept", t, arm(k)) for k in opts}
    if any(v is None for v in r.values()):
        return None, None
    best = max(opts, key=lambda k: (sum(r[k].values()), -opts.index(k)))
    return arm(best), {("zs" if k is None else str(k)): int(sum(v.values())) for k, v in r.items()}


def select(do_launch):
    p = json.load(open(PLAN))
    kept, acc, missing = {}, {}, []
    conds = {n: [(a["task"], a["candidates"]) for a in arr] for n, arr in p["arrivals"].items()}
    conds["every_demo"] = [(int(t), [int(t)]) for t in p["info"]]
    for n, items in conds.items():
        kept[n] = {}
        for t, cands in items:
            a, ac = keep(t, cands)
            if a is None:
                missing.append(f"{n}:{t}")
                continue
            kept[n][str(t)] = a
            acc[f"{t}:" + "+".join(str(c) for c in cands)] = ac
    print("[q17] kept forms", {n: sum(v != "zs" for v in k.values()) for n, k in kept.items()},
          ("missing " + str(len(missing))) if missing else "", flush=True)
    if missing:
        return
    json.dump(dict(kept=kept, acceptance=acc), open(KEPT, "w"), indent=1)
    if do_launch:
        need = {}
        for k in kept.values():
            for t, a in k.items():
                need.setdefault(int(t), {"zs"}).add(a)
        launch("q17_main", 30, 20, {t: sorted(v) for t, v in need.items()})


def read(record):
    p, k = json.load(open(PLAN)), json.load(open(KEPT))
    res, missing = dict(K=p["K"], demos=p["demos"], orders=p["orders"], kept=k["kept"]), []
    hyp = {}
    for h, n in (("H1", "A"), ("H2", "B"), ("H3", "every_demo")):
        tab = {}
        for t, a in k["kept"][n].items():
            x, z = load("q17_main", t, a), load("q17_main", t, "zs")
            if x is None or z is None:
                missing.append(f"{n}:{t}")
                continue
            tab[t] = {"kept": x, "zs": z}
        pairs = stats.paired(tab, "kept", "zs", list(tab))
        d = stats.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(condition=n, x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)],
                      sign_p=float(stats.sign_test(b, c)), kept_forms=int(sum(v != "zs" for v in k["kept"][n].values())),
                      per_task={t: [int(sum(v["kept"].values())), int(sum(v["zs"].values()))] for t, v in tab.items()})
    order = sorted(("H1", "H2"), key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run < 0.05)
    res["hypotheses"], res["missing"] = hyp, missing
    res["bytes"] = {n: int(p["demos"][n] * PROTO_BYTES) for n in ("A", "B")} | {"every_demo": int(len(p["info"]) * PROTO_BYTES)}
    for h, v in hyp.items():
        print(f"[q17] {h} ({v['condition']}): {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p "
              f"{v['sign_p']:.3g}" + (f", Holm {v['holm_p']:.3g} -> passes {v['passes']}" if "holm_p" in v else "")
              + f"; forms kept {v['kept_forms']}", flush=True)
    print("[q17] demos", p["demos"], "bytes", res["bytes"], ("missing " + str(missing)) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "q17_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q17_stream", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=res["bytes"]["A"], samples=0, teacher_copies=0, total=res["bytes"]["A"]),
                            config={}, script="experiments/q17_stream.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--launch-accept" in sys.argv:
        p = json.load(open(PLAN))
        per = {}
        for t, c in p["accept_pairs"]:
            per.setdefault(t, ["zs"]).append(arm(c))
        launch("q17_accept", 0, 10, per)
    if "--select" in sys.argv:
        select("--launch" in sys.argv)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
