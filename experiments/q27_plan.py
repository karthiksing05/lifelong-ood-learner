"""Plan of the consolidated-primitive read (ledger m3_prereg tag q27_joint): every pool task (q17_stream's 48) with a step whose key
is shared by at least 3 OTHER LIBERO-90 tasks gets, for each such step, ONE primitive fitted jointly on those other tasks' demos
(c0_fit --joint; leave the task out); a step without one runs the policy itself.  No demonstration of the scored task.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q27_plan.py
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
MIN_OTHERS = 3


def main():
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy import primitives as P
    d = {t.index: t for t in F.load() if t.suite == "libero_90"}
    by_key = collections.defaultdict(list)
    for t, tk in d.items():
        for c in P.steps_of(tk.language):
            by_key[P.step_key(c)].append(t)
    pool = sorted(int(t) for t in json.load(open(os.path.join(R, "q17_plan.json")))["info"])
    plan = {}
    for t in pool:
        steps = P.steps_of(d[t].language)
        per = []
        for s, c in enumerate(steps):
            k = P.step_key(c)
            others = sorted(u for u in set(by_key[k]) if u != t)
            per.append(dict(step=s, clause=c, key=k, donors=others if len(others) >= MIN_OTHERS else []))
        if any(p["donors"] for p in per):
            plan[str(t)] = dict(language=d[t].language, steps=per)
    json.dump(plan, open(os.path.join(R, "q27_plan.json"), "w"), indent=1)
    fits = sum(1 for v in plan.values() for p in v["steps"] if p["donors"])
    for t, v in plan.items():
        print(f"[q27] {t:>3} " + " + ".join(f"{p['key']} <- {len(p['donors'])}" if p["donors"] else f"{p['key']} (policy)" for p in v["steps"]),
              flush=True)
    print(f"[q27] {len(plan)} tasks, {fits} joint fits", flush=True)


if __name__ == "__main__":
    main()
