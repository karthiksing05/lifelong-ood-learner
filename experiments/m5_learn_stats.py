"""Learning from practice, the REDO read (pre-registered: ledger m3_prereg tag learn_from_practice).

Instances = init states 0-19 of the 21 tasks the prior fails.  y0 = the first composition's practice outcome (m5_practice,
recorded); retry = the SAME first composition run again on the same instances (m5_redo, tag dimfold_s1_last2 / dimcode_s1):
what re-running without learning gives; learned = compose -> practise -> consolidate (m5_redo tag learned); learned_mem0
= the refit fold without the stored compositions; lora_learn = a LoRA on the demo + the same successes.
  L1: on instances with y0 = 1, success of `learned` (>= 0.9 pre-registered) vs `retry`
  L2: on instances with y0 = 0, success of `learned` vs `retry`
Pooled over tasks, paired by instance, two-level cluster bootstrap.   python experiments/m5_learn_stats.py [--record]
"""
import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger, stats                                  # noqa: E402


def outcomes(experiment):
    out = {}
    for r in ledger.load(experiment):
        arts = [x for x in (r.get("artifacts") or []) if x.endswith(".json")]
        if not arts or not os.path.exists(arts[0]):
            continue
        for tid, t in json.load(open(arts[0]))["tasks"].items():
            for e in t["episodes"]:
                if e.get("error") is None:
                    out.setdefault((tid, r["tag"]), {})[int(e["trial"])] = int(bool(e["success"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    prac, redo = outcomes("m5_practice"), outcomes("m5_redo")
    tasks = sorted({t for t, _ in prac}, key=int)
    res = dict(tasks=[], L1={}, L2={}, overall={})
    for scope in ("L1", "L2"):
        want = 1 if scope == "L1" else 0
        table = {}
        for t in tasks:
            first = next((prac[(t, g)] for g in ("dimfold_s1_last2", "dimcode_s1") if (t, g) in prac), None)
            retry = next((redo[(t, g)] for g in ("dimfold_s1_last2", "dimcode_s1") if (t, g) in redo), None)
            if first is None or retry is None:
                continue
            inst = [i for i, y in first.items() if y == want]
            if not inst:
                continue
            row = {"retry": {i: retry[i] for i in inst if i in retry}}
            for arm in ("learned", "learned_mem0", "lora_learn"):
                if (t, arm) in redo:
                    row[arm] = {i: redo[(t, arm)][i] for i in inst if i in redo[(t, arm)]}
            table[t] = row
        res["tasks"] = sorted(set(res["tasks"]) | set(table), key=int)
        for arm in ("retry", "learned", "learned_mem0", "lora_learn"):
            ts = [t for t in table if arm in table[t] and table[t][arm]]
            k = sum(sum(table[t][arm].values()) for t in ts); n = sum(len(table[t][arm]) for t in ts)
            if n:
                lo, hi = stats.wilson(k, n)
                res[scope][arm] = dict(k=int(k), n=int(n), rate=k / n, wilson=[lo, hi], tasks=len(ts))
        for x in ("learned", "learned_mem0", "lora_learn"):
            pairs = stats.paired(table, x, "retry", list(table))
            if pairs:
                d = stats.bootstrap_diff(pairs, B=a.B)
                b, c = stats.discordant(pairs)
                d.update(discordant=[b, c], sign_p=stats.sign_test(b, c))
                res[scope][f"{x}-retry"] = d
        label = "instances the first composition SOLVED" if scope == "L1" else "instances the first composition FAILED"
        print(f"[{scope}] {label}:")
        for k, v in res[scope].items():
            if "-" in k:
                print(f"   {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}], discordant {v['discordant']}, sign p {v['sign_p']:.4f}")
            else:
                print(f"   {k:14s} {v['k']}/{v['n']} = {v['rate']:.3f} ({v['wilson'][0]:.2f}-{v['wilson'][1]:.2f}) over {v['tasks']} tasks")
    json.dump(res, open(os.path.join(ROOT, "results", "robot", "m5_learn_stats.json"), "w"), indent=1)
    if a.record:
        ledger.record("m5_learn_stats", bench="libero_90", tag=f"redo_{len(res['tasks'])}tasks", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/m5_learn_stats.py", command="python experiments/m5_learn_stats.py --record")
        print("[m5] ledger row written")


if __name__ == "__main__":
    main()
