"""The PoE pilot's read (ledger m3_prereg tag q_pilot): successes of 10 per task and arm on practice states 0-9, and the
pre-registered go rule.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q3_pilot_read.py [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
TASKS = [35, 36, 39, 42, 1, 6]
ARMS = ["zs", "poe_0.3", "poe_0.5", "poe_0.7", "poeopt_0.5", "poenopol_0.5"]


def main():
    tab, missing = {}, []
    for t in TASKS:
        for a in ARMS:
            f = os.path.join(R, f"q_pilot_{t}_{a}.json")
            if not os.path.exists(f):
                missing.append(f"{t}:{a}")
                continue
            eps = [e for v in json.load(open(f))["tasks"].values() for e in v["episodes"]]
            tab.setdefault(str(t), {})[a] = int(sum(e["success"] for e in eps))
    tot = {a: sum(tab[str(t)].get(a, 0) for t in TASKS if str(t) in tab) for a in ARMS}
    gains = {a: int(sum(tab[str(t)].get(a, 0) - tab[str(t)].get("zs", 0) >= 3 for t in TASKS if str(t) in tab)) for a in ARMS}
    cands = [a for a in ("poe_0.5", "poe_0.7") if gains[a] >= 3]
    best = max(cands, key=lambda a: tot[a]) if cands else None
    bypass = bool(best and tot["poenopol_0.5"] >= tot[best] - 2)
    only03 = bool(gains["poe_0.3"] >= 3 and not cands)
    go = bool(best and not bypass)
    for t in TASKS:
        print(f"[q3p] {t:>3} " + " ".join(f"{a}={tab.get(str(t), {}).get(a, '-')}" for a in ARMS), flush=True)
    print(f"[q3p] totals {tot}; tasks gaining >= 3 {gains}; best {best}; no-policy matches {bypass}; only lambda 0.3 {only03} -> go {go}"
          + (f"; missing {missing}" if missing else ""), flush=True)
    res = dict(per_task=tab, totals=tot, gains=gains, best=best, bypass=bypass, only_lambda_03=only03, go=go, missing=missing)
    json.dump(res, open(os.path.join(R, "q3_pilot_read.json"), "w"), indent=1)
    if "--record" in sys.argv and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q3_pilot", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/q3_pilot_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
