"""The token-practice read (ledger m3_prereg tag q7_tokpractice): the practised token (demo + its own practice successes, tag _tp)
against the demo token (zc_fold) and the one-demo LoRA on the same 420 paired episodes (21 tasks, init states 30-49).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q7_read.py [--record]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TASKS = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
SRC = {"tp": "q7_main_{t}_ftoken_self_tp", "tok": "m3_final_{t}_ftoken_self", "lora": "m3_final_{t}_lora", "zs": "m3_final_{t}_zeroshot"}


def eps(name):
    f = os.path.join(R, name + ".json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def compare(tab, x, y):
    pairs = stats.paired(tab, x, y, list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    return dict(x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))


def main():
    tab, missing = {}, []
    for t in TASKS:
        row = {}
        for k, pat in SRC.items():
            e = eps(pat.format(t=t))
            if e is None:
                missing.append(f"{t}:{k}")
            else:
                row[k] = e
        tab[str(t)] = row
    prac = {}
    for t in TASKS:
        e = eps(f"q7_p1_{t}_ftoken_self")
        prac[str(t)] = None if e is None else int(sum(e.values()))
    res = dict(missing=missing, practice_successes=prac,
               per_task={t: {k: int(sum(v.values())) for k, v in r.items()} for t, r in tab.items()})
    H1 = compare(tab, "tp", "tok")
    H2 = compare(tab, "tp", "lora")
    H2["noninferior"] = bool(H2["diff"]["lo90"] > -0.10)
    ps = sorted([("H1", H1["sign_p"]), ("H2", None)], key=lambda x: x[1] if x[1] is not None else 1)
    H1["holm_p"] = min(1.0, 2 * H1["sign_p"])
    H1["passes"] = bool(H1["x"] > H1["control"] and H1["holm_p"] < 0.05)
    res.update(H1=H1, H2=H2, vs_zero_shot=compare(tab, "tp", "zs"))
    print(f"[q7] practised token {H1['x']} vs demo token {H1['control']} of {H1['n']} (discordant {H1['discordant']}, sign p "
          f"{H1['sign_p']:.3g}, Holm {H1['holm_p']:.3g}) -> passes {H1['passes']}", flush=True)
    print(f"[q7] vs LoRA {H2['x']} vs {H2['control']}: diff {H2['diff']['diff']:+.3f}, lower bound {H2['diff']['lo90']:+.3f} -> "
          f"non-inferior {H2['noninferior']}; practice successes {prac}" + (f"; missing {missing}" if missing else ""), flush=True)
    print("[q7] per task", res["per_task"], flush=True)
    json.dump(res, open(os.path.join(R, "q7_read.json"), "w"), indent=1)
    if "--record" in sys.argv and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q7_tokpractice", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=18432 * len(TASKS), tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=18432 * len(TASKS)), config={},
                            script="experiments/q7_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
