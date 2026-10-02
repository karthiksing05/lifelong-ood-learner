"""Plan of the grammar read (ledger m3_prereg tag q24_grammar): per qualifier task, the TYPE prototype = the mean of the library's
prototypes of its moved type built from OTHER tasks' demos (results/robot/q24_type_protos.npz), and the number of peaks the grammar
orders, learned for each word from the OTHER tasks that use it (leave one task out over ledger q23_typemap's mean-map scores, ties to
the smaller count).  No demonstration of the scored task enters.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q24_plan.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")


def main():
    q = json.load(open(os.path.join(R, "q23_typemap.json")))["per_task"]
    P = np.load(os.path.join(R, "q4_object_protos.npz"), allow_pickle=True)
    protos, plan = {}, {}
    for t, v in q.items():
        others = [u for u, w in q.items() if u != t and w["word"] == v["word"]]
        if not others:
            continue
        n = max((3, 4), key=lambda n: (sum(q[u][f"mean_n{n}"] for u in others), -n))
        m = np.mean([P[str(k)] for k in v["donors"]], 0)
        protos[t] = (m / np.linalg.norm(m)).astype(np.float32)
        plan[t] = dict(word=v["word"], donors=v["donors"], npk=int(n), offline_pointing=v[f"mean_n{n}"])
        print(f"[q24] task {t:>3} {v['word']:6s} peaks {n} (from {others}) donors {len(v['donors'])} offline pointing "
              f"{v[f'mean_n{n}']:.2f}", flush=True)
    np.savez(os.path.join(R, "q24_type_protos.npz"), **protos)
    json.dump(plan, open(os.path.join(R, "q24_plan.json"), "w"), indent=1)
    print(f"[q24] {len(plan)} tasks; mean offline pointing {np.mean([p['offline_pointing'] for p in plan.values()]):.3f}", flush=True)


if __name__ == "__main__":
    main()
