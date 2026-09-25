"""Rollout JSON (experiments/libero_client.py) -> one ledger row.

    python experiments/roll_record.py --json results/robot/x.json --experiment r1_rollout --mode openpi --job 123
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--experiment", default="rollout")
    ap.add_argument("--mode", default="openpi")
    ap.add_argument("--tag", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--program-bytes", type=int, default=0)
    a = ap.parse_args()
    r = json.load(open(a.json))
    tasks = {k: dict(language=v["language"], success_rate=v["success_rate"], n=len(v["episodes"]),
                     mean_steps=float(np.mean([e["steps"] for e in v["episodes"]])),
                     errors=int(sum(e["error"] is not None for e in v["episodes"])))
             for k, v in r["tasks"].items()}
    allsr = [e["success"] for v in r["tasks"].values() for e in v["episodes"]]
    m = dict(success_rate=float(np.mean(allsr)) if allsr else None, n_episodes=len(allsr), tasks=tasks,
             secs=r.get("secs"), complete="success_rate" in r)
    b = int(a.program_bytes)
    p = ledger.record(a.experiment, bench=r["suite"], tag=a.tag or a.mode.split(":")[0], seed=int(r["args"]["seed"]),
                      keys=dict(mode=a.mode, tasks=r["args"]["tasks"], first_trial=r["args"]["first_trial"], trials=r["args"]["trials"]),
                      metrics=m, bytes=dict(experts=0, tree=0, codes=b, samples=0, teacher_copies=0, total=b),
                      config=dict(mode=a.mode, job=a.job, client=r["args"]), script="experiments/libero_client.py",
                      command=f"roll.sh MODE={a.mode} SUITE={r['suite']} TASKS={r['args']['tasks']} TRIALS={r['args']['trials']} (job {a.job})",
                      artifacts=[a.json])
    print(f"[record] {a.experiment} {r['suite']} mode={a.mode}: success {m['success_rate']} over {m['n_episodes']} -> {p}")


if __name__ == "__main__":
    main()
