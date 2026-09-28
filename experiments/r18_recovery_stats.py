"""Experiment 2, part b (pre-registered: ledger m3_prereg tags branch, branch_amend1-4): the recovery LoRA r18rec -- Experiment
1's data and loss plus the stall-point branches' rescued segments (positives) and failed post-branch stalls (negatives) --
against r17bal (the same without the branches) on init states 30-49.

    R1: the 9 concentrated tasks (180 paired episodes), exact sign test p < 0.05
    R2: every task with a rescue (secondary)
    budget-matched: r18rec vs the round-3 success-only learner (loop L3), which used 630 more practice episodes

    python experiments/r18_recovery_stats.py [--record]
"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger, stats                                 # noqa: E402

R = os.path.join(ROOT, "results", "robot")
CONCENTRATED = [42, 32, 1, 35, 40, 36, 52, 62, 41]
ARMS = {"r18rec": "lora_r18rec", "r17bal": "lora_r17bal", "L3": "lora_r16L3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    B = json.load(open(os.path.join(R, "r18_branch_stats.json")))
    rescued = sorted(int(t) for t, r in B["per_task"].items() if r["rescued"] > 0)
    trained = sorted(int(t) for t, tag in fin if tag == ARMS["r18rec"])
    scopes = {"R1_concentrated": CONCENTRATED, "R2_rescued": rescued, "all_trained": trained}
    res = dict(tasks=dict(scopes), arms=ARMS)
    for scope, tasks in scopes.items():
        table = {str(t): {k: fin[(str(t), tag)] for k, tag in ARMS.items() if (str(t), tag) in fin} for t in tasks}
        out = {}
        for k in ARMS:
            ts = [t for t in table if k in table[t]]
            kk = sum(sum(table[t][k].values()) for t in ts)
            n = sum(len(table[t][k]) for t in ts)
            out[k] = dict(k=int(kk), n=int(n), tasks=len(ts), rate=kk / n if n else float("nan"))
        for x, y in (("r18rec", "r17bal"), ("r18rec", "L3"), ("r17bal", "L3")):
            pairs = stats.paired(table, x, y, list(table))
            if pairs:
                d = stats.bootstrap_diff(pairs, B=a.B)
                b_, c_ = stats.discordant(pairs)
                d.pop("samples", None)
                d.update(discordant=[b_, c_], sign_p=stats.sign_test(b_, c_), n=int(sum(len(v[0]) for v in pairs.values())))
                out[f"{x}-{y}"] = d
        res[scope] = out
        print(f"[{scope}] tasks {tasks}")
        for k in ARMS:
            v = out[k]
            print(f"   {k:7s} {v['k']}/{v['n']} = {v['rate']:.3f} over {v['tasks']} tasks")
        for k, d in out.items():
            if "-" in k:
                print(f"   {k}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] over {d['n']} pairs, discordant {d['discordant']}, "
                      f"sign p {d['sign_p']:.4f}")
    json.dump(res, open(os.path.join(R, "r18_recovery_stats.json"), "w"), indent=1)
    if a.record:
        ledger.record("r18_branch", bench="libero_90", tag="recovery", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/r18_recovery_stats.py", command="python experiments/r18_recovery_stats.py --record",
                      artifacts=[os.path.join(R, "r18_recovery_stats.json")])
        print("[r18] ledger row written")


if __name__ == "__main__":
    main()
