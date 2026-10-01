"""Pre-registration of P3c: cross-task composition under rules v2, through the frozen prior (ledger m3_prereg tag p3c)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

tab = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/p3c_table.json"))["table"]
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="p3c", seed=0,
    metrics={
        "plan": "archive/2026-10-program/PLAN_TRANSFER.md C3 (composition = transfer), retested after rules v2 made each task's own phases reliable",
        "why": ("P3a/P3b composed bindings with the no-policy executor under rules v1 and failed on new targets (11/80); v2 (ledger "
                "p1_bindings tag v2) executes a task's own phases through the frozen prior at 385/420; P3c asks whether a task can be "
                "composed from OTHER tasks' phases with no demo of its own"),
        "program": ("lifelong_ood/policy/phase.py compose: the approach of the grasp source (a demo that manipulates the target's object "
                    "TYPE) + the carry and the rest of the place source (a demo with the target's goal RELATION: predicate + receptacle "
                    "region), relabelled onto the target scene's objects, re-anchored on their current poses (simulator, oracle), "
                    "followed by the frozen prior as in phasebind2_nofold; the target's own demo is not used"),
        "table_rule": "experiments/p3c_table.py (lowest-numbered other source; place source other than the grasp source when possible)",
        "table": {t: dict(g=v["g"], p=v["p"]) for t, v in tab.items()},
        "arms": {"compose2": "grasp source + place source", "sibling2": "the place source's WHOLE demo (another object type's grasp): control",
                 "own": "phasebind2_nofold (the target's own demo; existing m3_final rows)"},
        "tasks": f"targets {sorted(int(t) for t in tab)}; init states 30-49; {20 * len(tab)} paired episodes per arm",
        "hypotheses": {
            "C1": "PRIMARY: compose2 non-inferior to own at -0.10: one-sided 95% lower bound of the paired difference (two-level bootstrap) >= -0.10",
            "C2": "compose2 > sibling2: exact two-sided sign test p < 0.05 (the object's grasp binding carries across tasks)",
            "descriptive": "per target; compose2 vs the control (borrowing + fold) and vs LoRA + failure negatives",
        },
        "written_before": "any compose2 or sibling2 rollout at init states 30-49 (a 3-episode smoke test on task 48, experiment smoke, is excluded)",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/p3c.py", command="pre-registration of P3c"))
