"""Pre-registration of the binding-aware primitive read (ledger m3_prereg tag q28_bindprim), written before any of its fits and
episodes: on the two step keys whose tasks differ in WHICH object (\"on the plate\", \"in the basket\"), each task gets two
leave-one-out primitives fitted on the other tasks' demonstrations: plain (c0_fit --joint) and trained under attention binding
(c0_fit --joint --bind), the latter run with binding at test (the scored task's own demonstration's object prototype as the query).
The acceptance test keeps one of {policy, plain, binding-aware} per task."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/q28_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q28_bindprim", seed=10,
    metrics={"tasks": sorted(int(t) for t in plan), "plan": plan,
             "fits": "rank 16, 1000 steps (as c1b_joint and c4_bind); the 8 fits of tasks 12 13 14 30 from c1b_joint / c4_bind reused",
             "acceptance": "experiment q28_accept: zs, plain primitive, binding-aware primitive + binding on stored init states 0-9, 10 "
                           "episodes, noise seed 10; keep the most successes, ties to the policy, then plain",
             "test": "experiment q28_main: all three arms on the 20 fresh start states, noise seed 10",
             "hypotheses": {"H1": "kept > zs pooled over the 16 tasks (exact sign test)",
                            "H2": "binding-aware + binding > plain primitive pooled", "correction": "Holm over H1, H2"},
             "context": "c4_bind (exploratory, the 4 plate tasks 12 13 14 30, states 0-9, seed 0) was read before: 15 vs 6 of 40",
             "written_before": "any fit or episode of q28"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q28_bindprim.py", command="pre-registration of the binding-aware primitive read"))
