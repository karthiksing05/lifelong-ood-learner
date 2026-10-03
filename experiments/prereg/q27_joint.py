"""Pre-registration of the consolidated-primitive read (ledger m3_prereg tag q27_joint), written before any of its episodes: plan C
with consolidation (docs/PLAN_PRIMITIVES.md).  For each of the 20 pool tasks in results/robot/q27_plan.json, each step whose key is
shared by >= 3 OTHER LIBERO-90 tasks gets one rank-16 primitive fitted jointly on those tasks' demos (c0_fit --joint; the scored
task's own demo never enters); other steps run the policy itself.  The acceptance test keeps the primitive arm or the policy."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/q27_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q27_joint", seed=7,
    metrics={
        "tasks": sorted(int(t) for t in plan),
        "steps": {t: [(p["key"], p["donors"]) for p in v["steps"]] for t, v in plan.items()},
        "fits": "experiments/c0_fit.py --joint (rank 16, 1000 steps); the 7 fits of ledger c1b_joint (tasks 12 13 14 30 15 25 31) "
                "were made by this procedure and are reused",
        "acceptance": "experiment q27_accept: zero-shot and the primitive arm on init states 0-9, 10 episodes, noise seed 7; keep the "
                      "primitive arm only with MORE successes",
        "test": "experiment q27_main: both arms on init states 30-49, 20 episodes, noise seed 7",
        "hypotheses": {"H1": "kept > zero-shot pooled over the 20 tasks (exact sign test on discordant pairs)",
                       "H2": "the primitive arm unfiltered > zero-shot pooled", "correction": "Holm over H1, H2"},
        "context": "c1b_joint (exploratory, states 0-9, seed 0) was read before this registration: 'on top of the cabinet' 19/30 vs "
                   "2, 'on the plate' 7/40 vs 17",
        "bytes": "one primitive per step key in a stream (18.7 MB each); this leave-one-task-out read fits one per task and key",
        "written_before": "any episode of q27_accept or q27_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q27_joint.py", command="pre-registration of the consolidated-primitive read"))
