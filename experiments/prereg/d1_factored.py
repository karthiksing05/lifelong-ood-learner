"""Pre-registration of plan D's main read (ledger m3_prereg tag d1_factored), written before any of its fits and episodes
(docs/PLAN_FACTORED.md).  The 30 tasks of the stream; for each, a leave-one-out GOAL primitive (rank 4, the post-grasp frames of
every other LIBERO-90 task with the same goal key; c0_fit --joint --goal; the scored task's demonstration never enters it).
Arms: zs; goal (the primitive from the policy's grasp to its release); goal_bind (the same + object binding before the grasp with the
task's own demonstration's object prototype as the query; absent for task 51, which has no demonstration); lora (the per-task
one-demo LoRA, the strong baseline).  Acceptance on stored init states 0-9, 10 episodes, noise seed 11 (experiment d1_accept): the
scheme keeps the best of {zs, goal, goal_bind} (ties: zs, then goal); the baseline keeps lora only if it beats zs.  Test: all arms
on the 20 fresh start states per task, noise seed 11 (experiment d1_main)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/d1_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="d1_factored", seed=11,
    metrics={"tasks": sorted(int(t) for t in plan), "plan": plan,
             "hypotheses": {"H1": "the scheme's kept arm > zero-shot pooled over the 30 tasks (exact sign test on discordant pairs)",
                            "H2": "NON-INFERIORITY to the per-task LoRA with the acceptance test, on the 29 tasks with a LoRA: the "
                                  "task-level bootstrap 95% lower bound of (scheme - LoRA_acc) success rate above -0.05",
                            "correction": "H1 alone in the Holm family; H2 is a separate pre-specified criterion"},
             "reported": "bytes (one rank-4 goal primitive per goal key: 4 x 4.7 MB; per-task LoRA 29 x 18.7 MB), demonstrations used "
                         "(the scheme: as queries for binding and in other tasks' primitives; the baseline: one fitted per task), "
                         "practice episodes",
             "context": "d0_goal (exploratory, 14 of these tasks, stored states, seed 0) was read before this registration",
             "written_before": "any fit or episode of d1"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/d1_factored.py", command="pre-registration of plan D's main read"))
