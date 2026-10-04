"""Pre-registration of plan E's main read (ledger m3_prereg tag e1_factored), written before any of its fits and episodes
(docs/PLAN_FACTORED.md, plan E).  The stream's 30 tasks.  Every task is composed from two consolidated prototypes fitted on each
LIBERO-90 task's OWN scene's demonstration (data.episodes_of_task; ledger l90_scenes): the OBJECT primitive of its object key
(rank 4, approach frames of every demonstration with that object; c0_fit --joint --approach) switched on until the policy's grasp,
and the GOAL primitive of its goal key (rank 4, transport frames of every demonstration with that goal; c0_fit --joint --goal)
from the grasp to the release.  Arms: zs; fac (the factored library); lora (the per-task one-demo LoRA on the task's own scene's
demonstration, refit for the 5 stream tasks whose earlier LoRA had another scene's; launch_pack arm lorasc).  Acceptance on stored
init states 0-9, 10 episodes, noise seed 12 (experiment e1_accept): the scheme keeps fac only if it beats zs; the baseline keeps
lora only if it beats zs.  Test: all arms on the 20 fresh start states per task, noise seed 12 (experiment e1_main)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import data as D
plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/e1_plan.json"))
d90 = D.Demos90()
donors = sorted({t for v in plan.values() for t in v["obj_tasks"] + v["goal_tasks"] if d90.episodes_of_task(t)})
protos = sorted({v["obj_name"] for v in plan.values()} | {v["goal_name"] for v in plan.values()})
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="e1_factored", seed=12,
    metrics={"tasks": sorted(int(t) for t in plan), "plan": plan, "prototypes": protos, "demonstration_tasks": donors,
             "hypotheses": {"H1": "the scheme's kept arm > zero-shot pooled over the 30 tasks (exact sign test on discordant pairs, alpha 0.05)",
                            "H2": "NON-INFERIORITY to the per-task LoRA with the acceptance test, on the 29 tasks with a LoRA: the "
                                  "task-level bootstrap 95% lower bound of (scheme - LoRA_acc) success rate above -0.05",
                            "correction": "H1 alone in the Holm family; H2 is a separate pre-specified criterion"},
             "reported": "bytes (the library: one rank-4 primitive per object key and per goal key; per-task LoRA 29 x 18.8 MB), "
                         "demonstrations used (the scheme: one per LIBERO-90 task sharing an object or goal key with the stream; the "
                         "baseline: the scored task's own), fac vs zs, lora vs zs, fac vs lora, per goal key",
             "context": "e0_object (exploratory; the stream's 30 tasks, stored states 0-9, seed 11; first-episode demonstrations, "
                        "another scene's for 5 tasks) was read before this registration, and the data defect (ledger l90_scenes) was "
                        "found and fixed before it",
             "written_before": "any fit or episode of e1"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/e1_factored.py", command="pre-registration of plan E's main read"))
