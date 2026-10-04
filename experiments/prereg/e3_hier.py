"""Pre-registration of plan E's confirmatory read (ledger m3_prereg tag e3_hier), written before its tray refit and any of its
episodes (docs/PLAN_FACTORED.md).  The stream's 30 tasks; the HIERARCHICAL factored library (results/robot/e3_plan.json): E1's
object and goal prototypes, an instance child for each step that names an instance of its object (E2's from-scratch children:
back, front, middle black bowl), and the tray goal prototype refit with task 60 after the parser fix.  Arms: zs; hfac (the
library); lora (the per-task one-demo LoRA on the task's own scene, launch_pack arm lorasc); descriptive on the 4 instance tasks:
type (the black bowl's type prototype in place of the child).  Acceptance on stored init states 0-9, 10 episodes, noise seed 13
(e3_accept): the scheme keeps hfac only if it beats zs; the baseline keeps lora only if it beats zs.  Test: a NEW set of fresh start
states (results/robot/fresh_states_b: LIBERO's placement sampler at seeds 200000-200019, never used before), 20 per task, noise
seed 13 (e3_main)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/e3_plan.json"))
protos = sorted({v["object_proto"] for v in plan.values()} | {v["goal_proto"] for v in plan.values()})
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="e3_hier", seed=13,
    metrics={"tasks": sorted(int(t) for t in plan), "plan": plan, "prototypes": protos,
             "hypotheses": {"H1": "the scheme's kept arm > zero-shot pooled over the 30 tasks (exact sign test on discordant pairs, alpha 0.05)",
                            "H2": "NON-INFERIORITY to the per-task LoRA with the acceptance test, on the 29 tasks with a LoRA: the "
                                  "task-level bootstrap 95% lower bound of (scheme - LoRA_acc) success rate above -0.05 (the margin of "
                                  "D1 and E1, unchanged)",
                            "correction": "H1 alone in the Holm family; H2 is a separate pre-specified criterion"},
             "power": "projected from E1's test with the 4 instance tasks at the LoRA's level: difference -0.022, 95% interval -0.079 to "
                      "0.034; between-task differences carry 73% of the variance, so more episodes per task would not narrow it "
                      "(half-width 0.051 at 20, 0.048 at 40): H2 is expected to be inconclusive unless the other tasks' differences "
                      "shrink too",
             "reported": "bytes (22 rank-4 prototypes against 29 per-task LoRAs), the instance children against the type prototype "
                         "and against the LoRA on the 4 instance tasks, per goal key",
             "context": "e2_instance (exploratory; the 4 instance tasks, stored states 0-9, seed 12) and E1's test (fresh set a, seed 12) "
                        "were read before this registration; the test states here are a new set",
             "written_before": "the tray refit and any episode of e3"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/e3_hier.py", command="pre-registration of plan E's confirmatory read"))
