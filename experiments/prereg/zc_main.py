"""Pre-registration of the main composition read (ledger m3_prereg tag zc_main), after the pilot's go (ledger zc_pilot)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

TASKS = [18, 26, 48, 49, 52, 53, 58, 59, 62, 65, 66, 71]
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="zc_main", seed=0,
    metrics={
        "pilot": "zc_pilot: fcomp@8 14 of 70 vs cfg@4 4 and placebo@4 8 on practice states (go)",
        "parser_hash": F.parser_hash(),
        "tasks": TASKS,
        "task_rule": "every LIBERO-90 task pi0.5 fails zero-shot (< 0.6 in ledger/zeroshot90) whose factor_hooks.code_for has an "
                     "object factor (a readable same-scene sibling with the same relation); 5 of them are among the 21 "
                     "held-out tasks (48, 52, 53, 58, 62)",
        "arms": {"fcomp@8": "the code: base = the readable sibling, + 8 [v(sibling with the task's object) - v(sibling)], "
                            "gated to the approach (gripper open)",
                 "cfg@8": "control: 8 v(task) - 7 v(empty prompt)",
                 "placebo@8": "control: a random training-instruction pair at weight 8",
                 "zeroshot": "pi0.5 under the task's instruction"},
        "episodes": "20 per arm and task, init states 30-49, SERVE_SEED 0, light recordings with every object's pose",
        "hypotheses": {"H1": "fcomp@8 > zeroshot: exact sign test p < 0.05 AND two-level bootstrap one-sided 95% lower bound of "
                             "the paired difference > +0.05",
                       "H1b": "fcomp@8 > cfg@8 (exact sign test; Holm with H1c)",
                       "H1c": "fcomp@8 > placebo@8",
                       "descriptive": "per task success and binding accuracy (experiments/f1_binding.py); code bytes"},
        "written_before": "any episode of this read",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zc_main.py", command="pre-registration of the main composition read"))
