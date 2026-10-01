"""Pre-registration of the composition pilot (ledger m3_prereg tag zc_pilot): the zc_bind 'otherwise' branch."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="zc_pilot", seed=0,
    metrics={
        "why": "zc_bind: object words are read in LIBERO-Object layouts 0.43 of the time (neither >= 0.7 nor <= 0.3) and not "
               "in living rooms (non-habit objects 0.0-0.1); zc_probe failed its validity gate (LIBERO-Goal first-chunk "
               "pointing 0.43 < 0.8) but its contrast cosines were +0.72 in Goal and +0.5-0.7 in the living rooms; so the "
               "pre-registered branch is the composition pilot",
        "parser_hash": F.parser_hash(),
        "tasks": [48, 49, 52, 53, 58, 59, 62],
        "code": "factor_hooks.code_for: base = the readable same-scene sibling with the same relation, object factor = "
                "(the base with the task's object, the base), gated to the approach (gripper open) unless 'all'",
        "arms": {"fcomp@2": "lambda 2, approach", "fcomp@4": "lambda 4, approach", "fcomp@8": "lambda 8, approach",
                 "fcompall@4": "lambda 4, always on", "cfg@4": "control: 4 v(task) - 3 v(empty prompt)",
                 "placebo@4": "control: a random training-instruction pair at lambda 4"},
        "episodes": "10 per arm and task, practice init states 0-9, light recordings with every object's pose; zero-shot "
                    "reference from zc_bind's own-instruction runs",
        "decision": "go to the pre-registered 30-49 read only if the best composition arm reaches >= 10 of 70 successes and "
                    "beats both cfg@4 and placebo@4 by >= 5 of 70; otherwise composition of object words is stopped on "
                    "pi0.5 and the fold (factor tokens) is the main line",
        "written_before": "any episode of this pilot",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zc_pilot.py", command="pre-registration of the composition pilot"))
