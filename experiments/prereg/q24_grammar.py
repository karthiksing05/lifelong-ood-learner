"""Pre-registration of the grammar read (ledger m3_prereg tag q24_grammar), written before any of its episodes: the instance the
instruction's qualifier names, picked by the grammar (ledger q22_grammar) from the peaks of a TYPE prototype (the mean of other
tasks' prototypes of the type; ledger q23_typemap), with no demonstration of the scored task (experiments/q24_plan.py)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/q24_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q24_grammar", seed=6,
    metrics={
        "tasks": sorted(int(t) for t in plan),
        "arm": "bindw4@rho0.7_approach_type_qual_npk<n>: attention mass 0.7 while the gripper is open on the patches of the "
               "instance the grammar picks among the n strongest peaks of the type prototype's map; n per word learned from the "
               "other tasks with that word (results/robot/q24_plan.json)",
        "plan": {t: {"word": v["word"], "npk": v["npk"]} for t, v in plan.items()},
        "test": "experiment q24_main: the grammar arm and zero-shot on init states 30-49, 20 episodes, noise seed 6",
        "hypotheses": {"H1": "named object moved first (client bind tracking): grammar > zero-shot pooled over the 14 tasks "
                             "(exact sign test on discordant pairs by init state)",
                       "H2": "success: grammar > zero-shot pooled", "correction": "Holm over H1, H2"},
        "note": "many of these tasks also need a step binding does not supply (a shelf placement, 'and close it', a stack), so H2 "
                "is expected to be weaker than H1; no acceptance test here: this reads the grammar itself",
        "written_before": "any episode of q24_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q24_grammar.py", command="pre-registration of the grammar read"))
