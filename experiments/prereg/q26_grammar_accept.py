"""Pre-registration (ledger m3_prereg tag q26_grammar_accept), written before any of its episodes: the acceptance test of q13 applied
to the grammar arm of q24_grammar.  Acceptance: the grammar arm and zero-shot on init states 0-9, 10 episodes, noise seed 6
(experiment q26_accept); keep the grammar only with MORE successes.  Test: q24_main's episodes (init states 30-49, seed 6) of the kept
arm against zero-shot."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q26_grammar_accept", seed=6,
    metrics={"tasks": "the 14 tasks of q24_grammar", "rule": "the acceptance rule fixed in q13_main (ties to the policy)",
             "hypotheses": {"H1": "success: kept > zero-shot pooled over 14 tasks (exact sign test)",
                            "H2": "named object moved first: kept > zero-shot", "correction": "Holm over H1, H2"},
             "disclosure": "q24_grammar's test episodes were read before this registration and are reused; the acceptance "
                           "episodes are new and decide alone which arm each task keeps",
             "written_before": "any episode of q26_accept"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q26_grammar_accept.py", command="pre-registration: acceptance test on the grammar arm"))
