"""Amendment to q20_stream (ledger m3_prereg tag q20_confirm), written before any confirmation episode: a CONFIRMATION round for
the acceptance test.  A form kept on init states 0-9 is stored only if it again has more successes than the policy on init states
10-19 (10 episodes each, noise seed 5, experiment q20_confirm); otherwise the policy is kept.  The q20_main test episodes (init
states 30-49) are reused: a dropped form is scored by the zero-shot episodes."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q20_confirm", seed=5,
    metrics={
        "rule": "keep a form only if it beats the policy on BOTH the acceptance states 0-9 and the confirmation states 10-19",
        "hypotheses": {"H1": "order A with confirmation: kept > zero-shot pooled over 48 (exact sign test)", "H2": "order B: the same",
                       "correction": "Holm over H1, H2"},
        "disclosure": "proposed after reading q20_stream, where one kept form lost at test (task 70, 4 vs 16) as in q18_self; the "
                      "confirmation episodes are new and the rule is fixed here; the test episodes are q20_main's",
        "written_before": "any episode of q20_confirm",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q20_confirm.py", command="amendment: confirmation round"))
