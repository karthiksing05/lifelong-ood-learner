"""Pre-registration of the acceptance-test read (ledger m3_prereg tag q13_main), written before any of its episodes: the learner
keeps what its own fresh practice validates -- the policy alone, the binding form, or the folded token -- and the kept arm is
scored on fresh noise against zero-shot."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q13_main", seed=1,
    metrics={
        "tasks": [12, 2, 13, 14, 79, 15, 70, 43, 52],
        "arms": {"policy": "zs", "form": "bindw4@rho0.7_approach (the demo's 4 KB object prototype, no simulator state)",
                 "token": "ftoken@self_q102 (tasks of q10_main) / ftoken@self_w2 (43, 52 of q6_main): the tokens already folded"},
        "acceptance": "experiment q13_accept: each arm on init states 0-9 (never used to fit a token), 10 episodes, noise seed 1",
        "rule": "keep the arm with the most acceptance successes; ties go to the policy, then the token, then the form",
        "test": "experiment q13_main: all three arms on init states 30-49, 20 episodes, noise seed 1 (fresh draws; q6/q10 used seed 0)",
        "hypotheses": {"H1": "kept arm > policy, pooled over the 9 tasks (exact sign test on discordant pairs by init state)",
                       "H2": "token > policy pooled (replication of q10 H1 at a new seed)",
                       "H3": "form > policy pooled (replication of q10 H2)", "correction": "Holm over H1-H3"},
        "disclosure": "the rule was proposed after seeing q6_main and q10_main: there, choosing between form and token by the "
                      "fold-round practice (states 10-29) would have given 91 of 180 against 52 zero-shot (form 83, token 75). "
                      "The tasks are the same 9 (no other LIBERO-90 task passes the screen); the episodes, noise and acceptance "
                      "states are new",
        "bytes": "kept arm: policy 0, token 18432, form 4096 per task",
        "written_before": "any episode of q13_accept or q13_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q13_main.py", command="pre-registration of the acceptance-test read"))
