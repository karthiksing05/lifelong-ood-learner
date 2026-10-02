"""Pre-registration of the token-practice read (ledger m3_prereg tag q7_tokpractice), written before any of its episodes: does
one round of the folded token's OWN practice (rapid, no further demonstration) close the gap between the 18.4 KB demo token
(140 of 420) and the 18.7 MB one-demo LoRA (252 of 420)?"""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q7_tokpractice", seed=0,
    metrics={
        "tasks": [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63],
        "procedure": "the demo token of each task (results/robot/ftoken_<t>.pt, fitted on demo 0; ledger p1_bindings tag zc_fold: "
                     "140 of 420) practises alone (the task's instruction + the token) on init states 10-29, 20 episodes, real scene "
                     "recorded; a new token is fitted on demo 0 PLUS the successful episodes (experiments/f4_fold.py --source "
                     "demo+rollouts, tag _tp); a task with no practice success keeps its demo token",
        "scoring": "init states 30-49, 20 episodes per task, the same paired episodes as zc_fold",
        "hypotheses": {"H1": "practised token > demo token (zc_fold's token arm), exact sign test on discordant pairs",
                       "H2": "practised token non-inferior to the one-demo LoRA (252): one-sided 95% lower bound of the pooled "
                             "difference > -0.10 (two-level bootstrap)", "correction": "Holm over H1, H2"},
        "bytes": "18.4 KB per task (token) against 18.7 MB (LoRA)",
        "written_before": "any episode of q7_p1 or q7_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q7_tokpractice.py", command="pre-registration of the token-practice read"))
