"""Pre-registration of the fully deployable broadened scheme's read (ledger m3_prereg tag q10_main), written before any of its practice or scoring
episodes: the binding form found by the broad screen (ledger b2_q2 tag q9_w4screen) is practised, folded into a token, the token
practises alone and is refit; at test the token alone (instruction + 18.4 KB) against zero-shot."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q10_main", seed=0,
    metrics={
        "screen": "the 48 tasks of q8_screen with the DEPLOYABLE form: attention at mass 0.7 on the 4 patches most like the demo's object prototype (experiments/q4_template.py), no simulator state anywhere; 10 practice episodes on states 0-9 (ledger b2_q2 tag q9_w4screen); task 43 was read in q6_main and is excluded",
        "tasks": [12, 2, 13, 14, 79, 15, 70],
        "form_rule": "practice success >= 0.2 in the deployable screen",
        "procedure": "experiments/q6_chain.py --form bindw4@rho0.7_approach --pre q10: the form on states 10-29 (recorded) -> token "
                     "(_q101) -> the token alone on 10-29 -> refit on both rounds (_q102) -> init states 30-49, 20 episodes: zs, form, "
                     "token alone",
        "hypotheses": {"H1": "token alone > zs pooled over the 7 tasks (exact sign test on discordant pairs)",
                       "H2": "form > zs pooled", "correction": "Holm over H1, H2"},
        "deployability": "nothing but the demo's 4 KB object prototype and the frozen policy's own attempts; the token arm uses only the instruction",
        "written_before": "any episode of q10_p1, q10_p2 or q10_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q10_main.py", command="pre-registration of the fully deployable broadened scheme's read"))
