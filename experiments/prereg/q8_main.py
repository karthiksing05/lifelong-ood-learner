"""Pre-registration of the broadened scheme's read (ledger m3_prereg tag q8_main), written before any of its practice or scoring
episodes: the binding form found by the broad screen (ledger b2_q2 tag q8_screen) is practised, folded into a token, the token
practises alone and is refit; at test the token alone (instruction + 18.4 KB) against zero-shot."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q8_main", seed=0,
    metrics={
        "screen": "48 failing LIBERO-90 tasks with a moved object, not screened before; form = the action expert's attention at mass "
                  "0.7 on the moved object's tokens (simulator segmentation: an oracle teacher during practice only) while the gripper "
                  "is open; 10 practice episodes on init states 0-9 (ledger b2_q2 tag q8_screen)",
        "tasks": [12, 13, 14, 2, 43, 70, 82],
        "form_rule": "practice success >= 0.2 in the screen",
        "procedure": "experiments/q6_chain.py --form bind@rho0.7_all_approach --pre q8: the form on states 10-29 (recorded) -> token "
                     "(_q81) -> the token alone on 10-29 -> refit on both rounds (_q82) -> init states 30-49, 20 episodes: zs, form, "
                     "token alone",
        "hypotheses": {"H1": "token alone > zs pooled over the 7 tasks (exact sign test on discordant pairs)",
                       "H2": "form > zs pooled", "correction": "Holm over H1, H2"},
        "deployability": "the form uses the simulator's segmentation (an oracle teacher during practice); the token arm uses only the "
                         "instruction and the frozen policy's inputs",
        "written_before": "any episode of q8_p1, q8_p2 or q8_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q8_main.py", command="pre-registration of the broadened scheme's read"))
