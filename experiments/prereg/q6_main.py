"""Pre-registration of the DEPLOYABLE scheme's read (ledger m3_prereg tag q6_main), written before any of its practice or
scoring episodes: the one demonstration supplies only an object prototype (where), everything else is the frozen policy's own
attempts; no simulator state anywhere."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q6_main", seed=0,
    metrics={
        "scheme": "demo 0's first grasp -> the moved object's patch in its first frame -> the policy's own image features there = "
                  "object prototype (experiments/q4_template.py, 4 KB); at every query the 4 most similar agent-view tokens get the "
                  "action expert's attention raised to mass 0.7 while the gripper is open (bindw4@rho0.7_approach); its successes "
                  "on practice states 10-29 fold an 18.4 KB token (_w1), the token practises alone on 10-29 and is refit on both "
                  "rounds (_w2); at test ONLY the task's instruction + the token",
        "form_rule": "tasks whose practice success with the form on states 0-9 is >= 0.2 (ledger b2_q2 tag q5_w4): 43 (0.8), 52 (0.2); "
                     "the other 20 tasks get no form",
        "tasks": [43, 52],
        "scoring": "init states 30-49, 20 episodes per arm: zs, form, token alone",
        "hypotheses": {"H1": "token alone > zs pooled over the 2 tasks (exact sign test on discordant pairs)",
                       "H2": "form > zs pooled", "correction": "Holm over H1, H2"},
        "context": "with the oracle where (simulator segmentation) the same binding solved tomato sauce into the tray 10/10 "
                   "(b_q2full); this read measures the scheme with no oracle at all",
        "written_before": "any episode of q6_p1, q6_p2 or q6_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q6_main.py", command="pre-registration of the deployable scheme's read"))
