"""Pre-registration of the prototype-PoE pilot P3 (ledger m3_prereg tag q_pilot), written before any of its episodes, with the
revision P0's kill rule required."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q_pilot", seed=0,
    metrics={
        "p0_outcome": "KILL fired (ledger q0_share): an early prototype weight moves the output by 0.38 of ||q - a_pi|| even with the "
                      "policy alone at the last three steps (limit 0.2); the last step does not set the policy's share alone",
        "revision": "one CONSTANT expert weight over all Euler steps; the policy family's share is a FLOOR lambda on every action "
                    "dim (lifelong_ood/policy/poe.make_beta), the gripper from the policy family; the selected prototypes enter as "
                    "Wang's precision-weighted product (poe.proto_product), and only prototypes reached FROM the demo phase are "
                    "candidates (poe.query_nodes); experts are mixed in velocity space (a one-hot weight is the policy exactly); "
                    "observation sensitivity at a constant prototype weight w (P0): w 0.5 -> 0.33, so lambda >= 0.5 keeps it >= 0.3",
        "p2_outcome": "passes (ledger q2_explain): options cut the fit NLL by 0.57; the demo-0 code beats the policy on demo 1 in 21 "
                      "of 21 tasks (by rms 15 of 21 with prototypes, 19 of 21 options only); recomputed options beat frozen ones in "
                      "only 10 of 21",
        "tasks": [35, 36, 39, 42, 1, 6],
        "codes": "experiments/q3_build.py on demo 0 of each task (results/robot/poe_codes/<task>_d0.json)",
        "arms": {"zs": "pi0.5 zero-shot", "poe@0.3 / poe@0.5 / poe@0.7": "the phase codes with policy-share floor lambda",
                 "poeopt@0.5": "the codes without their prototypes (policy experts only)",
                 "poenopol@0.5": "the no-policy control: the prototype product alone on the arm dims"},
        "episodes": "10 per arm and task, practice init states 0-9",
        "go_rule": "go to the pre-registered main read with the lambda >= 0.5 arm of highest pooled success if at least 3 of the 6 "
                   "tasks gain >= 3/10 over zero-shot at that lambda; success only at lambda = 0.3 (sensitivity < 0.3) or with the "
                   "no-policy control matching ours (within 2 of 60) = the policy is bypassed -> stop",
        "written_before": "any episode of this pilot",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q_pilot.py", command="pre-registration of the PoE pilot"))
