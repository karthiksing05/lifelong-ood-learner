"""Pre-registration of the prototype-PoE plan's offline probes P0-P2 (ledger m3_prereg tag q_probes), written before any of them ran."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero", tag="q_probes", seed=0,
    metrics={
        "plan": "~/.claude/plans/i-m-interested-in-re-workshopping-quizzical-crane.md (Prototype PoE planning: Wang et al. "
                "2605.07078 ported to pi0.5); one demonstration may be the query, used once for selection, never executed",
        "P0": {"script": "experiments/q0_share.py", "question": "does the policy's share follow the expert weight at the last "
               "Euler steps (a ~= (1 - w_late) a_pi + w_late q)?", "kill": "the early weight moving the output by more than 0.2 of "
               "||q - a_pi|| (median over frames) -> revisit the sampler model before anything else"},
        "P1": {"script": "experiments/b2_steer_offline.py (scene-option generators)", "question": "do observation-grounded options "
               "exist and point at objects?", "kill": "the null-prompt generator is dropped if it shows <= 1 cluster holding >= 10% "
               "of the ascent starts; scene options are killed if no generator reaches contrast >= 0.4 and pointing gain >= 0.1"},
        "P2": {"script": "experiments/q2_explain.py", "question": "does the expert pool explain held-out demos, and does a code built on "
               "demo 0 transfer to demo 1 with options recomputed at demo 1's observations?", "kill": "options cut the demo NLL by "
               "< 10%, or the recomputed code beats the policy alone on < 12 of 21 tasks -> stop before rollouts"},
        "written_before": "any output of q0_share.py, the P1 generators or q2_explain.py",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q_probes.py", command="pre-registration of P0-P2"))
