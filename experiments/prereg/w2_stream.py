"""Pre-registration of plan W's read W2 (ledger m3_prereg tag w2_stream), written before any of its fits and episodes
(experiments/w2_stream.py).  W1's stream (same orders, prior library, request rule: no demonstration when the better of the policy
and the composition reaches 8 of 10 practice successes) with RE-VALIDATION: after every refit, each earlier arrival served by the
library whose composition changed re-runs its practice (states 0-9, 10 episodes, seed 13) and must pass the rule it was kept under
(beat the policy, and reach 8 of 10 without a demonstration of its own); one that fails and never gave its demonstration requests it
then; one that gave it falls back to the policy.  Served: the final library's composition for arrivals still served by it, else the
policy; tested on E3's fresh start states (set b), seed 13."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
st = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/w2_state.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="w2_stream", seed=13,
    metrics={"orders": {o: st[o]["order"] for o in ("a", "b")}, "theta": 8,
             "hypotheses": {"H1": "per order, served arms > zero-shot (exact sign test; Holm over the two orders)",
                            "H2": "NOTHING DAMAGED, per order: on the arrivals that first kept a composition, the arm serving them at the "
                                  "end (the final library, or the policy after a fall-back) is non-inferior to the arrival-time composition "
                                  "(task-level bootstrap 95% lower bound of the difference above -0.05)",
                            "reported": "demonstrations requested (late ones counted), fall-backs, re-checks; served against the per-task "
                                        "LoRA with acceptance, against E4's and W1's served arms, against E3's all-at-once library"},
             "context": "E4 was read before this registration (no damage held in order a, not in order b; losses on thin-margin keeps "
                        "and on an arrival whose concepts later refits changed); W1 is running",
             "written_before": "any fit or episode of w2"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/w2_stream.py", command="pre-registration of plan W's read W2"))
