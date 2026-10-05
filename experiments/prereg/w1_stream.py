"""Pre-registration of plan W's read W1 (ledger m3_prereg tag w1_stream), written before any of its fits and episodes
(experiments/w1_stream.py; docs/PLAN_WANG.md).  E4's stream (same two orders, same prior library, same library operations) with
the thesis's request rule: an arrival keeps the better of {the policy, the library's composition} (ties: the policy) WITHOUT a
demonstration when it succeeds in at least 8 of its 10 practice episodes (stored states 0-9, noise seed 13); otherwise it requests
its one demonstration, which refits every concept on its path, and keeps the refit composition only if it beats the policy.
Served at the end: the final library's composition where kept, else the policy; tested on E3's fresh start states (set b), seed 13."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
st = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/w1_state.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="w1_stream", seed=13,
    metrics={"orders": {o: st[o]["order"] for o in ("a", "b")}, "theta": 8,
             "hypotheses": {"primary": "demonstrations requested per order, against E4's on the same order (counts)",
                            "H1": "per order, served arms > zero-shot (exact sign test; Holm over the two orders)",
                            "H2": "per order, served arms NON-INFERIOR to E4's served arms on the same tasks and start states (task-level "
                                  "bootstrap 95% lower bound of the difference above -0.05)",
                            "H3": "no damage, as E4's H2 (final against arrival-time compositions, non-inferiority -0.05)"},
             "context": "E4 is running (its arrivals so far were seen); W0 (query selection by flow-loss gain) failed its gates; the "
                        "threshold of 8 of 10 was fixed before any W1 episode",
             "written_before": "any fit or episode of w1"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/w1_stream.py", command="pre-registration of plan W's read W1"))
