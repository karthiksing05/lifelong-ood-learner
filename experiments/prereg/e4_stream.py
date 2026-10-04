"""Pre-registration of plan E's stream read (ledger m3_prereg tag e4_stream), written before any of its fits and episodes
(experiments/e4_stream.py; docs/PLAN_FACTORED.md).  The hierarchical factored library learned as a stream: it starts from the
demonstrations of the LIBERO-90 tasks outside the stream (9 of the 22 concepts the stream needs); the 30 stream tasks arrive in two
seeded orders (a: seed 4, b: seed 5), each order with its own library.  An arrival composes from the current library (instance
child, else type concept; goal concept; a missing factor left to the policy) and requests its one demonstration only if the
composition fails the acceptance test (stored states 0-9, 10 episodes, noise seed 13, zero-shot from e3_accept; keep only if it
beats the policy); a demonstration updates every concept on its path (instance child, type concept, goal concept; each refit on all
its demonstrations so far, rank 4, c0_fit --joint), then acceptance again.  Served at the end: the FINAL library's composition where
the arrival kept the library, else the policy.  Test: E3's fresh start states (set b) at noise seed 13, paired with E3's zero-shot
and per-task LoRA runs (experiment e3_main); served and arrival-time compositions in experiment e4_main."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
st = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/e4_state.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="e4_stream", seed=13,
    metrics={"orders": {o: st[o]["order"] for o in ("a", "b")}, "concepts": st["_meta"]["need"],
             "prior_library": {k: e["tasks"] for k, e in st["a"]["library"].items() if e["tasks"]},
             "hypotheses": {"H1": "per order, the served arms > zero-shot pooled over the 30 tasks (exact sign test; Holm over the two orders)",
                            "H2": "NO DAMAGE, per order: on the arrivals that kept the library and whose composition changed after "
                                  "they arrived, the final library's composition is non-inferior to the arrival-time composition (task-level "
                                  "bootstrap 95% lower bound of the difference above -0.05)",
                            "reported": "demonstrations requested per order (of 29 that exist; task 51 has none), arrivals kept without a "
                                        "demonstration, prototypes and bytes; served arms against E3's per-task LoRA with acceptance and "
                                        "against E3's all-at-once library (descriptive)"},
             "context": "E3 (all demonstrations at once) was read before this registration; this read changes no method, only the "
                        "order in which demonstrations arrive and the rule that requests them",
             "written_before": "any fit or episode of e4"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/e4_stream.py", command="pre-registration of plan E's stream read"))
