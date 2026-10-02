"""Pre-registration of the stream read (ledger m3_prereg tag q17_stream), written before any of its episodes: the 48 failing
LIBERO-90 tasks with a moved object arrive one at a time; a demonstration is requested only for an object type the library lacks;
the acceptance test keeps a library form or the policy alone (experiments/q17_stream.py; plan in results/robot/q17_plan.json)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/q17_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q17_stream", seed=3,
    metrics={
        "tasks": sorted(int(t) for t in plan["info"]),
        "orders": plan["orders"],
        "rule": "an arrival whose object type is not in the library requests one demonstration, used once: its first grasp gives "
                "a 4 KB object prototype (experiments/q4_template.py) that joins the library and is its only candidate; an arrival "
                "whose type is known requests nothing and its candidates are up to K=4 library prototypes of the type (same scene "
                "first, then other scenes, each in arrival order)",
        "acceptance": "experiment q17_accept: zero-shot and every candidate form (bindw4@rho0.7_approach_src<k>) on init states "
                      "0-9, 10 episodes, noise seed 3; keep the most successes, ties to the policy, then the earlier candidate",
        "test": "experiment q17_main: the kept arm and zero-shot on init states 30-49, 20 episodes, noise seed 3 (a kept policy "
                "reuses the zero-shot episodes)",
        "hypotheses": {"H1": "order A: kept > zero-shot pooled over the 48 arrivals (exact sign test on discordant pairs)",
                       "H2": "order B (the mirror): the same", "correction": "Holm over H1, H2",
                       "H3 (descriptive)": "a demonstration for every task (candidate: its own prototype): kept vs zero-shot"},
        "reported": "demonstrations requested per order (planned: " + json.dumps(plan["demos"]) + " of 48), bytes stored "
                    "(4 KB per requested prototype), forms kept",
        "context": "the 48 tasks are those of the broad screen q9_w4screen (states 0-9, seed 0, seen); q13_main and q16_library read "
                   "9 of them; all episodes here are new (seed 3)",
        "written_before": "any episode of q17_accept or q17_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q17_stream.py", command="pre-registration of the stream read"))
