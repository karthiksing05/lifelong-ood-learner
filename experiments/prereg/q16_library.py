"""Pre-registration of the library read (ledger m3_prereg tag q16_library), written before any of its episodes: a new task gets NO
demonstration; its candidate binding forms point attention with object prototypes ALREADY IN THE LIBRARY (other tasks' demos, same
object type), and the acceptance test on the learner's own practice keeps one of them or the policy alone."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q16_library", seed=2,
    metrics={
        "tasks": [12, 2, 13, 14, 79, 15, 70, 43],
        "excluded": {"52": "no library prototype of its object type (milk)"},
        "library": "results/robot/q4_object_protos.npz: one 4 KB prototype per task demo, typed by the moved referent; the task's "
                   "OWN prototype is never a candidate",
        "candidates": "up to 6 library prototypes of the moved object's type from other tasks: same scene first, then other scenes, "
                      "each by task index (12: 13 14 15 16 17 2; 2: 12-17; 13: 12 14 15 16 17 2; 14: 12 13 15 16 17 2; 15: 12 13 14 "
                      "16 17 2; 79: 77 78 80 73 74 75; 70: 69 5; 43: 36 37); arm bindw4@rho0.7_approach_src<k>",
        "acceptance": "experiment q16_accept: zero-shot and every candidate on init states 0-9, 10 episodes, noise seed 2; keep the "
                      "arm with the most successes, ties to the policy, then the earlier candidate",
        "test": "experiment q16_main: the kept arm, zero-shot and the task's own-demo form (reference) on init states 30-49, 20 "
                "episodes, noise seed 2",
        "hypotheses": {"H1": "kept > zero-shot pooled over the 8 tasks (exact sign test on discordant pairs by init state)",
                       "H2 (descriptive)": "kept vs the own-demo form: successes and the two-sided sign test",
                       "correction": "H1 alone is confirmatory"},
        "bytes": "kept unit = a library index (no new prototype stored)",
        "context": "the borrowed-prototype screen q14_borrow (states 0-9, noise seed 0) was seen before this registration",
        "written_before": "any episode of q16_accept or q16_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q16_library.py", command="pre-registration of the library read"))
