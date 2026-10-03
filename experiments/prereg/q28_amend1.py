"""Amendment 1 to q28_bindprim (ledger m3_prereg tag q28_amend1), written before the rerun: a defect in the first run.  Six of the
16 tasks (9 72 46 47 50 54) had no stored object prototype, so (a) their binding arm could not run at test and (b) c0_fit --bind
silently trained without binding on those tasks' frames whenever they were donors, so most binding-aware primitives were trained
under binding on only part of their frames.  The first run's binding arm is void.  Rerun: prototypes for the six tasks
(q4_template.py, the same construction), c0_fit --bind now refuses a frame without a prototype, all 16 binding-aware primitives
refit (names ...b2), their arm rerun on the same acceptance (stored states 0-9) and test (fresh states) episodes, noise seed 10.
The plain and zero-shot arms of the first run are kept.  Hypotheses, rules and correction unchanged."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q28_amend1", seed=10,
    metrics={"defect": "missing prototypes for tasks 9 72 46 47 50 54: binding arm absent there, binding-aware training partial",
             "seen_before": "the first run's partial read (10 of 16 tasks): H1 37 vs 31 of 200, H2 26 vs 11 of 200",
             "rerun": "prototypes built; all 16 binding-aware primitives refit (...b2) with binding on every frame before the first "
                      "grasp; binding arm rerun at acceptance and test; plain and zs arms kept",
             "written_before": "any fit or episode of the rerun"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q28_amend1.py", command="amendment: q28 defect and rerun"))
