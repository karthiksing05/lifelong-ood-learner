"""Pre-registration of plan E's replication (ledger m3_prereg tag e5_replicate), written before any of its fits and episodes
(experiments/e5_read.py).  The 37 single-step pick-and-place LIBERO-90 tasks outside the stream that have a demonstration of their
own scene; none was used by any design decision.  The hierarchical factored library (E3's prototypes, plus 26 new object, instance
and goal prototypes fitted the same way on every LIBERO-90 demonstration of their concept) against the per-task one-demo LoRA on
the task's own scene.  Acceptance on stored states 0-9 (10 episodes, seed 14): the scheme keeps the library only if it beats the
policy; the baseline keeps its LoRA only if it beats the policy.  Test: 20 fresh start states per task (LIBERO's sampler at seeds
200000-200019), seed 14."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/e5_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="e5_replicate", seed=14,
    metrics={"tasks": sorted(int(t) for t in pl["tasks"]), "plan": pl,
             "hypotheses": {"H1": "the kept arm > zero-shot on the 37 tasks (exact sign test, alpha 0.05)",
                            "H2": "non-inferiority to the per-task LoRA with acceptance on the 37 tasks (task-level bootstrap 95% lower "
                                  "bound of the difference above -0.05)",
                            "H3": "the same non-inferiority POOLED with E3's 29 tasks (66 tasks; E3's own runs at seed 13). E3's half was "
                                  "read before this registration; H3 is the criterion fixed now for the combined evidence",
                            "correction": "H1 alone in the Holm family; H2 and H3 are separate pre-specified criteria"},
             "context": "E3 (non-inferiority not established on 29 tasks; between-task variance dominated), the streams, W0 and T0 were read "
                        "before; the parser now reads 'in the middle' as a qualifier (task 86)",
             "written_before": "any fit or episode of e5"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/e5_replicate.py", command="pre-registration of plan E's replication"))
