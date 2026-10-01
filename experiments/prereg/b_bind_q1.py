"""Pre-registration of the Bind plan's offline probes Q1a (where) and Q1b (does the bias turn the first chunk), ledger
m3_prereg tag b_bind_q1, written before either probe ran on any frame."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero", tag="b_bind_q1", seed=0,
    metrics={
        "plan": "~/.claude/plans/i-m-interested-in-re-workshopping-quizzical-crane.md (Bind and Chain, 2026-10-01 evening): a new "
                "task arrives as instruction + scene, NO demonstration; WHERE from the policy's own model first, an open-vocabulary "
                "detector as fallback, simulator segmentation only as an oracle",
        "parser_hash": F.parser_hash(),
        "data": "experiments/f0b_firstframes_seg.py: first frames of practice init states 0-9 of LIBERO-Goal, LIBERO-Object, "
                "LIBERO-90, LIBERO-10, LIBERO-Spatial with per-geom segmentation, depth, camera matrices",
        "tasks": "every task with one moved object and >= 2 candidate objects visible (oracle coverage >= 0.1 of some token); "
                 "groups: validity = LIBERO-Goal, object = LIBERO-Object, class_a = LIBERO-90 tasks 18 26 48 49 52 53 58 59 60 62 "
                 "65 66 71 38 70, other",
        "Q1a": {"script": "experiments/b1_where_probe.py",
                "W1": "pi0.5's PaliGemma attention from the referent's word tokens (instruction rewritten to name it) to the 256 "
                      "agent-view tokens, per layer; pointing = its peak token is covered >= 0.1 by the referent",
                "W3": "OWLv2 base patch16 ensemble, best box for the referent's phrase; same pointing",
                "W1_layer_rule": "the single layer with the highest pointing on validity is W1's layer everywhere else",
                "gate": "a where-source enters Q3 only if pointing >= 0.8 on validity AND >= 0.6 on class_a; W1 and W3 both "
                        "failing -> the deployable arm has no where-source and Q4 becomes a mechanism read (oracle + controls)",
                "diagnostic": "the action expert's attention mass on each candidate under the task's own instruction (no bias): "
                              "share of states where the most-attended candidate is the named object"},
        "Q1b": {"script": "experiments/b2_steer_offline.py",
                "configs": "target mass rho in {0.3, 0.5, 0.7} x layer band {all 18, mid 6-11}; fixed beta in {4, 8} all layers",
                "measures": "PA (first chunk's horizontal displacement points at the biased candidate), CC (contrast cosine "
                            "between two candidates' biased chunks and their positions), against PA/CC of the prompt swap "
                            "(the instruction naming the candidate) as calibration; override diagnostics OS, LS, SHIFT",
                "kill": "if CC < 0.1 AND PA gain over the unbiased chunk < 0.1 for EVERY config in the class_a group, BIND "
                        "stops before any rollout",
                "override_flag": "a config whose OS or LS median falls below 0.5 is marked overriding and excluded from Q2"},
        "written_before": "any output of b1_where_probe.py or b2_steer_offline.py",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/b_bind_q1.py", command="pre-registration of the Bind plan's offline probes"))
