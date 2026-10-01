"""Pre-registration of P0b, the offline readability probe (ledger m3_prereg tag zc_probe), written before it runs."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="zc_probe", seed=0,
    metrics={
        "plan": "the overhaul of 2026-10-01 (Read -> Compose -> Fold -> Consolidate): zero-shot composition of pi0.5's own "
                "conditionals needs the policy to carry WHERE a named object is; this probe measures it offline",
        "parser_hash": F.parser_hash(),
        "data": "first observations after the client's 10 settle steps, practice init states 0-9 only (experiments/"
                "f0_firstframes.py), every movable object's position; LIBERO-90 (90 tasks) and the training suites goal, "
                "object, 10 as controls",
        "prompts": "per task and candidate object X (movable types in the scene, the task's own targets excluded): the "
                   "task's instruction with its object phrase replaced by X's (factors.substitute), and the lowest-index "
                   "training instruction that moves X (factor-matched); all prompts of one observation share the noise",
        "measures": {"PA": "pointed-object accuracy: the first chunk's commanded hand displacement (0.05 m x sum of the "
                           "10 actions' xyz) points (xy cosine) at X more than at any other candidate; chance 1/m",
                     "CC": "contrast cosine cos(d(X) - d(X'), p_X - p_X') over candidate pairs (xy)",
                     "LS": "spread of d across the substituted prompts divided by its spread across two noise seeds",
                     "PA_w": "PA of the factor arithmetic d_ref + w (d(X) - d_ref), d_ref = the mean over the candidates' "
                             "prompts, w in {1, 2, 4, 8}"},
        "groups": {"validity": "LIBERO-Goal tasks that move an object", "object": "LIBERO-Object layouts",
                   "solved": "LIBERO-90 tasks pi0.5 solves zero-shot (>= 0.6)", "failed": "LIBERO-90 tasks it fails",
                   "held_B": "held-out group B (factors named only elsewhere; f0_coverage)",
                   "held_C": "held-out group C"},
        "decisions": {"validity": "LIBERO-Goal PA >= 0.8 at w = 1; otherwise the measurement is fixed before anything "
                                  "held-out is read",
                      "go": "held_B PA >= 0.6 at some w <= 4 -> pre-register the online composition pilot (P1)",
                      "kill": "held_B PA <= 1/m + 0.1 at every w <= 8 -> zero-shot object composition is dead on pi0.5; the "
                              "fold-first route (factor tokens) becomes the main line",
                      "between": "the online which-object-moves probe decides"},
        "written_before": "any first-chunk prediction of this probe",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zc_probe.py", command="pre-registration of the offline readability probe"))
