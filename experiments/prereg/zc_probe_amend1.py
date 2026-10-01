"""Amendment 1 to the readability probe's pre-registration (ledger m3_prereg tag zc_probe_amend1), before the full run."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="zc_probe_amend1", seed=0,
    metrics={"amends": "zc_probe",
             "change": "object substitution reads the task's own object phrase from its instruction (LIBERO-Goal says 'the "
                       "bowl', LIBERO-Spatial 'the black bowl') and names the other candidates as the task's own scene names "
                       "them (factors.own_phrase, factors.scene_phrases); before the fix the probe skipped every LIBERO-Goal "
                       "task (the validity group). Found in a smoke run on three files (LIBERO-Goal 1, LIBERO-Object 0, "
                       "LIBERO-90 14); nothing held-out was predicted",
             "parser_hash": F.parser_hash(),
             "unchanged": "measures, groups, decisions and data of zc_probe; f0_coverage's numbers (parse and coverage code "
                          "unchanged; only the phrase helpers were added)",
             "written_before": "the full probe run"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zc_probe_amend1.py", command="amendment to the readability probe pre-registration"))
