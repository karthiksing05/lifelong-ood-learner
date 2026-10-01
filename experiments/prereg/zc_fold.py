"""Pre-registration of the fold read (ledger m3_prereg tag zc_fold), written before any token episode at init states 30-49."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="zc_fold", seed=0,
    metrics={
        "plan": "the overhaul's FOLD: a factor token (lifelong_ood/policy/token_fold.py; a learned key and value per layer "
                "appended to the prompt's cache; every policy weight frozen; 18.4 KB fp16) fitted on the arrival's one demo "
                "(experiments/f4_fold.py: the same frames and chunks the one-demo LoRA trains on; the demo is never executed)",
        "parser_hash": F.parser_hash(),
        "arms": {"ftoken_self": "the task's own prompt + its demo-0 token (zero practice, zero weights changed)",
                 "ftoken_self_d1": "the same from demo 1 (IPEC episode 1)",
                 "lora": "the one-demo rank-16 LoRA (18.7 MB), demo 0 (recorded on 17 tasks; the 4 development tasks 35, 36, "
                         "39, 42 are scored now with the same adapters at the same states)",
                 "lora_d1": "the one-demo LoRA from demo 1 (recorded, 298 of 420)"},
        "tasks": "the 21 held-out tasks; init states 30-49; SERVE_SEED 0; 420 paired episodes per arm",
        "hypotheses": {"H2a": "ftoken_self non-inferior to lora at -0.10 (two-level bootstrap, one-sided 95% lower bound of "
                              "the paired difference)",
                       "H2b": "ftoken_self_d1 non-inferior to lora_d1 at -0.10",
                       "claim": "the token matches the LoRA only if BOTH H2a and H2b hold (the lesson of the second-demo "
                                "retraction)",
                       "Z": "ftoken_self > zero-shot pi0.5 (6 of 420): exact sign test, p < 0.05",
                       "cost": "bytes per task 18.4 KB against 18.7 MB"},
        "written_before": "any token episode at init states 30-49",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zc_fold.py", command="pre-registration of the fold read"))
