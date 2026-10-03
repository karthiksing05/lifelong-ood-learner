"""Amendment 1 to f1_fresh (ledger m3_prereg tag f1_amend1), written before any episode of f1_accept: the per-task LoRA given the
SAME acceptance test as the scheme (kept only if it beats zero-shot on stored init states 0-9, 10 episodes each, noise seed 9;
experiment f1_accept), scored with f1_main's fresh-state episodes.  Descriptive: it isolates what composition and consolidation add
beyond keeping only units that the learner's practice favours."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="f1_amend1", seed=9,
    metrics={"arm": "lora_acc: per task, lora if its f1_accept successes exceed zero-shot's, else zs",
             "reported": "lora_acc vs zs, vs lora, and the scheme's kept arm vs lora_acc, per set, with demonstrations used",
             "disclosure": "proposed after f1's Q27 half showed the per-task LoRA at 0 of 20 on two tasks zero-shot solves 14-15 times",
             "written_before": "any episode of f1_accept"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/f1_amend1.py", command="amendment: LoRA with the acceptance test"))
