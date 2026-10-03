"""Amendment 2 to f1_fresh (ledger m3_prereg tag f1_amend2), written before any of its episodes: the per-key JOINT LoRA (b_continual
joint) given the same acceptance test (kept only if it beats zero-shot on stored init states 0-9, 10 episodes, noise seed 9;
experiment f1_accept; zero-shot practice episodes reused), scored with f1_main's fresh-state episodes.  Descriptive: it separates the
stream's incremental consolidation and demonstration requests from training one adapter per step key on all demonstrations and
keeping it where practice favours it."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="f1_amend2", seed=9,
    metrics={"arm": "joint_acc: per task, bcl@joint_<key> if its f1_accept successes exceed zero-shot's, else zs",
             "reported": "joint_acc vs zs and the scheme's kept arm vs joint_acc on the stream's 30 tasks, with demonstrations and bytes",
             "disclosure": "proposed after f1's read showed the scheme above joint (419 vs 346) with the acceptance test on one side only",
             "written_before": "any episode of the joint arm in f1_accept"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/f1_amend2.py", command="amendment: joint per-key LoRA with the acceptance test"))
