"""Pre-registration of the self-prototype read (ledger m3_prereg tag q18_self), written before any of its episodes: no demonstration
of any kind; a task's object prototype comes from its own zero-shot successes (experiments/q18_selfproto.py, q18_read.py)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q18_self", seed=4,
    metrics={
        "tasks": "the 48 failing LIBERO-90 tasks with a moved object of q17_stream",
        "explore": "experiment q18_explore: zero-shot on init states 0-9, 10 episodes, recorded, noise seed 4",
        "build": "every task with >= 1 success gets a 4 KB prototype: in each successful episode the end effector at the grasp "
                 "(of the commanded closes sustained for 5 records, the one that starts lowest), projected into that episode's "
                 "first frame, marks the patch; the policy's SigLIP features at it and its 4 neighbours, averaged over successes. "
                 "Validity reported: pointing on the first frames of init states 0-9 (simulator segmentation, scoring only)",
        "grasp_rule_fixed_on": "the recorded q8_p1 practice episodes (oracle-binding rollouts of 7 tasks): pointing 0.5-1.0 per task "
                               "(results/robot/q18_validate.json), before any q18 episode",
        "acceptance": "experiment q18_accept: zero-shot and the self-prototype form (bindw4@rho0.7_approach_self) on init states "
                      "10-19, 10 episodes, noise seed 4; keep the form only if it has MORE successes",
        "test": "experiment q18_main: the kept arm and zero-shot on init states 30-49, 20 episodes, noise seed 4",
        "hypotheses": {"H1": "kept > zero-shot pooled over the 48 tasks (exact sign test on discordant pairs by init state)"},
        "written_before": "any episode of q18_explore, q18_accept or q18_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q18_self.py", command="pre-registration of the self-prototype read"))
