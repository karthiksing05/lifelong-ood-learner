"""Pre-registration of the Bind plan's first online read, Q2-mini (ledger m3_prereg tag b_bind_q2), written before any of its
episodes: can the frozen policy be ADDRESSED -- does biasing its action expert toward the named object's tokens (oracle maps)
make it pick up that object where it otherwise runs the room's routine?"""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="b_bind_q2", seed=0,
    metrics={
        "plan": "~/.claude/plans/i-m-interested-in-re-workshopping-quizzical-crane.md, Q2-mini; no demonstration anywhere",
        "parser_hash": F.parser_hash(),
        "tasks": [48, 49, 52, 53],
        "arms": {"zs": "pi0.5 under the task's own instruction, unbiased",
                 "bind@rho0.3_all, bind@rho0.5_all, bind@rho0.7_all": "the same, with the action expert's attention (all 18 "
                 "layers, every Euler step) raised to mass rho on the agent-view tokens of the current phase's referent: the "
                 "moved object while the policy's own commanded gripper is open, the target while closed; ORACLE maps "
                 "(simulator segmentation, coverage >= 0.1 per token) sent by the client (--seg)",
                 "bindplacebo@rho0.5_all": "the same bias toward a random OTHER movable object, fixed per episode (steerability: "
                 "does the policy pick the object it is biased toward?)"},
        "excluded_configs_rule": "a config that experiments/b2_steer_offline.py flags as overriding (OS or LS median < 0.5 on "
                                 "class_a) is dropped from this read before it runs",
        "episodes": "10 per arm and task, practice init states 0-9, client --bind stop (the episode ends at the first binding)",
        "measure": "BA = share of episodes whose first movable object displaced > 3 cm while the commanded gripper is closed is "
                   "the goal's moved instance; NONE = share binding nothing before the step limit; for the placebo arm, the "
                   "share binding the placebo object",
        "kill": "binding cannot be injected (BIND stops) unless, for some rho, BA >= 0.5 on at least 2 of the 4 tasks with NONE "
                "<= 0.2 on those tasks",
        "choice_for_Q2": "the rho with the highest mean BA among those meeting NONE <= 0.2 on every task (ties: the smaller rho)",
        "written_before": "any episode of this read",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/b_bind_q2.py", command="pre-registration of Q2-mini"))
