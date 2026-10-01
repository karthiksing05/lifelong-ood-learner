"""Amendment 1 to Q2-mini (ledger m3_prereg tag b_bind_q2_amend1), written before any episode of its new arms: ROLE
rebinding, the observation-side counterpart of attention binding."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="b_bind_q2_amend1", seed=0,
    metrics={
        "amends": "b_bind_q2 (Q2-mini: attention binding on tasks 48 49 52 53)",
        "why": "the offline probes (b1, b2) found that the action expert already attends the named object under the task's own "
               "instruction on most living-room frames while the policy still moves the routine's object; biasing attention "
               "turns the first chunk toward the biased object (contrast cosine) without changing which object it heads for. "
               "If the routine is driven by what the objects LOOK like, rebinding the role in the observation should work where "
               "attention does not.",
        "parser_hash": F.parser_hash(),
        "new_arms": {
            "role": "the task prototype the policy performs in the same scene with the same relation and target instance "
                    "(lifelong_ood/policy/roles.py role_code: 48 -> 46, 49 -> 46, 52 -> 50, 53 -> 50), under ITS instruction, with "
                    "the new task's object and the base's object exchanging appearance in the simulator's renderer (client "
                    "--swap a:b --swap-mode render; ORACLE: perfect rendering; physics untouched); the policy generates every action",
            "roleplacebo": "steerability: the same, with the base's object exchanging appearance with a movable object the task "
                           "does not name (roles.placebo_code: 48 tomato sauce, 49 ketchup, 52 tomato sauce, 53 tomato sauce)"},
        "episodes": "10 per arm and task, practice init states 0-9, client --bind track (episodes run to success or the step limit)",
        "measures": "BA = share of episodes whose first movable object displaced > 3 cm while the commanded gripper is closed is "
                    "the task's moved instance; success (the task's own BDDL check); for roleplacebo, the share binding the placebo object",
        "decisions": {"role_rebinds": "BA >= 0.5 on at least 2 of the 4 tasks",
                      "role_steerable": "roleplacebo binds its placebo object in >= 0.5 of its episodes pooled",
                      "if_role_rebinds": "a deployable role editor (detector masks, inpainting) and the role arm on every task "
                                         "with a role code (13 of the 71 failing tasks) on practice states follow; the attention "
                                         "arms of b_bind_q2 are read by their own rule"},
        "written_before": "any episode of the role or roleplacebo arms",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/b_bind_q2_amend1.py", command="pre-registration amendment: role rebinding"))
