"""Pre-registration of the v2 read (docs/PLAN_TRANSFER.md, after P6): the phase rules v2 and their generator."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="v2", seed=0,
    metrics={
        "plan": "docs/PLAN_TRANSFER.md section 0: iterate on the binding route; v2 fixes a defect found after the P1/P6/3.8 reads",
        "defect": ("the living-room demos' first kept steps catch the object still FALLING after the scene spawned (ketchup 12.2 cm, "
                   "milk and orange juice 8.4 cm above rest); v1 anchored the approach on that pose, so the retargeted grasp aimed "
                   "under the object (48 and 58: 0 of 40 through the prior)"),
        "rules_v2": ("lifelong_ood/policy/phase.py at_rest + cut: (1) the demo's objects read at rest (the evaluation placements' "
                     "convention); (2) the manipulated object = the first movable object DISPLACED > 1.5 cm while the gripper commands "
                     "close (lifted or pushed); (3) the carry and the rest anchored on the other object of interest, movable OR fixture; "
                     "(4) a demo that moves no object is anchored on its object of interest (a fixture) instead of the world"),
        "dev_evidence": ("no-policy generator on the practice init states 0-19 (ledger g38_generate tag demo_retarget_v2): v2 386/420 vs "
                         "v1 331/420, gains only on 42, 48, 52, 58, no task lower; the scoring states 30-49 were not used"),
        "arms": {
            "phasebind2_nofold": "rules v2, bindings alone through the frozen prior, simulator object poses (oracle)",
            "phasebind2_reader": ("rules v2 with the first-frame learned pose reader (experiments/p6_reader.py; readers of all 21 tasks, "
                                  "trained on states 0-29); no simulator state reaches the server"),
            "lora_r27gen": ("3.8's recipe (r15_consolidate --lora, rank 16, 1000 steps, seed 0) on the demo + the v2 generator's "
                            "successes (g27, states 0-19); the policy acts from images"),
        },
        "tasks": "the 21 held-out tasks where the prior fails its practice; init states 30-49; 420 paired episodes per arm (m3_final)",
        "hypotheses": {
            "V1": "PRIMARY: phasebind2_nofold > phasebind_nofold (v1, 314/420): exact two-sided sign test",
            "V2": "phasebind2_nofold > nft_bal (LoRA + failure negatives, 302/420; 0 weights vs 18.7 MB per task): exact two-sided sign test",
            "V4": "lora_r27gen > lora_r25gen (293/420): exact two-sided sign test",
            "holm": "Holm over V1, V2, V4 at 0.05",
            "V3": "phasebind2_reader non-inferior to phasebind2_nofold at -0.10: one-sided 95% lower bound of the paired difference "
                  "(two-level bootstrap) >= -0.10",
            "descriptive": "per task; phasebind2_reader and lora_r27gen vs nft_bal; bytes",
        },
        "later": "a near-object re-read of the pose (experiments/p6_near_reader.py) gets its own pre-registration before any rollout",
        "written_before": "any rollout of these arms at init states 30-49",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/v2.py", command="pre-registration of the v2 read"))
