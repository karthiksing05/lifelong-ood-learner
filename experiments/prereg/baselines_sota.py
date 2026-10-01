"""Pre-registration of the recent-baseline read (ledger m3_prereg tag baselines_sota)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="baselines_sota", seed=0,
    metrics={
        "why": "the user asked for recent state-of-the-art baselines (2025-26); the existing suite (LoRA variants, full fine-tuning, VINN, "
               "DemoDiffusion, replay, DiffusionNFT-style LoRA with failure negatives) predates the families below",
        "arms": {
            "strap": "STRAP (Memmel et al., ICLR 2025): rank-16 LoRA on the demo + K = 100 sub-trajectories retrieved from pi0.5's 40 "
                     "training tasks by subsequence DTW over DINOv2-B agent-view features (segments where the end-effector nearly stops, >= 20 "
                     "steps), all frames under the target instruction, 2,000 steps (experiments/b_strap.py; arm lora@strap)",
            "dsrl": "DSRL (Wagenmaker et al., CoRL 2025): the one-demo LoRA (r3_lora) switched on and frozen; a SAC agent chooses each "
                    "chunk's initial noise (10 x 32, tanh-bounded at 2) from frozen ResNet-18 features + state; 100 practice episodes per "
                    "task on init states 0-29 (10 warm-up with Gaussian noise, then 10 updates per chunk), sparse success reward; scored "
                    "with the deterministic actor (lifelong_ood/policy/dsrl.py; arms dsrl_train, dsrl)",
            "er": "experience replay / multi-task (continual-learning results for VLAs, arXiv 2603.03818): ONE rank-16 LoRA trained on all "
                  "21 demos, batches balanced across tasks, 3,000 steps (experiments/b_er.py; arm lora@er)",
        },
        "ours": "phasebind2_rjpred_pca512_rj: the one demo's bindings, rendered-scene reader head (7 KB), no simulator state at test, no "
                "practice -- the configuration of the storage claim; existing episodes (m3_final, seed 0)",
        "tasks": "the 21 held-out tasks; init states 30-49, SERVE_SEED 0; 420 episodes per arm, paired by init state",
        "hypotheses": {
            "B1": "ours vs strap", "B2": "ours vs dsrl", "B3": "ours vs er",
            "test": "exact two-sided sign test on discordant pairs, Holm over B1-B3; direction reported; the paired difference with the "
                    "two-level bootstrap interval",
            "descriptive": "each baseline vs the one-demo LoRA (252) and LoRA + failure negatives (302); storage; practice episodes "
                           "(DSRL 100 per task, ours 0); DSRL's practice success curve",
        },
        "written_before": "any scoring rollout (states 30-49) of strap, dsrl or er; only a 12-episode DSRL smoke run on practice states "
                          "(experiment dsrl_smoke) preceded it",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/baselines_sota.py", command="pre-registration of the recent-baseline read"))
