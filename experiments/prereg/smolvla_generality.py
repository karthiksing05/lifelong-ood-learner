"""Pre-registration of the generality read on a non-pi flow VLA, SmolVLA (ledger m3_prereg tag smolvla_generality)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="smolvla_generality", seed=0,
    metrics={
        "why": "the user asked whether non-pi models can be tested: SmolVLA is the closest public non-pi flow VLA whose LIBERO weights "
               "leave LIBERO-90 unseen",
        "model": "SmolVLA (Hugging Face LeRobot, lerobot/smolvla_libero, 450M parameters): a SmolVLM2 backbone cut to 16 layers with an "
                 "interleaved cross-attention flow-matching action expert, 50-step chunks, 10 Euler steps, MEAN_STD normalisation; "
                 "trained on lerobot/libero = the same 40 tasks (1,693 episodes) as the pi checkpoints, the same 8-D state and 7-D OSC "
                 "delta actions; run through lifelong_ood/policy/backbone_smolvla.py (LeRobot's own pre-processor and denoise_step)",
        "deviations_from_lerobot_eval": "the client's images are resized with padding to 224 (as for every arm; SmolVLA resizes to 512 "
                                         "itself), and the client replans every 5 steps (LeRobot's eval executes all 50); both arms "
                                         "share them",
        "arms": {"zeroshot_smolvla": "SmolVLA alone",
                 "phasebind2_rjpred_pca512_rj_smolvla": "the unchanged method on SmolVLA: the one demo's rules-v2 phases, the 7 KB "
                                                         "reader head, predicted carry, constant-weight guide w = 1, variance 1e-3, the "
                                                         "guide's actions normalised by SmolVLA's own statistics"},
        "tasks": "the 21 held-out tasks; init states 30-49; SERVE_SEED 0; 420 episodes per arm",
        "check": "decoding check first: SmolVLA zero-shot on libero_spatial tasks 0 and 5 (its training suite), 3 episodes each, must "
                 "succeed in most episodes",
        "hypotheses": {"Z": "bindings on SmolVLA > SmolVLA zero-shot: exact sign test, p < 0.05",
                       "descriptive": "vs the same method on pi0.5 (372 of 420) and pi0; per task; latency"},
        "written_before": "any SmolVLA rollout at init states 30-49",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/smolvla_generality.py", command="pre-registration of the SmolVLA generality read"))
