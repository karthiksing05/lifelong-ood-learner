"""Pre-registration of the generality read on another flow VLA (ledger m3_prereg tag pi0_generality)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="pi0_generality", seed=0,
    metrics={
        "why": "the user asked whether the method applies to other compositional flow / diffusion policies",
        "model": "pi0 (openpi pi0_libero, converted to PyTorch with openpi's converter): flow matching, 50-step action chunks, z-score "
                 "action normalisation, continuous state token; fine-tuned by Physical Intelligence on the same 40 LIBERO tasks",
        "arms": {"zeroshot_pi0": "pi0 alone",
                 "phasebind2_rjpred_pca512_rj_pi0": "the unchanged method on pi0: the one demo's rules-v2 phases, the 7 KB reader head, "
                                                     "predicted carry, constant-weight guide w = 1, variance 1e-3; the guide's actions "
                                                     "normalised by pi0's own statistics (Norm.for_backbone)"},
        "tasks": "the 21 held-out tasks; init states 30-49; SERVE_SEED 0; 420 episodes per arm",
        "hypotheses": {"Z": "bindings on pi0 > pi0 zero-shot: exact sign test, p < 0.05",
                       "descriptive": "vs the same method on pi0.5 (372 of 420); per task; latency"},
        "written_before": "any pi0 rollout at init states 30-49 (a smoke test on practice states may precede it)",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pi0_generality.py", command="pre-registration of the pi0 generality read"))
