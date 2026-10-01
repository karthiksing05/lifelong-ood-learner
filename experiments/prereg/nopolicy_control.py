"""Pre-registration of the no-policy control (ledger m3_prereg tag nopolicy_control), written before any no-policy rollout."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

chk = {b: json.load(open(f"/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/guide_override_{b}.json")) for b in ("pi05_libero", "smolvla_libero")}
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="nopolicy_control", seed=0,
    metrics={
        "why": "the guide the bindings use (constant weight 1, variance 1e-3) replaces the policy's noise prediction in the real action "
               "dims on every Euler step but the first; experiments/guide_override_check.py measured the sampled chunk at "
               f"{chk['pi05_libero']['guided_minus_target_rms']:.4f} rms from the guide's target (normalised units) on pi0.5 and "
               f"{chk['smolvla_libero']['guided_minus_target_rms']:.4f} on SmolVLA, and two unrelated observations' guided chunks "
               f"{chk['pi05_libero']['guided_obs0_minus_obs1_rms']:.4f} apart (the policy's own: {chk['pi05_libero']['own_obs0_minus_obs1_rms']:.2f}). "
               "If so, the policy contributes nothing to the bindings arm; this control executes the target with no policy at all",
        "arms": {"nopolicy_rjpred_pca512_rj": "the same phase program, reader head, predicted carry and events as the bindings arm; the "
                                               "server returns the guide's target chunk itself and never samples the policy",
                 "phasebind2_rjpred_pca512_rj": "the bindings arm through pi0.5 (372 of 420, recorded)"},
        "tasks": "the 21 held-out tasks; init states 30-49; 420 episodes",
        "hypotheses": {"N": "the no-policy arm is non-inferior to the bindings arm at -0.05 (two-level bootstrap, one-sided 95% lower "
                            "bound of the paired difference): the policy contributes nothing measurable",
                       "descriptive": "per task; the exact sign test both ways"},
        "consequence": "if N holds, every bindings result is a result about the phase program executed by the robot's controller, "
                       "and the frozen VLA is a pass-through in that arm; the README and the site say so",
        "written_before": "any no-policy rollout",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/nopolicy_control.py", command="pre-registration of the no-policy control"))
