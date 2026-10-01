"""Amendment to the pi0 generality pre-registration (ledger m3_prereg tag pi0_generality_amend1), written before any pi0 rollout at init
states 30-49: the smoke test found that pi0_libero's actions are state-relative, so the decoder and the guide both apply that transform."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="pi0_generality_amend1", seed=0,
    metrics={
        "amends": "pi0_generality",
        "finding": "openpi trains pi0_libero with extra_delta_transform: its first six action dims are the LIBERO action minus the robot "
                   "state (DeltaActions before Normalize, AbsoluteActions after Unnormalize); pi0.5 (pi05_libero) is trained without it. "
                   "Our decoder passed a zero state to openpi's output transform, so pi0's state was never added back (harmless for pi0.5).",
        "change": "Norm.for_backbone reads the flag; the guide subtracts the raw state from its target's first six dims before normalising; "
                  "the server passes the model's input state to the output transform (commit 31d9e54; tests/test_norm_delta.py)",
        "check": "a decoding check on pi0's own training suite (libero_spatial tasks 0 and 5, 3 episodes each, zero-shot) must succeed in "
                 "most episodes before the read's rollouts; the smoke run before the fix was cancelled before any episode finished",
        "unchanged": "arms, tasks, init states 30-49, seeds and hypothesis Z of pi0_generality",
        "written_before": "any pi0 rollout at init states 30-49",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pi0_generality_amend1.py", command="amendment to the pi0 generality pre-registration"))
