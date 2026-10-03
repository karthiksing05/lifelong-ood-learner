"""Pre-registration of the fresh-state benchmark (ledger m3_prereg tag f1_fresh), written before any episode on the fresh states:
the scheme and its baselines scored on 20 NEVER-USED start states per task (experiments/fresh_states.py: LIBERO's placement sampler,
seeds 100000-100019; every stored init state of LIBERO-90 has been used by earlier reads), noise seed 9, paired by state."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="f1_fresh", seed=9,
    metrics={
        "sets": {"C2": "the 30 arrivals of c2_stream (4 step keys)", "Q27": "the 20 tasks of q27_joint"},
        "arms": {"zs": "the frozen policy",
                 "lora": "per-task one-demo LoRA, rank 16 (r3_fewshot.py; 18.7 MB per task, a demo for every task)",
                 "bcl@joint_<key>": "one LoRA per step key trained on all the key's demos at once (b_continual.py; 2000 steps)",
                 "bcl@seq_<key>": "one LoRA per key fine-tuned task after task in the stream order (500 steps per task)",
                 "bcl@er_<key>": "the same with experience replay (half of each batch from earlier demos)",
                 "c2_kept": "each C2 arrival's kept arm (c2_state.json)", "c2_final": "each C2 key's final primitive",
                 "q27_kept": "each Q27 task's kept arm (q27_kept.json)"},
        "episodes": "20 per task and arm, fresh states 0-19, experiment f1_main",
        "hypotheses": {"H1": "C2 set: c2_kept > zs pooled (exact sign test on discordant pairs by state)",
                       "H2": "Q27 set: q27_kept > zs pooled", "correction": "Holm over H1, H2"},
        "descriptive": "two-sided sign tests and bootstrap intervals of c2_kept against lora, bcl joint, bcl seq, bcl er; seq against "
                       "lora on each key's earlier arrivals (forgetting); costs per method: demonstrations used, bytes stored, "
                       "practice episodes run before test",
        "written_before": "any episode of f1_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/f1_fresh.py", command="pre-registration of the fresh-state benchmark"))
