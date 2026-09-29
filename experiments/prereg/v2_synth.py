"""Pre-registration of the rendered-scene reader read (after the v2 read; ledger m3_prereg tag v2_synth)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="v2_synth", seed=0,
    metrics={
        "plan": "docs/PLAN_TRANSFER.md P6: the headline condition (a learned reader within 0.10 of the simulator's poses), after V3 failed",
        "why": ("V3 (ledger p1_bindings tag v2): the first-frame reader trained on 30 placements per task lost on the three bottles "
                "(362 vs 385 of 420, lower bound -0.121). Offline (ledger p6_reader): re-reading near the object 0.94 cm (no gain), "
                "spatial feature maps 0.58 cm (no gain on the bottles), 400 simulator-rendered placements per task 0.25 cm pooled and "
                "0.18 cm spatial"),
        "selection": ("on the practice states 20-29 only, by the rule stated before the spatial variant's result was seen: the lower "
                      "median error; spatial4_synth 0.18 cm vs pooled synth 0.25 cm -> spatial4_synth"),
        "arm": ("phasebind2_reader_spatial4_synth: rules v2 (as phasebind2_nofold) with the reader p6_reader_spatial4_synth_<t>.npz: "
                "standardized ridge (dual form) from the last ResNet-18 feature map of both cameras pooled to 4 x 4 (VinnPolicy."
                "_embed_spatial) of the episode's first frame to the objects' positions; trained on the practice frames of states 0-29 and "
                "400 simulator-rendered placements per task (experiments/p6_synth.py: fresh draws from the task's placement regions; "
                "privileged simulator state LABELS the training scenes only); no simulator state reaches the server; ~0.6 MB per task"),
        "tasks": "the 21 held-out tasks; init states 30-49 (never rendered, never trained on); 420 paired episodes",
        "hypotheses": {
            "V5": "PRIMARY: phasebind2_reader_spatial4_synth non-inferior to phasebind2_nofold (385/420) at -0.10: one-sided 95% lower "
                  "bound of the paired difference (two-level bootstrap) >= -0.10",
            "V6": "phasebind2_reader_spatial4_synth > nft_bal (302/420): exact two-sided sign test p < 0.05",
            "descriptive": "vs phasebind2_reader (the 30-placement reader, 362/420); per task; the reader's online error on the scored "
                           "episodes (experiments/p6_online_error.py)",
        },
        "written_before": "any rollout of this arm at init states 30-49 (a 3-episode smoke test on task 52, experiment smoke, is excluded)",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/v2_synth.py", command="pre-registration of the rendered-scene reader read"))
