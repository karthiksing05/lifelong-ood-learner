"""Pre-registration of the compressed pose reader read (ledger m3_prereg tag reader_compress)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="reader_compress", seed=0,
    metrics={
        "plan": "docs/PLAN_TRANSFER.md P6, storage: a task's bytes are dominated by its pose reader (443 KB of fp32 ridge weights over "
                "16,384 ResNet-18 map features) once its phases take 2-3 KB; can each task keep only a small head over a SHARED code?",
        "offline": ("experiments/p6_compress.py (results/robot/p6_compress.json; practice first frames of states 20-29, 50 tasks): full reader "
                    "median 0.152 cm at 443 KB per task; PCA-512 head 0.214 cm at 6.9 KB; PCA-256 0.242 cm at 3.5 KB; seeded random "
                    "projection 512 0.244 cm at 6.9 KB (larger k in p6_compress_k1024_2048.json, descriptive)"),
        "selection": ("by a rule on the offline errors only: the smallest per-task head whose median error is within 1.5 x the full "
                      "reader's (<= 0.228 cm) -> pca512"),
        "arm": ("phasebind2_rjpred@pca512_rj: exactly phasebind2_rjpred (own demo, rules v2, predicted carry, reader trained on 400 "
                "rendered scenes only) with the reader replaced by a ridge head on the 512 principal directions of the OTHER half of "
                "the tasks' rendered scenes (cross-fitted: a task's scenes never shape its basis; head stored fp16, 6.9 KB per task on "
                "average; the basis 16.8 MB fp16 stored once for all tasks); folded into W so the server is unchanged"),
        "tasks": "the 50 stream-pool tasks (results/robot/p5_pool.json); init states 30-49, SERVE_SEED 0; 1,000 episodes paired with "
                 "phasebind2_rjpred (m3_final, 874 of 1,000)",
        "hypotheses": {
            "RC1": "PRIMARY: phasebind2_rjpred@pca512_rj non-inferior to phasebind2_rjpred at -0.05: one-sided 95% lower bound of the "
                   "paired difference (two-level bootstrap, tasks then states) >= -0.05",
            "descriptive": "the same at -0.10; the exact sign test; per task; bytes per task (program + head) and the shared basis; the "
                           "cumulative bytes over the 50 tasks against a rank-16 LoRA per task (18.7 MB)",
        },
        "written_before": "any rollout of this arm",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/reader_compress.py", command="pre-registration of the compressed pose reader read"))
