"""Pre-registration of plan C's stream (ledger m3_prereg tag c2_stream), written before any of its episodes and fits
(experiments/c2_stream.py)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("c2", "/coc/flash10/ksingara3/lifelong-ood-learner/experiments/c2_stream.py")
c2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c2)
st = c2.init()
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="c2_stream", seed=8,
    metrics={
        "keys": {k: v[0] for k, v in c2.KEYS.items()},
        "orders": {k: v["order"] for k, v in st.items()},
        "rule": "per key, arrivals in order: the key's current primitive (if any) vs the policy on init states 0-9, 10 episodes, "
                "noise seed 8; kept if MORE successes; else the arrival requests its one demonstration, which joins the key's "
                "library, the key's primitive is refit on all its demonstrations (c0_fit --joint, rank 4, 1000 steps) and tried "
                "again against the same zero-shot episodes; kept if more successes",
        "test": "c2_main: each arrival's kept arm, zero-shot and the key's FINAL primitive on init states 30-49, 20 episodes, seed 8",
        "hypotheses": {"H1": "kept > zero-shot pooled over the 30 arrivals (exact sign test)",
                       "H2": "the final primitive > zero-shot pooled over all arrivals (consolidation keeps earlier arrivals' gains)",
                       "correction": "Holm over H1, H2"},
        "reported": "demonstrations requested per key and in all (of 30 arrivals), primitives stored (4.7 MB each), per-arrival success",
        "context": "c1b_joint (rank 4 = rank 16 on the cabinet key, exploratory) and q27_joint (pre-registered) were read before this",
        "written_before": "any episode or fit of c2_stream",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/c2_stream.py", command="pre-registration of plan C's stream"))
