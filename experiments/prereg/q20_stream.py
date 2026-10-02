"""Pre-registration of the stream-with-instances read (ledger m3_prereg tag q20_stream), written before any of its episodes: q17's
stream, with a known type's single library prototype tried at each of its 4 instances (experiments/q20_stream.py)."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

plan = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/q20_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="q20_stream", seed=5,
    metrics={
        "tasks": sorted(int(t) for t in plan["info"]), "orders": plan["orders"],
        "rule": "as q17_stream: an arrival whose object type is not in the library requests one demonstration, used once (its "
                "4 KB prototype joins the library and is its only candidate); an arrival whose type is known requests nothing and its "
                "candidates are the library's prototype of the type at each of 4 instances: the prototype's cosine map over the "
                "policy's image patches cut into its 4 strongest peaks at least 3 patches apart, ordered row-major, the r-th tracked "
                "through the episode (bindw4@rho0.7_approach_src<k>_inst<r>_npk4)",
        "motivation": "q17_stream: one prototype per type points at its source's instance; q19_inst (exploratory): instances "
                      "chosen by practice recover the back and middle bowls with a foreign prototype",
        "acceptance": "experiment q20_accept: zero-shot and every candidate on init states 0-9, 10 episodes, noise seed 5; keep the "
                      "most successes, ties to the policy, then the earlier candidate",
        "test": "experiment q20_main: kept arm and zero-shot on init states 30-49, 20 episodes, noise seed 5",
        "hypotheses": {"H1": "order A: kept > zero-shot pooled over 48 arrivals (exact sign test)", "H2": "order B: the same",
                       "correction": "Holm over H1, H2"},
        "reported": "demonstrations requested (planned " + json.dumps(plan["demos"]) + " of 48), bytes, forms kept",
        "written_before": "any episode of q20_accept or q20_main",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/q20_stream.py", command="pre-registration of the stream-with-instances read"))
