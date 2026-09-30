"""Pre-registration of the compositional scheme's confirmation (ledger m3_prereg tag p5_scheme).  CARRIES is filled from the carry
rule's output (experiments/carry_read.py, ledger p5_stream dev_carries) before this is recorded."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

CARRIES = sys.argv[1].split(",")
dev = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/carry_read.json"))

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="p5_scheme", seed=0,
    metrics={
        "plan": "docs/PLAN_TRANSFER.md C3/C4: confirm the compositional scheme on fresh orders after the failed confirmation (p5_confirm: "
                "157 vs 162 of 180 over nine arrivals; one composition passed practice and scored 9 of 20)",
        "changes_since_p5_confirm": [
            "the lookup (docs/GRAMMAR.md 3): grasp sources of the target's object type from the same scene first; place sources with the "
            "target's goal relation, of the same object type first, then the same scene (p5_validated.py plan --rank)",
            "the carry is chosen by practice from a set (kind best): " + ", ".join(CARRIES) + "; the set by greedy forward selection on the "
            "dev read (experiments/carry_read.py: the carry passing practice most often, then the one adding most passes, while it adds >= 2; "
            "passes per carry " + json.dumps(dev["passes"]) + ", the set's union " + str(dev["passes_union_set"]) + ")",
            "waypoints and gripper events are passed on arrival where a carry bridges (events=1; the fix for the generic transport)",
        ],
        "unchanged": "one candidate pair per arrival; compose if the best carry succeeds in >= 8 of 10 practice episodes on init states 0-9, "
                     "else request the task's demo, whose keys enter the table if its no-policy generator check passes (amendment 1); "
                     "rendered-scene reader, predicted in-hand pose, no simulator state at test",
        "orders": "fresh: E, F (random.Random(2) and its mirror) and G, H (random.Random(3) and its mirror); the plans are fixed by the "
                  "practice results alone (results/robot/p5_stream_plan_practice_orders{2,3}_best_rank.json) before any scoring rollout",
        "scoring": "SERVE_SEED 3 (never used for these arms), init states 30-49: each UNIQUE composition (task, grasp source, place source, "
                   "carry) arriving in any of the four orders once, and the same tasks' own demos (phasebind2_rjpred) at the same seed",
        "hypotheses": {
            "H1": "PRIMARY: the unique compositions non-inferior to their own demos at -0.10: one-sided 95% lower bound of the paired "
                  "difference, two-level bootstrap with the compositions as clusters, >= -0.10",
            "descriptive": "demos requested per order; compositions per half of each order; the sign test; per composition; the carry chosen; "
                           "the same compositions' guided carry where it was practised",
        },
        "read": "experiments/p5_validated.py read_scheme --order-seeds 2,3 --score-seed 3 --carries " + ",".join(CARRIES) + " --record",
        "written_before": "any rollout at SERVE_SEED 3 of these arms",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={"carries": CARRIES},
    script="experiments/prereg/p5_scheme.py", command="pre-registration of the compositional scheme's confirmation"))
