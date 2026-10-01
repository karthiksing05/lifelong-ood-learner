"""Pre-registration of the online binding read (ledger m3_prereg tag zc_bind), written before any of its episodes."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="zc_bind", seed=0,
    metrics={
        "question": "when the instruction names object X, which object does pi0.5 move? (does it READ object words, or does "
                    "the scene key a trained habit?)",
        "parser_hash": F.parser_hash(),
        "arms": {"own": "zero-shot pi0.5 on LIBERO-90 tasks 46-59 (living rooms 1-3: the same scenes with different named "
                        "objects), each under its own instruction",
                 "swap_object": "LIBERO-Object (in-distribution layouts) tasks 0-4 under their instruction with the object "
                                "replaced by each other candidate present in the layout (factors.substitute, scene phrasing)"},
        "episodes": "10 per arm and task, practice init states 0-9, light recordings with every object's pose",
        "measure": "binding accuracy BA: share of episodes whose first object displaced > 3 cm with the gripper command "
                   "closed is the named object (experiments/f1_binding.py); also which object is bound otherwise (habit)",
        "decisions": {"object_words_read_in_distribution": "LIBERO-Object swap BA >= 0.7",
                      "object_words_not_read": "swap BA <= 0.3 AND living-room BA <= 0.3 for named objects that are not "
                                               "the scene's habit -> zero-shot object composition is dead on pi0.5; the "
                                               "fold (factor tokens) is the main line",
                      "otherwise": "composition pilots on practice states decide"},
        "written_before": "any episode of this read",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zc_bind.py", command="pre-registration of the online binding read"))
