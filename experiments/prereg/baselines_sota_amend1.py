"""Amendment 1 to the recent-baseline read (ledger m3_prereg tag baselines_sota_amend1): implementation fidelity."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

print(ledger.record(
    "m3_prereg", bench="libero_90", tag="baselines_sota_amend1", seed=0,
    metrics={
        "amends": "baselines_sota (read p1_bindings tag sota: ours 372 vs STRAP 168, DSRL 234, experience replay 138)",
        "why": "a check of the authors' code after the read (user request: account for the existing solutions' implementations) found "
               "our DSRL and STRAP differ from them in ways that may matter",
        "dsrl_differences": "authors' pi0.5 LIBERO config (github.com/gonora24/dsrl_pi0 run_libero_pi05.sh, train_utils_sim.py, jaxrl2 "
                            "pixel_sac): a 7-dim latent tiled over the chunk in action dims 0-6 (ours: all 320 dims), bound 1.0 (ours 2.0), "
                            "10 critics with min reduction (ours 2), no entropy backup, auto temperature from 1.0, 20 updates per "
                            "transition (ours 4), discount 0.999 (ours 0.99), hidden 128; ours added a sigmoid critic and per-dimension "
                            "entropy to survive 320 dims",
        "strap_differences": "authors' retrieval (github.com/WEIRDLabUW/STRAP retrieval_utils.py, retrieval.py): segmentation = cut "
                             "after every step with L1 end-effector displacement < 2.5 mm, merge pieces < 20 steps (ours: speed below "
                             "its 20th percentile); S-DTW steps (1,1) and (1,2) only (ours also allowed stalls); top_k / n_segments "
                             "per segment with no global trim (ours trimmed to K overall)",
        "new_arms": {"dsrlo": "DSRL with the authors' configuration (kept from ours: frozen ResNet-18 features + state instead of their "
                              "64 x 64 CNN; the one-demo LoRA as the base policy, since pi0.5 alone succeeds ~0 on these tasks; 100 "
                              "practice episodes on states 0-29)",
                     "strap2": "STRAP with the authors' retrieval rules (kept from ours: a LoRA on pi0.5 instead of a policy trained from "
                               "scratch; one target demo instead of five; frames capped at 3,000)"},
        "hypotheses": "B1' ours vs strap2, B2' ours vs dsrlo, B3 ours vs er (unchanged), Holm over the three; the earlier strap and dsrl "
                      "arms are reported as our variants (descriptive)",
        "written_before": "any rollout of dsrlo or strap2",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/baselines_sota_amend1.py", command="amendment: baseline implementation fidelity"))
