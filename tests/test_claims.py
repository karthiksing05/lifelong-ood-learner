"""Every headline number cited in the documentation must match `results/aggregate.json`.

The documents in this repository quote specific numbers — README, CONTRIBUTIONS, SCOREBOARD, RESULTS
— and those numbers were correct when written. Results here have been recomputed many times: a
baseline was rerun at its own best configuration, a nan-poisoned mean was fixed, guidance was
calibrated, two claims were retracted after replication at three seeds. Every one of those changed
numbers that prose had already quoted.

So this asserts the prose against the aggregate rather than trusting it. A claim that no longer holds
fails here instead of surviving into a paper.

Adding a claim: put it in CLAIMS with the tolerance the text implies (a number written to three
decimals should be checked to three).

    python tests/test_claims.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(HERE, "results", "aggregate.json")

# (cited value, tolerance, experiment, arm, metric, what the docs call it)
#
# REMOVED 2026-08-11: the four n25 pretrained-composition claims (score-composed and
# path-union, `both` and `vendi`).  They were measured through a field that fed the
# pretrained UNet a pixel permutation, so the samples were tiled mosaics and the probes
# were scoring those.  Corrected, every generated arm scores 0.000 -- see RESULTS 30b/30d
# and README section 6.  They are deleted rather than re-pinned because there is no
# surviving positive result to assert.
# ── THE PIXEL-FEATURE LEDGER, frozen 2026-08-19 ──────────────────────────────
#
# `evals._feat` is a PIXEL feature space, and its own docstring records a retraction: recall reads
# 0.000 in pixel space and 0.180 in Inception's on the same 1600 images (RESULTS 5e-bis).  Every
# claim below whose metric is `recall`, `f1`, `memorized`, `precision`, `density` or `coverage` is
# computed through it -- FIFTEEN of them, traced individually:
#
#     n9_showdown          f1      x4   (ours, poe-guided, inversion, inversion-free)
#     n20_celeba_..._pair  recall  x2   (ours, composable-mcmc)
#     n27_factor_cond.     f1      x1
#     n30_proto_dist_fine  f1      x2 + recall x1
#     n71_absorb           recall  x1
#     n72_donation         recall  x2 + memorized x1
#     n75_grid_pooled      memorized x1
#
# NOT in the ledger, and the distinction matters: `joint`/`pair`/`ACC` come from probes, not `_feat`;
# `dist_prototype`, `d_query` and `real_spread` are raw-pixel L2; and n64's `recall` is computed by a
# trained multi-object DETECTOR, not by `precision_recall`.
#
# FOUR of the fifteen read exactly 0.000 (n30 poe recall, n72 address memorized, n72 query-copy
# recall, n75 memorized) and a fifth reads 0.003.  That subclass is the dangerous one: a k-NN metric
# reading exactly zero is precisely what a wrong feature space produces, so those numbers cannot be
# CITED as evidence until an Inception twin exists -- even though they correctly describe what the
# code produces today and must keep passing.
#
# The migration is additive: `space=` is added as a parameter with the default unchanged, so no
# number here moves, and new claims are pinned on new keys (`*_inception`) rather than replacing
# these.  See docs/NEXT_STEPS.md.
CLAIMS = [
    # ── composition, ColorMNIST (README §2, SCOREBOARD §2, RESULTS §11d) ──────
    (0.630, .01, "n9_showdown", "ours", "f1", "ColorMNIST ours F1"),
    (0.556, .01, "n9_showdown", "ours", "joint", "ColorMNIST ours joint"),
    # NOT the companion paper's published F1 (75.3).  This is their method measured under OUR
    # protocol -- 24 samples from ONE query, one reference set, pixel features -- where §4.2 of the
    # paper pools 100 queries against two reference sets in Inception-V3.  Under their protocol
    # their method reproduces here (RESULTS §5e-bis).  Pinned because it is what n9 produces and
    # every arm in that table shares the protocol; named so it cannot be quoted against their paper.
    (0.003, .01, "n9_showdown", "poe-guided", "f1", "ColorMNIST poe-guided F1 (our 1-query protocol)"),
    (0.839, .01, "n9_showdown", "poe-guided", "joint", "ColorMNIST poe-guided joint (our protocol)"),
    (0.000, .01, "n9_showdown", "composable-diff", "joint", "ColorMNIST composable-diff joint"),
    (0.531, .01, "n9_showdown", "inversion", "f1", "ColorMNIST inversion F1"),
    (0.311, .02, "n9_showdown", "inversion-free", "f1", "ColorMNIST inversion-free F1"),
    # ── composition, CelebA strong split (README §2, SCOREBOARD §2, RESULTS §11c)
    (0.903, .01, "n20_celeba_composition_pair", "ours", "pair", "CelebA ours pair"),
    (0.934, .01, "n20_celeba_composition_pair", "composable-diff", "pair", "CelebA comp-diff pair"),
    (0.965, .01, "n20_celeba_composition_pair", "composable-mcmc", "pair", "CelebA comp-mcmc pair"),
    (0.579, .01, "n20_celeba_composition_pair", "ours", "recall", "CelebA ours recall"),
    (0.483, .01, "n20_celeba_composition_pair", "composable-mcmc", "recall", "CelebA mcmc recall"),
    # ── retention (CONTRIBUTIONS 1, SCOREBOARD §1) ───────────────────────────
    (0.825, .015, "n5_classinc", "ours", "ACC", "n5 ours ACC"),
    (0.863, .01, "n5_classinc", "independent", "ACC", "n5 independent ACC"),
    (1.000, .001, "n5_classinc", "ours", "retention_exact", "n5 ours bit-identical"),
    (0.250, .001, "n5_classinc", "ewc", "retention_exact", "n5 protection-based bit-identical"),
    (0.000, .001, "n5_classinc", "ours", "FGT", "n5 ours FGT"),
    (0.708, .02, "n5_classinc", "replay", "ACC", "n5 generative-replay ACC"),
    (0.763, .01, "n22_celeba_lifelong", "ours", "ACC", "n22 ours ACC"),
    (1.000, .001, "n22_celeba_lifelong", "ours", "retention_exact", "n22 ours bit-identical"),
    # ── the pretrained backbone (CONTRIBUTIONS 5/6a, RESULTS §12e) ───────────
    (1.000, .001, "n26_batch_pretrained", "batch", "retention_exact", "n26 batch bit-identical"),
    # ── the analysis experiments (RESULTS §13b, §14) ─────────────────────────
    (0.042, .01, "n27_factor_conditionals", "factor-score", "joint", "n27 factor-score joint"),
    (0.000, .01, "n27_factor_conditionals", "slot-score", "joint", "n27 slot-score joint"),
    (0.490, .02, "n27_factor_conditionals", "factor-score", "f1", "n27 factor-score F1"),
    (0.562, .02, "n27_factor_conditionals", "ours", "joint", "n27 ours joint"),
    (0.010, .002, "n28_update_rules", "rehearse", "drift", "n28 rehearse drift"),
    # ── the prototype-distance dial (CONTRIBUTIONS 8, RESULTS §20) ───────────
    # the DEFAULT operating point, and the two endpoints the "2.32x span" is computed from
    (0.669, .02, "n30_prototype_distance_fine", "ours@0.9", "f1", "n30 ours F1 at the default s=0.9"),
    (0.410, .02, "n30_prototype_distance_fine", "ours@0.8", "f1", "n30 ours F1 at superseded s=0.8"),
    (0.071, .01, "n30_prototype_distance_fine", "ours@0.6", "dist_prototype", "n30 ours d(proto) s=0.6"),
    (0.165, .01, "n30_prototype_distance_fine", "ours@1", "dist_prototype", "n30 ours d(proto) s=1.0"),
    # the contrast the claim rests on: theirs does not move, and never covers
    (0.058, .01, "n30_prototype_distance_fine", "poe-guided@0.6", "dist_prototype", "n30 poe d(proto) s=0.6"),
    (0.055, .01, "n30_prototype_distance_fine", "poe-guided@1", "dist_prototype", "n30 poe d(proto) s=1.0"),
    (0.000, .01, "n30_prototype_distance_fine", "poe-guided@1", "recall", "n30 poe recall at s=1.0"),
    # ── the router's operating point (RESULTS §6c-quinquies, VLA_PATH, OUTLINE) ──────
    # The claim that 0.354 was the OPERATING POINT and not the backbone.  Held-out queries,
    # calibration and evaluation on DISJOINT slots.
    (0.861, .02, "n62_router_calibrate", "Shapes3D(pretrained)", "auroc", "n62 Shapes3D held-out AUROC"),
    (0.669, .02, "n62_router_calibrate", "Shapes3D(pretrained)", "shipped", "n62 Shapes3D at the shipped u=0.10"),
    # 6c is NOT overturned: on ColorMNIST the density ablation still matches the conditional.
    # Pinned so that a change which quietly separates them fails here instead of rewriting 6c.
    (1.000, .01, "n62_router_calibrate", "ColorMNIST(nobg3)", "auroc", "n62 ColorMNIST held-out AUROC"),
    (1.000, .01, "n62_router_calibrate", "ColorMNIST(nobg3)", "ablation", "n62 ColorMNIST density ablation"),
    (0.585, .05, "n62_router_calibrate", "Shapes3D(pretrained)", "ablation", "n62 Shapes3D density ablation"),
    # the label-free calibration: exact on Shapes3D, and expected NOT to help on ColorMNIST
    (0.929, .02, "n65_u_labelfree", "Shapes3D(pretrained)", "auroc", "n65 Shapes3D label-free AUROC"),
    (0.929, .02, "n65_u_labelfree", "Shapes3D(pretrained)", "oracle", "n65 Shapes3D oracle AUROC"),
    (0.826, .03, "n65_u_labelfree", "ColorMNIST(nobg3)", "auroc", "n65 ColorMNIST label-free AUROC"),
    # ── multi-object: a scene of k known objects is a composition over addresses (§7bb) ──
    # 3 sampling seeds on one backbone. Pinned now that it replicates; the ADDRESS number is the
    # claim, and BOTH controls are pinned with it because each one licenses it: a weak detector
    # poisons every downstream number, and without the self-conditioning control a low ADDRESS
    # score cannot be told apart from a model that never learned to condition at all.
    (0.317, .02, "n64_multiobject_address", "address", "exact", "7bb ADDRESS held-out pairs exact-set"),
    (0.608, .02, "n64_multiobject_address", "address", "recall", "7bb ADDRESS held-out per-object recall"),
    (0.178, .03, "n64_multiobject_address", "floor", "recall", "7bb unconditional recall floor"),
    (0.946, .02, "n64_multiobject_address", "controls", "detector", "7bb detector on HELD-OUT pairs"),
    (0.889, .05, "n64_multiobject_address", "controls", "self_cond", "7bb self-conditioning control"),
    # the basis fit, against its own permutation null -- the null is pinned too, because an R² that
    # drifts toward it would otherwise still 'match the docs' while meaning nothing
    (0.931, .02, "n64_multiobject_address", "basis", "r2", "7bb object-membership basis R2"),
    (0.875, .03, "n64_multiobject_address", "basis", "r2_cv", "7bb leave-one-scene-out CV R2"),
    (0.257, .05, "n64_multiobject_address", "basis", "perm_null", "7bb permutation null"),
    # ── the loop WITH storage in it (§7cc, N71) ──────────────────────────────
    # `stored` is the claim -- samples drawn from the CONSOLIDATED concept, not from the composed
    # address. `address` is pinned beside it as the ceiling, so a future regression that quietly
    # reports the pre-storage number instead would fail here rather than read as an improvement.
    (0.525, .05, "n71_absorb_and_generate", "stored", "joint", "7cc stored-concept joint"),
    (0.808, .05, "n71_absorb_and_generate", "stored", "recall", "7cc stored-concept recall"),
    (0.500, .05, "n71_absorb_and_generate", "address", "joint", "7cc address ceiling joint"),
    (0.000, .01, "n71_absorb_and_generate", "stream", "forgetting", "7cc forgetting over the stream"),
    (1.000, .01, "n71_absorb_and_generate", "stream", "frozen", "7cc base bit-identical"),
    # ── is the method DONATING a capability? (§7dd, N72) ─────────────────────
    # nearest-class is the adversarial control: the model's OWN best existing embedding, chosen by
    # proximity to our composed address. Pinned at 0.000 because that number is the whole claim --
    # if it ever rises to meet ADDRESS, the capability was already there and we are rediscovering.
    (0.599, .05, "n72_donation", "address", "joint", "7dd ADDRESS joint"),
    (0.000, .02, "n72_donation", "nearest-class", "joint", "7dd nearest-class (already-had-it) joint"),
    (0.871, .04, "n72_donation", "address", "recall", "7dd ADDRESS recall"),
    (0.000, .02, "n72_donation", "address", "memorized", "7dd ADDRESS memorised fraction"),
    # query-copy scores joint 1.000 at recall 0.000 -- perfect accuracy by copying. Pinned because
    # it is the standing demonstration that accuracy alone cannot distinguish composing from
    # copying, which is why recall is a headline number in this repository.
    (1.000, .01, "n72_donation", "query-copy", "joint", "7dd query-copy joint (copying scores 1.0)"),
    (0.000, .01, "n72_donation", "query-copy", "recall", "7dd query-copy recall (and covers nothing)"),
    # ── CelebA grid, the DENSITY holdout (RESULTS §7ee) ───────────────────────
    #
    # These are pinned because they are a SCOPED NEGATIVE and negatives drift back most easily.
    # The margin over retrieval was reported at +0.102 won 3/3 before the readability gate existed;
    # it was measured on cells whose REAL images score at chance, where every arm reads ~0. Only 16
    # of 36 (backbone, cell, seed) measurements survive the gate, and on those ADDRESS beats the
    # unconditional floor decisively and RETRIEVAL not at all. Both are pinned so neither can be
    # quoted without the other.
    (0.297, .01, "n75_grid_pooled", "address", "joint", "7ee grid ADDRESS (readable cells)"),
    (0.279, .01, "n75_grid_pooled", "nearest-class", "joint", "7ee grid nearest-class"),
    (0.117, .01, "n75_grid_pooled", "unconditional", "joint", "7ee grid unconditional floor"),
    (0.180, .01, "n75_grid_pooled", "margin_vs_uncond", "joint", "7ee grid margin over the floor"),
    (0.018, .01, "n75_grid_pooled", "margin_vs_retrieval", "joint",
     "7ee grid margin over RETRIEVAL (does not clear the bar)"),
    (0.375, .01, "n75_grid_pooled", "margin_vs_retrieval", "won",
     "7ee grid retrieval win rate (below half)"),
    # The copying evidence, and the pair is the point: samples sit as far from the query as two
    # REAL images of that cell sit from each other, which is the same signature 7dd reports on
    # ColorMNIST (9.03 against 9.06). Before the range fix this read 79.4 against 40.0 -- the
    # inflation was the model-space/data-space mismatch, not distance from the query.
    (42.1, 0.5, "n75_grid_pooled", "copying", "d_query", "7ee grid d(sample, query)"),
    (40.0, 0.5, "n75_grid_pooled", "copying", "real_spread", "7ee grid real-vs-real spread"),
    # the basis claim does NOT hold on this table, and the pair is what says so
    (0.875, .01, "n75_grid_pooled", "basis", "r2", "7ee grid basis R2 in-sample"),
    (0.207, .03, "n75_grid_pooled", "basis", "r2_cv", "7ee grid basis R2 leave-one-cell-out"),
    # the density holdout itself, which is what CelebA does support
    (0.000, .01, "n75_grid_pooled", "copying", "memorized", "7ee grid memorised (nothing copied)"),
    (16.0, .01, "n75_grid_pooled", "coverage", "n_readable", "7ee grid readable measurements"),
    (36.0, .01, "n75_grid_pooled", "coverage", "n_total", "7ee grid total measurements"),
]


def main():
    if not os.path.exists(AGG):
        print(f"[claims] no aggregate at {AGG} — run `bash reproduce.sh figures` first")
        return 0
    A = json.load(open(AGG))
    bad = []
    for want, tol, exp, arm, key, label in CLAIMS:
        cell = A.get(exp, {}).get("table", {}).get(arm, {}).get(key)
        got = None if not cell else cell.get("mean")
        if got is None:
            bad.append(f"  MISSING  {label:40s} ({exp} / {arm} / {key})")
        elif abs(got - want) > tol:
            bad.append(f"  MISMATCH {label:40s} docs say {want:.3f}, aggregate has {got:.3f}")
    for line in bad:
        print(line)
    print(f"[claims] {len(CLAIMS) - len(bad)}/{len(CLAIMS)} documented numbers match the aggregate")
    if bad:
        # ASCII, deliberately: stdout defaults to latin-1 on this cluster when redirected, so an
        # em-dash here raised UnicodeEncodeError *in the failure branch only* -- the test crashed
        # before it could print which claim was stale, which is the one moment it has a job to do.
        print("[claims] FAIL -- update the prose, or the claim is stale")
        return 1
    print("[claims] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
