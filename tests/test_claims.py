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
CLAIMS = [
    # ── composition, ColorMNIST (README §2, SCOREBOARD §2, RESULTS §11d) ──────
    (0.605, .01, "n9_showdown", "ours", "f1", "ColorMNIST ours F1"),
    (0.539, .01, "n9_showdown", "ours", "joint", "ColorMNIST ours joint"),
    (0.003, .01, "n9_showdown", "poe-guided", "f1", "ColorMNIST companion-method F1"),
    (0.839, .01, "n9_showdown", "poe-guided", "joint", "ColorMNIST companion-method joint"),
    (0.000, .01, "n9_showdown", "composable-diff", "joint", "ColorMNIST composable-diff joint"),
    (0.545, .01, "n9_showdown", "inversion", "f1", "ColorMNIST inversion F1"),
    (0.332, .02, "n9_showdown", "inversion-free", "f1", "ColorMNIST inversion-free F1"),
    # ── composition, CelebA strong split (README §2, SCOREBOARD §2, RESULTS §11c)
    (0.903, .01, "n20_celeba_composition_pair", "ours", "pair", "CelebA ours pair"),
    (0.934, .01, "n20_celeba_composition_pair", "composable-diff", "pair", "CelebA comp-diff pair"),
    (0.965, .01, "n20_celeba_composition_pair", "composable-mcmc", "pair", "CelebA comp-mcmc pair"),
    (0.579, .01, "n20_celeba_composition_pair", "ours", "recall", "CelebA ours recall"),
    (0.483, .01, "n20_celeba_composition_pair", "composable-mcmc", "recall", "CelebA mcmc recall"),
    # ── retention (CONTRIBUTIONS 1, SCOREBOARD §1) ───────────────────────────
    (0.839, .01, "n5_classinc", "ours", "ACC", "n5 ours ACC"),
    (0.863, .01, "n5_classinc", "independent", "ACC", "n5 independent ACC"),
    (1.000, .001, "n5_classinc", "ours", "retention_exact", "n5 ours bit-identical"),
    (0.250, .001, "n5_classinc", "ewc", "retention_exact", "n5 protection-based bit-identical"),
    (0.000, .001, "n5_classinc", "ours", "FGT", "n5 ours FGT"),
    (0.693, .01, "n5_classinc", "replay", "ACC", "n5 generative-replay ACC"),
    (0.763, .01, "n22_celeba_lifelong", "ours", "ACC", "n22 ours ACC"),
    (1.000, .001, "n22_celeba_lifelong", "ours", "retention_exact", "n22 ours bit-identical"),
    # ── the pretrained backbone (CONTRIBUTIONS 5/6a, RESULTS §12e) ───────────
    (0.417, .01, "n25_pretrained_case", "ours (score-composed)", "both", "n25 score-composed both"),
    (0.417, .01, "n25_pretrained_case", "ours (path-union)", "both", "n25 path-union both"),
    (1.436, .02, "n25_pretrained_case", "ours (score-composed)", "vendi", "n25 score-composed Vendi"),
    (1.280, .02, "n25_pretrained_case", "ours (path-union)", "vendi", "n25 path-union Vendi"),
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
        print("[claims] FAIL — update the prose, or the claim is stale")
        return 1
    print("[claims] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
