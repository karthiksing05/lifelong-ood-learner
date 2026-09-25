# Archive: the repository before the 2026-09-25 rework

This directory is a record, not a codebase. It holds the experiments, tests, docs, launchers, figures
and modules of the address / extend-basis / block architecture that the September 2026 review
(`docs/REVIEW_2026-09.md`, kept at the top level) audited, and that the rework replaced with
prototype composition and folding (`docs/PLAN_2026-09.md`). Paths were moved with `git mv`, so
`git log --follow <file>` shows each file's full history. The old top-level README is
[README_pre_rework.md](README_pre_rework.md); the old docs index is `docs/README.md`.

## Running anything in here

Nothing in this directory runs at HEAD: it imports package paths that no longer exist. Check out the
tag and run from there:

    git checkout pre-rework-2026-09-25
    python experiments/n41_addressing.py --help

The per-run result files these scripts wrote are still in the untracked `results/` directory at the
top level; the summary rows the review quotes are in the tracked `ledger/` (backfilled, marked
`backfilled: true`, source commit af8f07a).

The c-series fixtures (c1, c5, c_report, diag_read_by_u) are NOT here: they were re-homed to
`experiments/legacy/` and still run, because the fold-frontier reference arm must reproduce their
numbers.

## Retired claims, and the review's verdict on each

| retired claim | where it was made | verdict (docs/REVIEW_2026-09.md unless noted) |
|---|---|---|
| Composition F1 0.605 vs poe-guided 0.003 (n9) | README, SCOREBOARD | pixel-space k=3 recall rewards diversity (audit fault 7); the canonical run gives 0.630; superseded by Inception-space coverage |
| Bit-identical preservation (n5/n22, "0.25 bit-identical") | README, FINAL_REPORT | true by construction (old classes sampled with no adapter on); "0.25" is 1/(number of batches); a construction property, not a result |
| Router AUROC 0.980 / 0.995 and the 9.5x / 8.9x / 12.4x savings (n18, n24) | README, VLA_PATH | the router never drove storage; the savings are a no-model simulation with ground-truth tiers; the CelebA row was random bit-vectors (fault 6) |
| EXTEND BASIS 1 KB renders a new value at 0.755 (n11/n21) | README, THEORY | measured on a value the backbone was trained on; on a genuinely new value it renders 0.000 (fault 5, Part 3a/3b) |
| Storage cost +0.000, forgetting +0.000 on Shapes3D (n77, 7kk) | RESULTS 7kk | the adapter was never sampled (fault 1); every storage arm is self-distillation (fault 2) |
| Flagship 0.983 from oracle labels on 12 of 40 cells (n78, 7jj) | RESULTS 7jj, FINAL_REPORT | composed from TRUE labels on a backbone pretrained on the held-out images; remeasured 0.980 over 40 cells from a 1-image read; shows addressing, not learning (Part 3d) |
| Composer beats score-addition 4.5x (n43, RESULTS 28) | RESULTS 28 | a 3-epoch smoke run; the full run gives 0.446 vs 0.180 (fault 8) |
| Pretrained CelebA composition (n25/n26) | RESULTS 12 | retracted 2026-08-11 (CelebA guided at 2.4x the paper's strength; pixel permutation) |
| Representation headline (N50) | RESULTS | withdrawn: the adapters were inert (N53) |
| Hierarchy buys transfer (n14, n34, n36, n38) | RESULTS 24 | placement at chance, caterpillar tree; the cu rule fixes purity but buys nothing downstream at +23% parameters |
| Reconsolidation / orthogonal merge / deletion (n12, n17, n31) | RESULTS | negatives; consistent with C1's failed consolidation (17% drift) |
| Router MDL attempt 6 (score_mdl_rules) | RESULTS 6c-ter | loses to the cascade; but diagnoses that a tier is a property of the query AND the count, the accrual rule the rework adopts |
| e0-e4 analytic Cobweb track | docs/README, e-series | unused by every later result |
| C2 / C4 router loops | REVIEW Part 3c | superseded by C5 (re-homed) |
| CLEVR_VERDICT, VLA_PATH, OUTLINE, NEXT_STEPS* | docs | superseded by REVIEW Part 4 and PLAN_2026-09 |
| The three-tier ladder 72 B / 1 KB / 69K as the headline | README, CONTRIBUTIONS, THE_TASK | the middle tier is dead, the third is a label-gated LoRA of 276 KB to 1.2 MB, and the first only reaches what the backbone already renders; replaced by decompose / fold / consolidate over a label-free prototype tree |
