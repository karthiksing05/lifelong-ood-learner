# CelebA grid: state, and what to do next

Written 2026-08-19. The working note for the CelebA thread, so the state survives a context reset.

## What exists

`ddpm_celeba_grid.pt` — trained FROM SCRATCH, 49 min, on the nine SEEN cells of a
`hair(3) x Male(2) x Smiling(2)` grid. 24 432 training images; **6 402 held-out images never
shown**. This is the only CelebA backbone here with a **density** holdout: every previous one was
fine-tuned from `google/ddpm-celebahq-256`, so held-out combinations were already in the pretrained
density and "held out" meant only "no embedding row".

## What it measures (N73, `results/n73_celeba_grid_compose.json`)

| arm | joint | % of probe ceiling | recall | vendi |
|---|---|---|---|---|
| address | **0.194** | 47 % | 1.000 | 13.62 |
| nearest-class | 0.125 | 30 % | 0.928 | 3.49 |
| unconditional | **0.000** | 0 % | 0.922 | 3.47 |

* **ideal-words basis R² 0.872** over nine seen cells — ColorMNIST scores 0.870, so compositional
  organisation of the class table is not an artefact of scale or of pretraining.
* **probe ceiling 0.414** (hair 0.568 · Male 0.849 · Smiling 0.859 on REAL images), chance 0.111.
  Always report against the ceiling: 0.194 is 47 % of what the instrument can register.

## The mechanism, confirmed (`tests/diag_grid_protos.py`)

| cell | init | d(proto, query) | d(real, query) |
|---|---|---|---|
| 2 | query | 93.06 | 33.82 |
| 7 | query | 85.07 | 40.66 |
| 9 | query | 86.37 | 51.46 |

Even QUERY-SEEDED chains end far from the query — farther than genuine other images of the same
class. On the pretrained backbones they returned the query pixel-for-pixel. That is the density
holdout working: the query is not a mode of a model that never saw its combination.

## RESOLVED 2026-08-19 — and the headline changed twice under measurement

Read `docs/RESULTS.md` §7ee for the full account. The short version:

**What CelebA supports.** A genuine DENSITY holdout — the only non-ColorMNIST backbone here where
held-out combinations were never in the model. ADDRESS beats the unconditional floor by
**+0.242 ± 0.069, won 12/16**, nothing is copied (0.000 memorised; d(query) 79.4 against a
real-vs-real spread of 40.0), and storage preserves every pre-existing weight bit-identically with
+0.000 forgetting.

**What it does NOT support.** Donation over retrieval. **ADDRESS − nearest-class = +0.049 ± 0.027,
won 8/16** — a coin flip. On ColorMNIST the same control scores 0.000 against ADDRESS's 0.599; here
it scores 0.273, because a sample conditioned on the NEIGHBOURING cell often reads as the held-out
one. That is loose conditioning in a 9.3M-parameter UNet trained half an hour, and it caps what this
dataset can discriminate.

**Nor the basis claim.** 0.875 in-sample against **0.207** leave-one-cell-out, at 9 classes over 8
free vectors. ColorMNIST's 6.3 ratio gives 0.826 / 0.772.

**And the grid cannot be enlarged to fix it.** Measured, not assumed:

| probe | readable attributes | best table | ratio |
|---|---|---|---|
| pixel, 64px | Smiling, Male, Blond_Hair | 8 cells / 7 vectors | 1.1 |
| pixel, 128px | Smiling, Male | 4 / 5 | 0.8 |
| Inception, 64px | + Wavy_Hair | 16 / 9 | **1.8** |

All well below the ~4 where in-sample and cross-validated R² start to agree.

## Two measurement rules this cost, both now standing artifacts

1. **`n75_grid_pooled.py` gates on a real-data arm** — real images of *that cell*, scored by *that
   run's* probes, on 200 images. 20 of 36 measurements fail it. The pre-gate margin (+0.102, won
   3/3) was almost entirely those cells.
2. **`diag_basis_cv.py` reports the classes-to-free-vectors ratio** beside every R². Below ~4, quote
   only the cross-validated number.

Both are pinned in `tests/test_claims.py` so the withdrawn margin cannot drift back.

## A wrong mechanism, corrected

The first blind cells were all brown-hair and `Brown_Hair` scores −0.154 as a binary probe, so
"brown is unreadable" looked settled. It is wrong — the benchmark scores hair with one 3-way softmax
refit per seed, and on the 9-seen backbone the single READABLE held-out cell is the brown one while
blond and `other` read blind. Legibility is a property of the fitted probe and must be measured per
cell per run.

## Still open

* **N74 (storage) has only run on the h3 backbone at `--quick` scale.** The loop closes — base
  bit-identical, +0.000 forgetting — but its ceiling arm has never cleared the gate at full scale,
  and the enumeration reader (2/3 vs gradient inversion's 1/3) has not been measured at 64 draws.
* **The realism gap is the backbone's** (precision 0.000 composed, ~0.20 for a TRAINED class, 0.833
  for real images) and would need a bigger/longer-trained UNet, not a better composer — the norm
  hypothesis is refuted at +0.000.
