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

## Open, in priority order

1. **Statistical power.** ADDRESS beats nearest-class on 2 of 3 cells (0.375 vs 0.292; 0.208 vs
   0.083) but the mean margin is **0.069**, under the +0.10 bar, at n=3. Fix by holding out MORE
   cells (a second backbone at `n_holdout=5` leaves 7 seen) and by a second training seed. The grid
   is capped at 12 cells by attribute readability, so beyond that it needs a second grid over a
   different readable factor set rather than a bigger one.
2. **Cell 2 fails for every arm** (blond·male·not-smiling, 518 images — the smallest cell). Either
   raise its support or report it as a support floor.
3. **Storage + donation on this backbone.** N71 (absorb, then sample the stored concept) and N72
   (the four donation conditions) have only ever run on ColorMNIST. This backbone is the first
   non-ColorMNIST one where they would mean anything, because it is the first with a density
   holdout.

## Traps already paid for here

* `n1_train.py` passed `layout` to `train_ddpm` but not to its eval field, which defaulted to
  `permuted` against an `hwc` model. The run self-reported joint **0.111 — exactly 1/9 chance** on a
  checkpoint that scores 0.292. `tests/diag_grid_layout.py` re-scores a checkpoint both ways.
* `--n-holdout` defaults to 100 (added for shapes3d) and held out 8 of 11 cells.
* The eval hardcoded `rng.choice(uniq, 16, replace=False)`, fatal for a 9-class grid.
* `--max-rows 25000` shrinks the smallest cell below `min_cell` and drops it from the grid.
