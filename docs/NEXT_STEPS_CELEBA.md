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

## Added 2026-08-19

**Three sampling seeds** (`n73` seeds 0/1/2, same backbone): ADDRESS 0.204 +/- 0.048, nearest-class
0.102 +/- 0.030, unconditional 0.009 +/- 0.006.  The paired margin over ADDRESS's retrieval control
is **0.102 +/- 0.046 over three DISTINCT cells, won on 3/3** -- so it clears the +0.10 bar, but the
n that matters is 3, not the 9 cell x seed pairs.  Two more backbones (`_h5`, 5 held-out cells,
train seed 1; `_h4`, 4 cells, train seed 2) are training to raise it to twelve.

**The copying pair, both clean.** ADDRESS memorises **0.00** of the training set and sits **87.4
+/- 2.3** from a real held-out image where two real images of that cell sit **40.05** apart.  Not a
copy of the training set, and not a copy of the query.

**Instrument calibration is now part of every run.** A disjoint set of REAL images of the held-out
cell runs through the identical scoring pipeline, so precision / vendi / recall each have a measured
ceiling.  Prompted by ADDRESS scoring precision 0.000 against 0.083 for the other arms, which reads
as "off-manifold" and cannot be read at all until real data is scored the same way.

**The read, not the composer, is the CelebA bottleneck.** Gradient inversion (`learn_factored`)
recovered 1 of 3 held-out cells at 400 steps, and each wrong read drags the whole downstream loop:
N74's first full run scored address 0.167 on a cell where N73's true-factor composition scores 0.375.
Two fixes are in: `read_by_enumeration` scores all twelve candidates on common random draws, which is
exact on a grid this size, and N74 gained an `oracle` arm so read cost and storage cost are separable
rather than confounded.

## Open, in priority order

1. **Statistical power.** The margin is 0.102 +/- 0.046 at n=3 distinct cells -- over the bar, but
   on three cells from one backbone. The `_h5` and `_h4` backbones take it to twelve cells over
   three training seeds. Note the trade: more held-out cells means FEWER seen cells, and at 7 seen
   rows against 5 free basis vectors the in-sample R^2 stops being evidence -- report `cv_r2`.
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
