# Where this work stands against everything it is compared to

A single place to read what the method does and does not beat, with the comparison drawn on the axis
each baseline was actually built for. Numbers are 3-seed means from `results/aggregate.json` unless
marked otherwise.

**Reading note.** Two of the tables below are being re-derived under defaults that changed on
2026-08-11 (`cu` placement, plain-gradient mode ascent) and are marked **[re-deriving]**. They are
shown at their last canonical values, which is what the rest of the documentation currently quotes.

---

## 1. The claim, in one line

Cost should track the **vocabulary of the world**, not the **length of the stream**. Everything
below is an attempt to find out whether that holds and where it breaks.

## 2. Storage — the axis where the result is largest and least contested

| route | per concept | vs a block | quality |
|---|---|---|---|
| **ADDRESS** — factor coordinates | **72 B** | **31,895×** | F1 0.678 ± 0.042 |
| EXTEND BASIS — one direction | 1 KB | 538× | F1 0.755 ± 0.037, zero per-concept params |
| BLOCK rank 1 | 69 K | 8× | — |
| BLOCK rank 8 (`block-every`) | 2,296,467 B | 1× | F1 0.587 ± 0.020 |

Twenty-four concepts cost **1.7 KB against 55 MB**, and the backbone is never written to.

**Why this one is solid:** it is arithmetic on tensor shapes, not a measured quantity, and it did not
move through any of the five faults found on 2026-08-11. The *quality* column is 3 seeds and the
ordering held in every one.

**What it does not claim.** Addressing needs a factor basis to exist. There is none on an
unconditional pretrained model, which is why §12's pretrained track uses adapters instead. And the
router cannot yet tell EXTEND BASIS from BLOCK — a 69× cost decision currently made by hand.

## 3. Composition — against the methods built for it

ColorMNIST, held-out compositions **[re-deriving]**:

| method | F1 | joint accuracy | recall |
|---|---|---|---|
| **ours** | **0.605** | 0.539 | 0.778 |
| `poe-guided` (Wang et al., the companion paper) | 0.003 | **0.839** | 0.001 |
| `inversion` (textual inversion, span-restricted) | 0.545 | 0.403 | 0.799 |
| `inversion-free` (textual inversion, free vector) | 0.332 | 0.169 | **0.845** |
| `composable-diff` (Liu et al. 2022) | 0.030 | 0.000 | 0.134 |
| `composable-mcmc` (Du et al. 2023) | 0.000 | 0.003 | 0.134 |
| `nearest-class` | 0.005 | 0.000 | 0.066 |

**The honest reading is a trade, not a win.** `poe-guided` is *more accurate than us* — 0.839 against
0.539 — and has **essentially no coverage** (recall 0.001). It renders one correct image repeatedly.
We give up accuracy for coverage and win on F1, which is the product of the two. Anyone who cares
only about "is the attribute right" should prefer the companion method.

`composable-diff` and `composable-mcmc` score near zero here, and that is a **setting mismatch we
should not take credit for**: both are handed ground-truth factor labels at composition time and are
built for a regime where the concepts already exist as conditionals. On CelebA, where that regime
holds, `composable-mcmc` **beats us** (pair 0.965 against our 0.903).

## 4. Lifelong retention — against continual-learning baselines

ColorMNIST class-incremental, 11 arms **[re-deriving]**:

| arm | ACC | FGT | bit-identical | params |
|---|---|---|---|---|
| `c-lora` | **0.847** | 0.004 | 0.250 | 7.16 M |
| `independent` | 0.863 | **0.000** | **1.000** | **7.16 M** |
| **ours** | 0.839 | **0.000** | **1.000** | 8.82 M |
| `mas` | 0.839 | **0.000** | 0.250 | 7.16 M |
| `si` | 0.835 | 0.021 | 0.250 | 7.16 M |
| `ttt` | 0.828 | 0.021 | **1.000** | 7.16 M (+768 images kept) |
| `der++` | 0.821 | 0.049 | 0.250 | 7.16 M |
| `gen-distill` | 0.800 | 0.047 | 0.250 | 7.16 M |
| `finetune` | 0.752 | 0.123 | 0.250 | 7.16 M |
| `ewc` | 0.760 | 0.112 | 0.250 | 7.16 M |
| `replay` | 0.693 | 0.159 | 0.250 | 7.16 M |

**We do not win this table and should not claim to.** `independent` matches us on every column at
**fewer parameters**. The separation that is real is *categorical rather than incremental*: arms that
**remove** shared parameters from the optimizer reach bit-identity (ours, `independent`, `ttt`); arms
that **protect** them do not, at any regularization strength. `ttt` gets there only by keeping every
image of the stream.

Bit-identity is also **not novel** — PNN, PackNet, Piggyback and SupSup have it; `docs/RELATED.md`
says so and separates the methods that genuinely achieve it from those routinely miscited as doing
so. The differentiator against `independent` is contribution 2 (routing), measured elsewhere.

## 5. Unsupervised concept discovery — a baseline we could not run

Liu et al. (ICCV 2023) is structurally the closest published work to our addressing tier: a shared
library plus a small per-image coordinate, backbone frozen. It was implemented
(`lifelong_ood/neural/unsup_concepts.py`) and **it does not transfer to our setting**:

| arm | SEEN control | held-out |
|---|---|---|
| identity check (library = the true class embedding) | — | **0.802** |
| K = 8 / K = 18 / K = 18 no-simplex | 0.010 – 0.031 | 0.010 – 0.021 |
| *chance* | 0.006 | 0.006 |

Its control sits at chance, so **its held-out numbers are uninterpretable and quoting a win over it
would be fabricated**. The identity check passing at 0.802 proves the composition machinery is
correct and locates the failure in discovery. The paper itself gives the likely reason: its concepts
are word embeddings in a pretrained text-to-image model, and it states that the semantic density of
that space is what makes discovery tractable. Our from-scratch class table is the hard case it was
designed around. **Recorded as a gap in baseline coverage, not a victory.**

## 6. Against the companion paper's own results, on its own backbone

This is the axis that was broken until 2026-08-11 and is the reason for the fault list below. On
`google/ddpm-celebahq-256`, after the fixes, roughness against real faces (1.0 = matches real):

| arm | before | after |
|---|---|---|
| prototypes (mode ascent) | 0.78 – 1.40 | **0.56 – 0.69** |
| `poe-guided` | **2.50 – 4.82** | **0.82 – 1.11** |
| `query-only` | 3.53 – 4.27 | 0.93 – 1.19 |

The companion method now renders coherent faces at real-image texture on its own backbone. A
like-for-like composition comparison on a **fine-tune of that backbone** (their Table 5 setup) is
running as N47 → N48.

## 7. Five faults that were suppressing all of the above

Found and fixed on 2026-08-11. Each was silently degrading comparisons, and three of them produced
the visible CelebA artifact:

| # | fault | evidence |
|---|---|---|
| 1 | the UNet was trained on a **pixel permutation** | seam at rows H/3, 2H/3: 2.509 → **0.908** (real 0.906) |
| 2 | the **pretrained** backbone was fed permuted rows | 3×3 tiled mosaic; denoise RMSE 0.380 → **0.076** |
| 3 | CelebA guided at **2.4× the published strength** | w₀ 1.2 (their Color-MNIST value) → 0.5 |
| 4 | **Adam** marched past modes instead of finding them | 11.1% out-of-gamut → **0.0%**, eps 0.0113 → 0.0046 |
| 5 | the fine-tune **diverged** at their lr | loss → exactly 1.0 (network emitting zero) |

Faults 3 and 5 are cases where following the paper's stated setting was wrong *for our scale*, and
both are recorded as measured deviations rather than silent changes.

## 8. What is currently not supported

* **The adapter bank on a pretrained backbone produces noise** (§30d). Preservation is bit-identical
  and the adapter fits its pool to 0.0002 — which together prove only that bit-identity cannot detect
  a destructive delta. The image-quality half of "works on a model we did not train" is open.
* **The hierarchy earns nothing.** Insert-only category utility fixes the tree's *shape* (purity 0.27
  → 0.60) and buys no downstream gain (n5 −0.019, n9 +0.027, both inside error bars) at +23%
  parameters. It is bookkeeping, not a contribution.
* **Merge and delete are damaged.** Repeated merges destroy the model (0.837 → 0.073); deletion
  reclaims a median 0.250 of a path.
* **Growth is monotone, not constant.** Cost flattens only if the world's vocabulary is finite.
