# Where this work stands against everything it is compared to

A single place to read what the method does and does not beat, with the comparison drawn on the axis
each baseline was actually built for. Numbers are 3-seed means from `results/aggregate.json` unless
marked otherwise.

**Reading note.** Every table below was re-derived on 2026-08-11 under the current defaults, after
a day of fault-finding. One result is **retracted** in §8 and the retraction is the most important
line in this document.

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
`figures/f16_addressing.png` plots the three curves against concept count: four orders of magnitude
apart on a log axis, and the gap widens with every concept absorbed.

**Why this one is solid:** it is arithmetic on tensor shapes, not a measured quantity, and it did not
move through any of the five faults found on 2026-08-11. The *quality* column is 3 seeds and the
ordering held in every one.

**What it does not claim.** Addressing needs a factor basis to exist. There is none on an
unconditional pretrained model, which is why §12's pretrained track uses adapters instead. And the
router cannot yet tell EXTEND BASIS from BLOCK — a 69× cost decision currently made by hand.

**On how much an OOD query warrants (RESULTS §6c-ter, f32).** The router's residual over-spend —
0.200, and the whole oracle-vs-measured gap in the growth accounting — resisted six attempts to
close it. The sixth reframed the decision as Cobweb does, comparing hypothetical tiers under a
description length rather than thresholding a novelty statistic. At the benchmark's operating point —
one query per concept — it is far worse than the cascade (0.333 against 0.833); given **32 or more
instances of a concept it becomes indistinguishable from it** (0.722–0.778 against 0.833, a
one-concept gap on 18) while calibrating **no threshold at all**, and its promotion curve does what
nothing else here can: in-distribution is never promoted at any depth, new-combination converges to
exactly ADDRESS and stops, new-vocabulary climbs past it toward EXTEND. The cascade is not being
replaced. But the rule explains the other five attempts: under `L(tier) = C_tier + N × bits-per-example`, the fit term scales with the number of examples
and the storage cost does not, so **the tier a query warrants is not a property of the query** — it is
a property of the query and how many instances have arrived. A threshold is a function of the query
alone, so all six were answering an underdetermined question. Measured, one novel image buys ~33 bits
of coding gain against ADDRESS's 576-bit cost: **≈ 15 instances justify an address, ≈ 36 a vocabulary
extension**, where the benchmark supplies one. The cascade stays; the accounting is now stated.

## 3. Composition — against the methods built for it

ColorMNIST, held-out compositions:

| method | F1 | joint accuracy | recall |
|---|---|---|---|
| **ours** | **0.630** | 0.556 | 0.766 |
| `poe-guided` (Wang et al., the companion paper) | 0.003 | **0.839** | 0.001 |
| `inversion` (textual inversion, span-restricted) | 0.531 | 0.372 | 0.778 |
| `inversion-free` (textual inversion, free vector) | 0.311 | 0.156 | **0.840** |
| `composable-diff` (Liu et al. 2022) | 0.030 | 0.000 | 0.134 |
| `composable-mcmc` (Du et al. 2023) | 0.000 | 0.003 | 0.134 |
| `nearest-class` | 0.005 | 0.000 | 0.066 |

**The honest reading is a trade, not a win.** `poe-guided` is *more accurate than us* — 0.839 against
0.556 — and has **essentially no coverage** (recall 0.001). It renders one correct image repeatedly.
We give up accuracy for coverage and win on F1, which is the product of the two. Anyone who cares
only about "is the attribute right" should prefer the companion method.

> **Read the `poe-guided` row as an internal comparison only, never as that paper's result.** Every
> row above is measured **our** way: 24 samples from **one** query, scored against held-out images in
> a pixel feature space. Wang et al. measure **1600 samples from 100 queries** in **Inception-V3**
> features, against **two** reference sets, and report **80.7** where this row reads 0.001. Under
> *their* protocol their method reproduces here — Generalization precision 0.664 against their 0.670,
> Faithfulness F1 0.689 against their 0.753 (RESULTS §5e-bis). The comparison *between the rows
> above* is sound: all arms are measured identically, and the one asymmetry that existed
> (`var_scale`) was refuted by measurement. What the 0.001 must not be used for is a claim about
> their published numbers. `experiments/n49_paper_protocol.py` is the like-for-like table.

`composable-diff` and `composable-mcmc` score near zero here, and that is a **setting mismatch we
should not take credit for**: both are handed ground-truth factor labels at composition time and are
built for a regime where the concepts already exist as conditionals. On CelebA, where that regime
holds, `composable-mcmc` **beats us** (pair 0.965 against our 0.903).

### 3b. The same comparison on the companion paper's own protocol and metric space

The table above is *our* measurement. Run their §4.2 protocol instead — 100 queries × 16 generations
per class, Inception-V3 features, both of their reference sets — and their method reproduces
(§5e-bis, §5e-ter). At the **matched 4-query budget**, on Generalization:

| arm | precision | recall | F1 | **joint accuracy** |
|---|---|---|---|---|
| **ours** | 0.195 | **0.797** | **0.291** | 0.435 |
| PoE (Wang et al.) | **0.547** | 0.137 | 0.194 | 0.596 |
| query-only | 0.549 | 0.080 | 0.130 | **0.680** |
| top-1 | 0.208 | 0.610 | 0.267 | **0.000** |
| top-3 | 0.042 | 0.418 | 0.048 | 0.005 |

**The last column is not optional.** `top-1` out-covers PoE at joint accuracy **0.000** — it renders
the wrong composition thoroughly. The k-NN metrics score distributional coverage, not correctness,
which is why the paper's own Table 1 puts top-1 at F1 0.062 despite recall 42.2. PoE and query-only
sit at the opposite corner: accurate, and covering nothing.

**Ours is the only arm on both axes** — coverage 0.797 with accuracy 0.435 — and at a **single
query** the gap is starker: recall **0.648** against PoE's **0.073**, where PoE needs ~100 queries to
reach 0.300. That is the architectural difference, not a tuning one: their Eq. 13 guides toward a
unimodal Gaussian under a deterministic η = 0 sampler, so one teacher yields one image and their
diversity comes from having a hundred of them; we use the teacher only to *locate* the concept and
let the model's own multimodal conditional generate.

**Still a trade, not a sweep.** Our Generalization precision (0.195) is a third of PoE's (0.547) —
our pool is more dispersed than the real class. And **`+LoRA` did not reproduce**: their best arm
(published F1 87.6 / 76.3) comes out at 0.169 / 0.005 in our hands, so no claim here rests on
beating it.

**And the precision gap is not a knob we left in the wrong place.** The obvious suspect is the
guidance used when the manifold generates our pool — turn it up, concentrate on the mode, trade
coverage for precision. `tests/diag_ours_frontier.py` sweeps it and finds no such trade:

| pool guidance | G. precision | G. recall | G. F1 | joint |
|---|---|---|---|---|
| 1.0 | 0.156 | 0.818 | 0.228 | 0.297 |
| **1.5 — calibrated** | **0.219** | 0.833 | **0.303** | **0.469** |
| 2.5 | 0.078 | 0.887 | 0.133 | 0.094 |
| 4.0 | 0.062 | 0.443 | 0.059 | 0.000 |

Raising the guidance costs precision *and* composition accuracy while buying almost no coverage. So
the distance to their 0.670 is a property of the route on this backbone, not a setting to be tuned —
and it should be read as a real limitation rather than an unexplored option. Worth noting separately:
the calibration picks 1.5 using **trained slots only**, having seen none of these held-out numbers,
and it lands on the best of the four. That is evidence the label-free calibration works, and it is
the reason none of these numbers were used to choose it.

### 3b-CelebA. The same protocol on their own backbone — where our arm holds up

N48 runs their Table 2 on the N47 fine-tune of `google/ddpm-celebahq-256` at 128 px — their own
backbone configuration, not ours — over 6 held-out classes, 16 queries pooled, Inception-V3.

| arm | mean attr | **joint** | General. P | General. R | General. F1 |
|---|---|---|---|---|---|
| `poe-guided` (Wang et al.) | 0.885 | **0.544** | **0.753** | 0.034 | 0.065 |
| `query-only` | 0.879 | 0.528 | 0.681 | 0.039 | 0.074 |
| **`ours (ADDRESS)`** | 0.858 | **0.439** | 0.585 | **0.305** | 0.401 |
| *unconditional (floor)* | *0.733* | *0.161* | *0.474* | *0.359* | ***0.409*** |
| **their Table 2, PoE** | — | — | ***0.745*** | *0.090* | *0.156* |

**Their method reproduces closely**: Faithfulness F1 0.187 against their published 0.203, and
Generalization *precision* **0.753 against their 0.745** — within 0.008, through an independent
reimplementation on a separately fine-tuned backbone.

**Ours clears the floor on accuracy by 2.7×** — joint 0.439 against 0.161 — and on every one of the
six classes separately (ratios 1.8×, 2.1×, 2.3×, 2.7×, 5.2×, 6.3×).

> **What this replaces.** An earlier version of this section concluded the opposite — "ours does not
> clear the floor here" — and scoped the composition claim to 3-factor benchmarks. That reading was
> produced by an indexing bug in `attr_accuracy` (the target combination was indexed by position in a
> *filtered* attribute list rather than by canonical position), which suppressed every arm's accuracy
> to near the floor. The scoping is withdrawn: the same accuracy/coverage pattern found on ColorMNIST
> is present on CelebA. RESULTS §5e-sexies carries the full corrected table and the audit trail.

**The accuracy/coverage scissors, on this benchmark too.** Neither column ranks the arms alone.
`poe-guided` and `query-only` are the most accurate and cover almost nothing — recall 0.034 and
0.039, i.e. they re-emit the query, which their own §4.3 says of query-only. `unconditional` covers
the most and carries the composition worst. **`ours` is the only arm on both axes**: 2.7× the floor's
accuracy at 9× `poe-guided`'s recall.

**What still does not clear the floor.** Generalization **F1** is 0.401 against the floor's 0.409 —
a tie. F1 mixes precision with recall, and an unconstrained sampler buys recall by being diffuse; on
that composite alone ADDRESS is not distinguishable from not conditioning. This number was never
affected by the bug and it stands. The claim the data supports is the joint one — coverage *at*
accuracy — not coverage on its own.

**Two limits on the accuracy column.** (i) Only 5 of 14 attribute probes are readable on this
backbone and `joint` requires all of them at once, so the absolute level depends on how they were
fit. The run was therefore done twice. With Inception-V3 probes instead of pixel probes — 6 readable
of 14, stricter, and free of `Bald` (base rate 0.000 in this split, hence a probe that cannot be
wrong) — every level drops and the comparison does not: **ours 0.245 against a floor of 0.108, a
2.3× margin** against pixel's 2.7×, clearing the floor on every class in both spaces. Read the ratio,
not the level. (ii) The control that validates the metric at all now passes:
conditioning on a class's own trained embedding scores **0.517** against an unconditional **0.200**,
with the basis reconstruction at **0.500** — 97 % of the ceiling (`tests/diag_basis_ceiling.py`).
Under the bug this control read 0.100 against 0.142, i.e. ceiling below floor, which is what
justified deleting the accuracy column in the first place.

*An explanation offered here earlier — that CelebA's weaker factor basis (R² 0.620 vs 0.826) bounds
what ADDRESS can express — was tested by that control and **refuted**: the basis reaches 97 % of the
own-embedding ceiling. It has been withdrawn.*

**So the composition claim is not scoped to 3 factors.** It holds on ColorMNIST and on a
14-attribute CelebA under the paper's own protocol and backbone. The storage claim (§2) is unaffected
— it is arithmetic on tensor shapes — and so is the preservation guarantee.
### 3c. The finding that explains the rest of §3, and generalizes past our method

Across all three backbones tested — our ColorMNIST DDPM, the raw pretrained CelebA model, and a
fine-tune of the paper's own Table 5 configuration — **`poe-guided` and `query-only` produce the same
images**. On the paper's own backbone they agree to 0.002 on attribute accuracy and *exactly* on
joint. That is worth explaining rather than reporting, because a composition method that cannot be
told apart from "spread a Gaussian around the query" has not demonstrated composition.

Three explanations were tested and two died: it is **not** memorization of held-out data (the split
is clean and 24/24 prototypes are nearest to the *training* set), and it is **not** the selection
objective preferring query-seeded candidates (forbidding them entirely leaves the composition
identical in 6/6 slots).

The third is the answer, and it is about the **channel**, not the teacher. The two teachers' means
sit 0.082 apart; sweeping the guidance weight shows how much of that reaches the image:

| w₀ | fraction of the teacher difference transmitted | joint accuracy (both arms) |
|---|---|---|
| **1.2 — the paper's** | **0.33** | **0.56 – 0.64** |
| 3.0 | 1.79 | **0.00** |

**There is no setting where the two arms are distinguishable and the composition survives.** Guidance
strong enough to transmit the difference drives the trajectory off the manifold; guidance weak enough
to stay on it cannot express a difference of 0.08. This is a property of Eq. 13/14, not of our
reimplementation.

**Why this matters beyond a baseline row.** It is the quantitative case for the one place our
pipeline departs from theirs (Stage 4 of `figures/f0_architecture.png`): if the guidance channel
cannot carry the difference between two teacher means, then no choice of teacher distinguishes itself
through it, and using the teacher as the *sampling distribution* is the wrong use of it. Reading the
composition into `c_new` and letting the model's own multimodal conditional generate is what produces
coverage from a single query (recall 0.648 against PoE's 0.073) — and the measurement above is why
that is a structural fix rather than a tuning preference.

## 4. Lifelong retention — against continual-learning baselines

ColorMNIST class-incremental, 11 arms:

| arm | ACC | FGT | bit-identical | params |
|---|---|---|---|---|
| `c-lora` | **0.847** | 0.004 | 0.250 | 7.16 M |
| `independent` | 0.863 | **0.000** | **1.000** | **7.16 M** |
| **ours** | 0.825 | **0.000** | **1.000** | 8.82 M |
| `mas` | 0.839 | **0.000** | 0.250 | 7.16 M |
| `si` | 0.835 | 0.021 | 0.250 | 7.16 M |
| `ttt` | 0.828 | 0.021 | **1.000** | 7.16 M (+768 images kept) |
| `der++` | 0.821 | 0.049 | 0.250 | 7.16 M |
| `gen-distill` | 0.800 | 0.047 | 0.250 | 7.16 M |
| `finetune` | 0.752 | 0.123 | 0.250 | 7.16 M |
| `ewc` | 0.760 | 0.112 | 0.250 | 7.16 M |
| `replay` | 0.708 | 0.159 | 0.250 | 7.16 M |

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
| identity check (library = the true class embedding) | — | **0.750** |
| K = 8 / K = 18 / K = 18 no-simplex | 0.010 – 0.031 | 0.010 – 0.062 |
| *chance* | 0.006 | 0.006 |

Its control sits at chance, so **its held-out numbers are uninterpretable and quoting a win over it
would be fabricated**. The identity check passing at 0.750 proves the composition machinery is
correct and locates the failure in discovery, which is the reason this section reports a gap rather
than a win. *(The figure that showed it, `f22_liu_unsup_ident`, was retired 2026-08-13 — it
illustrates a baseline-coverage gap rather than any compositional claim, and the table above carries
the whole argument. `experiments/n46_unsup_concepts.py` still produces it on demand; the panel is in
`figures/archive/`.)* *(Re-run 2026-08-12 so the figure and
these numbers come from the same job: identity 0.750, SEEN control 0.010 against a chance of 0.006,
held-out 0.062. The earlier run read 0.802 / 0.010; the conclusion is unchanged and the control is
still at chance.)* The paper itself gives the likely reason: its concepts
are word embeddings in a pretrained text-to-image model, and it states that the semantic density of
that space is what makes discovery tractable. Our from-scratch class table is the hard case it was
designed around. **Recorded as a gap in baseline coverage, not a victory.**

## 6. Against the companion paper's own results, on its own backbone

This is the axis that was broken until 2026-08-11 and is the reason for the fault list below. On
`google/ddpm-celebahq-256`, after the fixes, roughness against real faces (1.0 = matches real):

| arm | before | after |
|---|---|---|
| prototypes (mode ascent) | 0.78 – 1.40 | 0.60 – 0.86 *(and they clip when rendered — §30c)* |
| `poe-guided` | **2.50 – 4.82** | **0.89 – 1.02** |
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
| 4 | *(withdrawn)* I changed the ascent optimizer believing it fixed the artifact | the **layout fix alone** already had: poe-guided 2.50–3.88 → 1.00/0.89/1.23 with Adam untouched. Reverted; see §5c |
| 5 | the fine-tune **diverged** at their lr | loss → exactly 1.0 (network emitting zero) |

Faults 3 and 5 are cases where following the paper's stated setting was wrong *for our scale*, and
both are recorded as measured deviations rather than silent changes. Fault 4 is listed because it was
*claimed and then withdrawn*: the repair had already been delivered by fault 1, and the unnecessary
change was costing `poe-guided` 0.100 of accuracy on Color-MNIST. Every benchmark now runs one
algorithm, with per-dataset settings only where they are the paper's own column or scale with the
input's dimensions.

## 8. What is currently not supported

* **"Works on a model we did not train" is now preservation-only.** The bank attaches to
  `ddpm-celebahq-256` at exactly 0.000e+00 and bit-identity holds — but the composition result that
  accompanied it (§12/N25, `both` = 0.417) was measured through a field feeding the UNet a pixel
  permutation, and **corrected it is 0.000 for every generated arm**, the untouched model included.
  Retracted 2026-08-11. The adapter fits its pool to 0.0002 while producing noise, which together
  with exact preservation proves only that bit-identity cannot detect a destructive delta (§30d).
* **The hierarchy earns nothing.** Insert-only category utility fixes the tree's *shape* (purity 0.27
  → 0.60) and buys no downstream gain (n5 −0.019, n9 +0.027, both inside error bars) at +23%
  parameters. It is bookkeeping, not a contribution.
* **Merge and delete are damaged.** Repeated merges destroy the model (0.837 → 0.073); deletion
  reclaims a median 0.250 of a path.
* **Growth is monotone, not constant.** Cost flattens only if the world's vocabulary is finite.
