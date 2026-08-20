# A theory of compositionality for lifelong, OOD learning

Written 2026-08-15. `docs/THE_TASK.md` states the task; this states the **mechanism** the task is
being solved by, in the order the claims depend on each other, with the measurement that supports
each and the measurement that would kill it. Everything here is measured on this repository's
backbones; nothing is assumed from the source paper.

---

## The claim, in one line

**A conditional diffusion model already holds a compositional coordinate system over its own
conditioning space. A new experience is a POINT in that system, so absorbing it costs an address
rather than a model — and the two hard parts, *identifying which point* and *covering the class it
names*, are separable and should be solved by different machinery.**

That last clause is the part this repository adds, and §4 below is the experiment that isolates it.

---

## 1. The substrate: the class table factorises without being told the factors

A class-conditional diffusion model trained on `slot = (digit, fg, bg)` — one free embedding row per
combination, never told those rows have parts — nonetheless arranges them so that

```
C[s]  ≈  μ + Σ_f  V_f[ value_f(s) ]
```

fits with **R² 0.870 (ColorMNIST) · 0.620 (CelebA) · 0.538 (Shapes3D) · 0.832 (latent DiT)**, and
**0.931 / CV 0.875** for the multi-object scene table against a permutation null of 0.254 (§7bb).
Four backbones, three datasets, two architectures, and one case (multi-object) where the "factors"
are *objects in a set* rather than attributes of one object.

**Why it is not circular:** the fit is performed *after* training, on a table that received no
factor supervision, and it is scored against a permutation null and leave-one-out CV where the
sample is small enough for R² to come cheap.

**The claim has a size condition, and it is now measured.** In-sample R² is a fit statistic; how
much it is worth depends on how far the classes outnumber the free vectors. `tests/diag_basis_cv.py`
reports both, per backbone:

| backbone | classes | free vectors | ratio | R² | R² leave-one-out |
|---|---|---|---|---|---|
| ColorMNIST | 120 | 19 | 6.3 | 0.826 | **0.772** |
| ColorMNIST (no-bg3) | 90 | 18 | 5.0 | 0.870 | **0.816** |
| ColorMNIST (stream) | 56 | 15 | 3.7 | 0.883 | **0.818** |
| CelebA grid, 9 seen | 9 | 8 | 1.1 | 0.872 | **0.474** |
| CelebA grid, 7 seen | 7 | 8 | 0.9 | 0.883 | **0.038** |

Above a ratio of ~4 the two agree and the in-sample number is a real statement about the table.
At a ratio near 1 they diverge completely — an additive model with almost as many parameters as
observations fits anything, and the CelebA grid's 0.872 is mostly arithmetic. **So §1 rests on
ColorMNIST, the multi-object table and the DiT, not on the CelebA grid**, whose contribution is the
*density* holdout (§7ee) rather than the basis. The ladder makes the same point from the other side:
CV R² tracks training almost exactly (−0.145 at 2 epochs → 0.790 at 130), so it measures something
real rather than sample size alone.

**The VLA consequence.** A policy's task table must have several times more task rows than
(factor values summed) before this claim can even be tested on it. That is a concrete design
constraint on the benchmark, and it is cheap to check before any training.

**But a high R² is NOT a precondition for ADDRESS to work, and §7jj is the sharpest evidence.** On
the pretrained Shapes3D table the additive fit reads **0.516 in-sample against 0.454 leave-one-out**
— agreeing, because the ratio is 16.4, so the number is trustworthy — and ADDRESS still reaches
**0.983 against a real-image ceiling of 1.000**, with retrieval at 0.010. Composition works through
a basis that explains *half* the embedding variance. §5t-ter found the same on a different axis: R²
is highest at the learning-rate rung where ADDRESS is near its worst.

So the size condition above is about **identifying** the basis, not about the basis explaining
everything. Read this section as: the table must be big enough for the decomposition to be
estimable, and the estimate must then be *checked by composing*, never by its own R².

**What would kill it:** an R² at the permutation null on a new backbone, or a cross-validated R²
far below the in-sample one at a healthy ratio. That is the first thing to run on a policy
(`VLA_PATH` Stage A) precisely because everything below is void without it.

## 2. A held-out combination is a POINT, not a direction

`ADDRESS` places `c_new = μ + Σ_f g_f (w_f · V_f)` — a new point in the span the basis already has.
Measured, it adds **no new direction at all**: the affine residual of a composed address against the
existing span is **0.0000**, where a *trained* embedding for the same concept has residual 0.7359
(§6e-bis).

This is what makes the storage claim arithmetic rather than empirical: 72 B for an address against
69 K for a rank-1 block. It also predicts, correctly, that ADDRESS cannot add representational
capacity — "composition-strong and representation-flat are one fact", which is why the N50
representation headline was withdrawn rather than defended.

## 3. The basis geometry predicts what can and cannot be addressed

Not every point in the span is reachable. On multi-object scenes, ADDRESS composes 6 of 9 held-out
pairs and fails 4 — and the failures split into two mechanisms (§7bb-bis):

| failure | signature | fix |
|---|---|---|
| **suppression** — one direction dominates the sum | corr(\|p_a − p_b\|, exact) = **−0.63** | a larger gain recovers it (0.33, 0.25 at g=2.5–4.0) |
| **destructive interference** — the two directions cancel | cos(W_a, W_b) = **−0.50, −0.75**; ‖W_a+W_b‖ = 0.95, 0.61 vs a 1.31 median | nothing: *g* scales a sum that already cancelled |

The second kind is **predictable from a dot product before anything is generated**. That is the
theory earning its keep: a compositional store can know in advance which compositions it must
*store* rather than *address*, which is the routing decision stated in geometry rather than in a
threshold.

## 4. Identity and coverage are separable, and want different machinery

This is the load-bearing claim of the repository, and the one the case-study figures obscured.

Producing a new instance of an unseen combination decomposes into two questions:

* **IDENTITY** — *which* composition is this? A property of one query.
* **COVERAGE** — what does the *class* look like? A property of a distribution.

They are usually solved by one mechanism and should not be. Wang et al. answer both with a Product
of Experts teacher: mode ascent (Eq. 4) discovers prototypes, submodular selection (Eq. 8–9) picks K
complementary ones, and a per-dimension product (Eq. 7/10) aggregates them into `μ_T`, which then
*guides sampling* (Eq. 12–14).

The identity half works: **`μ_T` probe-reads the correct composition** (§5e-quater) — but only as
the *aggregate*. Raw mode ascent on the composed conditional does not: on the pretrained Shapes3D
backbone the **mode** of `p(x | c_new)` reads the combination at **0.62** where **sampling** the same
address reads it at **1.00**, and it is no less diverse either (vendi 1.16 against 1.11, §7ll). So
the seam below takes identity from their *selection and product*, not from ascent alone, and on a
sharply peaked conditional there is no identity/coverage gap left for a mode to exploit. The coverage
half cannot, and the reason is structural rather than a tuning failure — **a product of Gaussians is
itself a Gaussian**, so under a deterministic sampler one query determines one image. Measured:
**recall 0.000, vendi 1.06 over 24 draws**, at joint accuracy 0.958. Their §4.2 recovers diversity by
running 100 different queries, i.e. by paying for coverage in queries.

Sampling the model's **own conditional** instead is multimodal by construction, and covers the class
from **one** query: recall **0.648–0.99**.

So the theory says to split the pipeline at the seam:

```
their discovery + selection + aggregation  ->  μ_T   ->  READ the factor values
                                                          |
                                    our factor basis  ->  c_new
                                                          |
                                     our generation   ->  p_θ(x | c_new)
```

**N67 tests exactly this seam** and is the direct evidence for §4. Prior to it the two pipelines were
disjoint — our `compose_factored` does textual inversion against the query and never calls mode
ascent, despite a docstring claiming otherwise — so "identity from theirs, coverage from ours" was an
argument, not a measurement.

## 5. Absorption must not disturb the substrate

An address is written into a table the rest of the model reads. The claim is bit-exactness, not
"small drift": every parameter is compared by **SHA-256** before and after, on ColorMNIST, CelebA,
Shapes3D and the DiT, and FGT is 0.000. This is what makes the scheme *lifelong* rather than merely
*incremental* — nothing already learned is overwritten, so cost tracks the **vocabulary of the
world** rather than the **length of the stream** (savings 9.5× / 8.9× / 12.4×, and Shapes3D's
vocabulary saturates at experience 11 of 100).

---

## What the theory forbids

A theory that forbids nothing explains nothing. This one forbids:

* **reading a training loss as evidence of conditioning.** The ε-loss gap is ~0.0001 at *every* rung
  of the class-table LR ladder, including the rung that does not compose at all (§5t-ter).
* **reading basis R² as a health check.** R² is *highest* at the 5× rung where ADDRESS is near its
  worst. Only the composition margin tracks composition.
* **reading accuracy without coverage.** `poe-guided` and `query-only` score 0.958 joint at recall
  0.000 by re-emitting the query. Accuracy alone cannot distinguish composing from copying — which
  is why recall is a headline number here and not a diagnostic.
* **expecting one operating point to transfer.** `U_NOVEL = 0.10` reads AUROC 0.354 on a pretrained
  backbone and 0.861 once calibrated (§6c-quinquies).

## Where it is weakest, stated plainly

* **§4 is one dataset deep** until N67 runs on more than ColorMNIST.
* **§3's interference law rests on n = 9** — the two unrescuable pairs being the two most
  anti-aligned is clean; the overall correlation (+0.46) is not a law yet.
* **§1 on multi-object is one backbone**, three sampling seeds, no training-seed replication.
* The **router** transfers only with a per-backbone calibration, and `U_VOCAB` has never been
  calibrated at all.
