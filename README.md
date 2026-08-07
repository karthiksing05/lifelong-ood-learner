# lifelong-ood-learner

**Continually adding compositions — including out-of-distribution ones — to a learned generative
manifold, forever, without the manifold degrading.**

A model meets a composition it has never seen. It assembles that composition out of prototypes it
already has, samples the assembly through its own manifold to render it properly, and then absorbs
the result *back into the manifold* as a new, reusable concept. Then it does it again, and again,
and nothing it already knew gets worse.

```python
from lifelong_ood import LifelongLearner

L = LifelongLearner()
L.fit(stream)                                                    # incremental, no gradients
teacher, pool = L.learn_concept(x_ood_query, "cyan_on_deep_red") # compose -> sample -> consolidate
S = L.sample_concept("cyan_on_deep_red", n=16)                   # generate it, forever after
```

## Where this comes from, and what is new

Steps 1–3 of that loop are [Wang et al., *Test-Time Compositional Generalization in Diffusion Models
via Concept Discovery*](poe-diffusion.pdf), reimplemented faithfully: discover concept prototypes as
modes of a diffusion model's noisy marginals `p_t`, select a complementary few with a submodular
per-dimension coverage objective, multiply them into an analytic product-of-experts teacher, and
sample it through classifier-free guidance (their Eq. 4, 7–10, 12–14).

**Step 4 is ours, and the paper names the gap in its own conclusion:**

> *"human-level intelligence also requires learning new primitives and incorporating them into
> future compositions … A full mechanism for detecting, consolidating, and reusing genuinely new
> primitives remains an important direction for future work."*

Their distillation trains one flat `(LoRA, c_new)` pair per query and throws it away. Scaled to a
stream, that gives you either N isolated adapters that never help each other, or one adapter trained
sequentially that forgets. We make consolidation a first-class, accumulating operation with a
structural non-degradation guarantee.

### The mechanism: an insert-only hierarchy of adapters

The page we take from Cobweb is its *insertion discipline*, not its statistics. Consolidated concepts
are leaves of an incrementally grown, insert-only hierarchy, and a concept's effective adapter is the
**sum of the deltas along its root→leaf path**:

```
root delta                            shared by everything ever consolidated
 └─ "warm colours on dark" delta      shared by that neighbourhood
     └─ "cyan on deep_red" leaf       this concept alone
```

Two properties follow, and a lifelong learner needs both at once:

- **No forgetting, structurally.** Ancestors freeze the moment they are created; inserting a concept
  trains only its new leaf. An existing concept's path — and therefore its function — is *bit-identical*
  after any number of later insertions. We verify the strong form: not "accuracy stayed similar" but
  "the epsilon output on fixed noise is unchanged to the bit."
- **Forward transfer, structurally.** A new leaf inherits its ancestors' accumulated delta as a warm
  start, so a concept landing near relatives only has to learn its residual.

Ablations isolate each ingredient: `flat` (no sharing — the paper's per-query adapter, merely stored)
and `unfrozen` (hierarchy kept, ancestors trainable). Retention collapses in `unfrozen` exactly as
predicted.

## Two backbones, one method

`BaseScoreField` defines a diffusion manifold by a single primitive, `eps(x, u)`. Sampling, mode
ascent and the Laplace covariance are written against it once, so the composition-and-consolidation
stack runs *unchanged* on either backbone — results are attributable to the method, not the model.

| | **analytic** (`ScoreField`) | **neural** (`NeuralScoreField`) |
|---|---|---|
| what | a Cobweb tree read as a Gaussian mixture — score, Tweedie and ε all closed-form | a trained class-conditional DDPM |
| training | incremental `ifit`; **no gradient descent anywhere** | ~3.5 h on one GPU |
| discovery | free: the tree's own nodes *are* the modes of `p_t` | mode ascent on the learned score |
| consolidation | adapter trees merged at snapshot level; drift bounded by `log(1 + n_adapter/n_base)` | hierarchical LoRA bank |

The analytic backbone came first and gives something the neural one cannot: concept discovery for
free, with the hierarchy explicit. It also has a measured limitation that motivated adding the neural
one — see below.

## Results

![forgetting matrix](figures/n5_forgetting_matrix.png)

*Accuracy of every task after every step, on a class-incremental stream of real held-out images.
Every row is constant left-to-right for ours; rows visibly fade for sequential fine-tuning. A row
that darkens is a task being forgotten.* Full figure set and analysis in
[`RESULTS.md`](RESULTS.md).

ColorMNIST, 120 seen / 40 held-out compositional slots. Benchmark, palette and splits are identical
to [cobweb-poincare]'s reconstruction of the paper's Appendix B, so numbers are comparable
slot-for-slot across the two repositories. Every table reports the **probe ceiling** — the same
probes applied to *real* held-out images — because the metrics are probe-based and that ceiling is
what any method is actually competing against.

**The tree is a valid diffusion manifold (E1, full scale).** Unconditional DDIM sampling on the
analytic field lands in seen compositions at rate **1.000**; denoise R² 0.995. And its hierarchy *is*
the density-mode hierarchy the paper recovers from a DDPM by gradient ascent — ascending `p_t` and
matching to the nearest tree node:

| noise `u` | 0.10 | 0.25 | 0.40 | 0.60 | 0.80 |
|---|---|---|---|---|---|
| matched node depth | 7.12 | 7.25 | 7.25 | 7.25 | **1.62** |
| match RMSE | 0.0002 | 0.0006 | 0.0011 | 0.0076 | 0.0296 |
| fraction that are leaves | 1.00 | 1.00 | 1.00 | 0.88 | **0.00** |

At low noise the modes are *exactly* the tree's leaves (RMSE 2e-4); at high noise they migrate to
shallow internal nodes. Free, explicit, and no ascent required to obtain it.

**One OOD query is enough to recreate its composition (E2, all 40 held-out slots × 32 samples).**

| method | joint | digit | fg | bg | novelty | diversity | gen |
|---|---|---|---|---|---|---|---|
| `poe_tree` (closed-form discovery) | 0.603 | 0.69 | 0.99 | 0.84 | 0.222 | 0.296 | 0.244 |
| `poe_ascent` (the paper's discovery) | 0.583 | 0.65 | 0.97 | 0.83 | 0.230 | 0.304 | 0.246 |
| `query_only` | **0.749** | 0.80 | 0.94 | 0.99 | **0.075** | **0.094** | 0.131 |
| `top1` | 0.000 | 0.65 | 0.35 | 0.55 | 0.254 | 0.311 | 0.250 |
| `top3` | 0.054 | 0.64 | 0.42 | 0.44 | 0.249 | 0.309 | 0.246 |
| *reference set's own internal spread* | | | | | | *0.152* | |

Read the accuracy column together with the geometry columns, because alone it is misleading:

- **Nearest-trained-class retrieval does not recover the composition.** `top1`/`top3` get the *digit*
  right (0.65) and the colour combination wrong (fg 0.35, bg 0.55/0.44), for a joint score of ~0.
- **`query_only` wins on accuracy by memorizing.** Its novelty is 0.075 and its diversity 0.094 —
  against a reference set whose own internal spread is 0.152. It is producing near-copies of the one
  query, which is precisely the failure mode the paper reports for this baseline (best faithfulness,
  collapsed generalization). On accuracy alone it would "win", which is the whole reason the geometry
  columns are here.
- **PoE recovers the composition *and* varies within it**, at diversity 0.296 — about twice the
  reference spread. Being *more* varied than the true class is its own imperfection, and worth
  saying: some of that spread is off-manifold rather than useful variety.
- **Closed-form tree discovery ≈ the paper's mode-ascent discovery** (0.603 vs 0.583). On this
  backbone, concept discovery is free — which is the extra claim the Cobweb hierarchy buys.

**Consolidation does not degrade the manifold (E3, full scale).** Over 12 consolidations, drift in
held-out log-likelihood totals **−0.15 nats**, drift in denoise R² is **+0.0000** throughout, and
unconditional sample validity stays 0.88–1.00.

More usefully, the drift is *exactly* what theory says it must be. Grafting leaves every base
component untouched, so the merged density is `(1−a)·p_base + a·p_adapters` and drift on old data is
bounded below by `log(1−a)`, with equality when the adapters contribute nothing there. E3 computes
that bound from the mixture weights actually in use and compares:

| after consolidation | 1 | 2 | 3 |
|---|---|---|---|
| observed drift (nats) | −0.01 | −0.02 | −0.04 |
| predicted bound `log(1−a)` | −0.01 | −0.02 | −0.04 |

The bound is not merely satisfied, it is *attained* — the new concepts sit where the old data is not,
which is exactly what one wants from adding an out-of-distribution composition.

Two things to state plainly rather than bury:

- **The destructive control was not very destructive.** Classic Cobweb absorption (`ifit`) drifted
  −0.10 nats by step 8 against grafting's −0.10 — indistinguishable. Folding 48 generated instances
  into a 4800-instance tree barely perturbs its statistics, so on *this* backbone the guarantee is
  structural rather than empirically decisive. The mechanism earns its keep on the neural backbone,
  where gradient training genuinely does overwrite (`unfrozen` retention 1.00 → 0.00).
- **Consolidation loses some of the composition.** Conditional sampling from a grafted adapter
  retains roughly 50–80% of the pool's composition accuracy (pool 1.00 → 0.50, 1.00 → 0.81). The
  analytic `sample_concept` path has not had the tuning treatment `sample_teacher` got; see
  `PLAN.md` §3.4.

**The neural backbone is real (N1).** 6.8M-parameter UNet, 200 epochs, 146 min on one GPU:
unconditional valid-composition rate **0.938**, and conditional joint accuracy **0.789** on requested
seen slots — level with the 0.775–0.82 probe ceiling. Samples are clean digits in the requested
colours on the requested backgrounds (`results/n1_cond_samples.png`).

### What did not work, and what it cost to find out

- **The PoE teacher is far too sharp, and that is structural.** Its covariance comes from Laplace
  estimates at modes, which measure *local curvature*, not how far a class extends — and the product
  over K experts sharpens it further. Measured: teacher σ **0.0117** against a real intra-class
  spread of **0.1595**, a 13.6× gap. Guided sampling faithfully reproduces that tightness, so every
  chain lands on μ_T and a "pool" of 64 images is 64 near-copies. Two sweeps (initialization, then
  the whole guidance schedule from `wmin=0.0` to `0.8`) moved diversity from 0.016 to at most 0.042
  before the real cause was measured. Inflating the covariance recovers spread along a clean
  tradeoff — scale 1 → 0.13× reference at joint 0.94; scale 30 → 0.61× at 0.71; scale 100 → 0.78× at
  0.70 — and the scale that works matches the predicted variance ratio (13.6² ≈ 186).
- **On the neural backbone, `query_only` beats PoE.** With baselines held faithful (the variance
  correction is *not* applied to a baseline whose variance is defined rather than estimated),
  N2 reads `query_only` 0.833 vs `poe` 0.674, at near-identical geometry. When the backbone
  generalizes well enough to re-render a single query, concept discovery is not obviously buying
  anything on this benchmark — the opposite of the analytic backbone, where `query_only`'s
  memorization is stark. This is a real negative result, not a tuning gap we have closed.
- **The analytic `ifit` control is not destructive**, so the analytic non-degradation guarantee is
  structural rather than empirically decisive (see above).

[`RESULTS.md`](RESULTS.md) collects every number in one place, together with an honest account of
**why we do not reproduce the paper's query-only collapse** -- the one row where our table inverts
theirs. `PLAN.md` carries the working status, the open work, and the threats to validity.

### The headline: 10 OOD concepts consolidated in sequence (N3, neural backbone)

| arm | bit-identical retention | per-concept accuracy drift | accuracy range across steps | plasticity | frozen-base drift |
|---|---|---|---|---|---|
| **`hier`** (ours) | **1.00** | **+0.021** | **0.000** | 0.65 | +0.00000 |
| `flat` (no sharing) | 1.00 | +0.035 | 0.000 | 0.64 | +0.00000 |
| `unfrozen` (no freezing) | **0.00** | **−0.625** | 0.375 | 0.66 | +0.00000 |

Reproduce with `python experiments/analyze_n3.py`.

The middle columns are the claim. With ancestors frozen, **every consolidated concept's composition
accuracy is literally constant** — range 0.000 across all subsequent consolidations — and its
epsilon output is bit-identical. Remove only the freezing and concepts that worked at birth lose
**0.917** of their accuracy. Plasticity is unaffected (0.64–0.66 across all arms), so this is not a
stability/plasticity trade being paid for: it is free, because it is structural.

**Consolidation is faithful when the pool is.** Across the stream, `pool → consolidated` reads
0.98 → 0.94, 0.83 → 0.88, 1.00 → 1.00, 0.00 → 0.00: **98% of the pool's composition accuracy is
retained** on pools that worked, and a bad pool is faithfully reproduced as a bad concept.
Consolidation neither invents quality nor destroys it, so the quality of the whole loop is set by
composition — the step this repository inherits from the paper — not by the step it adds.

**Batch addition is free (N4).** Adding 8 concepts in batches of 4 rather than one at a time is
**1.71x faster at identical final accuracy (0.609) and identical perfect retention** -- discovery
shares its query-agnostic modes, the K pools come out of one reverse-diffusion run, and distillation
is one joint optimization. It costs parameters (6.61M vs 4.96M) because shared group nodes are extra
nodes, not safety. A side effect we did not design for: batching *partially mitigates forgetting on
its own* -- the `unfrozen` arm improves from ACC 0.008 to 0.336 when batched, because concepts
trained together do not overwrite each other and only batch boundaries do damage.

**It also does ordinary class-incremental learning on real data (N5).** 12 tasks x 64 *real* held-out
images arriving in batches, no composition anywhere:

| arm | ACC | BWT | FGT | bit-identical |
|---|---|---|---|---|
| ours (hierarchical, frozen) | 0.802 | **+0.000** | **0.000** | **1.00** |
| independent (parameter isolation) | **0.839** | +0.000 | 0.000 | 1.00 |
| finetune (shared, sequential) | 0.651 | -0.199 | 0.205 | 0.25 |
| finetune + generative replay | 0.589 | -0.205 | 0.205 | 0.25 |

Zero forgetting by both standard measures. Two honest negatives sit in that table: **generative
replay does not help** (0.589 < 0.651, at 1.8x the wall clock -- replayed pools come from concepts
that are themselves drifting, so replay reinforces the drift), and **`independent` beats `ours` on
final accuracy** with fewer parameters. That sharpens the attribution: freezing delivers the
no-forgetting result; the hierarchy buys the warm start and the batch speedup, not final quality.

**Sampling was collapsing, and fixing it needed an architectural change, not a knob.** Guiding the
sampler toward the analytic PoE Gaussian produces near-copies of the query -- recall ~0, content
diversity 0.45x the real class -- and guidance schedule, teacher covariance, query pooling and
stochastic sampling all fail to fix it, because CFG toward a *unimodal* Gaussian contracts every
trajectory to its mean. The fix is to **condition the manifold instead of guiding it**: learn the
concept as a point in the model's own conditioning space, `c_new = sum_k alpha_k c_k` over the
trained class embeddings, and sample the model's conditional. Over 3 seeds:

| | F1 | recall | content div / real | accuracy |
|---|---|---|---|---|
| teacher-guided | 0.003 ± 0.003 | 0.062 ± 0.026 | 0.774 | **0.585 ± 0.081** |
| **inversion** | **0.323 ± 0.057** | **0.649 ± 0.066** | **1.166 ± 0.033** | 0.234 ± 0.040 |
| *ceiling (base model, seen class)* | *0.96* | *0.95* | *1.01* | *0.97* |

Coverage is solved and certified; **fidelity is the open problem** -- accuracy 0.585 -> 0.234,
concentrated in background. `RESULTS.md` §5e-§5g has the full elimination and what is already ruled
out.

**The older defect, for the record:** Vetting the
case-study grids (`figures/case_studies_neural.png`) showed every sample in a row as the same digit
instance varying only by speckle. Two checks confirm it: `diversity / novelty = 1.04`, where
independent draws around the query would give 1.41, and content diversity after a low-pass is
**0.45x** the real class's spread against the 0.61x raw pixels implied. This is the same defect
behind `query_only` beating PoE -- both are copying the query.

**And the mechanism does not degrade with scale.** `tests/test_unbounded.py` adds 40 concepts in
mixed batch sizes and asserts every earlier concept stays bit-identical at every step, the frozen
base never moves, and parameter cost per concept is constant (max residual 1.3%) -- so "how many
concepts fit in a budget" has an exact answer. Linear, not sublinear: ~45K params/concept at rank 2.

**The hierarchy buys a warm start, not a better endpoint.** New leaves inherit their ancestors'
deltas, so distillation *starts* 30% lower in the hierarchical arm (first-epoch loss 0.113 vs
0.160 flat) — but converges to the same place (0.0495 vs 0.0499). On this benchmark, sharing
accelerates learning without improving it. The freezing is what carries the no-forgetting result;
the hierarchy is what makes it cheap.

**Composition works, and one fix is responsible (E0, all 40 held-out slots).** Composing the
out-of-distribution teacher and reading it with the attribute probes:

| K=3 | joint | digit | fg | **bg** |
|---|---|---|---|---|
| the paper's fixed τ=0.5 | 0.300 | 0.80 | 1.00 | **0.38** |
| adaptive temperature (ours) | **0.800** | 0.80 | 0.97 | **1.00** |
| *probe ceiling on real held-out images* | *0.775* | *0.82* | *0.97* | *0.97* |

Background accuracy **0.38 → 1.00** is the whole story. Cobweb's variances sit near `prior_var`
everywhere, which compresses every per-dimension log-likelihood gap by a common factor; at τ=0.5 the
per-dimension softmax gave three selected experts weights 0.42/0.29/0.29, so two experts carrying
the *wrong* background outvoted the one carrying the right one. Setting the temperature from the
data's own scale fixes it. (The teacher scoring slightly *above* the ceiling is not a paradox: μ_T is
a clean prototype, and these probes read a clean prototype more easily than a real noisy image.)

**Guidance schedule.** A Gaussian mixture assigns essentially zero density to an unseen composition,
so during sampling its unconditional score pulls the result back toward seen slots. Annealing
guidance up as noise falls — handing the late, identity-deciding steps to the teacher — retains
**89%** of the teacher's accuracy against **78%** for the paper's variance-aware schedule. The gap is
much larger on a small tree (71% vs 21%), which is the tell: the density hole shrinks as the mixture
gets denser, and it is precisely the hole a convolutional UNet does not have. That is why the neural
backbone exists.

Both are deviations from the paper, documented at their implementation sites with the measurement
that motivated them.

`PLAN.md` carries the living status, the open work, and the threats to validity.

## Running it

```bash
python tests/test_smoke.py      # analytic, synthetic, ~10 s, no data needed
python tests/test_neural.py     # neural, CPU, <1 min (incl. batch addition)
python tests/test_unbounded.py  # 40 concepts: preservation and cost are K-independent

# on SLURM (login nodes are contended; lab convention is to validate via sbatch)
sbatch --export=ALL,SCRIPT="experiments/e1_manifold.py"           experiments/run.sh      # CPU
sbatch --export=ALL,SCRIPT="experiments/n1_train.py --epochs 200" experiments/run_gpu.sh  # GPU
```

Every experiment takes `--quick` for a small local run. `experiments/e0_tune.py` is the
tuning-of-record: it chose τ, K and the guidance schedule, and reports each against the probe ceiling.

| experiment | question |
|---|---|
| `e0_tune` | how the defaults were chosen (teacher sweep, then sampler sweep) |
| `e1_manifold` | is the Cobweb tree a valid diffusion manifold, and is its hierarchy the mode hierarchy? |
| `e2_composition` | can one OOD query be recreated by composing prototypes? vs query-only and top-k |
| `e3_lifelong` | does repeated consolidation degrade the manifold? vs a destructive `ifit` control |
| `e4_closure` | are consolidated concepts reused by later compositions? (hypothesis under revision — see `PLAN.md` §3.2) |
| `n1_train` | train the neural backbone |
| `n2_composition` | E2 on the neural backbone |
| `n3_lifelong` | **the headline**: hierarchical consolidation vs `flat` and `unfrozen` ablations |
| `n4_batch` | batch vs sequential addition: speedup, and CL metrics (ACC/BWT/FGT) |
| `n5_classinc` | Split-ColorMNIST class-incremental on **real** data vs finetune / replay / EWC / TTT / isolation |
| `n8_idealwords` | the ideal-words factorization: is a composition expressible as a sum of factor directions? |
| `n9_showdown` | **the case-study figure**: every method on the same OOD composition, annotated |

## Results, in one table

Split-ColorMNIST class-incremental, 12 tasks of **real** held-out images, 3 seeds (`n5_classinc`),
all on one backbone (200 epochs / 200 images per slot, conditional joint accuracy 0.891).
ACC is final mean joint accuracy; forgetting is −BWT, where 0 means none.

| method | ACC | forgetting | wall (s) | keeps data? |
|---|---|---|---|---|
| **ours** (hierarchical, frozen ancestors) | **0.849 ± 0.013** | **0.000** | 322 | no |
| test-time training | 0.835 ± 0.021 | 0.002 ± 0.010 | 820 | yes — all of it, forever |
| parameter isolation | 0.833 ± 0.017 | **0.000** | 214 | no |
| EWC | 0.757 ± 0.042 | 0.068 ± 0.026 | 291 | no |
| finetune | 0.741 ± 0.035 | 0.089 ± 0.022 | 217 | no |
| generative replay | 0.656 ± 0.086 | 0.180 ± 0.084 | 384 | no (samples itself) |

Ours is top on accuracy at exactly zero forgetting, but read the top three honestly: they sit within
0.016 ACC of each other against a seed spread of ±0.013–0.021, so ours, TTT and parameter isolation
are **statistically tied on accuracy**. What separates them is cost and what they retain — TTT pays
2.5× the wall clock and must keep every task's data forever, and unlike the other two it has no fixed
artifact to hand to a later task. The three methods that *protect* shared parameters rather than
removing them (EWC, finetune, replay) all forget measurably, and that gap is the real result.

Two further findings that qualify the headline, both in `RESULTS.md`:

- **§5m** — a composed-pool stream delivers *no gradient* to the shared adapter path (Δ‖B@A‖ = 0.000
  against 7.219 for real images), so composed-pool experiments cannot test forgetting at all. N3's
  and N9's no-forgetting numbers show that nothing was at stake, not that the method protected
  anything.
- **§5n** — for a *composed* concept the semantics live in a private per-concept embedding, not in
  the shared path. The shared path can triple (‖B@A‖ 7.49 → 22.28) with accuracy going only
  0.46 → 0.38, while zeroing that one embedding drops it to 0.04. So compositions are preserved by
  **isolation of the conditioning**, and absorbed real instances are preserved by **freezing the
  path** — two mechanisms, covering different halves of the pipeline.

## What this is not

- It does not deprecate [cobweb-poincare]: that repo studies the *representation* (hyperbolic
  geometry, encode/decode/interpolate). This one studies the *generative loop* over the same kind of
  tree, and reuses its benchmark unchanged.
- It is not a reimplementation of Cobweb — the backend is [cobweb-private]'s `CobwebContinuousTree`
  (`covar_from=1`, `var = sum_sq/n + prior_var`, matching cobweb-poincare exactly). The numpy tree in
  `tree.py` is a portability fallback.
- The metrics are probe-based, not FID/CLIP, and the palette is reconstructed from the paper's colour
  names rather than its RGB values. Absolute numbers are not comparable to the paper's tables; only
  within-table comparisons are. `PLAN.md` §3.6 lists these plainly.

[cobweb-poincare]: https://github.com/karthiksing05/cobweb-poincare
[cobweb-private]: ../Chunkformer/cobweb-private
