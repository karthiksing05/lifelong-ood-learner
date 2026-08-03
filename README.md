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

**Consolidation does not degrade the manifold (E3, full scale).** Per consolidation, drift in
held-out log-likelihood is **−0.01 to −0.05 nats**, drift in denoise R² is **+0.0000**, and
unconditional sample validity stays 0.92–1.00.

**The mechanism holds in the strong form (N3).** `retention_exact = 1.00` at every step for the
hierarchical arm — every previously consolidated concept bit-identical — against **0.00** for the
`unfrozen` ablation as soon as a second concept lands, with frozen-base drift `+0.00000` throughout.

Two tuning results were needed to make composition work at all, and both are deviations from the
paper documented at their implementation sites:

- **Adaptive PoE temperature.** The paper's fixed τ=0.5 gives joint teacher accuracy **0.200**;
  setting the temperature from the data's own scale gives **0.625** against a **0.700** ceiling.
  Cobweb's variances sit near `prior_var` everywhere, compressing per-dimension log-likelihood gaps
  so that two experts with the *wrong* background outvote the one correct one.
- **Annealed guidance.** A Gaussian mixture assigns essentially **zero density to an unseen
  composition**, so during sampling its unconditional score drags the result back toward seen slots.
  The paper's variance-aware schedule retains **21%** of the teacher's accuracy; annealing guidance
  up as noise falls retains **71%**. This measured hole is precisely what a convolutional UNet does
  not have, and is why the neural backbone exists.

`PLAN.md` carries the living status, the open work, and the threats to validity.

## Running it

```bash
python tests/test_smoke.py      # analytic, synthetic, ~10 s, no data needed
python tests/test_neural.py     # neural, CPU, <1 min

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
