# Compositionality as the substrate for lifelong, incremental, out-of-distribution learning

A generative model meets one out-of-distribution query at a time, forever. It must produce the new
concept, keep it, and not damage anything it already had. This repository argues that **composition
is the mechanism that makes that possible**, and measures the argument on two benchmarks — one where
the factors occupy disjoint pixels and one where they overlap.

![the loop](figures/f1_loop.png)

The loop is the method. A query arrives; the system decides *for itself* whether it is a new
**arrangement** of parts it already has or a genuinely new **part**; arrangements cost nothing and
parts cost a small kept delta; the result is registered and every earlier concept is certified
bit-identical. Nothing in the loop retrains the backbone.

**Relation to prior work.** See [RELATED.md](RELATED.md) for the full positioning, which baselines
are fair for this setting and which are not, and the citation hazards found while checking. Wang et
al., *Test-Time Compositional Generalization in Diffusion Models via Concept Discovery*, supply the
first three stages (discover → select → compose → sample) and stop
where this repository begins: they distil one flat `(LoRA, c_new)` pair per query **and discard it**,
naming the gap in their own conclusion — *"a full mechanism for detecting, consolidating, and reusing
genuinely new primitives remains an important direction for future work."* Everything below the
composition stage is that mechanism.

---

## The setting, and how it differs from the closest prior work

The nearest neighbour to this project is Wang et al., *Test-Time Compositional Generalization in
Diffusion Models via Concept Discovery* (arXiv:2605.07078). We use their composition machinery
directly, so the difference is not in the operator — it is in **what happens to the result**.

### Their setting

A frozen diffusion model meets **one** out-of-distribution query. The pipeline:

```
mode-ascent discovery on s_θ(x_t,t)  →  submodular selection of K prototypes
       →  per-dimension Product-of-Experts teacher  →  CFG toward that teacher (Eq. 13),
          with the variance-aware weight schedule of Eq. 14
       →  (optionally) distil the resulting pool into one LoRA + one embedding
```

The question asked is *can a frozen model be steered to render a combination it never saw?* The
answer is evaluated on the samples produced for that query — FID, CLIP-I, and precision/recall/F1 in
Inception feature space. **The per-query LoRA is then discarded.** Nothing carries to the next query,
and there is no second query in the protocol.

### Our setting

The same frozen model meets an **unbounded stream** of out-of-distribution queries, one at a time,
forever. Each must be produced, **kept**, and must not damage anything kept earlier:

```
for each arriving query, forever:
    compose  (their four stages, unchanged)
    ROUTE    is this a new ARRANGEMENT of parts we have, or a genuinely new PART?
    CONSOLIDATE  arrangement → a shared basis direction (1024 bytes)
                 new part    → a new leaf in an insert-only adapter bank
    CERTIFY  every previously consolidated concept is checked bit-identical
```

The question asked is *can a model accumulate out-of-distribution concepts indefinitely without
forgetting, and does a concept it consolidated behave like one it was trained on?*

### The differences that matter

| | Wang et al. | this work |
|---|---|---|
| **horizon** | one query | an unbounded stream |
| **what persists after a query** | nothing — the LoRA is discarded | a routed, consolidated concept |
| **the decision made per query** | none; every query takes the same path | **route**: new arrangement (free) vs new part (a kept delta) |
| **cost of the *n*-th concept** | same as the first (nothing accumulates) | ~0 for an arrangement, one small delta for a new part |
| **interference between concepts** | not applicable | **certified**: earlier concepts bit-identical, measured |
| **what is evaluated** | sample quality for that one composition | that, **plus** retention across the stream, growth, and parity with a natively-trained backbone |
| **baselines it must beat** | composition methods | composition methods **and** the continual-learning literature |

Concretely, three things in this repository have no counterpart in theirs, and they are the
contribution:

1. **The routing decision.** Their pipeline spends a LoRA on every query. Ours first asks whether the
   query is expressible in the vocabulary it already has — decided from the model's own denoising
   error, no labels (AUROC 0.980 novel-vs-in-distribution, 0.995 vocabulary-vs-combination). An
   arrangement then costs a shared basis direction rather than an adapter, which is what makes cost
   scale with *vocabulary* rather than with *experience*.
2. **Consolidation that provably cannot interfere.** Ancestors are frozen and insertion is
   append-only, so a new concept has no write path to an existing one. This is verified rather than
   argued: every earlier concept is bit-identical after every insertion (fraction 1.00, against
   0.25–0.50 for every method that instead tries to *protect* shared weights).
3. **Native parity.** We ask whether a consolidated concept matches a backbone *trained* on that slot
   — a question that only arises once concepts are kept.

### What we take from them, unchanged

The discovery → selection → PoE → guided-sampling stages are theirs, and we run their method in
**their own configuration** (Eq. 14), not ours — a distinction that mattered: under this repository's
annealed schedule the same teacher produced speckle and looked broken, and its precision went from
0.039 to 0.839 once corrected. Their limitation section names the gap this project fills:

> *"A full mechanism for detecting, consolidating, and reusing genuinely new primitives remains an
> important direction for future work."*

### Two honest notes on the comparison

- **Their headline does not hold uniformly on CelebA.** In their own tables, query-only beats PoE on
  FID-Faithful (107.5 vs 120.9) and on F1-Faithful (64.0 vs 20.3). The claim is unambiguous on
  ColorMNIST; on CelebA it is not, and we do not lean on it.
- **The fair comparison target is their PoE row, not their LoRA row.** Their LoRA variant is not
  test-time-free, so a claim of "no retraining" should be measured against PoE.

---

## The argument, in six figures

### 1. Attribute accuracy cannot tell composing from copying. Coverage can.

![coverage vs copying](figures/f2_coverage_vs_copying.png)

This is the figure the project turns on. A method that re-emits the query scores **high attribute
accuracy** — the query is, after all, an image of the target — while covering none of the class. The
published method and the query-only baseline sit in the red band on **both** benchmarks.

The point is sharper than "the baseline is bad", and the baselines are run at their best to make it:
on ColorMNIST `poe-guided` beats us on accuracy (0.817 vs 0.314) **and** on precision (0.839 vs
0.400), so by every per-sample measure its output is better than ours. Its recall is **0.001**. Any
evaluation that reports accuracy without coverage would rank it first.

### 2. Composition recovers combinations that were never observed

![composition](figures/f3_composition.png)
![ColorMNIST case study](figures/f9_case_colormnist.png)

ColorMNIST F1 **0.498 ± 0.030** against the published method's **0.003 ± 0.003**. The published
method is run in **its own configuration** (their Eq. 14 variance-aware schedule), and that matters:
under this repository's annealed schedule the same teacher emitted speckle and looked broken. Run
their way it is *excellent* — joint accuracy **0.817** and precision **0.839**, higher than ours on
both — and its recall is **0.001**. In the case study the six samples are sharp, correct, and
identical.

On CelebA the split is stronger — `Smiling` and `Big_Nose` **never co-occur** in training, though each
appears alone tens of thousands of times, and they share the lower face. Producing both at once
scores **0.903 ± 0.094** where retrieval from the nearest trained combination reaches **0.355 ±
0.186**, and retrieval falling short is structural: every trained neighbour differs on at least one
of the two. The published method reaches precision **0.944** here — its samples are plausible faces —
at recall **0.000**.

**One baseline is genuinely competitive here, and the honest statement is narrow.** Composable
Diffusion (Liu et al. 2022) — zero parameters, no training — reaches `pair` **0.73** against our
**0.71** when run at the guidance scale that maximises its own metric. It buys that by collapsing
coverage: across the scale sweep its precision climbs 0.38 → 0.92 while recall falls 0.61 → 0.30 and
content diversity falls to **0.77**, passing *through* real data's 1.01. We reach equivalent accuracy
at roughly twice the coverage, and that is the whole of the CelebA composition claim. On ColorMNIST
the same operator scores **0.000 at every scale across a 23× range**, because that backbone has no
per-factor conditional for score addition to combine. **The composition result is therefore
benchmark-dependent** — it identifies *when* the cheaper operator suffices — and RESULTS §9g gives the
full sweep. What is not affected is anything below: composable diffusion produces a sample and keeps
nothing.

![CelebA case study](figures/f10_case_celeba.png)

### 3. The routing rule: arrangements are free, parts are not

![vocabulary](figures/f4_vocabulary.png)

A new **combination** of known factor values costs **one 256-float direction (1024 bytes)**, shared by
every combination that uses it, against 551K parameters for a per-concept adapter — and it is a basis
rather than a lookup, because combinations it was never taught on score the same as the ones it was.

A new **vocabulary item** is different, and the two benchmarks agree. On a backbone that never saw the
value, composing it scores **exactly 0.000** (ColorMNIST) and a learned conditioning vector is **not
separated from an untrained one** (CelebA). Only a weight update from **real** data recovers it.
Feeding a model its own samples cannot help: a composed pool delivers bit-exactly zero gradient.

### 4. Preservation is structural, not engineered

![preservation](figures/f5_preservation.png)

Arms that structurally *remove* shared parameters sit at exactly **0.000** forgetting on both
benchmarks; every arm that tries to *protect* them forgets (EWC 0.070, finetune 0.091, replay 0.183).
`tests/test_unbounded.py` holds the stronger form: after 12 batches out to 40 concepts, every earlier
concept's epsilon output on fixed noise is **bit-identical**, not merely close.

The reason is one sentence, and it took four failed attempts to find: **conditionality, not size, is
what buys preservation.** Every route that removes the ability to switch a delta off collapses
(merged low-rank 0.379, merged orthogonal 0.069, hypernetwork 0.073, classifier guidance 0.000).

### 5. Is a consolidated concept as good as one the model was trained on?

![native parity](figures/f6_native_parity.png)

The comparison the thesis is named after, and it needs an oracle: a backbone **trained on the
held-out slots**, which validates itself by sitting at the data ceiling. A concept consolidated from
a single query at zero retraining reaches most of native-level **coverage** and does not reach native
per-sample **precision**. The published per-query LoRA reaches neither.

### 6. The system routes itself, and cost grows with vocabulary rather than time

![router](figures/f7_router.png)
![growth](figures/f8_growth.png)

AUROC **0.980** novel-vs-in-distribution and **0.995** vocabulary-vs-combination, from the model's own
denoising error and no labels. The error is one-sided in the safe direction: a new vocabulary item
was routed to the free path **0 times in 120 queries**, while some combinations over-spend on an
adapter they did not need.

Charging that measured error back in, a 160-experience stream costs ~10× less than one adapter per
query, and the curve's slope is set by a *detector error rate* rather than by the arrival of
experiences. That is what makes "forever" the operative word.

**All numbers with error bars: [TABLES.md](TABLES.md). Full record, including every refuted
hypothesis: [RESULTS.md](RESULTS.md).**

---

## What is *not* claimed

Kept here rather than in a footnote, because several shaped the design.

- **The hierarchy does not earn its keep.** Given its best case, `clustered` placement is the *worst*
  arm (0.831) against `shuffled` (0.859) and `flat` (0.842) at identical parameter counts. The method
  is flat and insert-only as a result.
- **Consolidated concepts do not make later learning cheaper** on ColorMNIST — not in accuracy
  (selection frequency 0.00) and not in cost (`hier` vs `unfrozen` differ by 30× less than the seed
  spread). Likely a property of separable factors; stated as open.
- **The router is a density detector, not a basis test.** Its unconditional ablation matches the
  conditional signal, so it is *not* evidence that the factor basis recognises what it can express.
- **Composed samples cannot be sampled into the class.** The conditioning span can express one factor
  or the other but not both for a held-out pairing; an oracle fitted on 64 real held-out images does
  no better than inverting from one query. Thirteen interventions on the optimizer, objective, prior
  and sampler could not move it.
- **On CelebA the factor basis is supplied, not discovered.** A per-class embedding is untrainable at
  this data scale (measured: 34% of ceiling, unmoved by 2.3× data), so conditioning is factored by
  construction. CelebA tests whether composing known values *renders* when they overlap — not whether
  the basis can be found.
- **The composition operator is not better than score addition on CelebA.** Run at its own best
  scale, Composable Diffusion matches us on `pair` accuracy (0.73 vs 0.71) with zero parameters and
  no training; we win on coverage, not on accuracy. The operator is inherited from prior work and
  improved only on the benchmark where no per-factor conditional exists (RESULTS §9g).
- **Every score-addition method here, ours included, samples the wrong distribution.** Du et al.
  (ICML 2023) show the sum of diffused scores is the score of the *product of diffused marginals*,
  not the *diffusion of the product*, so the sampler — not the model — is the error source. Their
  MCMC correction is reported as `composable-mcmc`; it does not rescue ColorMNIST, where the missing
  ingredient is a per-factor conditional rather than a better sampler.

## Reproducing

```bash
bash reproduce.sh tests       # 7 assertions, no backbone needed
bash reproduce.sh backbones   # the 5 checkpoints everything else loads (~12 GPU-hours)
bash reproduce.sh colormnist  # ColorMNIST experiments, 3 seeds
bash reproduce.sh celeba      # CelebA experiments, 3 seeds
bash reproduce.sh analytic    # the analytic track (no gradient descent anywhere)
bash reproduce.sh supporting  # negative results and cost claims (n14/n16/n17/n24)
bash reproduce.sh figures     # aggregate + every figure + TABLES.md   (CPU, seconds)
```

Every figure and every table is generated from `results/aggregate.json` by
`experiments/make_paper_figures.py` and `experiments/make_paper_tables.py`, so a number in a figure
and the same number in a table cannot drift apart. Nothing in `figures/` is hand-drawn.

**Environment.** `chunks` conda env (torch 2.7.1+cu126) for the neural track; `cobweb` (numpy only,
no torch) for the analytic track. Data: MNIST idx files, and CelebA as HuggingFace parquet shards in
`data/celeba/`.

**Two things that will otherwise cost you hours.** The login node has no GPU and no torch, so
everything goes through `sbatch`. And stale `__pycache__` on NFS silently runs old code —
`reproduce.sh` clears it before every submission wave.

## Layout

```
lifelong_ood/          the method
  score.py             BaseScoreField: a diffusion manifold is one primitive, eps(x, u)
  compose.py           discovery, submodular selection, per-dimension product of experts
  neural/adapters.py   the insert-only adapter bank; preservation lives here
  neural/idealwords.py the factor basis, basis extension, new-vocabulary learning
  neural/router.py     the self-routing detector
experiments/           one file per experiment; e* analytic, n* neural.  `reproduce.sh`
                       covers every number in README and TABLES; the remaining scripts
                       reproduce the fuller record in RESULTS.md, including the
                       refuted hypotheses
tests/                 7 assertions, including the two that caught real bugs
results/               per-run JSON + aggregate.json (the source for every figure and table)
figures/               f1-f10, all generated
```

## A note on measurement

Three bugs in this round were caught by a **control row** rather than by a result, and all three had
a plausible innocent explanation that would have been believed:

- a training path that used a different pixel layout than the sampler — visible because
  controllability was *exactly* 0.000 on eight of eight attributes, which is too clean for
  undertraining;
- a probe-readability filter that admitted five probes unable to recognise their own attribute
  (Gray_Hair: 0.962 accuracy, 0.321 true-positive rate on a 4.3%-prevalence attribute);
- smoke runs writing to the same files as full runs, so a 2-case result silently replaced a 12-case
  one.

Each showed up as *every arm moving together*, which reads as "the method does not work" rather than
"this measurement is invalid" — the most expensive confusion available. The habit that caught them is
printing the real-data ceiling beside every method, and it is why every table here carries one.
`tests/test_layout.py`, `probe_usable()` and the `--tag _smoke` convention are the permanent guards.
