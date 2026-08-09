# Compositionality as the substrate for lifelong, out-of-distribution learning

A generative model meets out-of-distribution queries **forever**. Each one must be produced, **kept**,
and must not damage anything kept before — at a cost that grows with the model's *vocabulary* rather
than with its *history*.

This repository argues that **composition is the mechanism that makes that possible**, and measures
the argument on two benchmarks (one where the factors occupy disjoint pixels, one where they overlap)
and on a pretrained 113.7M-parameter diffusion model we did not train.

![the loop](figures/f1_loop.png)

---

## How this branches off Wang et al.

**[Test-Time Compositional Generalization in Diffusion Models via Concept Discovery](https://arxiv.org/abs/2605.07078)
(Wang, Gupta, Zhu, MacLellan) is a companion paper from our own group.** This work is not a
competitor to it; it is a continuation of it, and the composition machinery is used unchanged.

**What they built.** Given one out-of-distribution query and a frozen diffusion model:

```
mode-ascent discovery on the model's own score field
   → submodular selection of K prototypes
   → per-dimension product-of-experts teacher
   → classifier-free guidance toward it (their Eq. 13, with the Eq. 14 variance-aware schedule)
   → optionally distil the result into one LoRA + one embedding
```

It answers *can a frozen model be steered to render a combination it never saw?* — and the per-query
LoRA is then **discarded**. There is no second query in the protocol.

**Where we pick it up.** Their limitation section names the gap exactly:

> *"A full mechanism for detecting, consolidating, and reusing genuinely new primitives remains an
> important direction for future work."*

We keep their four stages and add a fifth and sixth, which is what turns a test-time trick into a
lifelong learner:

```
for each arriving query, forever:
    COMPOSE      their four stages, unchanged
    ROUTE        is this a new ARRANGEMENT of parts we have, or a genuinely new PART?
    CONSOLIDATE  arrangement → a shared basis direction (1024 bytes)
                 new part    → a new leaf in an insert-only adapter bank
    CERTIFY      every previously consolidated concept is checked bit-identical
```

| | Wang et al. | this work |
|---|---|---|
| horizon | one query | an unbounded stream |
| what persists afterwards | nothing — the LoRA is discarded | a routed, consolidated concept |
| decision made per query | none; every query takes the same path | **route**: arrangement (free) vs new part (a kept delta) |
| cost of the *n*-th concept | same as the first (nothing accumulates) | ~0 for an arrangement, one small delta for a new part |
| interference between concepts | not applicable | **certified** bit-identical, measured |
| evaluated on | sample quality for that one composition | that, **plus** retention, growth, and parity with a natively-trained backbone |

**The composition operator is inherited, not improved.** That is deliberate. Where a better operator
exists we adopt it: on CelebA, MCMC-corrected score addition (Du et al., ICML 2023) is more accurate
than ours and we say so. What this project contributes is everything below the composition stage.

---

## Contributions, in one line each

Full detail, with evidence and caveats for each: **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)**.

1. **Certified-preserving consolidation.** Concepts accumulate forever with every earlier concept
   provably bit-identical — **1.00 bit-identical** across 3 seeds, two datasets and **eleven** CL
   baselines, where every method that *protects* shared weights instead of removing them sits at
   0.25 and forgets.
2. **A routing rule that decides what an update costs.** New *arrangement* vs new *part*, from the
   model's own denoising error with no labels: **AUROC 0.980 / 0.995**, error one-sided in the safe
   direction. An arrangement costs a 1024-byte basis direction instead of a 551K-parameter adapter,
   which is what makes cost scale with vocabulary rather than experience.
3. **Batch addition that is strictly better than one-at-a-time.** K concepts in one update:
   **2.14× faster *and* +0.052 accuracy** at K=12, bit-identical at every K from 1 to 12.
4. **Native parity.** A consolidated concept is measured against a backbone *trained* on that slot —
   a question that only exists once concepts are kept.
5. **The mechanism transfers to a model we did not train.** On `google/ddpm-celebahq-256`, injecting
   the bank changes the model by exactly **0.000e+00** and concepts are acquired at 256px without
   quality loss.
6. **Two rules for adapting diffusion models**, both measured rather than argued: compose a bank in
   **score space** (path union collapses diversity and, on factor concepts, produces nothing), and
   train concept adapters at **high noise levels only** — which is the difference between working and
   not on a pretrained backbone.

---

## The evidence, in six figures

### 1. Attribute accuracy cannot tell composing from copying. Coverage can.

![coverage vs copying](figures/f2_coverage_vs_copying.png)

The figure the project turns on. A method that re-emits the query scores **high attribute accuracy**
— the query is, after all, an image of the target — while covering none of the class. On ColorMNIST
`poe-guided` beats us on accuracy (0.839 vs 0.539) **and** precision, and its recall is **0.001**. Any
evaluation reporting accuracy without coverage ranks it first.

### 2. Composition recovers combinations never observed

![composition](figures/f3_composition.png)
![ColorMNIST case study](figures/f9_case_colormnist.png)

ColorMNIST **F1 0.605 ± 0.043** against the companion method's **0.003 ± 0.003**, with every baseline
run in *its own* best configuration. On CelebA, where `Smiling` and `Big_Nose` never co-occur in
training, we reach `pair` **0.903 ± 0.094** — and MCMC-corrected score addition reaches **0.965**,
which we report rather than bury (see [docs/RESULTS.md §11c](docs/RESULTS.md)).

### 3. Preservation is structural, not learned

![preservation](figures/f5_preservation.png)

Eleven CL arms, two datasets, 3 seeds. Only arms that *remove* shared parameters reach bit-identity;
`ttt` also does but keeps all 768 stream images to manage it.

### 4. Cost scales with vocabulary, not experience

![growth](figures/f8_growth.png)

### 5. A consolidated concept behaves like a trained one

![native parity](figures/f6_native_parity.png)

### 6. It works on a model we did not train

![pretrained case study](figures/n25_pretrained_case_final.png)

`google/ddpm-celebahq-256`, 256×256. Two attributes learned as separate concepts, each blind to the
other, then composed. The last two rows are the **same bank read two ways** — score composition and
path union — which is the measurement behind contribution 6.

---

## Reproducing

```bash
bash reproduce.sh tests       # assertions, no backbone needed
bash reproduce.sh backbones   # every checkpoint the rest loads (~12 GPU-hours)
bash reproduce.sh colormnist  # ColorMNIST experiments, 3 seeds
bash reproduce.sh celeba      # CelebA experiments, 3 seeds
bash reproduce.sh pretrained  # the pretrained-backbone experiments
bash reproduce.sh analytic    # the analytic track (no gradient descent anywhere)
bash reproduce.sh supporting  # negative results and cost claims
bash reproduce.sh figures     # aggregate + every figure + tables   (CPU, seconds)
bash reproduce.sh all
```

## Documentation

| file | what it is |
|---|---|
| **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)** | each contribution in detail: claim, evidence, caveats |
| **[docs/SCOREBOARD.md](docs/SCOREBOARD.md)** | every axis vs every baseline, with the winner named |
| **[docs/](docs/)** | every result, in full |
| [docs/RESULTS.md](docs/RESULTS.md) | the complete record, including every refuted hypothesis |
| [docs/TABLES.md](docs/TABLES.md) | every number with error bars, regenerated from `results/` |
| [docs/RELATED.md](docs/RELATED.md) | positioning, which baselines are fair, citation hazards |
| [docs/ASSESSMENT.md](docs/ASSESSMENT.md) | publication readiness, assessed against the evidence |

## Layout

```
lifelong_ood/      the library: score field, adapters, router, composition, evals
experiments/       n1–n28, one file per question
tests/             assertions and the diagnostics that CHOOSE configurations
results/           per-seed JSON + the aggregate the figures are built from
figures/           every figure in this README
docs/              the full record
```

## A note on measurement

Several results in this repository are corrections of earlier results in this repository, and the
corrections are kept rather than quietly replaced. Three measurement bugs were found by *control
rows* rather than by the numbers they were supposed to support; one baseline was found to be running
in a configuration its authors would not recognise; one comparison was found to be varying two things
at once. [docs/RESULTS.md](docs/RESULTS.md) records each, because a repository that only shows what
survived is not evidence that anything was checked.
