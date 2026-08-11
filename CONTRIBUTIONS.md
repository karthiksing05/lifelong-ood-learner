# Contributions, in detail

Each entry states the **claim**, the **evidence**, what the claim **does not** cover, and where the
supporting numbers live. Ordered by how much weight the evidence can carry, not by how interesting
the idea is.

The composition stage is inherited from our group's companion paper
([Wang et al., 2605.07078](https://arxiv.org/abs/2605.07078)) and is **not** claimed here. Everything
below is what happens after a composition is produced.

---

## 1. Certified-preserving consolidation under an unbounded stream

**Claim.** Concepts can be added to a diffusion model indefinitely with every previously added
concept remaining **bit-identical**, at **zero stored data**, and this is a structural property of
the insert-only design rather than a learned or tuned one.

**Evidence.** Three seeds, two datasets, **eleven** continual-learning arms — `finetune`, `replay`,
`gen-distill`, `der++`, `ewc`, `si`, `mas`, `c-lora`, `ttt`, `independent`, ours:

| | ours | independent | ttt | every arm that *protects* shared weights |
|---|---|---|---|---|
| fraction bit-identical | **1.000** | **1.000** | **1.000** | 0.250 |
| forgetting (FGT) | **0.000** | **0.000** | 0.021 | 0.004 – 0.159 |
| images kept forever | **0** | **0** | 768 | 0 (200 for `der++`) |

The separation is categorical rather than incremental: arms that *remove* shared parameters from the
optimizer reach bit-identity, arms that try to *protect* them do not, and no amount of regularization
strength changes which side of that line a method is on. `ttt` reaches bit-identity only by keeping
every image of the stream.

The four strongest modern baselines here — SI, MAS, DER++ and generative distillation — were added
*after* a systematic literature review said their absence was the most obvious objection, and
generative distillation specifically because naive generative replay is a documented pathology for
diffusion (Masip et al., CoLLAs 2025) that would otherwise have been a strawman.

**What this does not claim.** Exact preservation is **not novel as a property** — PNN, PackNet,
Piggyback and SupSup all have it, and [docs/RELATED.md](docs/RELATED.md) says so with a table
separating the methods that genuinely achieve it from those routinely miscited as achieving it (HAT
and DEN do not). Our closest structural ancestor is PNN; the difference is that adapters read only
the frozen trunk and their own ancestors rather than every predecessor, which turns PNN's O(T²)
lateral wiring into roughly linear growth.

It also does **not** claim superiority over parameter isolation *on this table*: `independent`
matches us on every column at **fewer parameters** (ACC 0.863±0.015 / 7.16M against 0.839±0.025 /
8.82M). The differentiator against it is contribution 2, measured elsewhere.

That loss survives a fair capacity sweep. `independent` is this same bank with flat placement, so it
has the identical rank knob; sweeping **both** arms over r ∈ {1,2,4,8,16} at 3 seeds, we lead at
matched rank only at r = 4 and by less than a standard error (§24d). Bit-identity is 1.000 at every
rank for both, which is the point: preservation does not depend on capacity.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §7d, §10; [docs/TABLES.md](docs/TABLES.md) §3.
`experiments/n5_classinc.py`, `experiments/n22_celeba_lifelong.py`.

---

## 2. A routing rule that decides what an update costs

**Claim.** The system can decide *for itself*, with no labels, whether an arriving query is a new
**arrangement** of parts it already has or a genuinely new **part** — and the two cost different
amounts, which is what makes cost scale with vocabulary rather than with experience.

**Evidence.** From the model's own denoising error:

- **AUROC 0.980** novel-vs-in-distribution, **0.995** vocabulary-vs-combination;
- the error is **one-sided in the safe direction**: a new vocabulary item was routed to the free path
  **0 times in 120 queries**, while some arrangements over-spend on an adapter they did not need;
- an arrangement costs a shared basis direction (**1024 bytes**) where a new part costs an adapter
  (**551K parameters**);
- charging the measured error rate back in, a 160-experience stream costs **~10× less** than one
  adapter per query, and the curve's slope is set by a *detector error rate* rather than by the
  arrival of experiences.

**The price list is three tiers, not two, and the cheapest is the default.** §25 measured a route
below basis extension: if the basis *already spans* the target, the concept is stored as factor
**coordinates** and nothing is learned at all.

| tier | when | cost | measured |
|---|---|---|---|
| **ADDRESS** | the basis already spans it | **72 B** | 31,895× cheaper than a block and not worse — F1 0.678 ± 0.042 vs 0.587 ± 0.020 over 3 seeds, winning every seed |
| **EXTEND BASIS** | a new *value* of a known factor | **1 KB** | F1 0.755 ± 0.037 at zero per-concept parameters |
| **BLOCK** | a new *factor*, or nothing factored | **69 K** (rank 1) | the tier is real: conditioning-only routes score 0.000 |

**The honest gap in this claim.** The router has three outcomes and its third **lumps EXTEND BASIS
and BLOCK together** — nothing here yet distinguishes a new *value* (1 KB) from a new *factor*
(69 K), so a 69× cost difference is currently decided by hand. What is measured is that the
distinction is *real* (§5q), not how to detect it.

This is the piece with no counterpart in the companion paper, whose pipeline spends a LoRA on every
query. A dedicated literature sweep found **no** paper combining product-of-experts composition with
a growing adapter bank — an empty cell.

**What this does not claim.** The router is a **density detector, not a basis test**: its
unconditional ablation matches the conditional signal, so it is not evidence that the factor basis
recognises what it can express. That was measured and is reported as a limitation.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §6c, §5z. `experiments/n18_router.py`,
`experiments/n24_growth.py`, `lifelong_ood/neural/router.py`.

---

## 3. Batch addition is strictly better than one-at-a-time

**Claim.** A stream that arrives in groups can be absorbed in groups, and doing so is **free** —
2.15× faster at no cost to accuracy and none to preservation. Not a speed-for-quality trade.

**Evidence.** Twelve concepts, matched discovery configuration, **3 seeds**, full K sweep. ΔACC is
**paired** (batch − sequential on the same seed and the same queries), which is the right test here
since the two arms share everything but the grouping:

| K | speedup | ΔACC (paired) | bit-identical |
|---|---|---|---|
| 1 | 0.99× | −0.016 ± 0.021 | 1.000 |
| 2 | 1.29× | −0.005 ± 0.036 | 1.000 |
| 4 | 1.65× | +0.009 ± 0.040 | 1.000 |
| 8 | 1.84× | +0.007 ± 0.011 | 1.000 |
| **12** | **2.15×** | **+0.035 ± 0.037** | 1.000 |

**The speedup is the claim, and it replicates: 2.15× at K = 12** against the 2.14× measured on one
seed, rising monotonically with K exactly as amortizing one reverse-diffusion run over K pools
predicts. **Bit-identity is 1.000 at every K and every seed**, which is the safety half and was never
at risk.

**The accuracy gain does not survive three seeds as a claim.** At K = 12 it is +0.035 ± 0.037 —
positive, the largest in the sweep, and *not distinguishable from zero*. The earlier headline of
+0.052 came from a single seed. The honest statement is that batching is **free** (2.15× faster at no
cost to accuracy or preservation), not that it is *better*; the shared-ancestor effect is a mechanism
we can see in the parameters (§24a) but cannot demonstrate in accuracy at this sample size.

Verified independently on the pretrained backbone: all three batch arms reach bit-identical 1.00 on
`google/ddpm-celebahq-256`.

**What this does not claim.** The **speedup** amortizes the generative pool sampling, not the
optimizer. Where concepts are trained on real images rather than distilled from sampled pools (the
pretrained path), batch and sequential both take 777s. "Batch addition is safe and structural" is
unconditional; "batch addition is fast" is conditional on there being pools to amortize.

An earlier version of this section reported batching as *costing* accuracy. That was a confound in
our own experiment — the two modes were silently using different discovery configurations — and is
recorded rather than deleted.

This section previously reported **+0.052 ACC** from a single seed at two K values. The 5 × 3 grid
above supersedes it: the speedup replicated, the accuracy gain shrank to +0.035 ± 0.037 and is no
longer claimed. Recorded rather than quietly revised.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §15, §16, §16b. `experiments/n4_batch.py`,
`experiments/n26_batch_pretrained.py`.

---

## 4. Native parity: is a consolidated concept as good as a trained one?

**Claim.** A concept the system consolidated from a single out-of-distribution query can be compared
against a backbone that was **trained** on that slot, and the comparison is favourable.

**Evidence.** Recall **0.903 ± 0.024**, F1 **0.570 ± 0.072** against the native-trained reference.

This question only becomes askable once concepts are *kept*, which is why the personalization
literature does not ask it: there is nothing to compare against a trained model when the artifact is
discarded after one query.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §6d. `experiments/n23_native_parity.py`.

---

## 5. On a model we did not train: preservation transfers, acquisition does not

**Claim, as it stood, was that BOTH halves work on `google/ddpm-celebahq-256` (113.7M parameters,
256×256, unconditional, trained by someone else). The acquisition half is RETRACTED (2026-08-11).**

The evidence for it was measured through a field that handed that model a pixel *permutation*
(RESULTS §30b): it returned 3×3 tiled greyscale mosaics, and the attribute probes were scoring
those. With the input layout corrected, every generated arm scores **0.000** on "both attributes
correct" — score-composed, path-union, each single-concept arm, and the untouched model alike,
against a real-image ceiling of 1.000. So the bank is not adding measurable attribute control on
this backbone; it is not that we lose to a baseline, it is that nothing renders the attributes.

**Preservation is unaffected and is what this contribution now asserts.** It is a property of
optimizer *scoping* — the parameters are absent from the optimizer — and holds regardless of what
the samples look like. That independence is exactly why it stayed green while the images were
nonsense, and it is the caution worth carrying: **a passing preservation test is not evidence that
anything renders.** See §30d, where three explanations for the failure have been tested and are
dead, and it is recorded as open.

**Evidence.**

*Preservation*, certified by `tests/test_pretrained_bank.py` on that model and on
`google/ddpm-cifar10-32`:

| assertion | result |
|---|---|
| injecting the bank changes the pretrained model | **0.000e+00** |
| a freshly registered concept is a no-op | **0.000e+00** |
| training a second concept moves the first | **0.000e+00** |

*Acquisition*: **retracted, see above.** The previously reported per-concept 1.00 and the composed
0.417 were both measured through the broken field.

**What this does not claim.** This does **not** restate the out-of-distribution claim: the model's
training set is not ours to control, so we cannot say it never saw a smiling face in glasses. What
*is* controlled is what our **concepts** saw — A trained only on `(a=1, b=0)`, B only on `(b=1, a=0)`
— so their co-occurrence is held out of both by construction, and composing them is a held-out
combination *at the level of the representations this method builds*. The full lifelong loop has not
been run at that scale across seeds.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §12. `experiments/n25_pretrained_case.py`,
`lifelong_ood/neural/pretrained.py`, `tests/test_pretrained_bank.py`.

---

## 6. Two rules for adapting a diffusion model, both measured

### 6a. Compose a bank in score space, not by activating several adapter paths

**Claim.** Reading a bank of independently trained adapters by activating several paths at once is
the wrong operator; combining their *predictions* is the right one.

**Evidence**, at 3 seeds:

| setting | score composition | path union |
|---|---|---|
| ~~pretrained backbone, 2 attributes — accuracy~~ | ~~0.417 ± 0.083~~ | ~~0.417 ± 0.083~~ **both 0.000 once corrected** |
| ~~pretrained backbone, 2 attributes — diversity~~ | ~~Vendi 1.436~~ | ~~1.280~~ **measured on mosaics** |
| ColorMNIST factor concepts | joint **0.042**, F1 **0.490** | 0.000, F1 0.039 |

**The two pretrained rows are retracted** (2026-08-11): they were measured through a field feeding
the model a pixel permutation, so both operators were being compared on tiled mosaics. Corrected,
every arm scores 0.000 there and the comparison has no signal to carry. **This contribution now rests
on the ColorMNIST row alone**, which is unaffected — the ColorMNIST backbone is self-consistent in
its layout, and re-deriving it under the current defaults left it unchanged.

**The accuracy claim did not replicate, and then its substrate turned out to be broken as well.** A
single seed showed 0.50 against 0.17 on the pretrained backbone; at three seeds the two operators
tied; and the layout correction removed the measurement entirely. The diversity argument below is
likewise from the retracted rows and does not survive (Vendi 1.280 against the untouched model's 1.456, where score composition holds
1.436), and that on ColorMNIST factor concepts path union produces **nothing at all** while score
composition produces something. The recommendation rests on those two, and the retracted number is
kept in [docs/RESULTS.md](docs/RESULTS.md) §12e.

The reason is structural: activating two paths sums two deltas each trained to be applied **alone**,
and nothing in either one's training says what it should do while the other is on. This is the
failure the entire LoRA-merging literature exists to address. Score composition never evaluates an
adapter outside its own conditioning.

Implemented as `NeuralLifelongLearner.sample_composed`. Preservation is untouched — paths stay
insert-only and every concept remains bit-identical.

### 6b. Train concept adapters at high noise levels only

**Claim.** A factor's identity is settled at high noise; capacity spent at low noise goes to texture
the model already renders correctly.

**Evidence.** On the pretrained backbone this is the difference between working and not: all-timestep
training gives 0.25 on unrecognizable samples, `t ≥ 500` gives **0.75 at Vendi 1.48** against the
untouched model's 1.51 — and without it, no learning rate produced an attribute without destroying
the model.

On *our own* backbone at 3 seeds the effect is **on the concepts, not on the composition**: high-t
learns the factors better (bg **1.000 ± 0.000** and digit **0.729 ± 0.083** against uniform's 0.958
and 0.604 ± 0.021) while the composed-joint differences sit inside the noise (0.188 ± 0.130 against
0.167 ± 0.042). A single seed showed a clean composition win; it did not replicate, and the claim is
narrowed accordingly.

### 6c. The negative result that produced these

The hypothesis that motivated the study was that an update **disturbing the existing score field
least** should compose best. It is false, though for a narrower reason than a single seed suggested.
`rehearse` genuinely cuts drift by **4×** (0.010 ± 0.000 against 0.038–0.043 — the only rule that
moves drift at all), and it is **not** rewarded for it: composition is no better than any other rule,
and the concept itself is measurably **worse** (fg 0.875 against 1.000, bg 0.792 against 0.958–1.000).
Suppressing what an adapter changes suppresses what it should change. Drift is not a cost to
minimize — it is the mechanism by which a conditional differs from the unconditional model at all.

The more principled version of 6b — sampling noise levels in proportion to the frozen model's own
error — *loses* to the blunt threshold (0.042 against 0.104), because measured "need" peaks in the
noisiest tail where every model does badly rather than in the semantically decisive band.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §12e, §14. `experiments/n28_update_rules.py`.

---

## 7. A characterization of when each composition operator applies

**Claim.** Whether score addition or embedding composition wins is **not** a property of the dataset.
It is determined by the granularity — and the marginals — of the conditionals being composed.

**Evidence.** Giving the ColorMNIST backbone per-factor conditionals it does not have, and re-running
the identical operator on the identical backbone, takes score addition from **0.000 → 0.167**. Its
120 native conditionals are per-*slot* and none of them means "cyan foreground", so adding them
cannot isolate a factor none of them represents.

| the model has | concepts are | compose them in |
|---|---|---|
| no conditioning channel | deltas | **score** space |
| entangled per-combination conditionals | vectors in a factorizable space | **embedding** space |
| factored, compositionally-trained conditionals | per-attribute conditionals | **score** space |

Granularity alone does not close the gap to embedding composition (0.500), because a conditional
fitted to "images with fg = f" is `p(x | fg = f)` and drags the other factors' marginals along:
`fg3` alone forces bg to 0.00 rather than staying silent. CelebA's factored head is trained with
conditioning built *as* a sum of per-attribute embeddings, so its conditionals are compositional by
construction — which is why score addition wins there and not here.

**What this does not claim.** This is an **analysis experiment, not a method.** Its factor concepts
are trained from 96 real images each, which is far stronger supervision than the single query the
out-of-distribution setting provides. It answers *why* score addition fails on one benchmark and
succeeds on another; it is not a proposal for how to compose.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §13. `experiments/n27_factor_conditionals.py`.

---

## 8. The prototype/exemplar trade is a dial we can turn and the companion method cannot

**Claim.** How far a sample sits from its composed prototype is a **runtime argument**, not a fixed
property of a composition method — but only if the concept has been consolidated into a conditional
of its own. One consolidated concept covers the whole range from near-copies of the prototype to
unconstrained draws, with no retraining.

**Mechanism.** Start the reverse process from a partially noised prototype rather than from pure
noise: `x_init = sqrt(abar(s)) * prototype + sqrt(1 - abar(s)) * noise`, integrating from `u = s`.
`score.sample()` already did the noising, so this is one argument (`sample_concept(..., u_start=s,
x_init=mu)`), not new machinery.

**Evidence.** 3 seeds × 4 compositions × 16 samples, sweeping `s` over seven settings:

| | distance to prototype | recall | Vendi |
|---|---|---|---|
| **ours**, s = 0.60 → 1.00 | 0.071 → 0.165 (**2.32×**) | 0.003 → 0.843 | 1.15 → 1.88 |
| **poe-guided**, s = 0.60 → 1.00 | 0.058 → 0.055 (**0.94×**) | 0.000 throughout | 1.07 → 1.07 |

Every quantity moves together for ours and none of them moves for theirs. At `s = 1.0` the companion
method begins from essentially pure noise and its samples still land 0.055 from the prototype:
guidance toward the analytic teacher re-collapses the trajectory onto the mode regardless of where it
started. That is the same collapse behind its joint 0.839 at F1 0.003, now shown to be **structural
rather than a tuning choice** — and shown under a starting condition that *favours* it, since its
`x_init` is a draw from the teacher's Gaussian and is already diverse before noising, while ours is
the bare prototype mean.

**Recommended setting:** `s = 0.90`, where F1 peaks at 0.669 ± 0.018. A coarse single-seed sweep had
read the elbow at 0.8; the 3-seed fine sweep put 0.8 at 0.410 ± 0.058, four standard errors below,
and that earlier read is recorded as superseded. This is a setting to *pass* when generating
exemplars — `sample_concept` still defaults to `u_start = 1.0`, which is what every comparison table
was run at, because tuning our arm's operating point while leaving the baselines at theirs would not
be a comparison.

**What this does not claim.** The dial does not manufacture coverage the concept lacks — at `s = 1.0`
it is just sampling the concept's conditional, which is what every other table already reports. The
new facts are that the near end is reachable at all, and that the companion method can neither reach
it nor leave it.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §20, `figures/f11_prototype_dial.png`.
`experiments/n30_prototype_distance.py`.

---

## 9. Certified deletion: a concept can be removed without disturbing the others

**Claim.** The same structure that makes addition non-destructive makes **removal** exact. Deleting a
concept leaves every remaining concept, and the frozen base, **bit-identical** — no gradient step, no
re-fitting, no distillation — in O(1).

**Evidence.** 10 concepts on ColorMNIST, deleting the middle one:

| arm | survivors bit-identical | max drift | params freed | seconds |
|---|---|---|---|---|
| **ours** | **9 / 9** | **0.000e+00** | 551,408 | **0.002** |
| retrain without it | 0 / 9 | 0.993 | 0 | 258.4 |

Forgetting is total rather than a fallback: the field resolves names through the bank, so a deleted
name becomes genuinely unknown and raises, instead of quietly reverting to the base.

**What this does not claim.** Deletion makes a concept unreachable; it does **not** generally reclaim
its storage. Only the leaf is private (≤ ⅓ of a path's parameters), and `_descend` can adopt an
earlier leaf as a later concept's ancestor — 3/10 concepts here — after which only the embedding can
go. Measured reclaimable fraction: min 0.000, median 0.250, max 0.333. The accurate description is
*certified forgetting with partial reclamation*.

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §22. `experiments/n31_deletion.py`.

---

## 10. Cost that tracks vocabulary rather than experience

**Claim.** The parameter growth of this system is governed by how many genuinely new *parts* the
world contains, not by how many experiences arrive — and the per-part cost is far smaller than the
default configuration spends.

**Evidence.** Three levers, all measured:

| lever | effect | where |
|---|---|---|
| **addressing** a spanned composition instead of a new block | **72 B vs 2,296,467 B — 31,895×** | §25 |
| **routing** an arrangement to the shared basis instead of a new block | **1024 B vs 551K — 538×** | §6c |
| **rank 1** instead of the default rank 8 | **8×** fewer, ACC 0.840 ± 0.009 vs 0.842 ± 0.026 | §24d |
| **flat** instead of hierarchical placement | ~20% fewer | §24d |

Bit-identity is **1.000 at every rank and both placement rules**, so none of this trades preservation
for size. Combined, flat + rank 1 costs **75K per concept against the shipped 735K — 9.8×** — at no
measured accuracy cost.

**What this does not claim.** Neither low rank nor flat placement is novel; what is measured here is
that this system does not need the capacity it was configured with, and that the *routing* decision —
which has no counterpart in the companion paper, whose pipeline spends a full LoRA on every query — is
the lever that makes growth sublinear in experience. It also does not claim constant memory: growth
is monotone, and flattens only if the world's vocabulary is finite (see below).

**Where:** [docs/RESULTS.md](docs/RESULTS.md) §6c, §24d. `experiments/n24_growth.py`,
`experiments/make_rank_frontier.py`.

---

## What is explicitly not claimed

Kept here rather than in a footnote, because several of these shaped the design.

- **A better composition operator — and now we know when ours is needed at all.** §28 removed the
  confound in §13 by training two ColorMNIST backbones that differ *only* in conditioning scheme.
  Score addition goes from F1 **0.200** (entangled, one row per combination) to **0.902** (factored,
  one embedding per factor value), while ours is unchanged (0.179 → 0.198). **Given a
  compositionally-conditioned backbone, training-free score addition beats our composer by 4.5×.**
  The composition half of this work is a workaround for entangled conditionals; the lifelong half
  (routing, addressing, preservation) is orthogonal and unaffected. On CelebA — the one benchmark
  here with a factored head — MCMC-corrected score addition (Du et al., ICML 2023) already reached
  `pair` **0.965 ± 0.023** against our **0.903 ± 0.094**, which now has a mechanism rather than being
  a benchmark-dependent curiosity. We keep an advantage in *coverage* (recall
  0.579 vs 0.483; content diversity 1.068 vs 0.838, where real data is 1.007). On ColorMNIST our
  operator is the only one that works at all. The composition result is **benchmark-dependent**.
- **Order invariance.** Insert-only preservation suggests the same concepts in a different order
  should give bit-identical concepts. **They do not**, and our arm is indistinguishable from a
  control that freezes nothing: 0.000 bit-identical across 4 orderings, because placement is decided
  against whatever tree exists at insertion time (only 5.6% of concepts even keep their path) and a
  shared ancestor's contents depend on which concept created it. The real property is **forward-only**
  — once consolidated, nothing later changes a concept — which is what §5/§7 measure. §23.
- **Scale.** Every quantitative claim rests on backbones of 6.8M and 9.3M parameters that we trained.
  The pretrained work establishes the mechanism at 113.7M, not the full loop.
- **Image quality.** Our ColorMNIST samples carry visible high-frequency roughness. It is **not ours
  to fix at this scale**: walking the pipeline stage by stage, the frozen 6.8M backbone renders a
  class it was explicitly *trained* on at **1.593×** real total variation — worse than anything
  downstream of it, including our adapter at 1.278× (50 steps) or 1.183× (200 steps). No change to
  the composer, pool, adapter or sampler removes an artifact the frozen model puts there; a bigger
  backbone does. The obvious culprit — classifier-free guidance amplifying high-frequency error at
  low noise levels — was implemented, swept and **refuted**, and the negative result is kept because
  it is the evidence. [docs/RESULTS.md](docs/RESULTS.md) §19.
- **That the hierarchy SAVES parameters.** It costs them: every node is a full block, so depth
  *adds* ~23% over flat (8.82M vs 7.16M at rank 8) and ~31% under the better-shaped `cu` rule. The
  cost saving in this work comes from routing, not from sharing.
- **That the hierarchy TRANSFERS anything.** Shared nodes are real — a group node carries 2.3× the
  relative magnitude of a leaf, and its top-influence training instances come from four different
  members (entropy 0.61) — but routing a newcomer through its own family is worth **−0.004 F1**
  against a *mismatched* family's ancestor and **−0.063** against no ancestor at all. Depth does not
  organize abstraction either (internal layer-centroid 48.8 vs leaf 46.9). What is load-bearing is
  the sealing discipline, not the tree. §24.
- **That the hierarchy earns its keep.** Given its best case, `clustered` placement is the *worst*
  arm (0.831) against `shuffled` (0.859) and `flat` (0.842). The method is flat and insert-only as a
  result.
- **That consolidated concepts make later learning cheaper.** Measured on ColorMNIST and refuted.
- **That composed samples can be sampled into the class.** Thirteen interventions on the optimizer,
  objective, prior and sampler could not move it.
- **That the CelebA factor basis is discovered.** It is supplied by construction: a per-class
  embedding is untrainable at that data scale (34% of ceiling, unmoved by 2.3× data).
