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

**Claim.** A stream that arrives in groups can be absorbed in groups, and doing so is better on
*every* axis — not a speed-for-quality trade.

**Evidence.** Twelve concepts, matched discovery configuration:

| | wall | ACC | FGT | bit-identical |
|---|---|---|---|---|
| K = 1 batch | 521s | 0.370 | 0.000 | 1.00 |
| K = 1 sequential | 521s | 0.370 | 0.000 | 1.00 |
| **K = 12 (all in ONE update)** | **249s** | **0.422** | 0.000 | 1.00 |
| K = 12 sequential | 532s | 0.370 | 0.000 | 1.00 |

**2.14× faster and +0.052 accuracy**, with bit-identity at every K from 1 to 12 — so batch size is a
free efficiency knob and not a safety trade. The accuracy gain is the shared-ancestor effect and is
the one thing a sequential run cannot reproduce at any budget: a group node trained by all twelve
members inside one optimization, where sequential can only hand a later concept a frozen warm start.

Verified independently on the pretrained backbone: all three batch arms reach bit-identical 1.00 on
`google/ddpm-celebahq-256`.

**What this does not claim.** The **speedup** amortizes the generative pool sampling, not the
optimizer. Where concepts are trained on real images rather than distilled from sampled pools (the
pretrained path), batch and sequential both take 777s. "Batch addition is safe and structural" is
unconditional; "batch addition is fast" is conditional on there being pools to amortize.

An earlier version of this section reported batching as *costing* accuracy. That was a confound in
our own experiment — the two modes were silently using different discovery configurations — and is
recorded rather than deleted.

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

## 5. The mechanism transfers to a model we did not train

**Claim.** Both halves — preservation and concept acquisition — work on `google/ddpm-celebahq-256`
(113.7M parameters, 256×256, unconditional, trained by someone else).

**Evidence.**

*Preservation*, certified by `tests/test_pretrained_bank.py` on that model and on
`google/ddpm-cifar10-32`:

| assertion | result |
|---|---|
| injecting the bank changes the pretrained model | **0.000e+00** |
| a freshly registered concept is a no-op | **0.000e+00** |
| training a second concept moves the first | **0.000e+00** |

*Acquisition*, after three faults were removed (§12b): each concept reaches **1.00** on its own
attribute at a Vendi **above** the untouched model's 1.51 — no quality cost — producing recognizable
256×256 faces.

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
| pretrained backbone, 2 attributes — accuracy | both correct 0.417 ± 0.083 | **0.417 ± 0.083** (tie) |
| pretrained backbone, 2 attributes — diversity | Vendi **1.436** (untouched: 1.456) | 1.280 |
| ColorMNIST factor concepts | joint **0.042**, F1 **0.490** | 0.000, F1 0.039 |

**The accuracy claim did not replicate.** A single seed showed 0.50 against 0.17 on the pretrained
backbone; at three seeds the two operators **tie on accuracy**. What survives is that path union
*collapses diversity* (Vendi 1.280 against the untouched model's 1.456, where score composition holds
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

## What is explicitly not claimed

Kept here rather than in a footnote, because several of these shaped the design.

- **A better composition operator.** On CelebA, MCMC-corrected score addition (Du et al., ICML 2023)
  reaches `pair` **0.965 ± 0.023** against our **0.903 ± 0.094** and is more accurate on every
  accuracy metric, with zero parameters and no training. We keep an advantage in *coverage* (recall
  0.579 vs 0.483; content diversity 1.068 vs 0.838, where real data is 1.007). On ColorMNIST our
  operator is the only one that works at all. The composition result is **benchmark-dependent**.
- **Scale.** Every quantitative claim rests on backbones of 6.8M and 9.3M parameters that we trained.
  The pretrained work establishes the mechanism at 113.7M, not the full loop.
- **That the hierarchy earns its keep.** Given its best case, `clustered` placement is the *worst*
  arm (0.831) against `shuffled` (0.859) and `flat` (0.842). The method is flat and insert-only as a
  result.
- **That consolidated concepts make later learning cheaper.** Measured on ColorMNIST and refuted.
- **That composed samples can be sampled into the class.** Thirteen interventions on the optimizer,
  objective, prior and sampler could not move it.
- **That the CelebA factor basis is discovered.** It is supplied by construction: a per-class
  embedding is untrainable at that data scale (34% of ceiling, unmoved by 2.3× data).
