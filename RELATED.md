# Related work, and where this sits

Compiled from a systematic sweep (arXiv API + abs-page verification) rather than recall. Every entry
below was checked against its abstract page. The point of this file is to say **which methods are
fair baselines for our setting and which are not**, because most of the neighbouring literature
solves a different problem and comparing to it naively would flatter us.

## Our setting, stated precisely

> One out-of-distribution query arrives. Produce the held-out **combination of known attributes** it
> depicts, from a **frozen** backbone, with **no per-concept pre-training** and no layout input. Then
> keep it, forever, without degrading anything already there.

The second sentence is what most of the literature does not attempt, and the first excludes most of
what looks superficially comparable.

## The dividing line: the exemplar argument, not the retraining argument

It is tempting to dismiss the personalization literature with "those methods require retraining."
**That framing is wrong and a reviewer will say so in one line.** K-LoRA, LoRA Switch/Composite,
B-LoRA, OMG, UCE and Custom Diffusion's closed-form merge are all genuinely training-free *at
composition time*; UCE and the Custom Diffusion merge involve no gradients at all. The
encoder-based family (ELITE, IP-Adapter, BLIP-Diffusion, InstantID, FastComposer, PhotoMaker) is
training-free at *acquisition* time too — one forward pass per concept.

The defensible line is about data, not gradients:

> Every method in this family needs **real exemplar images of the concept** to build its per-concept
> representation — an embedding, a LoRA, a spectral shift, a neuron mask, an encoder input. For an
> unseen attribute *combination*, those images do not exist by construction. Optimization-free is
> not exemplar-free: the encoder methods traded gradient steps for a forward pass, they did not
> remove the image.

A second, independent argument: these methods **bind a concept to a new token**. We compose
attributes that are already in the model's factor basis, so there is no new token to learn — the
failure mode under test is *binding*, not *representation*.

One genuine counterexample, which we cite rather than hide: **Concept Sliders** (Gandikota et al.,
ECCV 2024, [2311.12092](https://arxiv.org/abs/2311.12092)) learns directions from *prompt pairs with
no images at all*, and composes 50+ sliders simultaneously without visible degradation. It needs a
text encoder, so it is not runnable on our class-conditional backbone, but it is the strongest
existing evidence against a blanket "composition does not scale" claim, and we do not make that
claim.

## Fair baselines for composition (no per-concept training, single query)

| method | operator | in our tables |
|---|---|---|
| **Composable Diffusion** — Liu, Li, Du, Torralba, Tenenbaum, ECCV 2022, [2206.01714](https://arxiv.org/abs/2206.01714) | add guidance residuals of several conditionals (product of experts in score space) | **yes**, `composable-diff` |
| **Test-Time Concept Discovery** — Wang, Gupta, Zhu, MacLellan, [2605.07078](https://arxiv.org/abs/2605.07078) | mode-ascent discovery → submodular selection → analytic-score PoE | **yes**, `poe-guided`, in its own Eq. 14 configuration |
| **Textual Inversion** — Gal et al., ICLR 2023, [2208.01618](https://arxiv.org/abs/2208.01618) | learn one embedding; compose by concatenation | **yes**, `inversion` (acquisition is not training-free; included as the cheap reference point) |
| **MCMC-corrected composition** — Du, Durkan, Strudel, Tenenbaum, Dieleman, Fergus, Sohl-Dickstein, Doucet, Grathwohl, ICML 2023, [2302.11552](https://arxiv.org/abs/2302.11552) | annealed Langevin/HMC transitions at each noise level, correcting the composed score | **yes**, `composable-mcmc` |
| **StructureDiffusion** — Feng et al., ICLR 2023, [2212.05032](https://arxiv.org/abs/2212.05032) | restructure cross-attention K/V by constituency parse | **not applicable** — our backbone is class-conditional with a single embedding added to the timestep embedding. There is no text encoder and no cross-attention over tokens to restructure. Stated rather than silently omitted. |

**Du et al. is the sharpest objection to this entire family, ours included**, and it is why the
`composable-mcmc` arm exists. Their result is that

```
grad log q~_t^prod(x_t)  !=  grad log q_t^1(x_t) + grad log q_t^2(x_t)
```

— the sum of diffused scores is the score of the *product of diffused marginals*, not the *diffusion
of the product*. Any method that pushes a composed score through ancestral or DDIM sampling is
therefore targeting the wrong distribution, and the blame lies with the **sampler**, not the model.
Spending extra MCMC transitions per noise level takes CLEVR 5-cube composition from 70.8% to 91.6%
and ImageNet classifier guidance from 18.6% to 94.6%, at roughly 5× sampling cost. It needs no
training and no exemplars, so it is unambiguously fair here. Neither the predecessor paper nor an
earlier draft of this work reported it.

Also unreported by both, and worth a citation even where we do not run it: **Feynman–Kac Correctors**
(Skreta et al., ICML 2025 Spotlight, [2503.02819](https://arxiv.org/abs/2503.02819)) is the current
SOTA inference-time correction for products of experts, and **CompLift** (Yu & Gao, ICML 2025,
[2505.13740](https://arxiv.org/abs/2505.13740)) achieves a cheaper version using only the base
model's own scores as a resampling criterion.

Composable Diffusion is the most important of these and it is instructive on both benchmarks:

* On **ColorMNIST** it can only add the scores of *entangled slot* conditionals, because the model
  has no conditional for "cyan foreground" independent of a background. It produces diverse but
  wrong images — the attribute-binding failure the method inherits from its base model.
* On **CelebA** our factored head provides a genuine conditional per attribute, which is the setting
  the method was designed for. There it composes *the same conditionals we do*, in **score** space
  rather than **embedding** space — the sharpest possible comparison of the two operators.

## Architectural comparisons for the bank (not composition baselines)

- **Orthogonal Adaptation** — Po, Yang, Aberman, Wetzstein, CVPR 2024, [2312.02432](https://arxiv.org/abs/2312.02432). Constrains each concept's LoRA to a concept-specific random orthogonal basis so merges are plain summation at zero cost. The strongest structural precedent for a bank that grows without interference; our N17 orthogonal-delta arm is the same idea tested for *preservation* rather than merge quality.
- **Custom Diffusion** — Kumari et al., CVPR 2023, [2212.04488](https://arxiv.org/abs/2212.04488). The only genuinely **closed-form** merge in the pre-2026 personalization literature (constrained least squares on cross-attention K/V). Demonstrated at 2–3 concepts.
- **SSR-Merge** — ICML 2026, [2606.10617](https://arxiv.org/abs/2606.10617). Closed-form OLS-equivalent signal routing with a **streaming** update, scaling to K=21 adapters. The closest published thing to incremental bank growth done properly.
- **LatexBlend** — CVPR 2025, [2503.06956](https://arxiv.org/abs/2503.06956). An explicit **concept bank** of compact latent-textual representations, combined in latent textual space; motivated by scaling to many concepts.
- **Cones 2** — [2305.19327](https://arxiv.org/abs/2305.19327). Concepts as small additive residuals on a token embedding, composed at inference for free. The earliest clean statement of "store cheap residuals, compose them later".

## Continual learning for diffusion (the retention axis)

Our N5/N22 tables compared only against CL methods derived from the classifier literature (EWC,
replay, finetune, TTT, parameter isolation). The continual-**personalization** line is the closer
neighbour and is now represented by **C-LoRA**:

- **C-LoRA** — Smith, Hsu, Zhang, Hua, Kira, Shen, Jin, *Continual Diffusion*, **TMLR 2024**, [2304.06027](https://arxiv.org/abs/2304.06027). Replay-free self-regularization: the penalty weights each parameter by the accumulated magnitude of *past* tasks' LoRA updates, so parameters earlier concepts wrote to are expensive to move. Implemented in `neural/baselines.py::CLoRA` and reported as an arm.
- **STAMINA** — CVPR-**W** 2024 (workshop), [2311.18763](https://arxiv.org/abs/2311.18763). LoRA + hard attention masks via Gumbel-softmax, folded back into the base model so inference cost is zero; benchmarked on 50 sequential concepts. The reference for long-horizon capacity.
- **CIDM** — NeurIPS 2024, [2410.17594](https://arxiv.org/abs/2410.17594). Names **concept neglect** — a learned concept disappearing when composed — which is the failure our preservation test is designed to exclude by construction.
- **Mining Your Own Secrets** — ICLR 2025, [2410.00700](https://arxiv.org/abs/2410.00700). Replay-free, zero overhead over plain LoRA, regularizes in parameter *and* function space using the model's own diffusion-classifier scores. Current SOTA on this axis.
- **CNS** — ICCV 2025, [2510.02296](https://arxiv.org/abs/2510.02296). Explicitly **fusion-free**: finetune only the neurons associated with a concept, no bank and no merge rule. The most direct architectural competitor to the bank premise.
- **CCDM** — TPAMI, [2606.04797](https://arxiv.org/abs/2606.04797). Attribute-decoupled LoRA plus relevance-guided aggregation — the nearest published thing to "a bank of attribute experts selected by relevance", and therefore our closest prior art.

## Bit-exact preservation is not novel as a property — and pretending otherwise invites the obvious attack

The single most predictable reviewer objection is *"exact preservation is old — see PNN, PackNet,
Piggyback."* It is correct, and the distinction has to be made before a reviewer makes it. Nor are
all "architecture-based" methods equivalent on this axis; lumping them together is itself an error.

| method | isolation | **exact zero-forgetting?** | cost |
|---|---|---|---|
| **PNN** ([1606.04671](https://arxiv.org/abs/1606.04671)) | additive frozen columns + laterals | **yes, unconditional** | **O(T²)** from all-pairs laterals |
| **PackNet** ([1711.05769](https://arxiv.org/abs/1711.05769)) | prune + freeze a shared trunk | **yes**, given the task ID | log₂(T)·\|θ\| bits; **capacity saturates** |
| **Piggyback** ([1801.06519](https://arxiv.org/abs/1801.06519)) | binary mask on a frozen trunk | **yes** — W is never trained | 1 bit/weight/task |
| **SupSup** ([2006.14769](https://arxiv.org/abs/2006.14769)) | mask on a *random untrained* trunk | yes with task ID | 1 bit/weight/task |
| **HAT** ([1801.01423](https://arxiv.org/abs/1801.01423)) | soft gates + soft gradient masking | **NO — approximate** | activation-sized |
| **DEN** ([1708.01547](https://arxiv.org/abs/1708.01547)) | selective retraining + expansion | **NO — approximate** | data-dependent |

**HAT is routinely miscited as zero-forgetting and is not.** Its gate `σ(s·e)` never reaches exactly
{0,1} for finite `s_max`, and its gradient-masking factor `1 − min(a^{≤t−1})` is *continuous*, so any
weight whose cumulative mask is < 1 still receives gradient and the shared weights implementing old
tasks genuinely drift. The paper reports 45–80% forgetting *reduction*, not elimination. DEN's
stage-1 "selective retraining" explicitly modifies weights shared with earlier tasks.

**PNN is our closest structural ancestor.** An insert-only adapter bank with frozen ancestors *is* a
column-per-concept design minus the all-pairs lateral wiring — which is exactly what turns quadratic
growth into roughly linear. The correct framing is therefore:

> We obtain PNN's unconditional bit-exact preservation without its quadratic cost, because adapters
> read only the frozen trunk and their own ancestors rather than every predecessor.

And the novelty claim must live somewhere other than exactness:

1. **the routing decision** — whether an arrival is a new *arrangement* (free) or a new *part* (a kept
   delta), decided by the model's own denoising error with no labels (AUROC 0.980 / 0.995);
2. **the growth curve that follows from it** — cost linear in vocabulary, flat in experience;
3. **native parity** — asking whether a consolidated concept matches a backbone *trained* on that
   slot, which the parameter-isolation literature does not measure;
4. **the generative setting** — CLoG reports parameter isolation *failing* on Oxford-Flowers/CUB for
   lack of per-task samples, so it is not a foregone conclusion that it works here.

Point 4 cuts both ways and is worth pre-empting rather than hoping for: an insert-only bank *is*
parameter isolation, so the few-shot-per-concept regime is where a reviewer will attack it.

## The gap we occupy

A dedicated arXiv query for `abs:"product of experts" AND abs:"LoRA"` returns **zero results**, and
`"product of experts" AND ("adapter" OR "adapters")` returns exactly one diffusion-and-concepts hit —
the predecessor this work builds on. **PoE-style composition combined with a growing adapter bank is
an empty cell.** Composable Diffusion composes prompts and keeps nothing; C-LoRA/STAMINA/CNS keep
concepts and do not compose held-out ones; the LoRA-merging literature composes pre-trained experts
and never faces a held-out combination from a single query.

What is ours, stated narrowly: the **routing decision** (is this a new arrangement or a new part?),
the **certified-preserving consolidation** that follows it, and the **native-parity** evaluation that
asks whether the result matches a backbone trained on the slot. The composition operator itself is
inherited and improved, not invented here.

## Three claims in the recent literature that cut against this work

Recorded because a reviewer will raise them and the paper is stronger for answering them directly.

**1. An impossibility-flavoured conjecture about our whole method class.** *Catastrophic
Compositional Generation* ([2606.23920](https://arxiv.org/abs/2606.23920), June 2026) conjectures
that *no* inference-time technique can efficiently sample the target distribution in well-motivated
OOD settings, and argues that **score-estimation error hurts more than sampler error** when the
target is out of distribution — which, if right, limits even Feynman–Kac correction. It is a
conjecture supported by generalization arguments, not a theorem. Our answer is constructive rather
than rhetorical: we do not stop at inference-time composition. We *consolidate from real data*, so
the composition only has to be good enough to bootstrap a concept that is then learned properly and
kept. That is the part of the pipeline the conjecture does not address.

**2. Our benchmarks are the easy regime.** MOSAIC (ICML 2026,
[2605.00273](https://arxiv.org/abs/2605.00273)) argues that single-object, densely-covered grids like
ColorMNIST and CelebA are where compositional generalization looks best, and that performance
collapses as more combinations are held out. Inheriting the predecessor's benchmarks means
inheriting this criticism, so `n1_train.py --extra-holdout` sweeps the severity and we report the
curve rather than one point.

**3. Precision and density are anti-correlated with our target property.** Both anchor every
neighbourhood to the *real* sample set. A held-out combination is absent from that set by
construction, so a *correct* novel generation scores precision ≈ 0 and density ≈ 0. This is a real
problem for the predecessor paper, which reports FID + CLIP-I + P/R/F1 with no attribute classifier
— but it is only fair to note that our headline numbers are also P/R/F1, so the fix has to be ours
too: our reference set is **real held-out images of the target combination**, which makes recall and
coverage meaningful, while precision-against-the-training-manifold would not be. The conjunctive
probe accuracy is the primary metric for exactly this reason.

## Evaluation protocol precedent

**Beyond Adapter Retrieval** ([2410.09908](https://arxiv.org/abs/2410.09908)) uses a leave-one-out
protocol — train adapters on all datasets but one, then retrieve and compose for the held-out target
with no fine-tuning. That is the closest published analogue of our held-out-combination protocol,
in the LLM task-transfer setting.

## Citation hazards found during the sweep

Recorded because two of these would be factual errors a reviewer catches immediately.

- **"Diffusion MoE" does not exist** as a paper. The work usually meant is **DiT-MoE**, *Scaling Diffusion Transformers to 16 Billion Parameters* ([2407.11633](https://arxiv.org/abs/2407.11633)) — which also reports that its expert routing is spatial/temporal and *largely insensitive to class conditioning*, i.e. MoE routing in diffusion does **not** naturally produce concept experts.
- **"LoRA-Switch" is two unrelated things.** "LoRA Switch" (unhyphenated) is a round-robin activation schedule inside Multi-LoRA Composition (TMLR 2024) — not a router, not an MoE. "LoRA-Switch" (hyphenated, [2405.17741](https://arxiv.org/abs/2405.17741)) is an **LLM systems** paper about CUDA kernel fusion.
- **CLoRA ≠ C-LoRA.** [2403.19776](https://arxiv.org/abs/2403.19776) is contrastive test-time composition; [2304.06027](https://arxiv.org/abs/2304.06027) is continual self-regularized LoRA. One letter apart, unrelated.
- **Venue corrections:** C-LoRA is **TMLR**, not a conference paper. STAMINA is **CVPR-W** (workshop). Both are routinely mis-cited.
- **MoLE** (ICLR 2024, [2404.13628](https://arxiv.org/abs/2404.13628)) requires **training its gates**, so it is not training-free at composition time despite being widely cited as an MoE-of-LoRAs solution. A second, unrelated diffusion paper also uses the acronym.
