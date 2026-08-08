# Publication readiness, assessed against what the numbers actually support

Written after a systematic literature review, the addition of every baseline that review identified,
and the correction of four defects it exposed in how this repository ran its own experiments. The
purpose is to say what can be claimed, what cannot, and what would have to be true for this to be a
top-venue paper rather than a workshop one.

## The one-line verdict

**A strong workshop paper as it stands; a conference paper if the scale question is answered.** The
lifelong-learning results are solid, adequately baselined, and now survive a literature review that
was explicitly looking for reasons they would not. The composition results do not support the claim
the project began with, and the paper is better for saying so.

---

## What is established

### 1. Exact preservation under an unbounded stream, at zero storage cost

Across 3 seeds and two datasets, with **eleven** continual-learning arms:

| | ours | independent | ttt | every other arm |
|---|---|---|---|---|
| fraction bit-identical | **1.000** | **1.000** | **1.000** | 0.250 |
| forgetting (FGT) | **0.000** | **0.000** | 0.021 | 0.004 – 0.159 |
| images kept forever (SSS) | **1.000** | **1.000** | 0.000 | 1.000 except der++ (0.740) |

Every arm that tries to *protect* shared parameters — EWC, SI, MAS, C-LoRA, generative replay,
generative distillation, DER++ — forgets and none reaches bit-identity. The two that *remove* shared
parameters do. TTT also reaches 1.000 but keeps all 768 images of the stream.

This is the strongest result here and it is now properly baselined: SI, MAS, DER++ and generative
distillation were all added *after* the review said their absence was the most obvious complaint.

**The honest qualification, stated in the paper (§10b):** `independent` matches us on every column in
this table *at fewer parameters* (ACC 0.863±0.015 / 7.16M vs 0.839±0.025 / 8.82M). n5 does not show
our method beating parameter isolation. The differentiator is the routing rule and the growth curve,
measured elsewhere.

### 2. The routing decision

AUROC **0.980** novel-vs-in-distribution and **0.995** vocabulary-vs-combination, from the model's own
denoising error with no labels, and the error is one-sided in the safe direction (a new vocabulary
item was routed to the free path 0 times in 120 queries). This is the piece with no counterpart in
the closest prior work, and the literature sweep found no paper combining PoE-style composition with
a growing adapter bank — an empty cell.

### 3. Native parity

A consolidated concept is compared against a backbone *trained* on that slot. This question only
arises once concepts are kept, and the personalization literature does not ask it.

### 4. Preservation transfers to a model we did not train

`tests/test_pretrained_bank.py` certifies on **google/ddpm-celebahq-256** (113.7M params) that
injecting the bank changes the model by exactly `0.000e+00`, a fresh concept is an exact no-op, and
training a second concept leaves the first bit-identical. The preservation mechanism is not an
artifact of our small backbone.

---

## What is not established

### 1. The composition operator is not better than the baselines on CelebA

This is the significant negative of the review, and it emerged only after the baselines were run
*properly*. On the strong split (3 seeds):

| | pair | target | recall | content div |
|---|---|---|---|---|
| ours | 0.903±0.094 | 0.727±0.081 | **0.579** | **1.068** |
| composable-diff (2022) | 0.934±0.066 | 0.740±0.084 | 0.280 | 0.750 |
| composable-mcmc (2023 correction) | **0.965±0.023** | **0.779±0.105** | 0.483 | 0.838 |

Both are training-free and parameter-free. We retain an advantage in *coverage* — recall and content
diversity, where we match real data's 1.007 and they do not — and lose on accuracy.

**On ColorMNIST our operator is the only one that works** (both composable arms at 0.000 across a 23×
guidance sweep and with the MCMC corrector) because that backbone has no per-factor conditional to
add. The composition result is therefore **benchmark-dependent**, and the defensible contribution is
characterizing *when* score addition suffices — not a better operator.

### 2. Transfer to a pretrained backbone: preservation yes, composition not yet

Preservation transfers exactly (`0.000e+00` on a 113.7M-parameter model). Concept *acquisition* now
works too, once adaptation is confined to high noise levels: `Eyeglasses` at 0.75 with Vendi 1.48
against the untouched model's 1.51 (§12). What has **not** been demonstrated end-to-end is
composition of two such concepts at this scale with a quantitative result — the case study exists,
the numbers are single-seed, and they are not part of any headline claim.

The route there is now clear rather than blocked, which is a different thing from being done.

### 3. Scale

Backbones are 6.8M (32×32) and 9.3M (64×64) parameters. Every quantitative claim rests on models we
trained ourselves at small scale. This is the single largest obstacle to a top venue.

---

## What a reviewer will attack, and the current answer

| objection | answer | strength |
|---|---|---|
| "Exact preservation is not novel — PNN, PackNet, Piggyback." | Conceded explicitly in RELATED.md with a table separating methods that achieve it from those miscited as achieving it. Novelty relocated to routing + growth + native parity. | **adequate** |
| "Your generative-replay baseline is undertuned." | Cited as a reproduced known failure mode (Masip et al.) *and* generative distillation added as the corrected arm. | **adequate** |
| "You report probe accuracy, not generation quality." | AFQ/AIQ/FR added on a Fréchet distance. Reported as a null result: 0.030–0.040 for every arm. | **adequate** |
| "Parameter isolation fails when per-concept data is scarce (CLoG)." | Not tested. An insert-only bank *is* parameter isolation, so this is the sharpest untested attack. | **open** |
| "A single held-out split is cherry-picked (MOSAIC)." | `--extra-holdout` implemented for a severity sweep; **not yet run**. | **partly addressed** |
| "Composable Diffusion + MCMC beats you." | True on CelebA; stated in README and RESULTS. | **conceded** |
| "Small models." | No answer. | **open** |

---

## What would move this to a conference paper

In order of expected value:

1. **Answer the scale question.** Preservation is certified at 113.7M and concept acquisition now
   works there (§12c). What remains is running the full loop — route, consolidate, compose, retain —
   at that scale across seeds, so the lifelong claims inherit it rather than only the mechanism.
   This is the highest-value remaining item by a wide margin.
2. **Run the held-out-severity sweep.** The machinery exists (`n1_train.py --extra-holdout`). Every
   recent paper that ran one found degradation, so reporting the curve pre-empts the criticism and
   costs a few backbone trainings.
3. **Test the few-shot-per-concept regime.** CLoG reports parameter isolation failing there, and our
   method is parameter isolation. If it holds up, that is a result; if it does not, better to know.
4. **Reframe the composition section** as a characterization of when score addition suffices, with
   the ColorMNIST/CelebA contrast as the finding. Already done in the docs; it should shape the
   paper's structure too.

## What to submit if submitting now

A workshop paper on **certified-preserving lifelong concept acquisition**, with:

- the eleven-arm retention tables as the primary evidence,
- the routing rule as the mechanism,
- native parity as the quality check,
- the composition section as inherited machinery with an honest benchmark-dependent characterization,
- and the pretrained preservation certificate as evidence the mechanism is not a small-model artifact.

That paper is defensible line by line. The version claiming a better composition operator is not.
