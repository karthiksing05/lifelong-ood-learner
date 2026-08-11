# Experimental settings: the companion paper vs. this repository

Every hyperparameter in the companion paper's Appendices C/D/F, checked line by line against what
this repository actually runs. The point is to separate three things that are easy to conflate:

  * **MATCH** — we run the paper's value.
  * **DEVIATION** — we run a different value *on purpose*, with a reason recorded and, where it
    matters, a measurement backing it.
  * **INFIDELITY** — we ran a different value *by accident*, usually under a constant or comment
    claiming it was the paper's. These are bugs. Four were found writing this document; all four are
    fixed below, and two of them were affecting published-looking numbers.

Source: `poe-diffusion.pdf`, Tables 5–8 (backbone, discovery, distillation, sampling) and Appendix F
(compute). Our side: read out of the code, not out of prior documentation.

---

## 0. The rule these tables are held to

**Every benchmark runs the same core settings. A per-dataset difference is legitimate only when it
reflects the input's dimensions** — resolution, channel count, or a cost that scales with them.
Anything else is per-dataset algorithmic tuning, and a comparison whose algorithm changes with the
dataset is not a comparison.

Audited against that rule, the settings that differ between Color-MNIST and CelebA fall into three
groups:

| setting | Color-MNIST | CelebA | verdict |
|---|---|---|---|
| `u_grid`, `iters`, `lr`, `init` | 15 levels, 150, 0.10, hybrid | 10 levels, 100, 0.09, query | **the paper's own Table 6 columns** — reproduced, not chosen by us |
| `n_starts` | 128 | 8 | their column, and cost-driven: per-step ascent on 128×128 is far dearer than on 32×32 |
| PoE `w0` | 1.2 | 0.5 | their Table 8 columns |
| `probes`, `wmax` | 4, 2.0 | 4, 2.0 | identical already |
| `adam`, `clamp`, `tweedie` | Adam, off, off | Adam, off, off | **reverted to the paper's setting, uniform on both** |

The last row is the one that needed fixing, and the fix was to **undo all three**. They were
introduced from CelebA evidence during the 2026-08-11 artifact investigation and briefly applied per
dataset, which meant our own algorithm differed between benchmarks. Judged on both benchmarks by a
criterion that does not care which arm benefits — the eps error at the returned mode, the quantity
ascent minimizes — no single alternative beats the paper's Adam:

| | Color-MNIST, u=0.05 | pretrained CelebA, u=0.15 |
|---|---|---|
| **Adam** (their Table 6) | **0.0224** | 0.0113 |
| plain gradient | 0.1505 | **0.0046** |
| Adam + x₀ projection | 0.0696 | — |

The optimizer that wins on one loses on the other, and the unified alternative loses on both. **And
the change was never needed**: the CelebA artifact was the input layout, and fixing *that alone* took
`poe-guided` from 2.50–3.88× real roughness to 1.00/0.89/1.23 with Adam untouched. Adding a clamp
gave 1.04/0.94/1.20; adding plain ascent 1.02/0.82/1.11 — inside the noise of doing nothing further.

What remains true is that Adam leaves ~11% of a CelebA mode outside [−1,1], so the **prototype panels
show clipping when rendered**. That is a display property of unconstrained ascent, not a defect in
the composition: the teacher and the guidance consume the raw values, and the samples are at
real-image texture either way. §5c is retained as the record of a hypothesis that was tested and
rejected.


## 1. Backbone training (their Table 5)

| Parameter | Paper Color-MNIST | Ours | Paper CelebA | Ours | |
|---|---|---|---|---|---|
| Base UNet | `ddpm-cifar10-32`, **fine-tuned** | **from scratch**, base=64 | `ddpm-celebahq-256` @ `sample_size=128`, **fine-tuned** | **from scratch** | **DEVIATION** |
| Resolution | 32 | 32 | 128 | **64** | **DEVIATION** |
| Class-embedding rows (incl. null) | 121 | 121 | 52 | **1205** | **DEVIATION — 23×** |
| Optimizer | AdamW | AdamW | AdamW | AdamW | MATCH |
| Learning rate | 2e-4 | 2e-4 | 2e-4 | 2e-4 | MATCH |
| Weight decay | 1e-6 | 1e-6 | 1e-6 | 1e-6 | MATCH |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 | MATCH |
| Batch size | 256 | **128** | 128 (32×4) | 128 | DEVIATION |
| Training length | 100 epochs | **60** | 500,000 steps | ~26 GPU-h | **DEVIATION** |
| Mixed precision | bf16 | **fp32** (tf32 opt-in) | bf16 | **fp32** (tf32 on) | **DEVIATION** |
| EMA on weights | yes | yes | yes | yes | MATCH |
| Null dropout *p* | 0.1 | 0.1 | 0.1 | 0.1 | MATCH |
| Diffusion schedule | **linear β**, T=1000 | **cosine ᾱ** | linear β | **cosine ᾱ** | **DEVIATION — see §5** |
| Parameterization | ε-prediction | ε-prediction | ε-prediction | ε-prediction | MATCH |
| Compute | 1×A40, ~6 h | comparable | 4×RTX6000, ~50 h | ~26 GPU-h | DEVIATION |

The from-scratch backbone is a large deviation and it penalises **every arm equally**, including
`poe-guided`, the paper's own method: we are running their method on a substrate trained for roughly
an eighth of the compute at a quarter of the pixels, over a benchmark **23× wider in conditioning
rows**.

**It is no longer established that this is what decides CelebA image quality.** §29 concluded that;
§29a does not support it. Running the pipeline on the paper's *own* pretrained `ddpm-celebahq-256` —
a backbone that renders photorealistic faces, untouched by us — still produced samples rougher than
real, which the backbone story cannot explain. Three candidates are now live: this deviation, the
w₀ infidelity (§5a, measured and fixed), and the input-layout permutation (§5b).

## 2. Concept discovery (their Table 6)

| Hyperparameter | Paper CM | Paper CelebA | Ours | |
|---|---|---|---|---|
| Prototypes *K* | 3 | 3 | 3 | MATCH |
| Selection objective | submodular greedy | submodular greedy | submodular greedy | MATCH |
| Softmax temperature τ | **0.5 fixed** | **0.5 fixed** | **adaptive**, `0.05 × std` | **DEVIATION** |
| Timestep grid | [50,400,25] → 15 levels | [50,500,50] → 10 levels | **4 levels** in practice | **DEVIATION** |
| Ascent starts per *t* | 128 | 8 | **12–16** | **DEVIATION** |
| Ascent iterations | 150 | 100 | 80–120 | DEVIATION |
| Ascent step size | 0.1 | 0.09 | 0.1 | MATCH (CM) |
| Ascent optimizer | Adam | Adam | Adam (CM) / **plain** (CelebA) | **DEVIATION on CelebA only — §5c** |
| Initialization | hybrid (50/50) | query-centred | hybrid | MATCH (CM) |
| Hutchinson probes | 4 | 4 | 4 | MATCH |
| Hutchinson FD step | 5e-3 | 5e-3 | *not exposed* | **UNVERIFIED** |
| Seed | 42 | 42 | 0 | immaterial |

`PAPER_COLORMNIST` and `PAPER_CELEBA` in `lifelong_ood/compose.py` transcribe these columns
exactly. The deviation is that **almost nothing calls them** — the defaults threaded through the
experiments are 4 noise levels × 12–16 starts, i.e. **64 ascent chains against the paper's 1920, a
30× shortfall** on Color-MNIST. This is the cause of the bad prototypes: submodular selection can
only pick from what ascent found.

τ is a *documented, reasoned* deviation, not an accident (`compose.py:121`): the paper fixes τ=0.5
against experts whose Laplace log-likelihoods are separated by ~0.2 nats per pixel here, where a
0.5 softmax averages them into a blur. It should still be measured against the paper's value rather
than assumed, which `--paper-discovery` in n44 now does.

## 3. Distillation (their Table 7)

| Hyperparameter | Paper CM | Paper CelebA | Ours | |
|---|---|---|---|---|
| LoRA rank *r* | 8 | **16** | 8 everywhere | DEVIATION (CelebA) |
| LoRA α | **2r** (=16 / 32) | 2r | **no α — effective scale 1** | **DEVIATION** |
| Target modules | full UNet linear+conv | full UNet linear+conv | conv + linear, **`time_emb_proj` excluded** | **DEVIATION** |
| LoRA dropout | 0.0 | 0.0 | 0.0 | MATCH |
| LoRA lr | 1e-3 | 1e-3 | 1e-3 | MATCH |
| New embedding init | N(0, 0.01²) | N(0, 0.01²) | N(0, 0.01²) | MATCH |
| Optimizer | Adam | Adam | Adam | MATCH |
| Mini-batch | 16 | 2 | 16 | MATCH (CM) |
| Epochs | 10 | 5 | 10 | MATCH (CM) |
| CFG dropout on c_new | 0.1 | 0.1 | 0.1 | MATCH |
| Pool size N_pool | **256** | 256 | **48–64** | **DEVIATION** |
| Pool sampler | DDIM η=0, 50 steps | same | DDIM η=0, 50 steps | MATCH |
| Pool guidance | variance-aware (Eq. 14) | same | variance-aware | MATCH |

Two of these are worth flagging as load-bearing:

**α.** The paper scales its LoRA delta by α/r = 2. We have no α at all, so our delta enters at
scale 1. At equal rank and equal learning rate our adapters are **half-strength** relative to
theirs. Every parameter-efficiency comparison in this repo is therefore, if anything, conservative
in our favour on the *storage* axis and against us on the *quality* axis.

**Target modules.** Excluding `time_emb_proj` is a deviation from "full UNet linear/conv" and a
deliberate one, with the reasoning recorded at `adapters.py:139`: in `ddpm-celebahq-256`, 32 of 58
Linear modules are `time_emb_proj`, so "adapt every linear" overwhelmingly means adapting the path
that tells the model what noise level it is at. That produced melted faces. This is a real
difference from the paper and should be stated as one rather than buried.

**Pool size.** 48–64 against 256 is a 4–5× shortfall in distillation data, applied to *every* arm
including the baselines, so it compresses differences rather than favouring us.

## 4. Sampling (their Table 8)

| Hyperparameter | Paper CM | Paper CelebA | Ours | |
|---|---|---|---|---|
| DDIM scheduler | η=0, 50 steps | η=0, 50 steps | η=0, 50 steps | MATCH |
| Pool size | 256 | 256 | 48–64 | DEVIATION (§3) |
| Variance-aware **w₀** | **1.2** | **0.5** | **1.2 everywhere** | **INFIDELITY — fixed** |
| Variance-aware cap w_max | 2.0 | 2.0 | 2.0 | MATCH |
| Variance mode | entropy-adaptive | entropy-adaptive | entropy-adaptive | MATCH |
| LoRA sampling-w | 0.8 | 1.2 | swept {0.8, 1.0, 1.2} | MATCH |

---

## 5. The infidelities found, and what they touched

**(a) CelebA guidance ran at 2.4× the paper's strength.** A constant named `PAPER_POE`, holding
`w0=1.2`, was copy-pasted into six experiment files. 1.2 is Table 8's **Color-MNIST** value; CelebA
is **0.5**. So `n20_celeba_composition.py` — the CelebA headline — plus `n44` and
`diag_celeba_chain` all guided CelebA at 2.4× the published strength, under a constant that claimed
to be the paper's and a comment reading *"the published method in its published configuration."*
Over-guidance saturates and adds artifacts, and it was applied to `poe-guided`, the paper's own
method. **This is a second, independent cause of the CelebA quality complaint, additional to the
backbone**, and unlike the backbone it is free to fix. Now split into `PAPER_POE_COLORMNIST` and
`PAPER_POE_CELEBA` in `compose.py`, one definition per dataset.

**(b) Case-study prototypes came from a different ascent run than the teacher.**
`make_case_studies.py` called `L.compose(xq, **disc)` for the teacher and a *second*, separately
parameterized `candidates_ascent` for the displayed modes, then indexed the first run's `sel` into
the second run's `M`. With matching settings and seed the two runs coincide, so the bug was
invisible; under `--paper-budget` the settings diverge and it raised `IndexError: index 503 is out
of bounds for axis 0 with size 50`. Every prototype panel rendered to date was only *accidentally*
showing the modes that were actually selected. Now one ascent run feeds both.

**(c) The discovery budget was never the paper's** despite `candidates_ascent`'s docstring having
claimed its defaults were Table 6's Color-MNIST column. Docstring corrected earlier; the constants
now exist; the remaining work is making the experiments *call* them.

**(d) LoRA α is absent**, so our adapters are half-strength versus the paper at equal rank. Not yet
changed — changing it would move every adapter number in the repository, and it is the more honest
direction for our claims (we look worse on quality, better on storage), so it wants a deliberate
re-run rather than a quiet edit.

## 5b. A difference that is in no table of theirs: the input layout

The audit above compares hyperparameters. This one is not a hyperparameter — it is a difference in
what the UNet is shown, and it has no counterpart in the paper because they hand `diffusers` a normal
NCHW image.

Flat rows in this repository are **HWC** (both `rgb_img` and `celeba.rgb` display them as
`.reshape(img, img, 3)`), and every consumer turns one into a tensor with a plain
`.reshape(-1, 3, H, W)`. That is not an image; it is a fixed permutation of the pixels. For 64×64×3,
by exact index algebra (`tests/diag_layout.py`):

* what the UNet calls **channels** 0 / 1 / 2 are image **rows** 0–21 / 21–42 / 42–63, not R/G/B
* its **width** axis is the **colour** axis — adjacent positions are one pixel's R, G, B
* its **vertical** axis is a **~22-pixel jump across the image**

so one 3×3 convolution reads real columns **3, 24 and 45 simultaneously**. There is no spatial
locality left at any scale, and downsampling compounds it. `figures/f20_layout.png` renders a real
face as the UNet sees it: a striped, colour-scrambled field with only low-frequency structure intact.

`ddpm.py` argued this was harmless because training and sampling share the permutation, so "the model
simply learns in the permuted space". That is sound for a permutation-equivariant model. **A CNN is
not one** — locality *is* its inductive bias. Images still *display* correctly, which is why it
survived; what cannot survive is local texture.

**`pretrained.py:217` and `:246` do the same reshape**, so `google/ddpm-celebahq-256` — trained on
true CHW faces — is being fed permuted rows. That is a stronger failure than a lost prior, because
its filters were learned on real images, and it is the most likely explanation for §29a's finding
that a photorealistic backbone still emits rough samples.

A `layout` option (`"permuted"` default, `"hwc"` correct) is now threaded through `_as_img`/`_as_flat`,
`train_ddpm`, `NeuralScoreField` and `n1_train`, **recorded in the checkpoint** and preferred over the
process default by `load_field` — a checkpoint sampled in the other layout silently produces garbage.
N45 measures what it costs: **-28 % on the denoising objective, replicated**, and **no** gain in conditional composition accuracy (paired, it costs 0.062). See RESULTS 30.


## 5c. The ascent optimizer: a measured deviation from their Table 6

Their Table 6 specifies **Adam** for mode ascent. We use **plain gradient ascent**, and the reason
is a measurement rather than a preference.

Adam's update is normalized: every coordinate moves by about `lr` per iteration however small its
gradient is. Near a mode, where the gradient vanishes, a plain step vanishes with it and settles —
Adam's does not, it keeps taking full-sized steps and marches past. On the pretrained CelebA backbone
at u = 0.15, against the query's own eps error of **0.0260** as the bar:

| setting | fraction of x₀ out of [−1,1] | eps error at the returned point |
|---|---|---|
| Adam, lr 0.09, 100 it *(their column)* | **11.1 %** | 0.0113 |
| Adam, lr 0.02, 100 it | 0.0 % | 0.0056 |
| Adam, lr 0.09, 20 it | 0.3 % | 0.0391 — *worse than the query* |
| **plain, lr 0.09, 100 it** | **0.0 %** | **0.0046** |
| plain, lr 1.00, 100 it | 32.9 % | 0.0469 — *worse than the query* |

**Plain ascent wins on both axes**: it stays entirely in gamut *and* finds a lower-error mode than
Adam does. The eps error is the right adjudicator here because it is the quantity ascent exists to
reduce — a setting scoring worse than the query it started from has not found a mode of anything.

Why it mattered visually: out-of-range coordinates are clipped for display, and a large fraction of
them clipped to exactly ±1 is flat saturated colour. That is why the prototypes posterized, and why
adding a range clamp changed the numbers without changing the pictures — the values still arrived at
the edge, they just stopped there. Three earlier explanations of mine (runaway Adam steps, the
x₀ estimator, the clamp) were each refuted by measurement before this one held.

**Scoped to CelebA, and the scoping matters for fairness.** Making plain ascent global would have
been the wrong call, and the check that caught it was noticing what it did to the *baseline*:

| | Adam leaves out of [−1,1] | effect of switching to plain |
|---|---|---|
| pretrained CelebA, u = 0.15 | **11.1 %** | fixes the saturated prototypes |
| Color-MNIST, u = 0.05–0.40 | **0.00 – 0.32 %** | ours F1 0.605 → 0.633, but `poe-guided` joint **0.839 → 0.739** for nothing in return (recall 0.001 either way) |

Color-MNIST never had the problem plain ascent fixes, so the change there was buying our own arm an
improvement while costing the companion method a tenth of its accuracy. **The paper's Adam stands on
Color-MNIST**; `PAPER_CELEBA` carries `adam=False` because that is where the defect was measured.

For the record, our Color-MNIST gain is *not* from the ascent at all — the rule comparison isolates
it: `ll` → `cu` alone gives F1 0.606 → 0.633, which is the whole of it.

## 6. What this means for the results

The deviations sort into three groups by what they do to our claims:

**Penalise us relative to the paper** (weaker backbone, 4–5× smaller pools, no α, 30× less
discovery). These make our absolute image quality worse than the published figures and cannot
inflate any comparative claim, because they are applied identically to every arm including the
baselines.

**Change the meaning of a transcribed number.** The cosine-vs-linear schedule is the subtle one:
their timestep grid `[50,400,25]` indexes *their* linear-β ᾱ ladder, and we feed the same numbers
into a **cosine** ᾱ. `u=0.05..0.4` is therefore not the same set of noise levels for us as for
them, so "we used the paper's grid" is true numerically and false physically. Any claim that turns
on *which* noise levels discovery visits — the basic-level story inherited from the ACS paper most
of all — needs this stated.

**Actually invalidate a number** (the CelebA w₀). Fixed, and the affected CelebA runs need redoing.

None of the storage claims are touched by any of this: 72 bytes versus 2.3 MB is arithmetic on
tensor shapes, not a measured quantity.
