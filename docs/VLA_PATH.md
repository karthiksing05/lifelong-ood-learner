# From three image datasets to embodied policies

Written 2026-08-14. Where the scheme stands, what Shapes3D taught that ColorMNIST and CelebA could
not, and the order in which to attack a diffusion policy. `docs/OUTLINE.md` is the structural index;
this is the forward plan.

---

## What is established, and how confident to be

| claim | evidence | confidence |
|---|---|---|
| A conditional diffusion model's class table organises **compositionally** without being told the factors | basis R² 0.870 ColorMNIST · 0.620 CelebA · 0.538 Shapes3D · 0.832 **DiT** | **high** — four backbones, three datasets, two architectures |
| **ADDRESS** composes held-out combinations at ~0 parameters | ColorMNIST joint 0.556 · CelebA 2.7× floor · Shapes3D **16× floor** · DiT 0.661 full-loop | **high** |
| The scheme is **not tied to the UNet** | latent DiT, adaLN conditioning, `factorize`/`compose` unchanged (§7z) | **high** — this is the VLA-relevant one |
| ADDRESS is **span-limited by construction** — a point, never a direction | affine residual 0.0000 vs a trained embedding's 0.7359 (§6e-bis) | **high** — it is algebra, then measured |
| The **router** transfers | ColorMNIST AUROC 0.980 · Shapes3D **0.354** | **low** — does not transfer |
| Absorption improves **representations** | N50 withdrawn; N53 corrected experiment | **unknown** — the honest state |

---

## The three lessons Shapes3D taught that the other datasets could not

### 1. A strong pretrained backbone starves the conditioning pathway

Fine-tuning `eurecom-ds/scoresdeve-ema-shapes3d-64` reached eps-loss **0.0015 within minutes** — the
model already denoises the domain nearly perfectly. The residual a *label* could explain is therefore
almost zero, so the class table receives almost no gradient. Four recipes failed on this before it was
understood; giving the table its **own 50× learning rate** took ADDRESS from 0.094 to **0.583**.

**Why this matters more for VLA than for images.** A pretrained policy backbone is *exactly* this
situation, and more extreme: behaviour-cloning loss on a competent policy is already low, and the
conditioning you want to add (a new object, a new goal) explains a small fraction of the action
residual. Expect the conditioning pathway to need its own optimiser group from the start. **Do not
read a low total loss as evidence that conditioning is being learned** — it is nearly independent of
it.

### 2. Convention mismatches are silent and expensive

Three results were destroyed by two bugs that produced plausible numbers:

* a sampler initialised at `0.5 + noise` where the VP prior is centred at **zero** — every trajectory
  in one basin, one image out, every arm scoring identically;
* `conditional_losses` reading HWC rows as CHW — **974×** worse eps-loss, measured.

Neither raised an exception; both produced numbers that read as "this dataset does not condition".
`tests/diag_backbone_conventions.py` now asserts layout, range, and sampler prior per backbone.
**Port that diagnostic before the first policy experiment**, because a policy adds more conventions to
get wrong: action normalisation, chunk ordering, proprioception scaling, and the observation stack.

### 3. The verdict statistic has to be the thing you claim

N58 judged conditioning by the eps-loss gap between right and wrong classes and printed
**NOT USEFULLY CONDITIONAL** on a checkpoint composing at **16× the floor**. The gap is an *average*
denoising improvement; on an easy marginal it stays tiny however well the label steers. For a policy
the analogue is worse — average action MSE will be dominated by the easy majority of timesteps.
**Judge on task success or on the composition margin, never on the training loss.**

---

## Next, on Shapes3D specifically

1. **The router does not transfer (AUROC 0.354, below chance).** ADDRESS composes but the
   novelty/vocabulary decision fails. This is now measured through the corrected path, so it is a real
   negative. Diagnose whether `s_novel` is the wrong signal here or whether the u-band (tuned on
   ColorMNIST at u=0.10/0.45) simply does not locate the informative noise level for this domain — a
   u-sweep is cheap and is the first thing to try.
2. **Recover `object_hue`.** Pinning `wall_hue` shrank the set to 9000 images and cost that probe, so
   `joint` is a two-factor joint. More images per composition should restore it and make the headline
   a true three-factor number.
3. **EXTEND BASIS on Shapes3D.** §6e-bis predicts the tier that adds a *direction* is where
   representational gain must come from; Shapes3D has clean held-out factor *values* to test it on.
4. **Sweep the class-table learning rate.** 50× was a first guess that worked. The curve from 1× to
   100× is the evidence for lesson 1 above and turns a lucky setting into a stated rule.

---

## The order to attack a diffusion policy

**Stage A — establish the substrate exists.** Take a pretrained diffusion policy and ask the only
question that matters first: *does its conditioning table factorise?* Run `factorize` on whatever
embeds the task/goal and report R². Everything downstream needs R² materially above chance; if it is
not there, ADDRESS has nothing to compose through and the honest move is to say so early. This is one
afternoon of work and it gates the rest.

**Stage B — port the conventions diagnostic before anything else.** Extend
`diag_backbone_conventions` to a policy: action normalisation, observation layout, chunk ordering.
Three results have already been lost to this class of fault on images alone.

**Stage C — hold out a task COMBINATION, never a primitive.** The distinction ADDRESS rests on is
combination-vs-vocabulary: hold out "red block into blue bowl" while red, block, blue and bowl all
appear elsewhere. `factored.holdout_mask` already enforces the analogous guarantee for images and
should be reused rather than reimplemented — a split that starves a primitive silently converts an
ADDRESS experiment into an EXTEND one.

**Stage D — condition with its own learning rate, from the first run.** Lesson 1 says the pathway will
otherwise be starved by a loss the backbone already minimises.

**Stage E — judge on rollout success.** Lesson 3. Report the composition margin against an
unconditional-policy floor, exactly as `n58` does now.

**What to expect, honestly.** The storage claim should transfer — it is arithmetic on tensor shapes,
and a 72-byte address is a 72-byte address whatever the backbone. Composition has now transferred
across three datasets and two architectures. The **router** has not transferred once beyond
ColorMNIST, and the **representation** question is open with its headline withdrawn. A policy paper
resting on ADDRESS is supportable today; one resting on autonomous routing is not.
