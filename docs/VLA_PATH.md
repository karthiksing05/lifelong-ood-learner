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
| The **router** transfers | held-out AUROC: ColorMNIST **1.000**, Shapes3D **0.861 ± 0.043** at a calibrated u (§6c-quinquies) | **medium** — transfers once `u` is calibrated per backbone; 0.354 was the operating point |
| The router's **operating point** can be chosen without held-out labels | Shapes3D: label-free `calibrate_u_novel` picks the oracle's u on 3/3 seeds, **0.929** vs the inherited constant's 0.662 | **medium** — exact on the pretrained backbone, fails on ColorMNIST for the reason §6c gives |
| A **multi-object scene** is a composition over addresses, not *k* separate blocks | scene table factorises over object membership (R² 0.931, CV **0.875** vs null 0.254); held-out PAIRS at 0.302 exact-set vs a 0.003 floor, recall 0.592 vs 0.156 (§7bb) | **medium** — one seed, 9 pairs, and 3 of the 9 fail outright |
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

*Now measured rather than argued* (§5t-ter): across a 1×–100× ladder the ε-loss gap moves by
**1.4 × 10⁻⁴** while composition moves from 1.8× the floor to 11.6×. Any multiplier ≥5× works, so the
setting does not need tuning — but leaving it at 1× costs the entire result, silently, with a training
curve that looks fine.

### 2. Convention mismatches are silent and expensive

Three results were destroyed by two bugs that produced plausible numbers:

* a sampler initialised at `0.5 + noise` where the VP prior is centred at **zero** — every trajectory
  in one basin, one image out, every arm scoring identically;
* `conditional_losses` reading HWC rows as CHW — **974×** worse eps-loss, measured.

Neither raised an exception; both produced numbers that read as "this dataset does not condition".
`tests/diag_backbone_conventions.py` now asserts layout, range, and sampler prior per backbone.
**Port that diagnostic before the first policy experiment**, because a policy adds more conventions to
get wrong: action normalisation, chunk ordering, proprioception scaling, and the observation stack.

### 2-bis. A broken READER looks exactly like a broken MODEL — check the unconditional arm

Added 2026-08-15, after it cost a 109-minute retrain and a nearly-published false negative (§7bb).
The multi-object samples came out as pure noise and the model scored **0.000** conditioned on its own
trained row. That reads as "conditioning never learned". It was a **clip range**: the field works in
MODEL space [-1,1] and the DATA range `(0.0,1.0)` was passed, clamping away the whole negative half
at every reverse step — fatal because those scenes are ~90 % black background, i.e. **-1** in model
space. The model was fine: eps-loss 0.0097, *better* than the reference backbone.

**The tell was available immediately and was missed: the UNCONDITIONAL arm was equally broken.** A
model that failed to learn *conditioning* still generates plausible unconditional samples. If both
arms are broken, suspect the path, not the training. Three cheap checks, in this order:

1. **eps-loss of real data through the same reader**, against a backbone known to work — an absolute
   loss is uninterpretable, since eps-prediction is trivial at high noise and impossible at low.
2. **the layout, by trying both** — `hwc` scored 0.0097 against `permuted`'s 0.3818 on identical
   weights, which settles it in one run.
3. **the range**, at every boundary: sampler out, detector in, probe in.

For a policy this list grows — action normalisation, chunk ordering, proprioception scaling — and
every one of them can produce a confident, wrong "the method does not transfer".

### 3. The verdict statistic has to be the thing you claim

N58 judged conditioning by the eps-loss gap between right and wrong classes and printed
**NOT USEFULLY CONDITIONAL** on a checkpoint composing at **16× the floor**. The gap is an *average*
denoising improvement; on an easy marginal it stays tiny however well the label steers. For a policy
the analogue is worse — average action MSE will be dominated by the easy majority of timesteps.
**Judge on task success or on the composition margin, never on the training loss.**

---

## Next, on Shapes3D specifically

1. ~~**The router does not transfer (AUROC 0.354, below chance).**~~ **RESOLVED — it was the u-band,
   exactly as this item guessed.** The sweep was cheap and it was the right first thing to try: at a
   calibrated `u` the same checkpoint reaches **0.861 ± 0.043** on held-out queries against a density
   ablation of 0.585, and the operating point itself is choosable **without held-out labels**
   (`router.calibrate_u_novel` recovers the oracle's u on 3/3 seeds). §6c-quinquies. What remains open
   is `U_VOCAB` — the three-way decision's second threshold has not been recalibrated the same way,
   and no result yet depends on it.
2. **Recover `object_hue`.** Pinning `wall_hue` shrank the set to 9000 images and cost that probe, so
   `joint` is a two-factor joint. More images per composition should restore it and make the headline
   a true three-factor number.
3. **EXTEND BASIS on Shapes3D.** §6e-bis predicts the tier that adds a *direction* is where
   representational gain must come from; Shapes3D has clean held-out factor *values* to test it on.
4. ~~**Sweep the class-table learning rate.**~~ **DONE — and it is now a rule** (§5t-ter,
   `figures/f_lr_ladder.png`). 1× reaches only **1.8×** the unconditional floor — it does not compose —
   while every rung from 5× up reaches 10–14×. 50× is the **plateau**, not a tuned peak (50× and 100×
   are identical at 0.542), so the setting was robust rather than lucky. Two things worth carrying to a
   policy: the ε-loss gap is ~0.0001 at *every* rung including the failing one, and **basis R² does not
   predict ADDRESS** (highest at 5×, where ADDRESS is near its worst). Neither the loss nor the R² is a
   usable health check; only the composition margin is.

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
across three datasets and two architectures. The **router** now transfers too, but only with its
operating point calibrated per backbone (§6c-quinquies) — and note that the calibration was validated
on two backbones, not on a policy, where the analogue of "class legibility" has still to be defined.
The **representation** question remains open with its headline withdrawn. A policy paper resting on
ADDRESS is supportable today; one resting on autonomous routing is now *plausible* rather than
unsupported, and the honest way to present it is with the operating-point calibration as part of the
method rather than as a hyperparameter that happened to work.

**Stage C-bis — a scene with several objects is now a solved shape, not an open one.** §7bb shows a
scene-conditional model whose table organises as a *sum over per-object addresses*, so an unseen
combination of known objects is generated from a composed address and costs no new storage. For a
policy this is the difference between "one address per task" and "one address per object, composed
per scene" — the second is what scales, and it is the formulation to carry into a manipulation
benchmark. Caveat it honestly: one seed, 9 held-out pairs, and 3 of them fail outright.

**Stage F — calibrate the operating point before reading any routing number.** This is new, and it is
the lesson with the highest chance of silently wasting a week: `U_NOVEL = 0.10` carried onto a
pretrained backbone reads **0.354**, below chance, and looks exactly like a method that does not work.
Run `router.calibrate_u_novel` on the policy first, and check `uncond_novel` against `s_novel` to see
whether conditioning or density is carrying the decision — the calibration only works in the former
regime, which is the one a pretrained policy should be in.
