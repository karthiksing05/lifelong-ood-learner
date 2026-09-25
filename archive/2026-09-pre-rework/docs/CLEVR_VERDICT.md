# CLEVR: why it does not fit the task as defined, and what would

Written 2026-08-14, after three separate attempts to bring CLEVR in. Recorded so it is not
re-litigated, and because the reason turns out to be more interesting than the answer.

---

## The short version

**Every distributed CLEVR set labels ONE object in a scene containing several.** Our task
(`docs/THE_TASK.md`) is *one composition = one image*: a query is an instance of a composition, and
generated samples are scored by whether they carry that composition. When the image holds four
objects and the label describes one of them, "does this sample have the right composition" has no
well-defined answer — a probe cannot know which object to read.

That is not a data-quality problem to be filtered around. It is a **different task**.

---

## What was actually checked

### Attempt 1 — a pretrained backbone (2026-08-13)

Ruled out on architecture. `nanlliu/compose-visual-relations` publishes CLEVR checkpoints
(Dropbox, 128 px) but the model is an EBM-style architecture conditioned on **relations**; every
CLEVR model on the HF Hub is a Stable-Diffusion LoRA conditioned on **text**. A class-embedding
factor basis reads neither. Shapes3D was taken instead.

### Attempt 2 — `clevr_rel_128.npz` (2.3 GB, 50,000 images)

Labels are `(50000, 11)`. Decoded: **cols 0–3 are a clean per-object description** — shape(3),
size(2), colour(8), material(2) = **96 compositions**, 430–600 images each. Cols 5–10 are a partial
relational description matching the repo's `--num_classes 4,3,9,3,3,7`, with 0 as a per-attribute
null.

The label is clean. The **images are not single-object**, and the nulls are per-attribute rather than
"no second object", so single-object scenes cannot be identified from the labels.

Two detectors were tried and both are too weak to build on:

| detector | result |
|---|---|
| spatial connected components | ~28 % flagged single-object, but **≈50 % precision** by inspection — it merges touching and occluding objects |
| distinct canonical colours | 16 % flagged, **72 %** agreement with the label's colour, only **74 of 96** compositions covered |

A benchmark built on a 50–72 % reliable filter would poison every number downstream.

### Attempt 3 — `clevr_pos_128.npz` (1.4 GB, 30,000 images)

The most promising lead: `obj_labels` is **exactly `(30000, 4)`** — shape, size, colour, material —
plus `coords_labels` for position. 96 compositions, median 361 images each. The composable-diffusion
method generates multi-object scenes by *composing single-object conditionals*, which suggested the
training data would be single-object.

**It is not.** Rendered samples show 2–5 objects with the label describing one of them.

### The one remaining path, and its cost

Generate single-object CLEVR from the original Blender scripts. That gives exact factor control and
guaranteed one-object scenes. **Blender is not installed on this box** (`blender` not on PATH, `bpy`
not importable), so it is a real setup task rather than a download.

---

## The reason this is worth knowing rather than just worth recording

**CLEVR's distinctive value over Shapes3D is precisely that it is multi-object** — and that is a
harder task than the one this repository has defined, not the same task with more factors.

* Shapes3D already covers the regime CLEVR would otherwise add: a 3D-rendered scene with discrete
  factors, one object, clean holdout. It works — ADDRESS at **16× the unconditional floor**
  (§5t-bis).
* Filtering CLEVR down to single-object scenes throws away the only thing it has that Shapes3D does
  not. It would cost days and produce a **worse Shapes3D**.

So the question CLEVR really poses is: **what happens when a scene contains several composed things
at once?** Our current formulation cannot express that — `slot` is one integer naming one
composition per image. Answering it needs a different formulation:

* a label that is a **set** of compositions, not one
* probes that are **detectors** rather than whole-image classifiers
* a coverage metric defined over sets
* and a storage question that becomes genuinely new: does a scene of *k* known objects cost *k*
  addresses, or does the scene itself become a composition over addresses?

**That last question is the interesting one, and it is the one a VLA actually faces** — a manipulation
scene has several objects and a policy must condition on the relevant one. It is a natural next
formulation rather than a detour, but it is a research direction, not a dataset swap.

---

## Recommendation

**Do not force CLEVR into the current contract.** Three datasets plus a DiT already establish the
claim across two architectures and three factor structures. If multi-object is wanted, take it as the
**next formulation** — with CLEVR as its natural benchmark, since the multi-object structure that
disqualifies it here is exactly what that formulation would need.

---

## Followed up: the formulation is now open (N63/N64, 2026-08-15)

The recommendation above was taken, on a substrate where the set label is exact **by construction**
rather than filtered out of someone else's renders: `lifelong_ood/data/multiobject.py` composites two
MNIST digits onto one canvas, so the label is the unordered set of digits present. Colour is
randomised per object so identity is carried by SHAPE, and the side is randomised so the class is
genuinely a set and a probe must be a detector. Everything this document asked for:

| what this doc said multi-object needs | what was built |
|---|---|
| a label that is a **set**, not one composition | `objs` is a sorted 2-subset; 45 scenes over 10 objects |
| probes that are **detectors** rather than whole-image classifiers | a 10-sigmoid CNN, trained on SEEN pairs and scored on HELD-OUT pairs — **top2 0.945** |
| a coverage metric defined over sets | exact-set match under top-2, plus per-object recall |
| the storage question | see below |

**On the storage question, there is a real answer at the representation level.** The scene-conditional
model gets one free embedding row per SCENE and is never told a scene contains objects — summing
per-object embeddings would answer the question by construction. Fitting a *shared* object table over
0/1 membership afterwards:

| | R² |
|---|---|
| object-membership basis | **0.841** |
| leave-one-scene-out CV | **0.711** |
| permutation null | 0.252 ± 0.040 (max 0.359) |
| ordered-pair control (2 tables, 2× the vectors) | 0.883 |

So the freely-learned scene table **does** organise as a sum over object addresses, and the shared
10-vector table nearly matches a 20-vector ordered control — the sharing is real, not a fit artefact.
*(Measured on the first backbone, `ddpm_multiobject.pt`, 36 scene rows. The table is a property of
the checkpoint, so these are re-measured on the longer run rather than carried over.)*

**And the generation-level claim now holds: a scene of *k* known objects costs NO new storage.**
ADDRESS composes a held-out PAIR from `mu + g*(W[a] + W[b])`, gain calibrated on seen scenes only:

| | value |
|---|---|
| self-conditioning control (model on its OWN trained row) | **0.889 ± 0.025** |
| held-out pairs, exact-set match | **0.317 ± 0.008** vs unconditional floor **0.002** |
| held-out pairs, per-object recall | **0.608 ± 0.008** vs floor **0.178** (3.4×) |

*(± is standard error over 3 sampling seeds on one backbone; training-seed variance is not measured.)*

**Read the recall ratio, not the exact-set ratio.** The exact-set floor is ~0.002 — an unconditional
model almost never lands on one specific pair out of 45 — so the ratio it produces is arithmetically
large and tells you nothing; one seed printed **329861.1×** from a floor of exactly zero. The recall
comparison (3.4×) and the absolute 0.317 are the honest numbers, and they carry a real limitation:
**4 of the 9 held-out pairs fail**, so this composes for most unseen pairs, not all.

Those failures are diagnosed rather than left as noise (RESULTS §7bb-bis). They are **one-sided** —
corr(|p_a − p_b|, exact) = −0.63 — and come in two kinds: two are *suppression*, which a larger gain
recovers (so 0.317 is a floor), and two are **destructive interference** between anti-aligned object
directions (cos −0.50 and −0.75, against −0.29 for the worst pair that still works), which no gain
recovers because *g* scales a sum that has already cancelled. Thin data and magnitude imbalance were
both tested and refuted. The useful part: the second kind is **predictable from a dot product before
anything is generated**.

### The control earned its keep, and the fault was mine

Before this, both arms generated pure noise and the model scored **0.000** conditioned on its own
trained row. The obvious reading — "the basis factorises but ADDRESS cannot compose" — would have
been a clean, plausible, entirely false negative about compositional storage. It was neither
undertraining nor the model:

* the backbone's eps-loss on real data is **0.0097**, matching its training loss and *better* than
  the ColorMNIST reference (0.0164–0.023), and its layout is confirmed (`hwc` 0.0097 vs `permuted`
  0.3818 on the same weights);
* the fault was a **clip range passed to the sampler**. The field works in MODEL space [-1,1], so
  `sample_embedding`'s default `clip=(-1,1)` is correct; passing the DATA range `(0.0,1.0)` clamped
  away the entire negative half at every reverse step. These scenes are ~90 % black background,
  which is **-1** in model space, so every trajectory was destroyed — and because the unconditional
  arm broke identically, the failure looked like a badly trained model rather than a broken reader.

A second wrong turn is worth recording with it: the diagnostic first declared the model "essentially
unconditional" on an eps-loss conditioning gap of +0.0003 — but the *reference* backbone, whose
router scores AUROC 1.000, has a gap of **+0.0001**. The gate would have condemned a backbone known
to work. That is §5t-ter's lesson recurring: **the eps-loss gap is not a verdict statistic**, and
adding a known-good reference is what caught it. All draws now route through one `draw()` helper so
the model/data conversion lives in a single place.
