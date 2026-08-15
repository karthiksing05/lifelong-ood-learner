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

**The generation-level claim is not yet measurable, and the control is what caught it.** Conditioned
on its OWN trained row the first backbone scored 0.000, and the figure showed why: even unconditional
samples were speckle. Undertrained, not miscomposed. A longer run is in progress; until it reproduces
its own training scenes on demand, no ADDRESS number from it means anything. That control exists
precisely because a low composition score otherwise has three indistinguishable causes.
