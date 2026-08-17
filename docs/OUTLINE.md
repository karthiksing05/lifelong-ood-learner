# The repository, organised as the target outline

Written 2026-08-13. This is the structural map: every item of the target outline, what evidence
exists for it, which figure carries it, and what is missing. `docs/FINAL_REPORT.md` is the argument,
`docs/RESULTS.md` is the evidence, `docs/NEXT_STEPS.md` is the working state. **This file is the
index between the outline and those three.**

Status key: **✔** measured and written up · **◐** partial · **✘** not done · **⊘** deliberately
dropped, with the reason.

---

## Framing

> *market the solution as lifelong learning via a theory of composition, NOT as an exploration of
> composition within diffusion models*

**✔ Adopted throughout.** README leads with the storage ladder and the lifelong loop; composition
appears as the mechanism that makes the ladder possible. The diffusion-composition commentary is
confined to §5e-* and the Stage 4 comparison, where it is a *reproduction* of the companion paper
rather than a claim of our own.

> *do we need to talk about compositionality? I think no*

**✔** No section argues for compositionality in the abstract. Every composition number exists to
support a storage or lifelong claim, or to reproduce Wang et al.

> *maintaining the importance of hierarchy*

**⊘ Dropped, on measurement and on your later instruction.** §24 measures the hierarchy as *adding*
parameters over flat — every node costs a full block, so depth does not amortise — and you said
directly, right after sending this outline the first time, "don't worry about the hierarchy, that's
an important thing that we leave in the past." §24 is kept as the recorded negative. **This affects
three outline items**, marked ⊘ below: the hierarchy core claim, the hierarchy visualisation case
study, and the flat-vs-hierarchy ablation. If you want it back as a claim, the blocker is that the
evidence currently contradicts it.

---

## Contributions

| # | contribution | status | evidence | figure |
|---|---|---|---|---|
| **C1** | Stages 1–3 unchanged from Wang et al.; **Stage 4 differs** — the teacher only *locates* the concept, the model's own conditional generates | **✔** | README "Stage 4…", §5e-bis, §5e-ter, N49 | **f0**, f2, f29 |
| **C2** | **iteratively select concepts** rather than picking K from a fixed pool | **✔ built, ✘ no gain** | §5s-bis | — |
| **C4** | *(added)* the scheme is **not tied to the UNet** — it transfers to a latent DiT | **✔** | §7z | `figures/initial/dit/` |
| **C3** | three-tier storage decided by `s_novel` / `s_vocab` | **✔** | §2, §6c, §25, §5q | **f16**, f7, f8, **f32** |

**C1 is the load-bearing one and it is done.** Their Eq. 13 guides toward a PoE teacher that is a
*product of Gaussians, hence itself a single Gaussian*, under a **deterministic DDIM η=0** sampler —
so the teacher determines the trajectory and one query yields one image (measured: **vendi 1.06** from
24 draws). Their §4.2 gets diversity from **100 queries per class**. We read μ_T only to identify
which factor values the composition needs, write them as `c_new = μ + Σ_f g_f(w_f·V_f)`, and sample
from `p_θ(x|c_new)` — multimodal where q_T is one Gaussian. On their own axis: Generalization recall
**0.648 at one query** against PoE's **0.073**; PoE needs ~100 queries to reach 0.300.

**C2 is built and does not pay for itself (§5s-bis).** `compose.iterative_teacher` interleaves ascent
and selection — pick one, see what the partial teacher still fails to explain, grow the next chains
from there. Budget-matched against the paper's Eq. 8–9 route it scores **0.656 against 0.664, a
0.04-pooled-s.e. difference**, and all three regrow directions land on *the same number to three
decimals* — including the control that does not condition on the selection at all. The direction of
regrowth is inert, not weak.

The diagnosis relocates the bottleneck usefully: `digit` and `fg` are **1.000 for every arm** and the
entire difference sits in `bg`, the factor §5q shows the backbone under-responds to. Discovery is not
what is failing, so no amount of better-aimed discovery fixes it — which also predicts the paper's own
remedy (a 20× bigger pool) would not fix this particular error. **The outline should not list C2 as a
feature.** Power is low (8 compositions, s.e. 0.134), so this rules out a large effect, not a small
one; the flatness across three variants is the stronger evidence.

---

## Datasets

| dataset | status | note |
|---|---|---|
| **ColorMNIST** | **✔** | the calm explanation; 3 named factors, disjoint pixels |
| **CelebA** | **✔** | overlapping factors, 14 binary attributes; also the companion paper's own 128px backbone |
| **Shapes3D** | **✔ works** | ADDRESS **0.583 ± 0.113** against an unconditional floor of **0.036 — 16×**, floor_hue 0.953, basis R² 0.538, on a pretrained backbone (`eurecom-ds/scoresdeve-ema-shapes3d-64`, VE→VP bridged) with the class table on a **50× learning rate**. Four earlier recipes failed and the reason is the lesson: a strong pretrained backbone reaches loss 0.0015 in minutes, leaving no residual for the label to explain, so the conditioning pathway must be trained on its own schedule. Two bugs of mine (sampler prior centred at 0.5 instead of 0; HWC rows read as CHW, a 974× loss error) produced an earlier 'cannot work' verdict that is **withdrawn**. §5t-bis. The **router transfers once its operating point is calibrated** — the inherited u=0.10 gives 0.354, a calibrated u gives **0.861 ± 0.043** on held-out queries (density ablation 0.585), and `router.calibrate_u_novel` finds that u from training data alone, matching the oracle on 3/3 seeds. §6c-quinquies. |
| **CLEVR** | **✘ different task, not a dataset gap** | Both distributed sets label ONE object in a scene of 2–5 (`clevr_rel` 50k, `clevr_pos` 30k — the latter has a clean 4-factor `obj_labels`, but the scenes are still cluttered). Two single-object detectors were tried and rejected: connected components ≈50 % precise, colour clusters 72 % with 74/96 compositions. Blender is not installed, so regenerating is a setup task. **The point:** CLEVR's value over Shapes3D *is* that it is multi-object, which is a harder task than 'one composition = one image', not the same one with more factors. See `docs/CLEVR_VERDICT.md`. |

"Strong performance" now holds on **three** datasets — ColorMNIST, CelebA and Shapes3D — plus a latent DiT on ColorMNIST. `docs/VLA_PATH.md` carries the forward plan and what each dataset taught that the others could not.

---

## Performance tests

| test | status | headline | figure |
|---|---|---|---|
| **Representational strength** — linear probe, k-NN, vs a model trained on everything | **✔ negative** | `ours` 0.271 ≈ `base` 0.266; oracle 0.406 (linear digit). Absorbing a concept does **not** measurably improve the representation over not adapting. | **f33** |
| **Incremental strength** — learning curves, no catastrophic forgetting | **✔** | retention **exactly flat**, FGT **0.000**, bit-identical 1.00; `finetune` falls 0.73→0.66 in one step | **f31**, f5 |
| **OOD strength** — FID / precision-recall, their metrics | **✔** | their protocol reproduces (G-precision 0.753 vs their 0.745); ours is the only arm on both coverage and accuracy | **f26**, f29, f3, f18 |
| **Preservation on ALL settings** | **✔** | bit-identical by sha256 + FGT 0.000 on Shapes3D (N59) and the DiT (N60), joining ColorMNIST (N5) and CelebA (N22). ADDRESS tier only — EXTEND/BLOCK preservation rests on the sealing discipline and is measured on ColorMNIST. | §7aa |
| **Cost curve on ALL settings** | **✔** | 9.5× / 8.9× / 12.4× against block-every; Shapes3D's vocabulary saturates at experience **11 of 100**, CelebA's at 21 of 1000 — after which the stream is free | §7aa |
| **Batch-size invariance** | **✔** | ACC flat in K within error; **1.98× speedup at K=12**; bit-identical at every K | **f34** |
| ↳ *instances per concept* (the other reading) | **✔** | the tier a concept warrants scales with instance count; ~15 justify ADDRESS, ~36 EXTEND | **f32** |
| **Parameter count vs others** | **✔** | ADDRESS **31,895×** cheaper than a block; 24 concepts = 1.7 KB vs 55 MB | **f16**, f8 |
| **Training time vs others** | **◐** | wall-clock is recorded per arm in §7/§16 tables and in f34, but there is no single cross-method timing figure | f34 |

---

## Case studies

| study | status | figure |
|---|---|---|
| instances sampled at **various distances from prototypes**, vs other solutions | **✔** | **f9** (ColorMNIST), **f26** (CelebA, paper's own backbone), f11 (the dial swept) |
| visualisation of the **hierarchy** stored | **⊘** | dropped with the hierarchy claim; f13/f14 exist and are cited only from §24's negative |

The distance-from-prototype dial is `--dial s`. **Default 0.8 since 2026-08-13**, set on colour
faithfulness in the renders rather than on the metric, which prefers 0.9 (§20b: 3 seeds, 0.8 is >4
s.e. worse; §20c records the cost — coverage roughly halves, and one displayed block goes to recall
0.000). Scope is one constant in one script; `sample_concept` still defaults to `u_start=1.0`, so **no
table moves**.

---

## Ablations and justifications

| ablation | status | where |
|---|---|---|
| flat vs hierarchy — "why are hierarchies important?" | **⊘ answered, negatively** | §24: hierarchy *adds* parameters; `independent` ties it |
| basis direction vs per-concept adapter | **✔** | §5q2: F1 0.756 vs 0.662 at **2153× less cost**, matched run |
| router calibration, six attempts | **✔** | §6c, §6c-bis, §6c-ter, **f32** |
| input layout / renderer faults | **✔** | §20, §30d, f20, f25, f30 |
| guidance and dial operating points | **✔** | §20b, §20c |
| pool guidance, discovery budget, severity | **✔** | `diag_ours_quality`, §5e-*, `n1 --extra-holdout` |

---

## What is missing, in priority order

1. **Shapes3D end-to-end.** The loader and holdout are in and asserted; the backbone is training on
   the existing n1 recipe. Then composition + router on it, which is what actually demonstrates the
   contract rather than asserting it.
2. **C2 needs a different idea, or dropping.** Built and null (§5s-bis). The diagnosis says discovery
   is not the bottleneck — background is — so the next attempt should target the weak factor, not the
   pool. Options: per-factor gain re-calibration during selection, or selecting against a
   factor-balanced objective rather than a per-dimension max.
3. **A cross-method training-time figure**, if timing is to be a headline rather than a column.
4. **Accumulate-then-promote** streaming router (NEXT_STEPS §2a) — the follow-on §6c-ter makes precise.

---

## The core figure set

Graphs, curves and case studies the outline asks for. **Every figure in `figures/initial/` is cited from at
least one document** — that is asserted, not assumed, and re-checkable in one line:

```bash
for f in figures/initial/f*.png; do b=$(basename $f); \
  [ "$(grep -rl "$b" docs/*.md README.md | wc -l)" -eq 0 ] && echo "UNCITED: $b"; done
```

The hierarchy panels (f13, f13b, f14, f14_cu) are **kept, not archived**, because §24 cites them as
the evidence for the negative. They are outside the core set but they are not orphans.

**Architecture** — `01_architecture`, `02_lifelong_loop`
**Storage / cost** — `03_storage_ladder`, `04_growth_per_concept`, `05_batch_invariance`
**Curves** — `06_learning_curves`, `07_router_exchange_rate`
**Composition / OOD** — `08_coverage_vs_copying`, `case_celeba`
**Representation** — `09_representation`
**Case studies** — `case_colormnist`, `case_celeba`, `case_shapes3d`, `case_prototype_distance`
**Preservation / parity** — `10_preservation`, `11_native_parity`
**The DiT track** — `figures/initial/dit/` — d1 AE reconstruction, d2 training + ADDRESS, d3 DiT case study,
d4 UNet vs DiT, d5 prototypes
**Everything else** — `figures/initial/extra/` (supporting evidence, ablations, fault post-mortems) and
`figures/initial/archive/` (superseded, gitignored)
