# What the task actually is

Written 2026-08-14. Read this before any results document. Every experiment here scores the same
task, and mistaking it for one of two nearby tasks makes half the design choices look arbitrary.

---

## The task, in one line

**Given ONE example of a combination the model has never seen, produce NEW instances of that
combination, store it for good, and damage nothing already learned.**

| | |
|---|---|
| **Input** | one real image of a held-out composition — 0 training rows and no class-embedding row for it, asserted against the checkpoint by `diag_holdout_leak` |
| **Output** | **new samples** of that composition |
| **Technique** | read the composition into the model's conditioning space → store it at the cheapest tier that can hold it → generate from the model's **own conditional** |

The query identifies *which* composition is wanted. It is not a target to be reproduced.

---

## What it is NOT

### It is not reconstruction

The output is not meant to resemble the query. It is meant to cover the *class* the query is an
instance of. This is not a stylistic preference — it is the axis the method is defined on, and the
baselines fail exactly here:

| arm | joint accuracy | recall |
|---|---|---|
| `poe-guided` (Wang et al.) | 0.958 | **0.000** |
| `query-only` | 0.958 | **0.000** |
| **ours** | lower | **0.73 – 0.99** |

`poe-guided` and `query-only` are *more accurate than us* and cover **nothing**: they re-emit the
query. A method that scored 1.000 on accuracy by returning the input would be worthless for the goal,
which is why **recall is a headline number and not a diagnostic**. RESULTS §5e-quater works through
why their teacher does this: the PoE teacher is a product of Gaussians, hence itself a single
Gaussian, and under a deterministic DDIM (η=0) sampler one query determines one image — measured at
**vendi 1.06 over 24 draws**. Their §4.2 gets diversity from running 100 different queries.

Ours reads μ_T only to identify which factor values the composition needs, writes them as
`c_new = μ + Σ_f g_f(w_f·V_f)`, and samples `p_θ(x | c_new)` — the model's own conditional, which is
multimodal where the teacher is one Gaussian. That single difference is why one query suffices:
Generalization recall **0.648 at one query** against PoE's **0.073**.

### It is not representation learning

N50 asked whether an absorbed concept yields *better internal representations*. Its headline is
**withdrawn** (the LoRA adapters were inert during the readout — RESULTS §6e) and N53 is the corrected
experiment. It is a probe on the side, not the task. §6e-bis explains why the two are related but
distinct: ADDRESS adds a **point** in the existing span (affine residual 0.0000 against a trained
embedding's 0.7359), so it can place a concept correctly while adding no new axis for a readout to
exploit. Composition-strong and representation-flat are one fact, not a tension.

---

## The four axes, all load-bearing

1. **Can you produce the held-out combination?** — joint accuracy, probe-read per factor
2. **Do you cover its distribution rather than copy the one example?** — recall / precision
3. **Did absorbing it damage anything?** — bit-identical preservation, FGT 0.000
4. **What did it cost?** — ADDRESS 72 B · EXTEND BASIS 1 KB · BLOCK 69 K

Axis 4 is the thesis. Axes 1–3 are what stop it being trivially satisfiable: storing nothing costs
nothing, and copying the query is perfectly accurate.

---

## Where it has been shown

| dataset / backbone | composition | note |
|---|---|---|
| ColorMNIST (UNet, from scratch) | joint 0.556, F1 0.630 | the calm explanation |
| CelebA (the companion paper's own 128 px backbone) | **2.7× / 2.3×** the unconditional floor, every class | overlapping factors |
| Shapes3D (**pretrained** unconditional backbone, VE→VP bridged) | **16×** the floor, floor_hue 0.953 | the setting closest to VLA |
| ColorMNIST (**latent DiT**) | full loop 0.661 from one query, basis R² 0.832 | not tied to the UNet |

The mechanism is the same in all four: the class-embedding table organises compositionally without
ever being told the factors (`factorize` recovers R² 0.538–0.870 afterwards), and ADDRESS composes a
held-out point in that recovered basis.

---

## The VLA translation

The mapping is one-to-one, which is the reason to care about the framing at all:

| here | in a diffusion policy |
|---|---|
| one image of an unseen (digit, fg, bg) | one demonstration of an unseen (object, container, verb) |
| **new samples** of that composition | **new trajectories** for that task — not replaying the demo |
| joint accuracy | rollout success |
| recall | generalisation *within* the task — new positions, new starts |
| bit-identical preservation | previously learned tasks still succeed |
| 72 bytes | what the new skill cost to keep |

**The claim being defended, in the form it must take for policies:** experiences accumulate as
*addresses in a compositional space*, so cost tracks the **vocabulary of the world** rather than the
**length of the stream**, and nothing already learned is overwritten.

`docs/VLA_PATH.md` carries the staged plan and — importantly — which parts of this have transferred
and which have not.
