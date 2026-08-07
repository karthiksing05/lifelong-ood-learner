# HANDOFF — state of `lifelong-ood-learner` as of 2026-08-07

Written to bootstrap a fresh session with no prior context. `RESULTS.md` (1505 lines) is the full
record with every number and every refuted hypothesis; this is the compressed version plus the
things a new session would otherwise have to rediscover the hard way.

---

## 1. The goal, unchanged

> Reframe lifelong learning as **repeated composition of prototypes** to create out-of-distribution
> prototypes, **sample from those prototypes** to recreate the OOD instances, and **add those
> instances to the existing manifold** — without degrading what is already there.

Heavy inspiration from `poe-diffusion.pdf` (Wang et al., *Test-Time Compositional Generalization in
Diffusion Models via Concept Discovery*), own architecture. All tests run via SLURM.

---

## 2. What is established (numbers, all multi-seed unless noted)

### The loop works end to end
Compose an OOD prototype → sample instances → consolidate into the manifold. Preservation is
**bit-exact**: `tests/test_unbounded.py` holds it against *all* previous concepts at every one of 12
batches out to 40 concepts, 45K params/concept, linear with 1.3% max residual, base drift +0.00000.

### We beat the published method decisively on coverage (N9, §5l)
| method | acc | **F1** | content div |
|---|---|---|---|
| *real held-out images (ceiling)* | *0.82* | *0.96* | *0.98* |
| **ours** | 0.41 | **0.50** | 1.11 |
| `poe-guided` (the paper's method) | **0.57** | **0.00** | 0.77 |
| `inversion` | 0.27 | 0.46 | 1.18 |
| `nearest-class` | 0.00 | 0.01 | 1.20 |

`poe-guided` scores *higher accuracy* and F1 0.00 because it emits near-copies of the query — right
attributes, zero coverage. That contrast is the central claim.

### Continual-learning baselines on real data (N5, §5k, 4 seeds)
| method | ACC | forgetting | carried |
|---|---|---|---|
| ours (frozen hierarchy) | **0.816 ± 0.034** | **0.000** | 8.82M |
| parameter isolation | 0.806 ± 0.030 | **0.000** | 7.16M |
| test-time training | 0.797 ± 0.041 | 0.004 | 7.16M + all data forever |
| EWC | 0.743 ± 0.032 | 0.058 | 7.16M |
| finetune | 0.725 ± 0.030 | 0.082 | 7.16M |
| generative replay | 0.651 ± 0.061 | 0.163 | 7.16M |

The top three are **statistically tied on accuracy**. The forgetting column is the result: arms that
*remove* shared parameters sit at exactly 0.000; arms that try to *protect* them all forget.

### Two mechanism findings that reshaped the project
- **§5m — a composed-pool stream cannot test forgetting.** Consolidating a factored pool delivers
  Δ‖B@A‖ = **0.000** to the shared path vs **7.219** for real images. It is self-distillation onto a
  target the model already generates. N3/N9's no-forgetting numbers show *nothing was at stake*.
- **§5n — the composition lives in the private embedding, not the adapter.** The shared path can
  triple (‖B@A‖ 7.49 → 22.28) with accuracy going 0.46 → 0.38; zeroing one 256-float embedding drops
  it to **0.04**. So there are two preservation mechanisms: embedding isolation preserves
  *compositions* (for every arm), path freezing preserves *absorbed real instances*.

### Composition is free; vocabulary is not (N11, §5q)
| | acc | **F1** | cost/concept |
|---|---|---|---|
| basis extension, **unseen** combinations | 0.662 | **0.755** | **0** |
| per-concept adapter | 0.628 | 0.658 | 551K |

One 256-float direction covers all 40 combinations using a factor value. **But** on a backbone that
never saw the value at all (§5q phase B), conditioning gives background accuracy **exactly 0.000** —
only real-data weight updates work (0.833). Routing rule: *new combination → basis, free; new
value → weights, from real data, never from the model's own samples.*

### The hierarchy does not earn its keep (N14, §5s)
Given the case built to favor it (families sharing two of three factors, back to back), `clustered`
is the **worst** arm (0.831) against `shuffled` (0.859) and `flat` (0.842); all transfers positive.
The tree forms (depth 3.2 vs 1.0) and shares (4 ancestors) — sharing that does not help.
**Recommendation: drop hierarchical placement, keep the insert-only discipline.**

### Alternatives to LoRA (N16/N17, §5v–§5w)
| route | ACC | verdict |
|---|---|---|
| frozen low-rank adapter | 0.837 | exact preservation, unbounded growth |
| **frozen orthogonal (Cayley) adapter** | **0.773 ± 0.048** | exact preservation, **2.8× less growth** (2.56M vs 7.16M) |
| merged low-rank + rehearsal | 0.378 | zero growth, forgets |
| merged orthogonal + rehearsal | 0.082 | zero growth, barely learns |
| hypernetwork (generated delta) | 0.072 | zero growth, does not fit |
| classifier guidance | digit 0.26 peak | never controls colour at any scale |

**Four independent failures at zero-growth absorption.** The consistent explanation:
**conditionality, not size, is what buys preservation** — a delta that can be switched off for
concepts it was not trained for cannot damage them.

---

## 3. Recommended architecture (what the evidence supports)

1. **Composition** — ideal-words factor basis in the model's embedding space (R² 0.826), read off
   the query, per-factor gain **calibrated on the backbone** (`experiments/common.py:backbone_gains`,
   never a hardcoded constant).
2. **New combinations of known factor values** — basis extension. Zero parameters, beats adapters,
   preservation structural (read-only basis).
3. **New vocabulary only** — a **kept orthogonal delta**, trained on **real instances**.
4. **No hierarchy.** Flat, insert-only.

That is 2.8× smaller than the current method, free for the combinatorial majority, exact
preservation.

---

## 4. Refuted / corrected — do not re-derive these

- **The interference hypothesis.** First "refuted" by `diag_interference_regime.py`, which drew its
  pool from a class branch then consolidated under a *fresh* embedding (first loss 0.0445 > real
  data's 0.0183 — its "on-manifold" pool was not), and measured ‖Δ‖ over raw LoRA factors including
  `A`, which moves freely while `B`≈0. Corrected measurement **confirms** it. **Always measure
  ‖B@A‖, and always sample the pool from the embedding consolidation will condition on.**
- **factored vs inversion flipped between backbones** (0.391/0.426 → 0.57/0.55). That comparison is
  **unsupported in either direction**. What does reproduce: factored reads (d,f,b) correctly, needs
  ~18 weights instead of 120, higher content diversity.
- **The embedding-overwrite fix.** Real inconsistency (adapter fit against a conditioning discarded
  before sampling), but pinning it measured *no* improvement (N9 0.41/0.50 → 0.36/0.44, matched
  config, one seed each). Kept for consistency, not for a gain.
- **Orthogonal merging.** Predicted to fix merging via norm preservation. **Wrong** — 0.082 vs
  low-rank's 0.378. Merging's damage is loss of conditionality, not norm distortion.
- **"My guided sampler is broken."** It was not. At guidance scale 0 the model samples
  *unconditionally*, and unconditional samples scored against one specific target **must** score at
  chance (digit 0.09, fg/bg 0.28 for 10/4/4 classes). That is correct behaviour, not a bug.

---

## 5. Open problems, in priority order

1. **CelebA conditioning is too weak to compose on** (§5o). Two backbones; scaling 2.3× data and
   140 epochs moved conditional/ceiling from 34% → 33%. **Not undertraining** — likely structural
   (14-attribute classes with ~25 examples, spatially overlapping attributes). Every conclusion
   still rests on ColorMNIST, where factors separate cleanly. **This is the biggest threat to the
   paper.**
2. **Pretrained transfer is untested, not refuted** (§5u). `google/ddpm-celebahq-256` is
   unconditional; a conditioning slot was added on the time embedding (zero vector reproduces the
   model at 0.000e+00) and the basis built from data. `composed` (0.146) = `uncond` (0.146), but
   `per-combo` — no composition involved — is also barely above unconditional (0.167). **The
   bottleneck is the conditioning channel, not the composer.** Next: more conditioning capacity
   (projection into block-level conditioning, not one additive vector) and a budget matched to a
   113.7M model.
3. **Hypernetwork was tested only as FiLM channel-scaling.** Two explanations remain unseparated:
   shared trunk cannot hold per-concept corrections, or channel scaling is too weak a delta.
   Generating a **low-rank** delta instead would separate them.
4. **N13 stream shows a forgetting gradient** (0.52 newest → 0.00 oldest) under repeated merges.
   Expected given §5r; the fix is to not merge (use kept orthogonal deltas).
5. **Narrow open mechanism question** (§5m): the composed-pool gradient at shared nodes is
   *bit-exactly* zero, not merely small — points at forward-pass routing, not residual size. Affects
   no conclusion.

---

## 6. Repo state

**Path** `/coc/flash10/ksingara3/lifelong-ood-learner` · remote
`github.com/karthiksing05/lifelong-ood-learner`, branch `main`.
**61 commits, NONE PUSHED** — no credentials on this machine. Requires `gh auth login`.

**Docs**: `RESULTS.md` (1505 lines, §5a–§5w + replication analysis + threats), `PLAN.md` (498),
`README.md` (388).

**Backbones** (`results/cache/`, all gitignored):
| file | what |
|---|---|
| `ddpm_colormnist.pt` | **the main one** — 200 epochs / 200 per slot, cond acc 0.891, R² 0.826 |
| `ddpm_colormnist_nobg3.pt` | background 3 withheld entirely (for basis-extension phase B) |
| `ddpm_colormnist_stream.pt` | bg3 + fg3 + digits 8,9 withheld, 56 classes, cond acc 0.953 |
| `ddpm_celeba.pt` | 140 ep / 400 per class, 64px — conditioning too weak to use |

**Experiments** `n1_train` (backbone; `--holdout-bg/-fg/-digits` withhold whole factor values) ·
`n3_lifelong` · `n4_batch` · `n5_classinc` (6 CL baselines) · `n8_idealwords` · `n9_showdown`
(case-study figure) · `n10_celeba` · `n11_basis_extension` · `n12_reconsolidate` · `n13_stream` ·
`n14_hierarchy` · `n15_pretrained_celeba` · `n16_alternatives` · `n17_ortho_merge`.

**Key modules** `neural/idealwords.py` (factorize, compose, learn_factored, **extend_basis**,
**learn_attribute_basis**) · `neural/adapters.py` (PathLoRA, bank, **merge_concept**,
**activation_subspace**) · `neural/alternatives.py` (**HyperDelta**, **OrthoDelta**,
**GuidanceHead**) · `neural/pretrained.py` (diffusers wrapper + added conditioning slot) ·
`neural/baselines.py` (EWC, TTT) · `celeba.py`.

**Tests** (all pass): `test_smoke`, `test_neural`, `test_unbounded`, `test_merge`,
`test_ortho_merge`, `test_pretrained`. Plus 16 `diag_*.py` diagnostics that print rather than
assert — **§5m and the gain calibration are not covered by an assertion**, so a regression there
would show up as quietly worse numbers.

---

## 7. Cluster gotchas that cost real time

- **Everything runs via `sbatch experiments/run_gpu.sh`** with `--export=ALL,SCRIPT="..."`. Logs land
  in `experiments/slurm/gpu-<jobid>.{out,err}`. The login node has **no torch** — `make_figures.py`
  and anything importing torch must go through SLURM.
- **Clear `__pycache__` and `sync` before submitting.** Stale NFS reads silently run old code.
- **`squeue` can hang** when the controller is unhealthy — always wrap in `timeout`.
- **`n1_train.py --out` once overwrote the ColorMNIST backbone** while training CelebA (it defaulted
  to a fixed name). Fixed to `ddpm_<dataset>.pt`, but the lesson is broader: the original was 200
  epochs / 200 per slot and my first retrain used 60/100, which collapsed the whole method
  (N9 `ours` 0.42 → 0.10). **Backbone quality dominates everything.**
- **Sample via `L.sample_concept(name)`**, which activates the adapter path — `sample_factored_pool`
  on the raw field silently omits the adapter.
- **Pretrained diffusers models must be integrated on their native schedule.** Walking our cosine
  u-grid produced a green colour cast (R 0.338/G 0.365/B 0.177 vs reference 0.531/0.477/0.232).
- `HF_HOME=/coc/flash10/ksingara3/.cache/huggingface` is exported in `run_gpu.sh`; `diffusers` is
  installed in the `chunks` env only.

---

## 8. If I were writing the plan

The defensible paper today is **"certified-preserving OOD composition by basis extension, with rare
weight updates for new vocabulary"** — not "hierarchical lifelong learning". The hierarchy has no
supporting evidence and should come out. The strongest single claim is §5q + §5w: combinations are
free and beat adapters, vocabulary costs a kept orthogonal delta 2.8× smaller than LoRA, and
preservation is exact throughout.

The one thing that would most change the paper's standing is **CelebA (item 5.1)** — every result is
currently on a dataset whose factors separate cleanly, which is exactly the assumption the method
needs. That is where I would spend the next block of compute.
