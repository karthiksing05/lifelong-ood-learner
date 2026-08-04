# PLAN — lifelong-ood-learner

Working plan and handoff document. Read top-to-bottom before touching code.
Last updated: 2026-08-03 (session 2).

## 1. The contribution (do not drift from this)

**Continually adding compositions — including out-of-distribution compositions — to a learned
generative manifold, forever, without the manifold degrading.**

Everything else in the repo is in service of that sentence. The loop:

```
loop forever:
  1. meet ONE out-of-distribution query x_q
  2. COMPOSE known prototypes into an OOD prototype        (product-of-experts, poe-diffusion Eq. 7-10)
  3. SAMPLE that prototype through the learned manifold     (analytic CFG, poe-diffusion Eq. 12-14)
  4. CONSOLIDATE the samples into the manifold as a new, additive, non-destructive concept
```

Steps 1–3 are the poe-diffusion paper (`poe-diffusion.pdf`), reimplemented faithfully. **Step 4 is
ours.** The paper explicitly stops here — it distils one flat `(LoRA, c_new)` pair per query,
discards it, and names the gap in its own conclusion: *"human-level intelligence also requires
learning new primitives and incorporating them into future compositions … A full mechanism for
detecting, consolidating, and reusing genuinely new primitives remains an important direction for
future work."*

### Two backbones, one method

`BaseScoreField` (in `score.py`) defines a diffusion manifold by ONE primitive, `eps(x, u)`.
Sampling, mode ascent and the Hutchinson Laplace covariance are written against it once. Two
backbones implement it, and `compose.py` + the lifelong loop run unchanged on both — which is what
lets results be attributed to the *method* rather than to the backbone.

| | `ScoreField` (analytic) | `NeuralScoreField` (neural) |
|---|---|---|
| what it is | a Cobweb tree read as a Gaussian mixture; score, Tweedie and eps in closed form | a trained class-conditional DDPM (UNet) |
| training | incremental `ifit`, no gradients anywhere | ~3.5 h on one GPU |
| concept discovery | free — the tree's nodes ARE the modes of `p_t` (validated in E1) | mode ascent on the learned score (the paper's route) |
| consolidation | a separate adapter tree, merged at snapshot level | `HierarchicalAdapterBank`: LoRA leaves in an insert-only hierarchy |
| **the catch** | **a Gaussian mixture has essentially ZERO density at an unseen composition**, so its unconditional score fights the composition (measured: 21% of teacher accuracy retained under the paper's guidance schedule, 71% under an annealed one) | a convolutional UNet shares weights across colours and positions, so it holds real density at unseen combinations |

That measured hole is *why* the neural backbone exists — not a change of direction.

### The consolidation mechanism (the novel part)

`neural/adapters.py::HierarchicalAdapterBank`. Concepts are leaves of an **insert-only** hierarchy
built Cobweb-style over their PoE teacher means; a concept's effective adapter is the **sum of the
LoRA deltas along its root→leaf path**.

- **No forgetting, structurally.** Ancestors freeze the moment they are created; inserting a concept
  trains only its new leaf. An existing concept's path is therefore bit-identical after any number
  of later insertions. Verified in the *strong* form — not "accuracy stayed similar" but "the
  epsilon output on fixed noise is bit-identical".
- **Forward transfer, structurally.** A new leaf inherits its ancestors' accumulated delta as a warm
  start, so a concept landing near relatives only learns its residual. Flat per-query adapters
  (the paper's) cannot do this.
- The analytic backbone gets the same guarantee from mixture additivity, with drift bounded
  analytically by `log(1 + n_adapter/n_base)`.

Sources: `poe-diffusion.pdf`; `../cobweb-poincare` (palette/splits/prior_var conventions — this repo
complements it, does not deprecate it); `../Chunkformer/cobweb-private` (the real Cobweb; the numpy
tree in `tree.py` is only a portability fallback).

## 2. Status

> Full results, with the paper-replication analysis, live in [`RESULTS.md`](RESULTS.md).

### Verified and working

| piece | file | evidence |
|---|---|---|
| ColorMNIST benchmark, palette/splits identical to cobweb-poincare & paper App. B | `datasets.py` | ✅ |
| cobweb-private backend + numpy fallback, one `snapshot()` interface | `backend.py`, `tree.py` | ✅ both smoke-tested |
| `BaseScoreField` interface; analytic mixture field | `score.py` | ✅ |
| PoE composition: pools, submodular selection, per-dim PoE, teacher eps, guidance schedules | `compose.py` | ✅ |
| analytic lifelong loop (frozen base + merged adapter snapshots) | `lifelong.py`, `backend.merge_snapshots` | ✅ |
| neural backbone: UNet, DDPM training, `NeuralScoreField` | `neural/unet.py`, `neural/ddpm.py` | ✅ trains, generates |
| **hierarchical adapter bank** | `neural/adapters.py`, `neural/lifelong_neural.py` | ✅ `tests/test_neural.py` |
| tuning-of-record | `experiments/e0_tune.py` | ✅ chose tau, K, guidance schedule |
| E1 manifold validity | `experiments/e1_manifold.py` | ✅ **full scale** |
| E2 OOD composition | `experiments/e2_composition.py` | ✅ **full scale, all 40 slots** |
| E3 analytic lifelong | `experiments/e3_lifelong.py` | ✅ **full scale, 12 concepts, both strategies** |
| E4 composition closure | `experiments/e4_closure.py` | ⚠️ runs; hypothesis needs redesign, see §3.2 |
| N1/N2/N3 neural experiments | `experiments/n{1,2,3}_*.py` | ✅ **all complete** (N1 trained 146 min; N3 10 concepts × 3 arms) |
| N0 neural sampler diagnosis | `experiments/n0_tune.py` | ✅ found the teacher-covariance bias |
| per-concept retention analysis | `experiments/analyze_n3.py` | ✅ |
| multi-seed pooling | `experiments/aggregate.py` | ✅ (no multi-seed runs yet — see §3.0) |

### Headline numbers so far

**E1 (full scale, 4800 instances, 7178-node tree, 1589 concepts, depth 11).** The tree IS a
diffusion manifold: unconditional DDIM samples land in *seen* compositions at **rate 1.000** for both
the leaf field and the depth-annealed frontier field; denoise R² 0.995. And the mode hierarchy is
real — ascending `p_t` and matching to the nearest tree node gives

| noise `u` | 0.10 | 0.25 | 0.40 | 0.60 | 0.80 |
|---|---|---|---|---|---|
| matched node depth | 7.12 | 7.25 | 7.25 | 7.25 | **1.62** |
| match RMSE | 0.0002 | 0.0006 | 0.0011 | 0.0076 | 0.0296 |
| fraction that are leaves | 1.00 | 1.00 | 1.00 | 0.88 | **0.00** |

i.e. at low noise the modes are *exactly* the tree's leaves (RMSE 2e-4), and at high noise they move
to shallow internal nodes. The hierarchy Cobweb built incrementally is the density-mode hierarchy the
paper recovers from a DDPM by gradient ascent (their Appendix A) — here explicit and free.

**E0 (full scale, all 40 held-out slots).** See §4 for the full story.

| K=3 teacher | joint | digit | fg | bg |
|---|---|---|---|---|
| paper's fixed τ=0.5 | 0.300 | 0.80 | 1.00 | **0.38** |
| adaptive temperature | **0.800** | 0.80 | 0.97 | **1.00** |
| *probe ceiling on real images* | *0.775* | *0.82* | *0.97* | *0.97* |

The teacher scoring slightly above the ceiling is expected, not a bug: μ_T is a clean prototype and
the probes read it more easily than a real noisy image. τ=0.02 fixed and τ=0.05 adaptive tie at
0.800; adaptive is kept because it is scale-free and transfers.

Sampler, on the best teacher: annealed guidance retains **89%** of teacher accuracy (joint 0.740),
the paper's variance-aware schedule **78%** (0.646), constant w=1.5 **85%**.

⚠️ **Scale-dependence worth knowing.** At quick scale (12/slot) the same comparison read 71% vs
**21%** — a far bigger gap. The density hole that annealing compensates for *shrinks as the mixture
gets denser*, so the advantage of annealing is largest on small trees. Any claim about the schedule
must state the scale it was measured at. This is also direct evidence for the argument that
motivates the neural backbone.

**E2 (full scale, complete — see RESULTS.md §4 for the final 40-slot table).** Per-slot joint accuracy over the first 11 held-out slots:
`poe_tree` 0.79 / 0.00 / 1.00 / 0.96 / 0.96 / 0.46 / 1.00 / 0.71 / 0.25 / 0.83 / 0.00, with
`poe_ascent` tracking it closely (0.71 / 0.08 / 1.00 / 1.00 / 0.88 / 0.54 / 1.00 / 0.00) and
`top1`/`top3` at essentially 0.00 throughout. That is the paper's qualitative result reproduced, plus
the extra claim this repo can make: **closed-form tree discovery ≈ the paper's mode-ascent
discovery**, so discovery is free. `query_only` scores high by memorizing, which is exactly what the
geometry triple (novelty / diversity / gen_rmse) is reported to expose.

Two slots read 0.00 for *every* method including `query_only` — those are probe failures on the real
image, not method failures, which is why the per-slot table matters and the mean alone would mislead.

**E3 (partial, full scale).** Per consolidation: drift in held-out seen log-likelihood ≈ **−0.01 to
−0.05 nats**, drift in denoise R² **+0.0000**, unconditional validity stays 0.94–1.00. Non-degradation
is holding at full scale.

**N1 (neural backbone, trained).** 6.8M-param UNet, 200 epochs, 146 min on one GPU. Unconditional
valid-composition rate **0.938**; conditional joint accuracy **0.789** on requested seen slots, level
with the probe ceiling. It generates clean digits in the requested colours.

**N2 (composition on the neural backbone, 12 slots).** `poe` 0.674, `query_only` **0.833**,
`top1`/`top3` 0.000. **query_only beats PoE here** — a negative result, and the opposite of the
analytic backbone. When the backbone generalizes well enough to re-render a single query, concept
discovery is not obviously buying anything on this benchmark.

**N3 (the headline — 10 OOD concepts consolidated in sequence, trained backbone).**

| arm | bit-identical retention | per-concept drift from birth | accuracy range across steps | plasticity | base drift |
|---|---|---|---|---|---|
| `hier` | **1.00** | **+0.021** | **0.000** | 0.650 | +0.00000 |
| `flat` | 1.00 | +0.035 | 0.000 | 0.637 | +0.00000 |
| `unfrozen` | **0.00** | **−0.625** (−0.917 on concepts that worked) | 0.375 | 0.662 | +0.00000 |

With ancestors frozen, a consolidated concept's accuracy is *exactly* constant (range 0.000) for the
rest of the run. Remove only the freezing and concepts that worked lose 0.917 of their accuracy.
Plasticity is unchanged across arms, so no stability/plasticity trade is being paid.

Consolidation retains **98%** of pool accuracy on pools that worked (0.98→0.94, 1.00→1.00), and
faithfully reproduces a bad pool as a bad concept. The hierarchy gives a **warm start** — first-epoch
distillation loss 0.113 (`hier`) vs 0.160 (`flat`), 30% lower — but converges to the same final loss
(0.0495 vs 0.0499). Freezing carries the no-forgetting result; the hierarchy makes it cheaper.

Reproduce the per-concept analysis with `python experiments/analyze_n3.py`.

## 3. Open work, in priority order

### 3.0 Finish the fidelity half of the inversion fix (IMPLEMENTED, NOT YET MEASURED)
The coverage half is done and measured (N7: F1 0.00 -> 0.45, recall 0.03 -> 0.83, content diversity
0.74 -> 1.21x real; RESULTS 5f). The remaining gap is fidelity: composition accuracy fell 0.58 ->
0.29, almost entirely on BACKGROUND (0.84 -> 0.60) while foreground held (0.83-0.94). Run this the
moment the cluster is back:

```
sbatch --export=ALL,SCRIPT="experiments/n7_inversion.py --n-slots 8 --n-samples 32 --inv-steps 400" \
       experiments/run_gpu.sh
```
It sweeps `--anchor 0.5 2.0 10.0` (penalty `anchor * ||alpha - alpha_teacher||^2`, already
implemented in `inversion.learn_embedding`) crossed with `--guidance 1.5 2.5`. The hypothesis: 120
free parameters are underdetermined by one query, so the teacher must act as a PRIOR throughout
rather than only as an initialization, and the anchor strength is the dial between "what the teacher
composed" and "what this query looks like". Target: accuracy back toward 0.58 without losing the
recall.

Already ruled out for this, so do not retry them: using the teacher's POOL as the inversion target
(recall 0.19 -- the pool is collapsed, so inverting on it teaches one image), and `nearest-class`
conditioning (F1 0.00).

If the anchor is not enough, the next candidates in order are: (a) invert on mu_T rather than the raw
query, since mu_T is the COMPOSED image and probe-reads correctly at 0.800 while the query carries
idiosyncratic detail; (b) add the K selected expert means as extra inversion targets -- they are
clean, on-composition, and genuinely different from each other, which is the diverse on-composition
set the pool failed to be; (c) a per-factor loss that weights background pixels, since that is
where the error concentrates.

### 3.0z Superseded: replace Gaussian-teacher sampling with model conditioning — DONE
The diversity investigation (RESULTS 5e) is finished and the answer is not a hyperparameter.
Measured: the base DDPM sampled class-conditionally reaches precision 0.89 / recall 0.90 / content
diversity 0.97x real -- the ceiling. Every teacher-guided configuration reaches recall 0.00. Ruled
out by direct test: guidance schedule (7 variants incl. the paper's Eq. 14), teacher covariance
(product vs mixture, 3 scales), pooling across queries as the paper does, ancestral sampling
(eta 0.5/1.0), and distillation.

The cause is structural: CFG toward a UNIMODAL analytic Gaussian contracts every trajectory to its
mean, and the guidance knobs only trade collapse against divergence. The base model is diverse
because its class embedding conditions a LEARNED MULTIMODAL conditional -- a different mechanism
entirely.

So: keep the PoE for locating the concept (mu_T probe-reads correctly, 0.800 vs a 0.775 ceiling) and
stop using it as the sampling distribution.
- **Learn `c_new` directly from the query + mu_T** (textual inversion, Gal et al.), then sample the
  model conditioned on it. `HierarchicalAdapterBank` already holds a per-concept embedding and
  `UNet.forward(emb_override=)` already routes it -- the plumbing exists; only the objective changes.
- Seed the embedding from the nearest trained class embeddings weighted by the same per-dimension
  PoE weights, so composition still determines what is learned.
- Validate with `tests/diag_base_diversity.py` as the ceiling and `experiments/n6_diversity.py` as
  the harness; the target is recall approaching 0.90, not merely above 0.

### 3.0aa Also worth fixing: the Laplace floor
99% of `laplace_diag` variances sit exactly on `var_floor=1e-4`, so Sigma_T is a constant rather
than a shape, and the anisotropy the PoE needs (stroke uncertain, background certain) is mostly
clipped away -- measured stroke/background ratio 1.91 against the real class's 3.87. Raising `delta`
does not help (99.0% -> 96.4% across 0.005-0.2). This is not what is capping diversity, but it does
mean the covariance carries almost no information and the per-dimension weighting is running on
less signal than it appears to.

### 3.0a Compress the adapter bank — the real scaling limit
Vetting the case-study figures showed PoE samples are near-copies of the query: `div/novelty` 1.04
(independent draws would give 1.41) and content diversity 0.45x the real class spread. Everything
else in the pipeline is healthier than this. Three leads, and the first is nearly free:

- **Raise `teacher_var_scale` further and re-check with the CONTENT metric.** N0 swept it against raw
  pixel diversity, which we now know overstates variety by ~35%. The sweep should be redone against
  `content_ratio`; the optimum is probably above the 30 currently defaulted.
- **Stochastic sampling.** The sampler is DDIM with eta=0 — fully deterministic given x_T, so all
  variety must come from the initial draw, which the too-tight teacher covariance controls. Ancestral
  (eta>0) sampling injects noise at every step and is the standard fix.
- **Estimate the teacher covariance from a batch rather than one query.** The Laplace estimate
  measures curvature at a mode, not class extent, which is *why* it is 13.6x too tight. With a batch
  of related queries (now supported, see `compose_batch`) the covariance can be estimated
  empirically instead. This is the principled fix and batch addition already provides the inputs.

### 3.0a Compress the adapter bank — the real scaling limit
Growth is **linear, not sublinear**: ~0.55M params/concept at rank 8, so 40 concepts already exceed
the 6.8M base. `tests/test_unbounded.py` proves the *guarantee* is K-independent; the *cost* is not
free. Three directions, cheapest first:
- **Lower rank for leaves, higher for shared nodes.** Leaves carry one concept; ancestors carry many.
  `inject(..., rank=)` is already per-bank; make it per-node.
- **Adapt a subset of layers.** `inject(model, rank, state, targets=("conv","linear"))` already
  supports restriction; sweep which blocks actually matter (attention and mid-block first).
- **Merge sibling adapters in an offline consolidation pass.** Sample from two sibling concepts,
  distil into their shared parent, and *keep the merge only if both fingerprints reproduce within
  tolerance* — that turns the guarantee from structural to audited for merged nodes, which is a
  defensible trade when memory is the binding constraint.

### 3.0b Make the hierarchy earn its keep
N5 is blunt: `independent` (no sharing) beats `ours` on final accuracy, 0.839 vs 0.802, with fewer
parameters. Freezing gives the no-forgetting result; the hierarchy currently buys only the warm start
and the batch speedup. Either find a regime where sharing pays — concepts with genuine overlap,
rather than ColorMNIST's spatially separable attributes — or drop the hierarchy and keep flat frozen
adapters, which would be a simpler and equally safe method. **Do not keep the hierarchy on the
strength of the idea alone; it has to show up in a number.**

### 3.0c Seeds — still unaddressed
Everything above is **single-seed**. `--seed` is plumbed through every experiment and
`experiments/aggregate.py` pools `results/<name>__seed<k>.json` into mean ± SE automatically (every
run writes its seed-tagged copy, so this needs no discipline beyond launching the jobs). Before any
of this is written up:
```
for s in 0 1 2; do
  sbatch --export=ALL,SCRIPT="experiments/e2_composition.py --n-slots 40 --seed $s" experiments/run.sh
  sbatch --export=ALL,SCRIPT="experiments/e3_lifelong.py --n-concepts 12 --seed $s" experiments/run.sh
  sbatch --export=ALL,SCRIPT="experiments/n3_lifelong.py --n-concepts 10 --n-pool 96 --epochs 60 --seed $s" experiments/run_gpu.sh
done
python experiments/aggregate.py
```
N3's headline (`range across steps = 0.000` vs `unfrozen` −0.917) is structural and will not move,
but every accuracy number will.

### 3.1 The composition step is now the bottleneck, not consolidation
Consolidation retains 98% of what it is given, so end-to-end quality is set by how good the pool is,
and pools vary wildly by slot (0.00 to 1.00). Two concrete leads:
- **Per-slot failures are systematic, not noise.** Slots on the `(cyan, deep_red)` pair fail far more
  than others across both backbones. Diagnose at teacher level (cheap, no sampling): which experts
  get selected, and does μ_T probe-read correctly?
- **Digit identity is the weak factor everywhere** (E2 full: fg 0.99, bg 0.84, digit 0.69). The
  per-dimension coverage objective has no notion of shape. Try masking `submodular_select`'s marginal
  gains by the digit region (pixels where the query deviates from its background) so shape-compatible
  experts are preferred.

### 3.2 E4's hypothesis needs a redesign (analysis done, fix not written)
E4 asks whether a consolidated concept gets *reused* by later compositions. The quick run measured
`adapter_expert_frac = 0.00` — consolidated concepts are never selected. On inspection **that is
probably correct behaviour, not a bug**, and the experiment is the thing that is wrong:

ColorMNIST's attributes are *spatially separable*. Background pixels depend only on `bg`, stroke
pixels only on `fg`. A per-dimension PoE can therefore always assemble an unseen `(fg, bg)` pair from
base concepts — a `yellow-on-deep_red` concept already supplies a perfect deep_red background — so
having the *combination* pre-learned in one expert buys nothing. Closure cannot show up here.

To test closure properly, use an attribute pair that is NOT separable, so the combination must be
carried by a single expert. Options, in order of preference:
(a) a second-order composition: consolidate `(cyan, deep_red)` and a held-out *digit style*, then
    query something needing both — the digit shape and the colour scheme overlap on the same pixels;
(b) switch the benchmark axis to non-separable attributes (stroke thickness × digit identity, or
    CelebA-style attributes as in the paper's second benchmark);
(c) keep ColorMNIST but measure closure by *cost* rather than accuracy — does consolidating concept
    k get cheaper (fewer epochs to the same loss) when relatives are already in the bank? This is
    the `transfer` column N3 already records and is the cleanest available test on this dataset.
Recommend (c) first — it needs no new benchmark — then (a).

### 3.3 poe_ascent (the paper's discovery route) underperforms
Full-scale E2 shows `poe_ascent` 0.47 vs `poe_tree` 0.81 on slot 46. The hyperparameters were
brought in line with the paper's Table 6 this session (low-noise grid u∈[0.05,0.4], 150 iters, lr
0.1, hybrid query/random init, scale-free dedup at 0.05 RMSE — the old raw-L2 threshold of 1.0
deduplicated *nothing* over 3072 dims). Still worth checking: does ascent actually converge (log
p_t rising, modes stationary), and are the Laplace variances hitting the clip bounds? The headline
claim to aim for is `poe_tree ≈ poe_ascent`: the tree gives the paper's discovery for free.

### 3.4 Analytic `sample_concept` is weak — and the neural fix probably applies
E3 reports `new joint` between 0.03 and 1.00 across concepts. The neural equivalent turned out to be
a **guidance bug, not a learning failure**: the default `guidance=2.0` scored 0.00 while `1.0` scored
0.38, because classifier-free guidance extrapolates away from a *weak* unconditional model, and here
the unconditional branch is a strong one (see `tests/diag_sample_concept.py`). The analytic
`sample_concept` still uses `guidance=2.0` against a full-mixture unconditional branch — the same
setup. **Try `guidance=1.0` first**; it is a one-line change and likely most of the gap.

### 3.6 Threats to validity to state plainly in the writeup
- Metrics are **probe-based**, not FID/CLIP as in the paper. The probes are linear-ish softmax
  models on raw pixels; digit accuracy tops out around 0.85–0.90 even on real images, which is why
  every table reports the **probe ceiling** next to the method.
- The ColorMNIST palette is *reconstructed* from the paper's colour names, not their RGB values
  (inherited from cobweb-poincare). Absolute numbers are not comparable to the paper's tables; only
  within-table comparisons are.
- The annealed guidance schedule and the adaptive PoE temperature are **deviations** from the paper,
  both documented at their implementation sites with the measurement that motivated them.
- `n_per_slot` is 40 (analytic) / 200 (neural) against the paper's 1000.

### 3.7 Later
- Compose *from* consolidated concepts across backbones (the E4 redesign above, on the neural side).
- `mode="frontier"` (depth-annealed reverse process) is implemented and E1-validated but unused in
  E2/E3 — worth an ablation.
- Cap total adapter mass (weight anchoring) vs count-proportional, on the analytic side.
- Second dataset: the union stream (mnist/fashion/kmnist) from cobweb-poincare.

## 4. Two results worth not re-deriving

**Adaptive PoE temperature.** The paper's τ=0.5 does not port. Their experts are DDPM modes with
per-mode Laplace covariances that vary widely; Cobweb variances sit near `prior_var` everywhere,
compressing every per-dimension log-likelihood gap by a common factor (a whole background colour is
~0.2 nats/pixel). At τ=0.5 the per-dim softmax gave three selected experts weights 0.42/0.29/0.29, so
two experts carrying the *wrong* background outvoted the one correct one. It also broke the method's
internal consistency: selection maximizes a per-dimension MAX (their Eq. 8) while composition was
applying a near-uniform blend. Fix: `T = tau · mean_r std_j l_{j,r}`, scale-free, transfers across
datasets and `prior_var`. Background accuracy 0.42 → 0.97.

**Annealed guidance, and the order of diagnosis.** A `(w0, u_start)` sweep across
`{1.2, 2.5, 4.0, 6.0} × {0.4, 0.55, 0.7}` moved joint accuracy barely at all, and two of four slots
sat at exactly 0.0 for *every* setting. That flatness was the signal: the teacher, not the sampler,
was wrong — μ_T probe-read correctly on exactly the slots that scored nonzero, and wrongly on the
ones that scored zero. **Sample accuracy is bounded by teacher accuracy**, so the sampler sweep was
uninformative until the teacher was fixed. Re-run afterwards it became informative immediately.

The reason annealing helps is the density hole in §1: the unconditional mixture is trustworthy for
generic image structure at high noise and actively wrong about the composition at low noise, where
identity is decided. **But the size of the effect depends on how dense the mixture is** — 71% vs 21%
on a 315-concept tree, 89% vs 78% on a 1589-concept one. Report the scale with the claim.

## 5. Environment / running

```
analytic:  ~/flash/miniconda3/envs/cobweb/bin/python    (cobweb-private, numpy, matplotlib; no torch)
neural:    ~/flash/miniconda3/envs/chunks/bin/python    (torch 2.7.1+cu126, cobweb, matplotlib)
data:      MNIST idx auto-found at ../cobweb-poincare/data/MNIST/raw (or $LOL_MNIST_DIR, or download)
tests:     python tests/test_smoke.py      (analytic, synthetic, ~10 s)
           python tests/test_neural.py     (neural, CPU, <1 min)
quick:     every experiment takes --quick
```

**Run everything through SLURM** (login nodes are heavily contended — load 51 on 40 cores during
this session — and lab convention is to validate via sbatch):
```
sbatch --export=ALL,SCRIPT="tests/test_smoke.py"                     experiments/run.sh      # CPU
sbatch --export=ALL,SCRIPT="experiments/e1_manifold.py"              experiments/run.sh
sbatch --export=ALL,SCRIPT="experiments/n1_train.py --epochs 200"    experiments/run_gpu.sh  # GPU
```
`run.sh` is CPU-only (account/partition `tail-lab`, qos `short`); `run_gpu.sh` requests one GPU and
uses the `chunks` env. Trees built by cobweb-private are live C++ objects: they do not pickle or
deepcopy, so experiments refit them from the stream (fast) rather than caching.

## 6. Repo/git

Remote `https://github.com/karthiksing05/lifelong-ood-learner.git`, branch `main`. **No credentials
are stored on this machine** — pushing needs `gh auth login`, a PAT, or an SSH key. Commit after each
milestone and update §2/§3 here.
