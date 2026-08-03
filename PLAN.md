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
| E2 OOD composition | `experiments/e2_composition.py` | 🔄 full-scale run in flight |
| E3 analytic lifelong | `experiments/e3_lifelong.py` | 🔄 full-scale run in flight |
| E4 composition closure | `experiments/e4_closure.py` | ⚠️ runs; hypothesis needs redesign, see §3.2 |
| N1/N2/N3 neural experiments | `experiments/n{1,2,3}_*.py` | N1 🔄 training; N2/N3 plumbing ✅, awaiting checkpoint |

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

**E0 (the two tuning results that made composition work).** See §4 for the full story.
- teacher: paper's τ=0.5 → joint **0.200**; adaptive temperature → **0.625**, vs a **0.700** probe
  ceiling measured on real held-out images (89% of ceiling).
- sampler: paper's variance-aware guidance retains **21%** of teacher accuracy; annealed retains
  **71%**.

**E2 (partial, full scale).** First slots: `poe_tree` 0.81 vs `top1`/`top3` 0.00. `query_only` scores
1.00 by memorizing — which is why the geometry triple (novelty / diversity / gen_rmse) is reported
alongside and is the thing that separates them.

**E3 (partial, full scale).** Per consolidation: drift in held-out seen log-likelihood ≈ **−0.01 to
−0.05 nats**, drift in denoise R² **+0.0000**, unconditional validity stays 0.94–1.00. Non-degradation
is holding at full scale.

**N3 (plumbing run on an untrained checkpoint — structure only).** `hier` keeps
`retention_exact = 1.00` at every step; the `unfrozen` ablation drops to **0.00** as soon as a second
concept is consolidated; frozen-base drift is **+0.00000** in all arms. The mechanism does what it
claims; the accuracy numbers need the trained checkpoint.

## 3. Open work, in priority order

### 3.1 Collect the in-flight full-scale runs (DO THIS FIRST)
Jobs submitted this session: E2 `3647390`, E3 `3647391`, DDPM training `3647581`. Check with
`squeue --me`; outputs in `experiments/slurm/`. E2 ≈ 7 h (40 slots × 5 methods × 32 samples), DDPM
≈ 3.5 h (200 epochs). When the checkpoint lands, run N2 and N3 at full scale:
```
sbatch --export=ALL,SCRIPT="experiments/n2_composition.py --n-slots 12" experiments/run_gpu.sh
sbatch --export=ALL,SCRIPT="experiments/n3_lifelong.py --n-concepts 12" experiments/run_gpu.sh
```
Then write the numbers into README §Experiments and into §2 above.

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

### 3.4 Analytic `sample_concept` is weak
E3 reports `new joint` between 0.03 and 0.88 across concepts — conditional generation from a grafted
adapter is unreliable, unlike `sample_teacher` which E0 tuned. It uses a fixed `guidance=2.0`. Give
it the same treatment: sweep guidance and `u_start` in `e0_tune.py` against consolidated concepts.

### 3.5 Seeds and error bars
Everything is single-seed. Before any of this is written up, run E1/E2/E3 and N3 over ≥3 seeds and
report mean ± SE. `--seed` is already plumbed through every experiment.

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

**Annealed guidance.** Diagnosis order matters here: a `(w0, u_start)` sweep across
`{1.2, 2.5, 4.0, 6.0} × {0.4, 0.55, 0.7}` moved joint accuracy barely at all and two of four slots
sat at exactly 0.0 for *every* setting. The teacher, not the sampler, was wrong — μ_T probe-read
correctly on exactly the slots that scored nonzero. Only after fixing the teacher did the sampler
sweep become informative, and it then showed the paper's variance-aware schedule retaining 21% of
teacher accuracy versus 71% for a schedule that anneals guidance up as noise falls. The reason is
the density hole in §1: the unconditional mixture is trustworthy for generic image structure at high
noise and actively wrong about the composition at low noise, where identity is decided.

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
