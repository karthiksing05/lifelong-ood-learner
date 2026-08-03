# PLAN — lifelong-ood-learner

Working plan and handoff document. Read this top-to-bottom before touching code.
Last updated: 2026-08-03 (initial build session, Claude Fable).

## 1. The idea (do not drift from this)

**Reframe lifelong learning as repeated composition of prototypes:**

```
loop forever:
  1. meet ONE out-of-distribution query x_q
  2. COMPOSE known prototypes into an OOD prototype        (product-of-experts, poe-diffusion Eq. 7-10)
  3. SAMPLE that prototype through the learned manifold     (analytic CFG, poe-diffusion Eq. 12-14)
  4. CONSOLIDATE the samples as a frozen-base additive adapter (the LoRA move, made structural)
```

Two claims the experiments must establish:

- **E1 (validity):** a Cobweb hierarchy of Gaussian prototypes IS a diffusion manifold. Its mixture
  score is analytic at every noise level; reverse diffusion on it generates; and its internal nodes
  coincide with the modes of the noisy marginals p_t at coarser t (what poe-diffusion recovers from a
  DDPM by gradient ascent, their Appendix A, we get for free from the tree).
- **E2+E3 (lifelong):** repeated compose→sample→consolidate grows the manifold with new OOD concepts
  **without degrading it** — old-slot log-likelihood, denoise R², and unconditional sample validity
  must stay flat across many consolidations (graft strategy), while the classic absorb-everything
  `ifit` control drifts. Non-degradation is by construction: adapters are separate trees merged at
  snapshot level, so base statistics are bit-identical forever; the only global effect is mixture
  weight renormalization, drift ≤ log(1 + n_adapter/n_base).

Primary sources: `poe-diffusion.pdf` (in ~/flash root; the method we mirror — mode ascent, submodular
selection, per-dim PoE, variance-aware CFG, LoRA distillation), `../cobweb-poincare` (palette, splits,
prior_var conventions, PoE compose precedent — this repo must complement it, not deprecate it),
`../Chunkformer/cobweb-private` (the TRUE Cobweb implementation; user explicitly requires it as the
backend — the numpy tree is only a portability fallback).

## 2. What exists and works (verified)

| piece | file | status |
|---|---|---|
| ColorMNIST benchmark (palette/split identical to cobweb-poincare & paper Appendix B) | `lifelong_ood/datasets.py` | ✅ works, reads MNIST idx from `../cobweb-poincare/data/MNIST/raw` |
| cobweb-private backend (covar_from=1, var=sum_sq/n+prior_var, same as cobweb-poincare) | `lifelong_ood/backend.py` | ✅ smoke-tested; env: `~/flash/miniconda3/envs/cobweb/bin/python` |
| numpy fallback Cobweb (4 operators, category utility) | `lifelong_ood/tree.py` | ✅ smoke-tested |
| analytic score field: p_t mixture score, DDIM sampler, mode ascent, Laplace diag, tweedie | `lifelong_ood/score.py` | ✅ E1 quick-validated |
| composition: candidate pools, submodular select, per-dim PoE, teacher ε (Eq.12), variance-aware CFG w/ floor (Eq.14) | `lifelong_ood/compose.py` | ✅ teacher μ_T probe-reads the right composition |
| lifelong loop: frozen base + adapter trees merged at snapshot; conditional sampling per adapter | `lifelong_ood/lifelong.py`, `backend.merge_snapshots` | ✅ smoke: consolidation drifts old loglik by only −0.03 nats |
| probes + metrics + figures | `lifelong_ood/evals.py` | ✅ (fg/bg probes 1.00 val acc, digit 0.785 at quick scale) |
| E1 manifold validity | `experiments/e1_manifold.py` | ✅ quick run: uncond valid-composition rate 0.875 (leaves) / 0.938 (frontier); mode ascent matches deep leaves at u≤0.6 (rmse 0.015–0.16), coarsens to depth ~1.75 internal nodes at u=0.8 |
| E2 composition | `experiments/e2_composition.py` | ⚠️ partially working — see §3 |
| E3 lifelong | `experiments/e3_lifelong.py` | ⚠️ not yet run after backend refactor |
| smoke test (both backends) | `tests/test_smoke.py` | ✅ passes on numpy AND cobweb-private |

Quick-run cache lives in `results/cache/` (numpy trees pickle; cobweb-private trees can't — they refit
each run, which is fast in C++). Delete the cache after changing tree/backend code.

## 3. Open issues, in priority order

### 3.1 E2 composition quality — ROOT CAUSE FOUND AND FIXED (adaptive PoE temperature)

**The bug was the teacher, not the sampler.** Diagnosis chain:

1. A `(w0, u_start)` sweep over `{1.2, 2.5, 4.0, 6.0} × {0.4, 0.55, 0.7}` moved joint accuracy
   barely at all (0.00–0.21) and two of four slots stayed at exactly 0.0 for every setting —
   guidance strength was not the lever.
2. Per-slot inspection showed μ_T probe-reads the queried triple on exactly the slots that scored
   nonzero, and misreads it on the ones that scored zero. **Sample accuracy is bounded by teacher
   accuracy**; the sampler was faithfully rendering a wrong teacher.
3. Expert dumps explained the misreads: on slot 19 (`digit 1, yellow on dark_brown`) the three
   selected experts read `(1,1,3)`, `(1,0,0)`, `(1,1,0)` — one correct background, two wrong ones.
   At the paper's τ=0.5 the per-dimension softmax gave them 0.42 / 0.29 / 0.29, so the two wrong
   backgrounds outvoted the one right one and μ_T came out `deep_red`.

**Why the paper's τ=0.5 does not port.** Their experts are DDPM modes with per-mode Laplace
covariances that vary widely; Cobweb's variances sit near `prior_var` almost everywhere, which
compresses every per-dimension log-likelihood gap by a common factor (a whole background-colour
difference is only ~0.2 nats/pixel at prior_var=0.05). A fixed τ tuned for the former near-uniformly
averages the latter — and that also breaks the internal consistency of the method, since selection
maximizes a per-dimension MAX (their Eq. 8) while composition was applying a near-uniform blend.

**The fix** (in `compose.poe_teacher`, documented there as a deviation): set the temperature from the
data's own scale, `T = tau · mean_r std_j l_{j,r}`, exactly as cobweb-poincare sets its profile
temperature. Scale-free, so it transfers across datasets and `prior_var` without retuning.

Measured on ALL 40 held-out slots (quick scale, teacher μ_T probe reading, no sampling):

| K | τ | joint | digit | fg | bg |
|---|---|---|---|---|---|
| 3 | 0.5 (paper) | 0.200 | 0.72 | 0.90 | **0.42** |
| 3 | 0.05·adaptive | **0.625** | 0.75 | 0.90 | **0.97** |
| 6 | 0.05·adaptive | 0.650 | 0.75 | 0.93 | 0.97 |
| — | *probe ceiling on REAL held-out images* | *0.700* | *0.90* | *0.90* | *0.90* |

0.625 against a 0.700 ceiling = **89% of what a genuine OOD instance scores**. Defaults changed to
`tau=0.05, adaptive_tau=True`; K stays at the paper's 3 (K=6 buys +0.025, not worth the deviation).

Remaining gap is digit identity (0.75 vs 0.90 ceiling). If worth closing later, the structured fixes
are: (a) mask `submodular_select`'s marginal gains by the digit region (pixels where the query
deviates from its background) so shape-compatible experts are preferred; (b) two-stage teacher —
pick the shape expert first, then let colour experts weigh only within the stroke/background
partition its μ implies.

Sampler settings are being re-swept against the corrected teacher (the first sweep measured a broken
one); update the defaults in `LifelongLearner.cfg` when that lands.

### 3.2 E2 metrics: faithfulness vs generalization — DONE
`evals.sample_geometry` returns novelty (RMSE to the query), diversity (mean pairwise RMSE),
gen_rmse (mean min-RMSE to the slot's OTHER held-out instances) and ref_spread (the reference set's
own internal spread, the yardstick for the other three). Wired into `e2_composition.py` rows and
summary table. The narrative to reproduce: query_only = high accuracy but near-zero novelty
(memorization), top-k = wrong composition, PoE = accuracy at reference-like novelty and diversity.

### 3.3 poe_ascent discovery is weak at quick scale (NOT STARTED)
`candidates_ascent` (the paper-faithful route) got joint 0.0 on the quick run. Before tuning, verify
at full scale (n_per_slot=40, iters=80, n_starts=8, u_grid=(0.15,0.3,0.45,0.6)). Check that (a)
ascent converges (log p_t increases, modes stop moving), (b) Laplace variances are sane
(var_floor..var_cap, not all clipped), (c) dedup_tol=1.0 isn't collapsing distinct modes (it's L2^2
over 3072 dims — probably should be ~ D*0.01·scale; MEASURE pairwise mode distances first).
The headline claim wants poe_tree ≈ poe_ascent: the tree gives the paper's discovery for free.

### 3.4 E3 full run (NOT STARTED beyond smoke)
After 3.1 lands, run `e3_lifelong.py --quick` (both strategies) with the cobweb env, then full scale
in SLURM. Expected: graft curves flat (drift ~ −log(1+n_ad/n_base), order 0.01–0.1 nats), ifit curves
degrade; plasticity (new-concept joint acc) high; retention (earlier adapters' joint acc) flat.
If conditional `sample_concept` quality is poor, raise `guidance` (currently 2.0) — the adapter
mixture is much tighter than the full field, so it may need less, not more.

### 3.5 SLURM (user requirement: tests must run via sbatch)
`experiments/run.sh` is the sbatch runner (account/partition `tail-lab`, qos `short`, CPU-only —
nothing here needs a GPU; cobweb ifit is serial C++, score field is numpy — request 8-16 CPUs and set
OMP/MKL/OPENBLAS thread env vars). Submit:
```
cd ~/flash/lifelong-ood-learner
sbatch --export=ALL,SCRIPT="tests/test_smoke.py" experiments/run.sh
sbatch --export=ALL,SCRIPT="experiments/e1_manifold.py" experiments/run.sh          # full scale
sbatch --export=ALL,SCRIPT="experiments/e2_composition.py --n-slots 40" experiments/run.sh
sbatch --export=ALL,SCRIPT="experiments/e3_lifelong.py --n-concepts 20" experiments/run.sh
```
Cluster gotchas (from prior sessions): login nodes SIGTERM torch processes (this repo avoids torch
entirely, and cobweb-private+numpy have run fine on login nodes so far — but validate everything
through sbatch anyway, per lab convention); check queue with `squeue --me`.

### 3.6 Full-scale numbers + README results section (NOT STARTED)
Defaults are n_per_slot=40 (4800 seen instances). After the full runs, paste the E1/E2/E3 headline
numbers + figures into README.md, and write the threats-to-validity list (probe-based metrics, no
FID, palette reconstruction not identical to paper's RGB, single seed unless swept).

### 3.7 Nice-to-have extensions (only after 3.1–3.6)
- Compose FROM consolidated concepts: after E3's sequence, hand the learner a query whose nearest
  explanation needs an OOD prototype learned earlier in the sequence — closure of composition under
  consolidation is the distinctive claim; make it an explicit experiment (E4).
- `mode="frontier"` study: depth-annealed reverse process vs leaf mixture (E1 already computes both
  valid rates; extend to E2/E3).
- Weight anchoring ablation: cap total adapter mass at β (e.g. 0.2) vs count-proportional.
- Second dataset (the union stream mnist/fashion/kmnist from cobweb-poincare) for a
  hierarchy-heavier setting.

## 4. Architecture invariants (keep these true)

1. **Everything downstream of the tree reads ONLY the snapshot dict** `(M, V, W, count, depth,
   parent, is_leaf, branch)`. New backends implement `ifit/fit/snapshot`, nothing else.
2. **The base is frozen.** Adapters are separate trees; `backend.merge_snapshots` concatenates
   snapshots (adapter roots at depth 1 under the base root). Never mutate base statistics in the
   graft path — the bounded-drift guarantee is the paper-level claim.
3. **No torch anywhere.** Pure numpy + the cobweb-private nanobind extension. Keeps login-node dev
   safe and the repo portable.
4. **Palette/splits stay byte-identical to cobweb-poincare** so numbers are comparable across repos.
5. **The paper's equation structure stays recognizable** in compose.py/score.py (Eq. 4, 7, 8-10,
   12, 13, 14 are cited at the implementation sites). Deviations must be documented as deviations
   (the guidance floor and teacher-init sampling already are).

## 5. Environment / running

```
env:      ~/flash/miniconda3/envs/cobweb/bin/python   (has cobweb-private + numpy + matplotlib)
fallback: any python3.10+ with numpy+matplotlib (numpy tree backend, slower, same results shape)
data:     MNIST idx auto-found at ../cobweb-poincare/data/MNIST/raw (or $LOL_MNIST_DIR, or download)
quick:    every experiment takes --quick (12/slot, small iters) and runs in 1–3 min on a login node
tests:    python tests/test_smoke.py   (synthetic, ~10 s, no MNIST needed)
```

## 6. Repo/git

Remote: `https://github.com/karthiksing05/lifelong-ood-learner.git` (branch `main`). No credentials
are stored on this machine — pushing needs the user (gh auth login, a PAT, or an SSH key). After any
milestone: commit with a body explaining WHAT changed and WHY, and update §2/§3 of this file.
