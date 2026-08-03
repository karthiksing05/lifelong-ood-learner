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

### 3.1 E2 sampling quality (IN PROGRESS — next action)
State: teacher μ_T composes correctly (probes read the queried (digit,fg,bg) off μ_T), but guided
DDIM samples lose digit identity on some slots (quick run: poe_tree joint 0.19–0.38; digit acc is the
weak factor). Fixes already in: candidate pool = ALL tree concepts (total-loglik ranking let
background pixels crowd out fg experts — do not undo this), guidance floor `wmin` in `cfg_weight`,
SDEdit-style teacher-init sampling (`sample_teacher(u_start=0.55)`).

Next action (script ready to paste, run with the cobweb env python from repo root):
sweep `(w0, u_start) ∈ {1.2, 2.5, 4.0} × {0.4, 0.55, 0.7}` on ~4 OOD slots, measure joint
composition accuracy AND novelty RMSE `sqrt(mean((S - x_q)^2))` (novelty guards against the
query-copy degenerate optimum). Pick the best joint accuracy with novelty ≥ roughly the query_only
baseline's novelty × 3, set as defaults in `LifelongLearner.cfg` / `sample_teacher`.

```python
# from repo root:  ~/flash/miniconda3/envs/cobweb/bin/python <this>
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
from common import base_parser, load_all, ood_queries
from lifelong_ood.evals import composition_accuracy
args = base_parser().parse_args(['--quick'])
data, learner, probes = load_all(args)
qs = ood_queries(data, n_slots=4, seed=0)
for w0, us in [(1.2,.55),(2.5,.55),(4.,.55),(2.5,.4),(4.,.4),(2.5,.7)]:
    js, novs = [], []
    for qi,q in enumerate(qs):
        xq = data['X'][q['query']]
        t = learner.compose(xq, discovery='tree')
        S = learner.sample_teacher(t, n=12, seed=100+qi, w0=w0, u_start=us)
        a = composition_accuracy(probes, S, q['d'], q['f'], q['b'])
        js.append(a['joint']); novs.append(float(np.sqrt(((S-xq)**2).mean())))
    print(f"w0={w0} u_start={us}: joint={np.mean(js):.3f} {[round(j,2) for j in js]} nov={np.mean(novs):.3f}")
```

If digit identity still breaks after the sweep: the likely cause is per-dim weight mixing of stroke
pixels across experts with different digits. Two structured fixes to try, in order:
(a) bias selection toward SHAPE-compatible experts — add the digit-region coverage (pixels where the
query deviates from its background) as a mask when computing marginal gains in `submodular_select`;
(b) two-stage teacher: first select the best shape expert (same-digit, any color), then restrict the
color experts' per-dim weights to the background/stroke partition implied by μ of the shape expert.

### 3.2 E2 metrics: faithfulness vs generalization (NOT STARTED)
query_only currently scores joint=1.0 by construction (teacher-init makes it copy the query). Mirror
the paper's split: report per method (i) joint composition accuracy, (ii) novelty RMSE to the query,
(iii) sample diversity (mean pairwise RMSE), and optionally (iv) mean min-RMSE to the slot's held-out
`rest` images (their "generalization" reference). Add to `e2_composition.py` row dict; the paper's
narrative to reproduce: query_only = faithful but degenerate, top-k = wrong composition, PoE = both.

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
