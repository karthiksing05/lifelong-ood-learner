# lifelong-ood-learner

**Lifelong learning as repeated composition of prototypes, on a manifold learned the way diffusion
models learn it — but where the score field is read, analytically, off a Cobweb hierarchy.**

No neural network, no gradient descent, no backprop. The generative model is an incremental Cobweb
tree of diagonal-Gaussian prototypes ([cobweb-private]'s `CobwebContinuousTree`); its mixture defines
noisy marginals `p_t` whose score, Tweedie denoiser and ε-prediction are closed-form, so everything a
DDPM does by learned approximation — reverse-diffusion sampling, mode ascent on `p_t`,
classifier-free guidance against a teacher — runs here in exact arithmetic.

## The idea

The [poe-diffusion paper] (Wang et al., *Test-Time Compositional Generalization in Diffusion Models
via Concept Discovery*) shows that a pretrained DDPM's time-indexed score field hides a hierarchy of
density modes; that these modes, discovered by gradient ascent from a single OOD query, can be
composed by a product-of-experts into a teacher for an unseen composition; and that the teacher can
be distilled back into the model with LoRA. Its stated limitation: a full mechanism for
*consolidating and reusing* discovered concepts — lifelong learning — is future work.

This repository is that future work, with the hierarchy made explicit:

1. **The manifold.** A Cobweb tree grown one instance at a time IS the diffusion model
   (`lifelong_ood/score.py`). Its effective leaves are the mixture components of `p_0`; its internal
   nodes are the modes of the smoothed marginals `p_t` at coarser noise — the density-mode hierarchy
   the paper recovers by ascent (their Appendix A), here explicit and free (validated in E1).
2. **Composition.** From one OOD query: discover prototypes (closed-form ranking over tree concepts,
   or the paper's mode ascent — both implemented), select K complementary ones with the submodular
   per-dimension coverage objective (their Eq. 8–9), weigh per-dimension (Eq. 10), multiply into an
   analytic PoE teacher (Eq. 7), and sample it through the manifold with variance-aware
   classifier-free guidance (Eq. 12–14) (`lifelong_ood/compose.py`, E2).
3. **Lifelong consolidation.** The paper distills its teacher into a LoRA adapter on a frozen UNet.
   We make the same move structural: the sampled pool grows a **separate adapter tree**; the base is
   frozen and the adapter is merged at snapshot level as new mixture components
   (`lifelong_ood/lifelong.py`, `backend.merge_snapshots`). Interference with old knowledge is
   bounded analytically — `|Δ loglik| ≤ log(1 + n_adapter/n_base)` — so **the manifold cannot
   degrade over time by construction**, and consolidated concepts become ordinary prototypes that
   later compositions can select: composition is closed under consolidation (E3).

```python
from lifelong_ood import LifelongLearner

L = LifelongLearner()                 # cobweb-private backend if installed, numpy fallback otherwise
L.fit(stream)                         # Cobweb's own incremental loop — the only training there is
teacher, pool = L.learn_concept(x_ood_query, "my_new_concept")   # compose -> sample -> consolidate
S = L.sample_concept("my_new_concept", n=16)                     # conditional generation, no c_new training
```

## Experiments (ColorMNIST, 120 seen / 40 held-out compositional slots)

Benchmark, palette and splits are byte-identical to [cobweb-poincare]'s reconstruction of the paper's
Appendix B benchmark, so numbers are comparable slot-for-slot across the two repositories.

| exp | question | quick-scale evidence so far |
|---|---|---|
| `e1_manifold.py` | is the tree a valid diffusion manifold? | pure-noise DDIM samples land in seen compositions at 0.88 (leaf field) / 0.94 (depth-annealed frontier field); ascended modes of `p_t` match deep tree nodes at low noise and coarsen to depth-2 internal nodes at `u=0.8` |
| `e2_composition.py` | can one OOD query be recreated by composing prototypes? | PoE teacher `μ_T` probe-reads the queried (digit, fg, bg); baselines (query-only, top-k retrieval) fail exactly the way the paper reports; guided-sampling tuning in progress — see `PLAN.md` §3 |
| `e3_lifelong.py` | does repeated consolidation degrade the manifold? | smoke: one consolidation drifts old-data loglik by −0.03 nats; full curves (graft vs destructive `ifit` control) pending |

Run everything through SLURM (`experiments/run.sh`; CPU-only by design):

```bash
sbatch --export=ALL,SCRIPT="tests/test_smoke.py" experiments/run.sh
sbatch --export=ALL,SCRIPT="experiments/e1_manifold.py" experiments/run.sh
```

or locally at toy scale: `python experiments/e1_manifold.py --quick` (1–3 min, any machine with
numpy+matplotlib; MNIST idx files are auto-located or downloaded).

## What this is not

- It does not deprecate [cobweb-poincare]: that repo studies the *representation* (hyperbolic
  geometry, encode/decode/interpolate); this one studies the *generative loop* (diffusion-style
  sampling, composition, lifelong consolidation) over the same kind of tree, and reuses its
  benchmark unchanged.
- It is not a reimplementation of Cobweb: the backend is [cobweb-private]'s `CobwebContinuousTree`
  (`covar_from=1`, `var = sum_sq/n + prior_var`, matching cobweb-poincare's configuration exactly);
  `lifelong_ood/tree.py` is a portability fallback only.

`PLAN.md` is the living working plan: current status, open issues in priority order, and the exact
next actions.

[poe-diffusion paper]: ../poe-diffusion.pdf
[cobweb-poincare]: https://github.com/karthiksing05/cobweb-poincare
[cobweb-private]: ../Chunkformer/cobweb-private
