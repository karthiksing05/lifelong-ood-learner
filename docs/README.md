# The full record

Everything this project measured, including what it refuted. Start at
[../README.md](../README.md) for the argument and [../CONTRIBUTIONS.md](../CONTRIBUTIONS.md) for the
claims; this folder is the evidence behind them.

| file | what it is |
|---|---|
| **[SCOREBOARD.md](SCOREBOARD.md)** | every axis vs every baseline, with the winner named — start here |
| **[RESULTS.md](RESULTS.md)** | the complete lab record, in the order things were learned |
| **[TABLES.md](TABLES.md)** | every headline number with error bars, regenerated from `results/` |
| **[RELATED.md](RELATED.md)** | positioning, which baselines are fair for this setting, citation hazards |
| **[FINAL_REPORT.md](FINAL_REPORT.md)** | where this stands against every baseline, on the axis each was built for |
| **[SETTINGS.md](SETTINGS.md)** | every hyperparameter vs the companion paper's Tables 5–8, marked MATCH / DEVIATION / INFIDELITY |
| **[ASSESSMENT.md](ASSESSMENT.md)** | publication readiness, assessed against what the numbers support |

---

## Every experiment, and the question it answers

Each writes per-seed JSON to `results/<name>__seed<k>.json`, combined by
`experiments/aggregate.py` into `results/aggregate.json`, which is what the figures and tables read.

### The main results

| experiment | question | answer | § |
|---|---|---|---|
| `n9_showdown` | can a held-out (digit, fg, bg) combination be produced, against every training-free baseline at its own best setting? | ours F1 **0.605 ± 0.043**; the companion method 0.003; score addition **0.000** | §11a, §11d |
| `n20_celeba_composition` | the same, where the factors overlap spatially | `pair` **0.903 ± 0.094**; MCMC-corrected score addition reaches 0.965 | §11b, §11c |
| `n5_classinc` | class-incremental ColorMNIST against eleven CL methods | bit-identical **1.00** vs 0.25; FGT **0.000** | §10, §11 |
| `n22_celeba_lifelong` | the same on overlapping factors | bit-identical **1.00**; FGT **0.000** | §7d |
| `n18_router` | can the system decide *arrangement vs new part* by itself? | AUROC **0.980 / 0.995**, error one-sided | §6c |
| `n23_native_parity` | is a consolidated concept as good as a trained one? | recall **0.903 ± 0.024** | §6d |
| `n24_growth` | what does the *n*-th concept cost? | linear in vocabulary, flat in experience | §5z |
| `n4_batch` | can concepts be absorbed in groups? | **2.15× faster** at K=12 (3 seeds), ACC +0.035 ± 0.037 (n.s.) | §16, §16b |

### The pretrained backbone (`google/ddpm-celebahq-256`, 113.7M params, 256×256)

| experiment | question | answer | § |
|---|---|---|---|
| `test_pretrained_bank` | does preservation transfer to a model we did not train? | **0.000e+00** on all three assertions | §12a |
| `n25_pretrained_case` | can concepts be acquired at 256px without damaging the model? | **1.00** on each attribute, Vendi *above* the untouched model | §12c, §12e |
| `n26_batch_pretrained` | do batch updates work there? | bit-identical **1.00** on all arms | §15 |
| `n15_pretrained_celeba` | the earlier attempt, and why it failed | conditioning capacity — superseded by §12 | §5u |
| `diag_pretrained_render` | is "the bank renders nothing" real? | **no — a renderer reading HWC rows as CHW**; gradient ratio 2.080 vs 0.963 | §30d-CLOSED |

### Reproducing the companion paper on ITS protocol

Every composition number elsewhere in this repository is measured *our* way: 24 samples from one
query, one reference set, pixel features. Wang et al. §4.2 pool **100 queries × 16 generations**
against **two** reference sets in **Inception-V3**. That difference is worth four experiments,
because measured their way their method reproduces here and measured ours it reads 0.001 against
their published 0.807.

| experiment | question | answer | § |
|---|---|---|---|
| `diag_paper_protocol` | how much of the gap is the protocol? | recall 0.000 → **0.590** as queries pool 1 → 100 | §5e-bis |
| `n49_paper_protocol` | their Table 1, all five arms plus ours | top-3 F1 **0.027/0.053** vs their 0.051/0.053; ours covers from **one** query | §5e-ter |
| `n47_paper_backbone` | their Table 5 CelebA column (~13 h) | fine-tune converges, loss 0.0104 — and it is **required**: from scratch never learns conditioning | §29(d) |
| `n48_paper_composition` | their Table 2, on their own backbone (`figures/case_celeba.png`) | PoE reproduces (F1 0.187 vs their 0.203; G-precision 0.753 vs 0.745) — **and ours clears the unconditional floor 2.7× on joint accuracy** (0.439 vs 0.161 pixel probes; 2.3× Inception), on every class in both spaces. Coverage-F1 still ties the floor (0.401 vs 0.409). | §5e-sexies |
| `diag_holdout_leak` | are the prototypes memorized held-out images? | **no** — 24/24 nearest to TRAIN, split clean | §5e-quinquies |
| `diag_selection_bias` | does discovery just find the query? | **no** — query-agnostic modes carry the composition intact | §5e-quater |

### Representational strength, and the layer that makes new datasets cheap

| experiment | question | answer | § |
|---|---|---|---|
| `n50_representation` | is an ABSORBED concept's representation as separable as a TRAINED one? | **no, and it barely beats no adaptation**: ours 0.271 ≈ base 0.266 vs oracle 0.406 (linear digit) | §6e |
| `make_n5_curves` | what do the incremental curves look like, not just their summary? | `figures/06_learning_curves.png` — ours/independent are **exactly flat**, `finetune` falls 0.73→0.66 in one step | §10, §11 |
| `make_performance_figures` | the two main performance tests that had numbers but no figure | `figures/09_representation.png` — the honest negative drawn: `ours` sits on `base`; `figures/05_batch_invariance.png` — ACC flat in K, **2.15× at K=12**, bit-identical at every point | §6e, §16/§16b |
| `n4_batch` (figure) | does absorbing concepts in GROUPS cost anything? | no: accuracy flat within error, preservation exact at every K | §16, §16b |
| `n18_router` (figure) | the three-way self-routing decision | `figures/extra/f7_router.png` — AUROC 0.980 / 0.995, self-routing 0.858 ± 0.018, **under-spend 0.000** | §6c |
| `n11/n21` (figure) | what a new *vocabulary* item costs | `figures/extra/f4_vocabulary.png` — one shared direction at 1 KB beats per-concept adapters at 551 K | §5q, §7c |
| `diag_iterative_select` | does selecting concepts iteratively beat one greedy pass over a fixed pool? | contribution C2, budget-matched against the paper's Eq. 8–9 route | `docs/OUTLINE.md` |
| `diag_mdl_router` | can description length replace the router's two calibrated thresholds? | **not at one query per concept** (0.333 vs 0.833), but **at N≥32 it draws level** (0.722–0.778) with no calibrated threshold, and its promotion curve halts each population at the right rung. `figures/07_router_exchange_rate.png` | §6c-ter |
| `diag_elbo_stability` | is that codelength measurable, or is it reading sampling noise? | measurable — paired \|mean\|/sd **17.3 → 93.6** on the hardest case; but its **sign flips between noise bands**, so any claim from it must name the band | §6c-ter |
| `n52_shapes3d` | does the contract actually make a THIRD dataset cheap? | Shapes3D on the existing n1 recipe — ADDRESS vs nearest-class vs unconditional on 100 held-out compositions, plus the router's `s_novel` | §5t-bis |
| `diag_shapes3d_probes` | can the Shapes3D factors be read from pixels before anything is scored with them? | yes at the settings n52 uses — floor_hue **0.999**, object_hue **1.000**, shape **0.973** vs base 0.130/0.114/0.260 | §5t-bis |
| `n55_dit_train` / `n56_dit_full_loop` | is the three-tier scheme UNet-specific? | **no** — a latent DiT recovers a basis at R² **0.832** (UNet 0.870) and `factorize`/`compose` run on it unchanged; full loop reads the composition from ONE query at **0.917** and scores joint **0.661** (oracle 0.859) | §7z |
| `n54_ae_train` | does the latent space keep every factor the composition is scored on? | accepted on PROBES not MSE — a first fit had mse 0.00044 and lost foreground (1.000→0.835); the accepted one is mse 0.00020, worst drop +0.002 | §7z |
| `n57` / `n58_shapes3d_results` | Shapes3D on a PRETRAINED unconditional backbone, the N47 recipe | `eurecom-ds/scoresdeve-ema-shapes3d-64` bridged VE→VP by `neural/ve_bridge.py`; conditioning gap reported FIRST because four from-scratch runs left the label decorative | §5t-bis |
| `diag_ve_parameterisation` | what does a variance-exploding checkpoint actually predict? | measured, not guessed: ε = −out·σ on [0,1] inputs, MSE **0.0002** against 0.64–441 for every alternative | §5t-bis |
| `lifelong_ood/data/factored.py` | one contract for any factored dataset | ColorMNIST's 3 named factors and CelebA's 14 binary attributes map onto the same shape, so adding Shapes3D/CLEVR is loader-sized rather than a refactor — **neither is registered or run** | `test_factored` |

`factored.py` is additive — nothing existing changes. Its guarantee is asserted rather than assumed:
its generic mixed-radix encoding reproduces ColorMNIST's own slot ids **exactly** and agrees with
`idealwords.slot_factors`, so the 39 sites reading `data["slot"]` cannot be silently relabelled. It
also enforces the distinction the three-tier router exists to make: `holdout_mask` never removes a
factor VALUE entirely, because a split that does has quietly converted a new *arrangement* into new
*vocabulary* — a different experiment.

### The analysis experiments

These answer *why* something behaves as it does. They are not methods, and §13 in particular uses
stronger supervision than the OOD setting provides.

| experiment | question | answer | § |
|---|---|---|---|
| `n27_factor_conditionals` | is score addition's failure the operator, or the conditionals? | the conditionals: F1 **0.016 → 0.490** on the same backbone | §13 |
| `n28_update_rules` | which update rule disturbs the score field least, and does that help? | it does **not** — 4× less drift buys no composition and a worse concept | §14 |
| `n11_basis_extension` | what does a new vocabulary *value* cost vs a new arrangement? | 1024 bytes vs 551K params | §5q2 |
| `n8_idealwords` | is the conditioning space additively factored at all? | **yes** — `c(d,f,b) ≈ mu + D + F + B`, R² 0.826. The whole composer rests on this | §5q |
| `n7_inversion` | can a concept be learned as a POINT in conditioning space rather than guided toward? | yes, and it is what covers the class where guidance cannot | §5f |
| `n29_slot_scan` | how does performance vary across ALL 40 held-out compositions? | median joint 0.438, range [0.00, 0.94]; strongly colour-dependent | §11e |
| `n21_celeba_vocabulary` | can an attribute value the backbone never saw be added? | only with a weight update from real images | §7c |

### The negative results

Kept because they shaped the design.

| experiment | hypothesis | verdict | § |
|---|---|---|---|
| `n14_hierarchy` | a Cobweb-style hierarchy helps placement | **refuted** — clustered is the worst arm | §5s |
| `n12_reconsolidate` | deltas can be merged back into the base | **refuted** without projection + rehearsal | §5m |
| `n17_ortho_merge` | orthogonal deltas make merging safe | partial — 2.8× smaller delta, merge still closed | §5w |
| `n16_alternatives` | three alternatives to a per-concept LoRA | all worse on the goal's own loop | §5v |
| `n6_diversity` | the PoE teacher can be sampled into a covering set | **partly restored** — 0.00 per query, but 0.59 pooled over 100 as the paper does | §5e-bis |
| `n49_paper_protocol` | their Table 1 reproduces under their own protocol | Generalization precision **0.664** vs their 0.670; recall does not | §5e-bis |
| `n13_stream` | closure: does an accumulated bank make the next concept cheaper? | **refuted** on ColorMNIST | §5y |

### The diagnostics that *choose* configurations

Not checks — these select the settings the headline numbers are produced with, so that every arm is
reported at its own best.

| diagnostic | what it decides |
|---|---|
| `tests/diag_poe_faithful.py` | that the companion method runs on **its** Eq. 14 schedule, not ours (precision 0.039 → 0.839) |
| `tests/diag_composable_scale.py` | Composable Diffusion's guidance scale, swept rather than defaulted to 1/n |
| `tests/diag_ours_quality.py` | our pool-guidance weight (ColorMNIST joint 0.125 → 0.375) |
| `tests/diag_pretrained_lora.py` | the pretrained backbone's adapter rank, learning rate and noise range |
| `tests/diag_celeba_guidance.py` | the CelebA guidance grid, extended to 6.0 on measured evidence |
| `tests/diag_router_signals.py` | the router's signal — absolute error, not a ratio (AUROC 0.199 → 0.985) |

### The assertions

`bash reproduce.sh tests` — these must pass before any number is believed.

| test | asserts |
|---|---|
| `test_layout.py` | images are not silently transposed (the bug that invalidated a whole CelebA run) |
| `test_merge.py` | merging a delta is exact for the concept just learned (1.1e-06) |
| `test_ortho_merge.py` | orthogonal deltas merge without disturbing what came before |
| `test_pretrained_bank.py` | the bank attaches to a pretrained UNet at **0.000e+00**, and preservation holds |
| `test_unbounded.py` | the bank keeps accepting concepts without degradation |
| `test_neural.py`, `test_smoke.py`, `test_pretrained.py` | end-to-end paths run |
| `test_claims.py` | every number quoted in these documents matches `results/aggregate.json` (33/33) |
| `test_figures_fresh.py` | no published figure is older than the code that draws it (advisory) |

### The figures

`figures/` holds only what the documents cite, plus the intermediates `make_paper_figures.py` copies
from. It had grown to 48 PNGs and 41 MB, of which 19 were referenced; the rest were smoke runs,
superseded variants and one-off diagnostics that no reader — and no staleness check — could tell
apart from the real ones.

`python experiments/curate_figures.py` re-derives that split mechanically (cited by path from any
markdown file, or named as a source in the rename map) and prints the plan; `--apply` moves the
remainder to **`figures/archive/`**. Archived, not deleted: several came from runs whose checkpoints
are gone, so restoring one is a `git mv` rather than a re-run. It also reports figures a document
cites but which do not exist, which is how the dangling `f9b_showdown_colormnist.png` reference was
found.

Current: **30 maintained figures, ~23 MB**, with 23 in the archive.

`tests/test_figures_fresh.py` cross-checks each figure's mtime against its generator's. It used to
report "11 with no generator found" because it only recognised names of the form `f<digits>_`, so
every *intermediate* — `n18_router.png`, `n24_growth.png`, `n26_batch_pretrained.png` — went
unchecked. Those are precisely the files the published figures are copied FROM, so a stale
intermediate propagated silently into a cited figure. It now attributes **every** figure to a
generator.

---

## The corrections

Results in this repository that corrected earlier results in this repository. They are kept because a
record that shows only what survived is not evidence that anything was checked.

| what was wrong | how it was found | effect |
|---|---|---|
| uint8 images trained in NCHW while sampled as CHW | **looking at a sample grid** — tiled mosaics | every prior CelebA number discarded |
| the companion method run on *our* guidance schedule | comparing case studies against the paper's | its precision 0.039 → **0.839** |
| Composable Diffusion pinned at w = 1/n | a literature audit of our own code | it became competitive, then beat us on CelebA |
| textual inversion confined to `span(C)` | the same audit | the baseline had been assuming our claim |
| DER++ replaying a stored ε against *fresh* noise | disbelieving ACC 0.250 for a strong baseline | 0.250 → **0.781** |
| `cl_metrics` using a nan-poisoning `np.mean` | a column of all-`nan` in a table | CelebA's discriminative metric recovered |
| batch vs sequential using different discovery configs | K=1 not being a no-op when it must be | the "cost of batching" disappeared |
| drift metric reading a constant 0.0000 | five different rules scoring identically | the metric became informative |
| our pool guidance never calibrated (CelebA's was) | asking why our case studies looked worse | ColorMNIST F1 0.498 → **0.605** |
| the companion method scored on **our** protocol and published as **theirs** | reading their §4.2 after doubting recall 0.001 against their 80.7 | their method reproduces: General. precision **0.664 vs their 0.670** (§5e-bis) |
| `evals._feat` asserting metric orderings survive a change of feature space | measuring it — the same 1600 images give recall 0.000 (pixel) and 0.180 (Inception) | `paper_metrics` added, naming its space |
| the CelebA panel labelling six *other* class images as "real query" | a colour cast that tracked the query but had nothing to compare against | the apparent contamination was the figure's indexing (§30e) |
| n48 passing the teacher **mean** where an ε-prediction belongs | auditing what "the published method on the published backbone" actually computed | the arm was neither; now Eq. 12/14 via the shared code |
| a bare `aggregate.py` silently demoting aliased canonical keys to stale smoke runs | running it, and watching `test_claims` drop 33/33 → 23/33 | it now **refuses** unless `--force`, with provenance recorded |
| concept vector written into a pretrained model's **time** embedding | melted faces at 256px | unblocked pretrained acquisition |
| adapters on `time_emb_proj` (32 of 58 Linear modules) | counting what "adapt the linears" actually hit | same |
