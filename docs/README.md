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
| `n4_batch` | can concepts be absorbed in groups? | **2.14× faster and +0.052 ACC** at K=12 | §16, §16b |

### The pretrained backbone (`google/ddpm-celebahq-256`, 113.7M params, 256×256)

| experiment | question | answer | § |
|---|---|---|---|
| `test_pretrained_bank` | does preservation transfer to a model we did not train? | **0.000e+00** on all three assertions | §12a |
| `n25_pretrained_case` | can concepts be acquired at 256px without damaging the model? | **1.00** on each attribute, Vendi *above* the untouched model | §12c, §12e |
| `n26_batch_pretrained` | do batch updates work there? | bit-identical **1.00** on all arms | §15 |
| `n15_pretrained_celeba` | the earlier attempt, and why it failed | conditioning capacity — superseded by §12 | §5u |

### The analysis experiments

These answer *why* something behaves as it does. They are not methods, and §13 in particular uses
stronger supervision than the OOD setting provides.

| experiment | question | answer | § |
|---|---|---|---|
| `n27_factor_conditionals` | is score addition's failure the operator, or the conditionals? | the conditionals: F1 **0.016 → 0.490** on the same backbone | §13 |
| `n28_update_rules` | which update rule disturbs the score field least, and does that help? | it does **not** — 4× less drift buys no composition and a worse concept | §14 |
| `n11_basis_extension` | what does a new vocabulary *value* cost vs a new arrangement? | 1024 bytes vs 551K params | §5q2 |
| `n21_celeba_vocabulary` | can an attribute value the backbone never saw be added? | only with a weight update from real images | §7c |

### The negative results

Kept because they shaped the design.

| experiment | hypothesis | verdict | § |
|---|---|---|---|
| `n14_hierarchy` | a Cobweb-style hierarchy helps placement | **refuted** — clustered is the worst arm | §5s |
| `n12_reconsolidate` | deltas can be merged back into the base | **refuted** without projection + rehearsal | §5m |
| `n17_ortho_merge` | orthogonal deltas make merging safe | partial — 2.8× smaller delta, merge still closed | §5w |
| `n16_alternatives` | three alternatives to a per-concept LoRA | all worse on the goal's own loop | §5v |
| `n6_diversity` | the PoE teacher can be sampled into a covering set | **refuted** — recall 0.00 in every configuration | §5e |
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
| concept vector written into a pretrained model's **time** embedding | melted faces at 256px | unblocked pretrained acquisition |
| adapters on `time_emb_proj` (32 of 58 Linear modules) | counting what "adapt the linears" actually hit | same |
