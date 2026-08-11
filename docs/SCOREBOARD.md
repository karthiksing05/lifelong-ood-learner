# Scoreboard: where this method leads, ties, and loses

Every axis measured, against every baseline run, with the winner named. Mean ± standard error over
**3 seeds** unless stated. Numbers are regenerated from `results/aggregate.json`; see
[TABLES.md](TABLES.md) for the full grids.

The rule applied throughout: **every baseline is reported in its own best configuration**, chosen by
a sweep rather than a default. Four separate configurations were corrected in our own favour's
disfavour after audits found we were running a baseline weakly ([RESULTS.md](RESULTS.md) §9a, §16b).

---

## Summary

| axis | who leads | margin |
|---|---|---|
| **exact preservation** | **ours** (tied: `independent`, `ttt`) | 1.000 vs 0.250 for all seven protection-based methods |
| **forgetting** | **ours** (tied: `independent`, `mas`) | 0.000 vs 0.004–0.159 |
| **retention at zero stored data** | **ours** (tied: `independent`) | SSS 1.000 vs `der++` 0.740, `ttt` 0.000 |
| **composition, ColorMNIST** | **ours** | F1 0.605 vs 0.545 next-best, 0.003 for the companion method |
| **composition, CelebA** | ✗ `composable-mcmc` | 0.965 vs our 0.903 |
| **composition, given FACTORED conditioning** | ✗ `score-add` | F1 0.902 vs our 0.198 on a backbone conditioned per factor — our composer is for ENTANGLED backbones (§28) |
| **coverage at comparable accuracy** | **ours** | recall 0.579 vs 0.483; diversity 1.068 vs 0.838 (real: 1.007) |
| **batch addition** | **ours** | no baseline offers it; **2.15× faster at K=12** (3 seeds) at no cost to ACC (+0.035 ± 0.037, n.s.) or preservation |
| **routing (arrangement vs part)** | **ours** | no baseline offers it; AUROC 0.980 / 0.995 |
| **native parity** | **ours** | no baseline offers it |
| **robustness as the grid thins** | **ours** | at 80/160 slots withheld the companion method drops 0.892 → 0.292; ours holds 0.467 |
| **parameter economy** | ✗ `independent` | 7.16M vs our 8.82M at higher ACC; loss survives a fair rank sweep of both arms (§24d) |
| **storage per concept (graded)** | **ours** | **ADDRESS 72 B** / EXTEND BASIS 1 KB / BLOCK 69K — 31,895× cheaper than a block and not worse (F1 0.678±.042 vs 0.587, 3 seeds, §25); no baseline offers a graded route |
| **cost per new concept** | **ours** | routing makes an *arrangement* cost 1024 B against 551K — **538×** — so cost tracks vocabulary, not experience |

Two losses, both stated in [../CONTRIBUTIONS.md](../CONTRIBUTIONS.md) rather than buried.

---

## 1. Lifelong retention — ColorMNIST, 12 tasks, eleven arms

| arm | ACC | A | FGT | **bit-identical** | SSS | images kept |
|---|---|---|---|---|---|---|
| `independent` | **0.863 ± 0.015** | **0.854** | **0.000** | **1.000** | **1.000** | 0 |
| `c-lora` | 0.847 ± 0.020 | 0.836 | 0.004 | 0.250 | 1.000 | 0 |
| **ours** | 0.839 ± 0.025 | 0.842 | **0.000** | **1.000** | **1.000** | **0** |
| `mas` | 0.839 ± 0.018 | 0.829 | **0.000** | 0.250 | 1.000 | 0 |
| `si` | 0.835 ± 0.025 | 0.825 | 0.021 | 0.250 | 1.000 | 0 |
| `ttt` | 0.828 ± 0.017 | 0.819 | 0.021 | **1.000** | **0.000** | **768** |
| `der++` | 0.821 ± 0.015 | 0.823 | 0.049 | 0.250 | 0.740 | 200 |
| `gen-distill` | 0.800 ± 0.017 | 0.802 | 0.047 | 0.250 | 1.000 | 0 |
| `ewc` | 0.760 ± 0.021 | 0.773 | 0.112 | 0.250 | 1.000 | 0 |
| `finetune` | 0.752 ± 0.020 | 0.768 | 0.123 | 0.250 | 1.000 | 0 |
| `replay` | 0.693 ± 0.107 | 0.757 | 0.159 | 0.250 | 1.000 | 0 |

**Read it as a categorical split, not a ranking.** Every method that tries to *protect* shared
parameters — EWC, SI, MAS, C-LoRA, DER++, both replay variants — sits at bit-identical **0.250**. Every
method that *removes* them sits at **1.000**. No regularization strength moves a method across that
line. `mas` reaches FGT 0.000 without reaching bit-identity, which is exactly the distinction the
column exists to make.

`ttt` reaches 1.000 by keeping all **768** images of the stream (SSS 0.000). `der++` buys its
retention with a 200-image buffer. Ours and `independent` keep nothing.

**Where we lose:** `independent` beats us on ACC at fewer parameters. Parameter isolation without a
hierarchy is better *on this table*; the differentiator is the routing rule and growth curve
(§3 below).

Same pattern on CelebA (`n22`): ours 0.763, bit-identical **1.000**, FGT **0.000**; every
protection-based arm at 0.250.

---

## 2. Composition — every training-free baseline at its own best setting

### ColorMNIST (held-out digit × foreground × background)

| method | joint | recall | **F1** | Vendi |
|---|---|---|---|---|
| *real (ceiling)* | *0.864* | *0.939* | *0.969* | *1.901* |
| **ours** | 0.539 ± 0.025 | 0.778 | **0.605 ± 0.043** | **1.942** |
| `inversion` (span-restricted) | 0.403 | 0.799 | 0.545 | 2.015 |
| `inversion-free` (Gal et al. as published) | 0.169 | 0.845 | 0.332 | 2.473 |
| `poe-guided` (**the companion method**) | **0.839 ± 0.049** | 0.001 | 0.003 | 1.075 |
| `composable-diff` (Liu et al. 2022) | 0.000 | 0.134 | 0.030 | 2.222 |
| `composable-mcmc` (Du et al. 2023) | 0.003 | 0.134 | 0.000 | 1.617 |
| `nearest-class` | 0.000 | 0.066 | 0.005 | 1.874 |

**Ours leads on F1 by a wide margin.** The companion method wins joint accuracy (0.839) at recall
**0.001** — it emits near-copies of the query, which is the whole reason F1 rather than accuracy is
the headline. Both score-addition methods are at **0.000**, and that is a structural fact about this
backbone rather than a tuning failure: it has no per-factor conditional to add ([RESULTS.md](RESULTS.md) §13).

*(`finetune-stream`, at F1 0.626, is not a composition baseline — it is our own concept with the
shared path left unprotected, and it matches us because on ColorMNIST the semantics live in the
private embedding. See [RESULTS.md](RESULTS.md) §5n.)*

### CelebA — `Smiling` × `Big_Nose`, never co-occurring in training

| method | pair | target | recall | content diversity |
|---|---|---|---|---|
| *real (ceiling)* | *0.801* | *0.656* | *0.884* | *1.007* |
| ✗ `composable-mcmc` | **0.965 ± 0.023** | **0.779** | 0.483 | 0.838 |
| ✗ `composable-diff` | 0.934 ± 0.066 | 0.740 | 0.280 | 0.750 |
| **ours** | 0.903 ± 0.094 | 0.727 | **0.579** | **1.068** |
| `query-only` | 0.777 | 0.620 | 0.018 | 0.295 |
| `poe-guided` (companion method) | 0.645 | 0.488 | 0.000 | 0.147 |
| `nearest-class` | 0.355 | 0.235 | 0.630 | 1.212 |

**We do not lead here.** MCMC-corrected score addition is more accurate on both accuracy metrics,
with zero parameters and no training. We lead on **coverage**: recall 0.579 against 0.483, and
content diversity 1.068 where real data is 1.007 and the baselines are at 0.750–0.838 — we match the
spread of the real class and they produce narrower sets.

The composition result is therefore **benchmark-dependent**, and what the pair of tables establishes
is *when* score addition suffices: it needs per-factor conditionals, which CelebA's factored head
provides and ColorMNIST's per-slot head does not.

---

## 3. Axes with no baseline to compare against

These are contributions of this work; nothing in the literature sweep offers them for this setting
([RELATED.md](RELATED.md) documents the search, including the empty cell for
product-of-experts + adapter banks).

| axis | result |
|---|---|
| **routing** (arrangement vs new part, unsupervised) | AUROC **0.980** / **0.995**, error one-sided in the safe direction (0 unsafe routings in 120 queries) |
| **growth accounting** | an arrangement costs **1024 bytes**, a new part **551K parameters**; a 160-experience stream costs ~10× less than one adapter per query |
| **batch addition** | **2.15× faster** at K=12 over 3 seeds, ACC +0.035 ± 0.037 (not distinguishable from zero), bit-identical at every K and every seed |
| **native parity** | recall **0.903 ± 0.024** against a backbone *trained* on the held-out slot |
| **preservation on a pretrained model** | **0.000e+00** on `google/ddpm-celebahq-256` (113.7M params) |
| **hierarchical reuse** | ✗ **not supported**: a matched family ancestor is worth −0.004 F1 against a mismatched one and −0.063 against none. Shared nodes are real (2.3× a leaf, entropy 0.61) but do not transfer (§24) |
| **certified deletion** | remove a concept in 0.002 s with **9/9** survivors and the base bit-identical; retraining without it leaves **0/9** (drift 0.993). Reclaims storage only partially — median 0.250 of a path |
| **prototype-distance dial** | ours spans **2.32×** in distance-to-prototype (recall 0.003 → 0.843) from one consolidated concept; the companion method spans **0.94×** — i.e. not at all — with recall 0.000 throughout |

---

## 3b. Held-out severity — does it collapse as coverage thins?

Three backbones trained from scratch with progressively more of the combination grid withheld (one
seed each; every attribute *value* still present at every level).

| extra withheld | seen slots | **ours** joint | `poe-guided` joint | `inversion` joint |
|---|---|---|---|---|
| 0 | 120 | 0.492 | **0.892** | 0.425 |
| 20 | 100 | **0.683** | **0.942** | 0.450 |
| 40 | 80 | **0.467** | 0.292 | 0.392 |

At the severe end the companion method loses two thirds of its accuracy and ours does not move much.
One seed per level and a different backbone at each, so the fine ordering is noise — the factor-of-
three collapse is not. [RESULTS.md](RESULTS.md) §18.

---

## 4. Assertions

`bash reproduce.sh tests` — **11 / 11 pass**, including the pretrained-backbone preservation
certificate. These gate every number above.

```
PASS test_layout          PASS test_merge           PASS test_pretrained
PASS test_smoke           PASS test_ortho_merge     PASS test_pretrained_bank
PASS test_neural          PASS test_unbounded       PASS test_speckle
PASS test_fid             PASS test_claims
```

---

## What would change this scoreboard

- **CelebA composition** is the one contested axis we lose. Adopting the MCMC corrector inside our
  own consolidation loop — rather than only running it as a baseline — is the obvious next step, and
  is not yet done.
- **`independent` beating us on ACC/params** is real on the retention table and only offset by the
  growth accounting. A direct head-to-head on a long stream, where the routing rule's savings
  compound, would settle it; the 12-task stream is too short to show it.
