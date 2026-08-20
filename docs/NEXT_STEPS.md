# Where this stands, and what to do next

Written 2026-08-13. Companion to `docs/FINAL_REPORT.md` (the argument) and `docs/RESULTS.md` (the
evidence, section by section). This file is the working state: what is settled, what is open, and
what the next run should be.

---

## PIVOT, 2026-08-19: pretrained backbones, and an instrument gate in front of everything

**What changed and why.** The from-scratch density holdout is retired. CelebA could never support it
(§7ee: 12 cells at ratio 1.1, a readout ceiling of ~0.55 against a 0.117 floor, and no enlargement
possible at any resolution or probe strength), and chasing image quality there was answering the
wrong question — quality was never the binding constraint, the instrument was. The pretrained regime
is also the only one a VLA will ever be in.

**The claim is restated.** Not *"the model never saw this combination"* but *"the conditioning
interface cannot ADDRESS it; we make it addressable at ~0 parameters, from one example, without
disturbing anything already learned."*

### Settled this session

* **`lifelong_ood/instrument.py`** — one gate every experiment calls before it measures: conventions,
  per-target readout ceiling on 200 real images, dynamic range against an *empirical* floor, and the
  classes-to-free-vectors ratio. `tests/test_instrument.py` calibrates it against known-broken
  instruments and reproduces §7ee's mechanism in eight lines of numpy. Wired into N73, it recovers
  the same 16-of-36 and every pooled number to three decimals.
* **A fourth fault, retroactive** (§7gg): at n=24 the smallest resolvable margin is 0.11–0.13 and
  N73 declares verdicts at +0.10. §7ee's ADDRESS-over-floor result (+0.180) is safe; the
  ADDRESS-over-retrieval margin (+0.018) was never resolvable by any run.
* **Conventions are declared, not defaulted** — `data_space` per field class, `clip="auto"` derived
  from it. The analytic field stays bit-identical; the VE bridge keeps its own `(0,1)`.
* **The feature-space migration was measured and mostly declined.** `tests/test_feature_space.py`
  calibrates pixel against Inception on four arms with known answers. Both pass on ColorMNIST →
  nuisance, keep pixel. Pixel *fails* on CelebA-grid data (it certifies scrambled pixels at
  precision 0.375) → a fix there, but that dataset is retired. 15 pinned claims left untouched
  because a calibration said so, not because nobody looked.
* **Shapes3D is the flagship, and it passes the gate decisively.** At `value_stride=1`: 400 classes
  against 25 free vectors (ratio 18.2), **every factor readable on real images** (floor_hue 1.000,
  object_hue 1.000, shape 0.994), ceiling 0.994 against a chance of 1/400, 12/12 targets readable.
  `object_hue`'s long-standing blindness was a *table-size* artifact — at stride 2 it reads 0.236.
* **The addressing claim does NOT transfer to language conditioning** (§7ff). On 400 CLIP phrases,
  composing ties the literal phrase (rank 36.00 vs 35.25 of 400). For a language-conditioned policy
  the contribution must be **storage and routing**, not addressing.
* **`n77_storage_any.py`** — the N71 loop against the generic contract, so it runs on any factored
  backbone. N71 is untouched because its numbers are pinned.

### Resolved after the summary, and it changed three conclusions

All three items on the "chase next" list were **one bug**: `n77` loaded its data with loader
defaults instead of the split recorded in its checkpoint (`wall_hue` pinned). Corrected, the loop
**closes on Shapes3D** — read 5/5, joint 0.975, recall 0.908, read and storage cost +0.000,
forgetting +0.000, base bit-identical. The "coverage does not transfer" negative and the
"`floor_hue` fails under both readers" finding are both withdrawn; `instrument.check_split` now
makes the fault fatal at the gate. And the low diversity (vendi 1.14) is the backbone's, not the
composer's — a genuine trained class reads 1.19.

### Open

1. ~~**The Shapes3D flagship run itself.**~~ **DONE** — §7jj (composition) and §7kk (storage). The class table is training at the full readable design via
   the N57 pattern (frozen backbone, `cemb` in its own optimiser group at 50× LR). Then: basis →
   ADDRESS vs nearest-class vs unconditional → routing → N77 storage.
2. **`router.calibrate_u_novel` is still called by no experiment**, and the inherited `U_NOVEL = 0.10`
   reads AUROC 0.354 on a pretrained backbone against 0.861–0.932 calibrated.
3. **EXTEND BASIS vs BLOCK is still decided by hand** — a 69× cost decision the router does not make.
4. **Sequence-valued conditioning** is unsupported anywhere, and Phase 4 needs to know whether a
   pooled instruction embedding costs task performance before that matters.


## 1. Settled this session

### 1a. The CelebA composition negative was a measurement bug, and it is reversed

`RESULTS §5e-sexies`, `FINAL_REPORT §3b-CelebA`. Both sections previously said "ours does not clear
the unconditional floor on CelebA" and scoped the composition claim to 3-factor benchmarks. The cause
was two bugs in `evals.py`:

* `attr_accuracy` indexed the target combination by position in a **filtered** attribute list instead
  of by canonical position, so probes were scored against other attributes' targets;
* `probe_usable` admitted `Bald` (base rate 0.000 in this split — a probe that cannot be wrong).

Re-run on their own backbone under their own protocol, 6 held-out classes:

| | ours `joint` | floor | ratio | Generalization F1 |
|---|---|---|---|---|
| pixel probes | 0.439 | 0.161 | **2.7×** | 0.401 vs floor 0.409 |
| Inception probes | 0.245 | 0.108 | **2.3×** | (coverage identical) |

Clears the floor on **every class in both probe spaces** (1.6×–6.3×). Their PoE reproduces:
G-precision **0.753 against their published 0.745**, F-F1 0.187 vs 0.203.

**The half that did not reverse, and must keep being said:** Generalization *F1* 0.401 against the
floor's 0.409 — a tie. That column never touched the probes. The supportable claim is coverage *at*
accuracy (ours is the only arm on both axes), not coverage alone.

The 3-factor scoping is **withdrawn**.

### 1b. The router: a sixth attempt, an honest negative, and a real diagnosis

`RESULTS §6c-ter`, `figures/initial/07_router_exchange_rate.png`. Reframed the decision the way Cobweb makes one —
compare hypothetical operators under an objective with a built-in complexity penalty — using the
storage ladder's own costs in bits:

    L(tier) = C_tier + N × (bits to code one example under that tier)

**At the benchmark's operating point (one query per concept) it is far worse than the cascade**:
0.333 against 0.833. **Given 32+ instances it draws level** — 0.722 at N=32, 0.778 at N=64 against a
flat 0.833, which on 18 concepts is a one-concept gap — with **no calibrated threshold anywhere**. As
a detector it is still worse: AUROC 0.950 on N* against 6c's 0.995. The cascade is not replaced.

The promotion curve is the part a threshold cannot produce: in-distribution **never** promotes at any
depth; new-combination converges to **exactly ADDRESS (1.00) and stops**, not overshooting to the tier
costing 14× more; new-vocabulary passes through it toward EXTEND (1.67 by N=64).

**What it establishes anyway.** The fit term scales with N and the storage cost does not, so the tier
a query warrants is **not a property of the query** — it is a property of the query *and* how many
instances have arrived. A threshold is a function of the query alone, so all six attempts were
answering an underdetermined question. That is one explanation for five separate failures.

Measured exchange rate — the direct answer to "how much does an OOD query warrant addition":

| population | ADDRESS gain | N* ADDRESS | EXTEND gain over ADDRESS | N* EXTEND |
|---|---|---|---|---|
| in-distribution | **−9.7 b/img** | 56 (7/10 never) | +21.1 | 388 |
| new-combination | +33.0 | **15** | +87.6 | 93 |
| new-vocabulary | +22.6 | 18.5 | **+218.0** | **36** |

**≈ 15 instances justify a 72-byte address; ≈ 36 justify a 1 KB vocabulary extension.** The benchmark
supplies one. The promotion ordering is correct — in-distribution is **never** promoted at any N,
vocabulary promotes fastest — and the rate is far too slow to be useful as it stands.

### 1c-bis. F9's case-study panels: a build-script flag, not the method

`RESULTS §20c`. The ColorMNIST `ours` row looked broken (0.33 / 0.00 / 0.00 / 0.00 across blocks).
Cause: `reproduce.sh` passed `--guidance 2.0` to `make_case_studies.py`, whose default is 1.5 and
whose own help says the figure must match the tables. At s=0.9 the same code scores joint **0.583 at
1.5 and 0.083 at 2.0**. Re-rendered at the default the row is **1.00 / 0.17 / 0.50 / 0.67** at recall
0.73–0.99. Override removed; interaction recorded in the `--dial` help.

**The dial was re-confirmed at 0.9 and lowering it is a trap.** At guidance 1.5: s=0.7 gives joint
0.833 / recall **0.116**; s=0.8 gives 0.792 / 0.427; s=0.9 gives 0.583 / **0.854**. The baselines
that re-emit the query score joint 0.958 / recall **0.000**. Lowering the dial makes the panels look
cleaner by moving `ours` toward that corner — accuracy without coverage — which concedes the claim
the arm exists to make. Judge the dial on both columns; never on the renders.

### 1c. Two silent bugs worth remembering

* `elbo_bits` returns **per-image** bits; the diagnostic divided by `len(X)` again, which cancels the
  batch-size scaling and surfaces as a **flat promotion curve** — indistinguishable from a real null.
* `sbatch --export=ALL,SCRIPT="..."` **splits its value on commas**. `--batch-sizes 1,2,4,8,16`
  arrived as `1`; every run for several hours measured N=1 only while the job header printed
  `--batch-sizes 1` and nothing failed. **Use colons in any sbatch-passed list.**

### 1d. Health

`tests/run_all.py` — **13/13 pass** (run via `sbatch`; the login node SIGTERMs torch, exit 143).

---

## 2. Open, in the order worth doing

### 2a. Accumulate instead of deciding on arrival  — the direct follow-on from §6c-ter

§6c-ter makes the shape of the fix precise: at one query per concept the correct MDL answer is
"store nothing", and the system stores something because the benchmark hands it one example where the
accounting needs fifteen. So stop deciding on arrival. Buffer novel queries keyed by their inferred
basis row, accumulate the coding gain, and promote when it crosses the tier cost. This is Cobweb's
actual regime — it restructures as counts grow rather than committing on first sight — and it is
untested here. `mdl_crossover` already returns the promotion point.

Needs: a streaming harness over n18's populations where concepts recur, and a decision on buffer
eviction. No new backbone.

### 2b. Whether §5e-sexies' Generalization-F1 tie can be broken

Ours ties the unconditional floor on coverage F1 (0.401 vs 0.409) while beating it 2.3–2.7× on
accuracy. Diagnose whether that is a sampler-diversity property (the floor buys recall by being
diffuse) or a real coverage limit. The f0 note on prototype-seeded generation is the place to start.

### 2c. The band-dependence of the codelength sign

`diag_elbo_stability` found the sign of the ADDRESS coding gain stable across seeds but **not across
noise bands** — the full variational bound and u ∈ [0.10, 0.70] disagree on a new-vocabulary query
(−17.2 vs +7.6 bits, both stable). The full bound is the principled choice and gives the semantically
right answer, but nothing here explains *why* the mid band inverts. Any future claim resting on that
sign must name its band until this is understood.

### 2d. Carried over, unchanged

* **EXTEND BASIS vs BLOCK** is still a 69× cost decision made by hand; §6c-ter did not touch it
  (the ColorMNIST benchmark has no population that needs BLOCK).
* **`n50`'s readout** — the representation is ε-error under each conditioning, which is defined
  identically across arms but scores low in absolute terms for all of them. A fairer stronger readout
  would settle whether "absorbing a concept does not improve the representation" is about the
  representation or the probe.
* ~~**f10's CelebA case study**~~ **RESOLVED 2026-08-13 — retired, no retraining needed.** f10
  aliased n20, which runs on `ddpm_celeba_factored.pt`: 64 px and carrying no `layout` key, so it
  loads as `permuted` and renders with CNN grid artifacts. **f26 already superseded it** — the same
  row structure (real / prototypes / ours / poe-guided / query-only / unconditional, one block per
  held-out class) on the companion paper's OWN backbone, `ddpm_celeba_paper128.pt`, `layout: hwc` at
  128 px, which n48 was already using. The correct backbone was on disk the whole time. f10 is in
  `figures/initial/archive/` and its alias is gone from `make_paper_figures.py`.

  *Audited while doing it:* every CelebA checkpoint's layout was checked. The whole
  `ddpm_celeba_factored*` family is 64 px/permuted and feeds n20/n21/n22 — but those are cited for
  their **numbers**, not their renders, and the numbers are valid (the model was trained in that
  layout, so loading it as permuted is correct; `test_layout.py` covers this, and the preservation
  claims are layout-independent). `ddpm_celeba_permuted.pt` and `ddpm_celeba_layout_permuted.pt` are
  the deliberate layout ablation. **Every cited CelebA figure — f18, f26 — is on an hwc backbone
  derived from `google/ddpm-celebahq-256`. No lasting artifacts.**
* **~90+ commits unpushed** — no GitHub credentials on this box; needs `gh auth login`.
* **Shapes3D / CLEVR are NOT registered.** `load_factored` accepts `colormnist` and `celeba` only;
  the docstring names Shapes3D/CLEVR as the *intended* next entries and describes what their factor
  tables look like, but neither has a loader and neither has been run. The unified contract means
  adding one is loader-sized rather than a refactor — that is the claim, and it is weaker than
  "they are now a loader", which is how I put it earlier and which overstated it.

---

## 3. Where to look

| what | where |
|---|---|
| the argument, start here | `docs/FINAL_REPORT.md` |
| evidence section by section | `docs/RESULTS.md` — CelebA §5e-sexies, router §6c, §6c-bis, §6c-ter |
| what each experiment answers | `docs/README.md` |
| tables | `docs/TABLES.md` |
| figures | `figures/initial/` — f32 is the router, f31 the learning curves, f0 the overview; superseded ones are in `figures/initial/archive/` |
| reproduce | `./reproduce.sh {tests\|figures\|paper\|all}` |
| this session's new code | `lifelong_ood/neural/router.py` (`elbo_bits`, `mdl_route`, `mdl_crossover`, `utility_route`), `tests/diag_mdl_router.py`, `tests/diag_elbo_stability.py`, `experiments/score_mdl_rules.py`, `experiments/make_mdl_figure.py` |

Re-score any routing rule offline, without a GPU, from the recorded codelengths:

    python experiments/score_mdl_rules.py results/diag_mdl_router.json
