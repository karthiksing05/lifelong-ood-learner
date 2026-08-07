# PLAN V2 — the lifelong OOD learner (2026-08-07)

Comprehensive plan for the next phase. Supersedes the priority ordering in `PLAN.md` §3;
`PLAN.md` §1–§2 (contribution + status), `RESULTS.md` (the record), and `HANDOFF.md` (compressed
state) remain the ground truth for what has been measured. Nothing here re-litigates an experiment
listed under "refuted / do not re-derive" in `HANDOFF.md` §4 or `PLAN.md` §4b.

---

## 1. The thesis, and how the evidence has bent it

The original framing, unchanged as the *goal*:

> Compositionality is a solution to lifelong learning (one part of which is OOD learning). The
> field studies **recombination**; it neglects (1) **decomposition** — what the parts are — and
> (2) **representation into the manifold** — how a new instance enters an existing representation
> *as if it had been learned natively*. Lifelong learning is reframed as repeated composition of
> prototypes into OOD prototypes, sampling from those prototypes, and adding the results to the
> existing manifold — batched, preserved, and more permanent than a series of LoRAs.

Wang et al. (`../cobweb-poincare/paper/poe-diffusion.pdf`) supply stages 1–3 of that loop
(discover → select → compose → sample) and stop exactly where we begin: their conclusion names
"detecting, consolidating, and reusing genuinely new primitives" as future work, and their
consolidation artifact is a flat, per-query, discarded `(LoRA, c_new)` pair. Our contribution is
the fourth stage — and now, after ~20 experiments, the evidence has *sharpened* the thesis in four
ways that this plan treats as fixed constraints, not open questions:

1. **"As if learned natively, without degrading anything" is established — and the mechanism is
   known.** Preservation is bit-exact across 12 batches to 40 concepts (`tests/test_unbounded.py`),
   forgetting exactly 0.000 vs real CL baselines (N5). The law extracted from four independent
   zero-growth failures (§5w): **conditionality, not size, buys preservation** — a delta that can
   be switched off cannot damage what it was not trained for. Any future mechanism must be
   conditional; merging into shared weights is settled as a dead end.

2. **"More permanent than a series of LoRAs" has a precise answer now.** The permanent object is
   not a pile of adapters — it is the **factor basis** (ideal-words decomposition of the model's
   own conditioning space, R² 0.826). New *combinations* of known factor values cost **zero
   parameters** and beat per-concept adapters (N11: F1 0.755 vs 0.658 at 551K params/concept).
   The "series" shrinks to **rare vocabulary events**: one kept orthogonal (Cayley) delta per
   genuinely new factor value, 2.8× smaller than LoRA, trained on real instances. This is the
   efficiency thesis made concrete: **cost grows with the size of the vocabulary, not with the
   length of the experience stream** — the combinatorial majority of novelty is free.

3. **The hierarchy is out as a placement mechanism; the basic level survives as a selection
   question.** N14 gave hierarchical placement its best case and it lost to flat and to shuffled.
   Keep the insert-only discipline, drop the tree of adapters. What the original thesis wanted from
   "the basic level of every parse" — a fixed abstraction level donating a shared set of composites
   — re-enters at a different point: *which level of the analytic tree supplies the experts for
   composition* (see WS4), where it is currently an untested depth heuristic.

4. **The loop's "sample → absorb" leg is sound only for combinations.** A composed pool delivers
   bit-exactly zero gradient to shared paths (§5m — self-distillation, nothing at stake) and
   conditioning on a never-seen factor value scores exactly 0.000 (§5q phase B) — only real-data
   weight updates work there (0.833). This is not a wound; it is the routing rule of the method:
   *new combination → basis, free, from the model's own knowledge; new value → weights, from real
   data, never from the model's own samples.*

**The defensible paper** (per `HANDOFF.md` §8, adopted here): *certified-preserving OOD composition
by basis extension, with rare weight updates for new vocabulary.* The lifelong claim is that this
loop runs forever: preservation is structural, growth is sublinear in experiences, and consolidated
concepts are first-class citizens of future compositions.

---

## 2. Stage map against Wang et al.

| Wang et al. stage | their mechanism | ours | status |
|---|---|---|---|
| Discover concepts | mode ascent on `s_θ(x_t,t)` at many noise levels, 10–17 min/query | Cobweb tree nodes ARE the modes of `p_t` (E1: leaf match RMSE 2e-4 at low noise, shallow nodes at high noise) — free, closed-form | ✅ validated, full scale |
| Select | greedy submodular per-dim coverage (Eq. 8–9) | same, transplanted (`compose.py:submodular_select`) | ✅ |
| Compose | per-dim PoE, fixed τ=0.5 (Eq. 7, 10) | same + two certified fixes: adaptive temperature (bg acc 0.42→0.97), extent-not-precision covariance | ✅ beats their config |
| Sample | analytic-teacher CFG (Eq. 12–14) | two certified operating points: teacher-guided (fidelity, acc 0.585) / inversion (coverage, recall 0.649); span oracle shows the limit is the conditioning space | ✅ closed — do not re-tune |
| Absorb | flat `(LoRA, c_new)` per query, discarded | **the contribution**: factor-basis extension (combinations, 0 params) + kept orthogonal deltas (vocabulary) + insert-only registry, bit-exact preservation | ✅ core measured; loop not yet self-routing |
| Detect / reuse | named as future work | **open** — routing is currently done by the experimenter (WS2); reuse of consolidated concepts unproven (WS3) | ⚠️ the plan's center |

Central empirical contrast already in hand (N9): ours F1 **0.50** vs the paper's method **0.00** —
`poe-guided` scores higher per-sample accuracy precisely by emitting near-copies of the query.
Coverage vs copying is the story; every workstream below either widens that contrast or defends it.

---

## 3. The loop, v2 (what the finished system looks like)

```
observe one OOD query x_q  (or a batch — batch addition is first-class, n4)
  1. READ    factor assignment off x_q via the factor basis + probes
  2. ROUTE   in-distribution        → nothing to do
             new COMBINATION        → extend basis: compose read-only direction, 0 params
             new VOCABULARY value   → flag; collect real instances; train kept orthogonal
                                      delta, conditionally activated
  3. REGISTER the concept (flat, insert-only) with its embedding — it is now a source
             for future compositions, not just a product
  4. CERTIFY preservation: fingerprint check (bit-exact ε on fixed noise) against every
             prior concept — cheap, already the test_unbounded protocol
```

Steps 1 and 3–4 exist and are measured. **Step 2 is the missing organ** — today the experimenter
decides what is a new combination vs a new value. Making the system route itself is what turns a
set of certified mechanisms into a lifelong *learner*, and it is exactly the "detecting" half of
the gap Wang et al. named. It is WS2.

---

## 4. Workstreams

Ordered by how much each changes the paper's standing. Compute notes assume the SLURM setup in
`HANDOFF.md` §7 (everything through `run.sh`/`run_gpu.sh`; login node has no torch; clear
`__pycache__` before submitting).

### WS1 — Break the separability crutch (the biggest threat)

Every positive result rests on ColorMNIST, whose factors occupy disjoint pixels — exactly the
setting where a per-dimension PoE and a spatially-factorized basis are easiest. `HANDOFF.md` §5
already ranks this first. Three prongs, cheapest first:

- **1a. A non-separable ColorMNIST-family benchmark (1–2 days, small GPU).** Factor pairs that
  overlap on the same pixels: stroke thickness × digit, rotation × digit, or digit × digit-style.
  Reuse the entire harness (`n1_train --holdout-*`, N2/N3/N11 unchanged). This is also the right
  substrate for closure (WS3) — `PLAN.md` §3.2 already argues separability is why closure can't
  show up on standard ColorMNIST.
- **1b. Fix CelebA conditioning (the real fix, ~1 week).** §5o's diagnosis is structural, not
  undertraining: 14-attribute classes × ~25 examples, one additive time-embedding vector. Change
  the *conditioning architecture*, not the budget: per-attribute embeddings summed (not one
  embedding per 14-bit class — this matches the factor-basis worldview and multiplies effective
  examples per conditioning unit), injected at block level (FiLM per resolution), and fewer
  attributes (6–8 well-separated ones) before scaling back up.
- **1c. Pretrained transfer (N15 continuation, after 1b works).** §5u showed the bottleneck is the
  conditioning channel bolted onto `ddpm-celebahq-256`, not the composer. Apply 1b's block-level
  conditioning there. If it works: the method retrofits onto frozen public backbones — a large
  claim. If not: reported as a scoped limitation.

**Acceptance:** the N9 contrast (coverage vs copying) and the N11 result (basis beats adapters)
reproduced on at least one benchmark whose factors overlap spatially. Without this the paper's
scope must be explicitly narrowed to separable-factor domains.

### WS2 — The router: self-directed novelty detection (the missing organ)

Mechanisms to test, all cheap, all read off existing machinery:
- **Basis residual**: solve for the factor assignment; the residual norm of the embedding-space
  solve separates "expressible as a known combination" from "needs new vocabulary" (the span
  oracle in §5h is precisely this signal, used diagnostically — promote it to a decision rule).
- **Probe confidence / factor-marginal likelihood**: per-factor read-off entropy — a new *value*
  should read as uniform over known values of that factor while other factors read confidently.
- **Teacher fit**: the submodular objective's achieved coverage `F(S_K)` on the query — low
  coverage on some dimensions = those dimensions are not spanned by any expert.

Experiment `n18_router.py`: stream mixing (i) seen slots, (ii) held-out combinations, (iii)
held-out factor values (the `nobg3` / `stream` backbones exist for exactly this), report detection
AUROC per route and end-to-end stream accuracy when the system routes itself, vs oracle routing
(the current results) and vs always-adapter / always-basis ablations. **Acceptance:** self-routed
stream within a stated margin of oracle-routed; misroute cost quantified. This converts N13-style
streams from choreographed to autonomous, and it is the single largest step from "mechanisms" to
"learner."

### WS3 — Closure: consolidated concepts as future primitives ("reusing")

The thesis says *repeated* composition — outputs must become inputs. E4 measured
`adapter_expert_frac = 0.00` and the post-mortem (`PLAN.md` §3.2) concluded the benchmark, not the
mechanism, is at fault (separable attributes make pre-learned combinations redundant). Plan, in
order:
- **3a (free, this week):** cost-based closure on existing data — does consolidating concept k get
  cheaper (epochs to threshold loss, warm-start gap) when relatives are already registered? The
  `transfer` column of N3 already records this; write the analysis.
- **3b (after WS1a):** second-order composition on the non-separable benchmark — consolidate
  (thickness×digit) then query something requiring it plus a colour; measure whether the
  consolidated concept is *selected* and whether ablating it hurts.

**Acceptance:** one measured instance of a consolidated concept being load-bearing in a later
composition (selection frequency > 0 AND ablation cost > 0), or an honest negative with the cost
result from 3a as the fallback closure claim.

### WS4 — The basic level: which stratum donates the composites

The original thesis' decomposition leg, in its testable form: *is there a privileged level of
abstraction whose concepts are the right expert pool for composition?* Today the analytic track
picks experts by depth heuristics (`score.py` `mode="leaves"` / `frontier_mask` / `level_gamma`)
— never by category utility. Meanwhile `cobweb-private` has grown a continuous `get_basic`
(`include/cobweb_continuous_node.h:111`, "mirrors CobwebDiscreteNode::get_basic", commit
`ec8b9a9`), but **both envs this repo runs in (`cobweb`, `chunks`) have pre-`get_basic` binaries**
— only `trellis` has the 2026-07-29 rebuild. Steps:
1. Commit/stash the dirty `cobweb-private` working tree (5 modified + 3 untracked files — do not
   build from an uncommitted state), then rebuild the extension into `chunks` and `cobweb`.
2. `e5_basic_level.py`: on the full-scale E1 tree, compare expert pools — leaves / fixed-depth
   frontier / `get_basic` selection — on teacher accuracy (cheap, no sampling: does μ_T probe-read
   correctly per slot?) and on E2 composition. Also check whether the basic level explains the
   known systematic per-slot failures (`(cyan, deep_red)` — `PLAN.md` §3.1).
3. If basic-level pools win: the thesis gets its decomposition claim back in measured form —
   "the basic level is the stratum that donates reusable composites." If they don't: the depth
   heuristic stays and the claim is dropped. Either way the analytic track's discovery story
   (E1: hierarchy for free) becomes sharper.

Note: the old pixel-space decomposition work (chunk/stroke vocabularies, BLC) lived in a repo that
is no longer on disk. In this paper, "decomposition" = the factor basis (embedding space) + the
basic level (concept space). If the stroke-level story is wanted back it is a separate rebuild —
out of scope here.

### WS5 — Native-equivalence: the eval the thesis is named after

"Adapt new instances into the manifold *as if they were learned natively*" — currently argued via
preservation + accuracy, never measured head-to-head. Build the oracle: train
`ddpm_colormnist_all160.pt` on all 160 slots (~2.5 h, one GPU, same 200/200 recipe). Then for each
held-out slot compare three arms: **native** (oracle, slot in training data), **consolidated**
(ours: composed → sampled/collected → registered), **paper** (per-query LoRA). Report the
**native-parity ratio** (consolidated / native) on: slot sample quality (probe acc, F1 vs real
held-out reference — the N9 protocol), conditional diversity, and drag on the rest of the manifold
(seen-slot metrics before/after). Batch addition (n4) folds in: parity of batch vs sequential vs
native. **Acceptance:** a stated parity number on every axis. Even ~0.8 parity at zero backbone
retraining is a strong headline; and this is the eval design that most cleanly differentiates us
from Wang et al., whose LoRA is never compared to native training.

### WS6 — The efficiency and permanence accounting

The thesis promises efficiency; make it a table, not a sentence:
- **Growth curve**: cumulative params vs #experiences under the router's expected mix (combinations
  free, values 2.56M orthogonal) vs per-query LoRA (0.55M × every query, the paper) vs full refit.
  The claim to certify: **sublinear in experiences, linear only in vocabulary.**
- **Adapter compression** (from `PLAN.md` §3.0a): per-node rank, restrict `inject` targets
  (attention/mid-block first). Secondary now that orthogonal deltas already give 2.8×.
- **Wall-clock**: discovery+composition ms (tree) vs 10–17 min/query (their mode ascent, D9 in
  `../cobweb-poincare/paper/CLAIMS.md`); consolidation cost per route.
- **Permanence audit**: fingerprint certification cost as the registry grows (it is O(concepts) —
  state it), and the N13 lesson (repeated merges create a forgetting gradient → merging stays out).

### WS7 — Hygiene that blocks the writeup

- **Seeds**: 3+ seeds on every number that reaches the paper (aggregate.py is plumbed; N5 has 4,
  N3/N9 have 3 — E-track and N11 need theirs).
- **Backbone re-verification**: `PLAN.md` §2 open item 1 — every ColorMNIST number re-checked on
  the correctly retrained 200/200 backbone; nothing read off the 60/100 one.
- **Metrics threat**: probe-based metrics with the probe ceiling reported alongside (kept);
  palette-reconstruction caveat stated (kept); consider adding the paper's FID/CLIP as secondary
  on the final N9/WS5 tables only, for reviewer comfort.
- **Assertion coverage**: §5m (composed-pool zero-gradient) and gain calibration currently have no
  asserting test — regressions would surface as quietly worse numbers. Add both to `tests/`.
- **Push the repo**: 61 commits, none pushed; needs `gh auth login` on an interactive session.

---

## 5. Paper skeleton

**Working title:** *Certified-Preserving Out-of-Distribution Composition: Basis Extension with
Rare Weight Updates for New Vocabulary* (lifelong learning as repeated composition).

Claims table (each row = one section, evidence already in hand unless marked WS):

| claim | evidence |
|---|---|
| Concept discovery is free: an incrementally-built tree IS the density-mode hierarchy a DDPM encodes | E1; discovery-cost table (WS6) |
| Composition transfers, with two certified fixes to the published method | E0/E2, RESULTS §4; adaptive τ + extent covariance |
| Coverage vs copying: the published method scores F1 0.00 by re-emitting the query; ours 0.50 | N9 (+ WS1 replication off-separability) |
| New combinations are free and beat adapters | N11 |
| New vocabulary requires real data — conditioning on model samples gives exactly 0.000 | §5q phase B (a *finding*, not a failure) |
| Preservation is structural and bit-exact; conditionality, not size, buys it; four zero-growth routes fail | N3/N5, test_unbounded, §5v–§5w |
| The system routes itself between these regimes | **WS2** |
| Consolidated concepts are reusable primitives (or: the cost-closure result) | **WS3** |
| Native parity: a consolidated concept approaches a natively-trained one at zero retraining | **WS5** |
| Honest negatives: hierarchy doesn't pay (N14), composed pools can't test forgetting (§5m), sampler diversity limit is the conditioning span (§5h) | already measured — report, don't hide |

Threats to state: separability (until WS1 lands), probe metrics, reconstructed palette, backbone
scale (6.8M/24k vs paper's ~35M/120k — the span-oracle capacity hypothesis in `PLAN.md` §3.0 is
the one untested explanation and WS1b's bigger backbone doubles as its test).

---

## 6. Sequencing

Two-week blocks, assuming current cluster access; WS7 hygiene runs continuously in the gaps.

1. **Block 1 — cheap and decisive:** WS3a (free, existing data) · WS1a benchmark build + backbone
   · WS2 router mechanisms offline (AUROC on existing backbones) · start WS5 oracle backbone
   training (fire-and-forget). Commit + push.
2. **Block 2 — the learner:** `n18_router` end-to-end self-routed stream · WS5 native-parity
   tables · WS1a replication of N9/N11 on the non-separable benchmark · WS3b second-order closure.
3. **Block 3 — scope:** WS1b CelebA conditioning architecture · WS4 basic level (rebuild first) ·
   WS6 accounting tables.
4. **Block 4 — write.** Freeze numbers, seeds complete, claims table → sections.

Explicitly deprioritized: further inversion/sampler tuning (certified closed), any zero-growth
absorption route (four failures), hierarchical placement (refuted), hypernetwork-with-low-rank-
output (interesting mechanism question, `HANDOFF.md` §5.3, but changes no conclusion — Block 4+
if ever), WS1c pretrained transfer before 1b works.

The two results that would most change the paper's standing, in order: **WS1** (kills or keeps the
separability objection) and **WS2** (turns certified mechanisms into a lifelong learner). Spend
compute there first.
