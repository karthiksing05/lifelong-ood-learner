# Certified-preserving out-of-distribution composition

Paper-facing synthesis. One row per claim, the experiment behind it, and — where it matters — the
control that makes it honest. `RESULTS.md` is the full record including every refuted hypothesis;
this is what the paper would assert, and nothing appears here that is not measured there.

Claims are graded:

* **A** — holds on both benchmarks, multi-seed, with a control that rules out the obvious
  alternative explanation.
* **B** — holds, but on one benchmark, or without a control strong enough to close the question.
* **X** — a claim the evidence does **not** support, kept on the record because it shaped the design
  or because a reviewer will ask.

---

## The thesis

Compositionality is a mechanism for lifelong learning. The literature studies **recombination** and
largely assumes the parts; this work is about the other two halves — what the parts are, and how a
new instance **enters an existing representation** so that it behaves as if it had been learned
natively rather than bolted on.

The loop, run forever:

```
observe one out-of-distribution query
  1. READ     the factor assignment off the query
  2. ROUTE    known combination -> compose, zero parameters
              new vocabulary    -> a weight update, from REAL data
  3. REGISTER the concept, flat and insert-only; it is now a source for future composition
  4. CERTIFY  preservation: every prior concept bit-identical
```

Steps 1, 3 and 4 were established before this round. Step 2's *decision* — which the source paper
names as open work and which every earlier experiment had the experimenter supply — is N18.

## Relation to Wang et al., *Test-Time Compositional Generalization in Diffusion Models via Concept
Discovery*

Their pipeline is discover → select → compose → sample → distil. We reimplement it faithfully and
diverge at exactly one place: they distil one flat `(LoRA, c_new)` pair per query **and discard it**,
naming the gap in their own conclusion ("a full mechanism for detecting, consolidating, and reusing
genuinely new primitives remains an important direction for future work"). Everything below the
composition stage is that mechanism.

Two of their settings do not port and both are documented at the implementation site: their fixed
`tau = 0.5` (background accuracy 0.42 -> 0.97 once made scale-free) and their product covariance,
which is sharper than any factor it is built from and samples near-copies of its own mean.

---

## Headline claims

| # | claim | grade | evidence |
|---|---|---|---|
| **C1** | Composition recovers a held-out combination that nearest-class retrieval never produces, and beats the published method on **coverage**: F1 **0.502 ± 0.036** against `poe-guided`'s **0.007 ± 0.007**, which scores *higher* per-sample accuracy (0.678 vs 0.331) precisely by emitting near-copies of the query — recall 0.013 against ours at 0.789, content diversity 0.82x the real class spread against ours at 1.18x. `nearest-class` scores F1 0.005 and background accuracy exactly 0.000. Real held-out images are at F1 0.969. | **A** (3 seeds) | RESULTS 5l (N9) |
| **C2** | A new **combination** of known factor values is expressible at **zero parameter cost** — one 256-float direction, shared by every combination using that value, against **551K parameters per concept** for an adapter. On quality the two are **comparable, not separated**: in the run containing both arms the adapter is ahead (F1 0.658 ± 0.017 against 0.569 ± 0.155), and the norm-matched basis run reaches 0.755 ± 0.052 but has no adapter arm to compare against. A matched norm-matched run is in progress; until it lands the defensible claim is *comparable quality at ~2000x less cost*, **not** "beats". | **B** | 5q (N11) |
| **C3** | A new **vocabulary** item does not. On a backbone that never saw the value, composing it scores **exactly 0.000** on that factor with **zero variance across 3 seeds**, and an adapter distilled from the model's own samples scores 0.002 — but the same adapter trained on **real** images reaches **0.997** (joint 0.833 against a real-data ceiling of 0.906). Feeding a model its own samples cannot add capacity: a composed pool delivers bit-exactly **zero** gradient to shared parameters. | **A** | 5q phase B (3 seeds), 5m |
| **C4** | Preservation is **structural and bit-exact**: every prior concept's epsilon output on fixed noise is bit-identical after any number of later insertions, held across 12 batches to 40 concepts. | **A** | 6b (`tests/test_unbounded.py`) |
| **C5** | Against named continual-learning baselines on **real** held-out data, the arms that structurally *remove* shared parameters sit at exactly **0.000** forgetting while every arm that tries to *protect* them forgets (EWC 0.058, finetune 0.082, replay 0.163). | **A** | 5k (N5, 4 seeds) |
| **C6** | **Conditionality, not size, is what buys preservation.** Every route that *removes* the ability to switch a delta off collapses — merged low-rank **0.379**, merged orthogonal **0.069** (0.082 with rehearsal), hypernetwork **0.073**, classifier guidance **0.000** — while both routes that *keep* it hold: LoRA 0.837 at 7.16M and a Cayley-orthogonal delta 0.773 at **2.56M, i.e. 2.8x less growth for the same exact preservation**. Four independent failures, one explanation. | **A** (3 seeds) | 5v, 5w (N16/N17) |
| **C7** | The system **routes itself**: AUROC 0.980 novel-vs-in-distribution and 0.995 vocabulary-vs-combination, self-routing accuracy 0.858. The error is one-sided in the safe direction — a new vocabulary item was routed to the free path **0 times in 120 queries across 3 seeds**, while 20% of new combinations over-spend on an adapter they did not need. | **A** | 6c (N18, 3 seeds) |
| **C8** | Cost grows with **vocabulary**, not with experience. Charging the router's own measured error rates, a 160-experience ColorMNIST stream costs 10.1M parameters against 95.5M for per-query LoRA, and a 419-experience CelebA stream 23.6M against 250.2M. Under oracle routing the curve is flat once vocabulary saturates; under measured routing its slope is set by a detector error rate rather than by the arrival of experiences. | **A** | 5z (N24) |

## Honest negatives that are part of the result

| # | claim | why it fails |
|---|---|---|
| **X1** | *"The hierarchy earns its keep."* | **No.** Given the case built to favour it — families sharing two of three factors, arriving back to back — `clustered` is the **worst** arm (0.831) against `shuffled` (0.859) and `flat` (0.842), at identical parameter counts. The tree forms and shares; the sharing does not help. Recommendation adopted: flat, insert-only. (5s) |
| **X2** | *"Consolidated concepts make later learning cheaper or get reused."* | **Not on ColorMNIST.** Selection frequency 0.00 (E4); and by cost, `hier` and `unfrozen` differ by −0.00018 on first-epoch distillation loss against a seed spread of 0.0053 — thirty times smaller than the noise. The likely reason is that separable factors make a pre-learned combination redundant, which is a hypothesis about the benchmark, so this is stated as open rather than refuted. (5y) |
| **X3** | *"The factor basis is what recognises a new combination."* | **Unsupported.** The router's unconditional ablation — no class, no basis, no conditioning — matches the conditional signal to within a standard error (0.981 / 0.998). The detector works; the mechanistic story it would have supported does not follow, and no candidate among ten measured beats the ablation. (5x) |
| **X4** | *"Normalising the novelty signal by the unconditional loss removes per-image difficulty."* | **Reversed, not merely weak** (AUROC 0.199). The normaliser is itself sensitive to novelty and degrades faster than the numerator. Recorded because it is a trap the next person will otherwise fall into. (5x) |
| **X5** | *"The composed teacher can be sampled to generate the held-out class."* | **No.** The class-embedding span can express one factor or the other but not both for a held-out pairing, and a span oracle fitted on 64 real held-out images does no better than inverting from one query. The limit is the conditioning space, not the optimiser, objective, prior or sampler — which is why thirteen interventions on those could not move it. (5h) |

## Threats to validity, stated up front

1. **Probe-based metrics.** Accuracies come from weak softmax probes on raw pixels, and every table
   reports the **real-data ceiling** beside the method for that reason. They are not FID/CLIP and
   should not be read as comparable to the source paper's tables.
2. **Reconstructed palette.** The ColorMNIST palette is this project's reconstruction of the paper's
   colour *names*; absolute numbers are not comparable to theirs, only within-table comparisons.
3. **Backbone scale.** 6.8M parameters on 24k images (ColorMNIST) and 9.3M on 96k (CelebA), against
   the paper's ~35M on 120k. Backbone-to-backbone variation on the same training command is up to
   0.07 F1 (5p), which bounds how finely any ordering here can be read.
4. **CelebA's factor basis is architectural, not discovered.** On ColorMNIST the basis is recovered
   from an entangled per-class embedding space (ideal-words R² 0.826). On CelebA a per-class
   embedding is untrainable at this data scale — measured, not assumed (5o: 34% of ceiling, and 2.3x
   data moved it to 33%) — so conditioning is factored by construction. CelebA therefore does not
   re-test "can the basis be found"; it tests whether composing known factor values renders correctly
   when the factors **overlap the same pixels**, which is the question ColorMNIST structurally cannot
   ask.

---

*CelebA-specific claims (composition under spatially overlapping attributes, vocabulary extension,
class-incremental retention) and the native-parity result are added as their runs complete; see
`RESULTS.md` for the state of each.*
