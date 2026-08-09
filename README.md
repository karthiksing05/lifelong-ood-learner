# Compositionality as the substrate for lifelong, out-of-distribution learning

A generative model meets out-of-distribution queries **forever**. Each one must be produced, **kept**,
and must not damage anything kept before — at a cost that grows with the model's *vocabulary* rather
than with its *history*.

This repository argues that **composition is the mechanism that makes that possible**, and measures
the argument on two benchmarks (one where the factors occupy disjoint pixels, one where they overlap)
and on a pretrained 113.7M-parameter diffusion model we did not train.

![the loop](figures/f1_loop.png)

---

## How this branches off Wang et al.

**[Test-Time Compositional Generalization in Diffusion Models via Concept Discovery](https://arxiv.org/abs/2605.07078)
(Wang, Gupta, Zhu, MacLellan) is a companion paper from our own group.** This work is not a
competitor to it; it is a continuation of it, and the composition machinery is used unchanged.

**What they built.** Given one out-of-distribution query and a frozen diffusion model:

```
mode-ascent discovery on the model's own score field
   → submodular selection of K prototypes
   → per-dimension product-of-experts teacher
   → classifier-free guidance toward it (their Eq. 13, with the Eq. 14 variance-aware schedule)
   → optionally distil the result into one LoRA + one embedding
```

It answers *can a frozen model be steered to render a combination it never saw?* — and the per-query
LoRA is then **discarded**. There is no second query in the protocol.

**Where we pick it up.** Their limitation section names the gap exactly:

> *"A full mechanism for detecting, consolidating, and reusing genuinely new primitives remains an
> important direction for future work."*

We keep their four stages and add a fifth and sixth, which is what turns a test-time trick into a
lifelong learner:

```
for each arriving query, forever:
    COMPOSE      their four stages, unchanged
    ROUTE        is this a new ARRANGEMENT of parts we have, or a genuinely new PART?
    CONSOLIDATE  arrangement → a shared basis direction (1024 bytes)
                 new part    → a new leaf in an insert-only adapter bank
    CERTIFY      every previously consolidated concept is checked bit-identical
```

---

## The method in pseudocode

What is actually stored, what is actually touched, and what is provably not.

### The state that persists between queries

```python
BACKBONE  eps_theta(x_t, t, c)         # a UNet, FROZEN forever. requires_grad_(False) on every
                                       # parameter at construction and never re-enabled.
                                       # Its class-embedding table has a NULL row: conditioning on
                                       # that row IS the unconditional model.

BANK = HierarchicalAdapterBank:
  .wrapped   [PathLoRA]                # every targeted nn.Linear / nn.Conv2d of the backbone,
                                       # replaced in-place by a wrapper. `targets`/`exclude` pick
                                       # which: all of them on our UNet, attention-only on a
                                       # pretrained one (32 of its 58 Linears are time_emb_proj,
                                       # and adapting the noise-level path destroys the model).
  .state["path"]  [node_id]            # THE ACTIVE PATH. One shared mutable list. Switching
                                       # concepts is switching this list — no weight copying,
                                       # no re-injection, no forward-pass branching.
  .tree      ConceptTree               # insert-only. node_id -> children. Never merged, never split.
  .concepts  {name: (path, leaf)}      # path = root..leaf node ids for that concept
  .frozen    {node_id}                 # sealed by an earlier insertion: NEVER in an optimizer again
  ._open     {node_id}                 # created by the insertion in progress: the only trainable set
  ._emb_extra {name: vector[d_emb]}    # the concept's conditioning vector

FACTOR_BASIS = factorize(C, labels)    # ideal words, fit ONCE by ridge regression on the backbone's
                                       # own trained class embeddings C:
                                       #     C[i] ~ mu + sum_f V[f][ label[i][f] ]
                                       # V[f] is [n_values_f, d_emb]. R^2 = 0.826 on ColorMNIST.
                                       # This is what makes a factor addressable at all.
```

Inside one wrapped layer, the delta is a **sum over the active path** — which is why a concept is
its ancestors plus its leaf, and why an unrelated concept contributes exactly zero:

```python
class PathLoRA(base_layer):            # base_layer.weight is frozen
    A, B = ParameterDict(), ParameterDict()      # keyed by node_id

    def add_node(nid):
        A[nid] = kaiming_uniform(rank, fan_in)
        B[nid] = ZEROS(fan_out, rank)            # <- a fresh node is an EXACT no-op, not a small one

    def forward(x):
        out = base_layer(x)                      # frozen weights
        for nid in BANK.state["path"]:           # <- only the active path contributes
            if nid in A:
                out = out + B[nid] @ (A[nid] @ x)
        return out
```

### One query

```python
def absorb(x_q):
    # ── 1. ROUTE ────────────────────────────────────────────────────────────
    # No labels. Score x_q under every conditional the bank can already express and take the best
    # denoising error, at two noise levels chosen by tests/diag_router_signals.py.
    s_novel = min_c  E_eps[ || eps_theta(noise(x_q, u=0.10), c) - eps ||^2 ]   # over trained classes
    s_vocab = min_c  E_eps[ || eps_theta(noise(x_q, u=0.45), c) - eps ||^2 ]   # over basis compositions

    if   s_novel <= t_novel:  return                      # already in distribution: do nothing
    elif s_vocab <= t_vocab:  kind = "new ARRANGEMENT"     # expressible in the factor basis
    else:                     kind = "new VOCABULARY"      # a factor VALUE we do not have

    # ── 2. COMPOSE — read the query IN THE FACTOR BASIS ─────────────────────
    # Not a free vector, and not a weighting over the ~120 entangled class embeddings: one softmax
    # per factor, so 10 + 4 + 4 = 18 parameters. Composition is structural rather than something
    # the optimizer has to discover, and 18 parameters are determined by a SINGLE image where 120
    # would not be.
    w = [softmax(logits_f) for f in factors]              # learned by inversion through the FROZEN model
    c_new = mu + sum_f  gain[f] * (w[f] @ V[f])           # gain[f] calibrated on TRAINED slots, label-free

    if kind == "new VOCABULARY":
        V[f] = concat(V[f], new_direction)                # extend the basis: 1024 bytes
        # (measured: a genuinely new VALUE cannot be composed — it needs a weight update
        #  from real images. §7c. The router's job is to tell these two cases apart.)

    # ── 3. SAMPLE the pool FROM THE MANIFOLD, not from the teacher ──────────
    # The PoE teacher is a unimodal Gaussian: guiding toward it collapses every trajectory onto its
    # mean (recall 0.00 in every configuration tried, §5e). So the teacher decides WHERE the concept
    # is and the model's own conditional generates it.
    pool = sample(eps_theta, cond=c_new, guidance=1.5)     # guidance calibrated on TRAINED slots

    # ── 4. CONSOLIDATE — the only place any parameter moves ─────────────────
    BANK._seal()                                  # everything _open from the last insertion joins
                                                  # .frozen. After this line it can never be trained
                                                  # again, by ANY future concept.
    path, leaf = BANK.tree.insert(mean(pool))     # insert-only placement
    BANK._open = {leaf}
    for nid in path:
        for layer in BANK.wrapped: layer.add_node(nid)     # B = 0, so the model is UNCHANGED so far
    BANK._emb_extra[name] = c_new                 # pinned, and frozen during distillation: the
                                                  # adapter is a residual to the ACTUAL composition

    params = [p for nid in path if nid in BANK._open for p in (A[nid], B[nid])]
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ THE PRESERVATION GUARANTEE, in one line.
    #        Ancestors are in .frozen, so they are not in `params`, so no gradient can reach them.
    #        Not a penalty, not a small learning rate — the parameters are absent from the optimizer.
    Adam(params).minimize( ||eps_theta(noise(pool), c_new) - eps||^2 )
    #   noise levels: uniform by default; `t >= t_min` is the RECOMMENDED setting (§14) and what the
    #   pretrained path uses at t_min=600, where it is the difference between acquiring an attribute
    #   and destroying the model. Not yet the default in `consolidate()` on our own backbones.

    # ── 5. CERTIFY ──────────────────────────────────────────────────────────
    for old_name in BANK.concepts:
        assert fingerprint(old_name) == stored_fingerprint[old_name]      # bit-identical, ==, not ~=
```

### Reading the bank afterwards

```python
def sample_concept(name):                          # one concept
    with BANK.activate(name):                      # state["path"] = concepts[name].path
        return sample(guide_eps = eps_theta(., c=_emb_extra[name]),  guide_w = 1.0)

def sample_composed(names):                        # SEVERAL concepts — in SCORE space
    eps_base = eps_theta(x, t, NULL)               # the frozen model: empty path
    eps_hat  = eps_base + sum_i w_i * (eps_theta(x, t, concept_i) - eps_base)
    #          ^ each term activates ONE path at a time, so no adapter is ever evaluated outside
    #            its own conditioning. Activating several paths at once instead sums deltas that
    #            were each trained to be applied ALONE — measurably worse (§12e).
```

### Batch addition: K concepts in one update

The only change is that `_open` holds K leaves plus any shared node, and one optimizer sees all of
them — which is what lets a shared ancestor learn from every member at once:

```python
BANK._seal()
paths, leaves, new_nodes = BANK.tree.insert_batch(mus, group=True)
BANK._open = set(new_nodes)                        # K leaves + the shared group node
params = union of _open nodes' (A, B) + the K embeddings
Adam(params).minimize( interleaved minibatches from all K pools )
```

`_seal()` runs once for the whole batch, so the guarantee is identical whatever K is — which is what
makes batch size a free efficiency knob rather than a safety trade (bit-identical **1.000 at every K
from 1 to 12**, and 2.14× faster at K=12).

---

| | Wang et al. | this work |
|---|---|---|
| horizon | one query | an unbounded stream |
| what persists afterwards | nothing — the LoRA is discarded | a routed, consolidated concept |
| decision made per query | none; every query takes the same path | **route**: arrangement (free) vs new part (a kept delta) |
| cost of the *n*-th concept | same as the first (nothing accumulates) | ~0 for an arrangement, one small delta for a new part |
| interference between concepts | not applicable | **certified** bit-identical, measured |
| evaluated on | sample quality for that one composition | that, **plus** retention, growth, and parity with a natively-trained backbone |

**The composition operator is inherited, not improved.** That is deliberate. Where a better operator
exists we adopt it: on CelebA, MCMC-corrected score addition (Du et al., ICML 2023) is more accurate
than ours and we say so. What this project contributes is everything below the composition stage.

---

## Contributions, in one line each

Full detail, with evidence and caveats for each: **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)**.

1. **Certified-preserving consolidation.** Concepts accumulate forever with every earlier concept
   provably bit-identical — **1.00 bit-identical** across 3 seeds, two datasets and **eleven** CL
   baselines, where every method that *protects* shared weights instead of removing them sits at
   0.25 and forgets.
2. **A routing rule that decides what an update costs.** New *arrangement* vs new *part*, from the
   model's own denoising error with no labels: **AUROC 0.980 / 0.995**, error one-sided in the safe
   direction. An arrangement costs a 1024-byte basis direction instead of a 551K-parameter adapter,
   which is what makes cost scale with vocabulary rather than experience.
3. **Batch addition that is strictly better than one-at-a-time.** K concepts in one update:
   **2.14× faster *and* +0.052 accuracy** at K=12, bit-identical at every K from 1 to 12.
4. **Native parity.** A consolidated concept is measured against a backbone *trained* on that slot —
   a question that only exists once concepts are kept.
5. **The mechanism transfers to a model we did not train.** On `google/ddpm-celebahq-256`, injecting
   the bank changes the model by exactly **0.000e+00** and concepts are acquired at 256px without
   quality loss.
6. **Two rules for adapting diffusion models**, both measured rather than argued: compose a bank in
   **score space** (path union collapses diversity and, on factor concepts, produces nothing), and
   train concept adapters at **high noise levels only** — which is the difference between working and
   not on a pretrained backbone.

---

## The evidence, in six figures

### 1. Attribute accuracy cannot tell composing from copying. Coverage can.

![coverage vs copying](figures/f2_coverage_vs_copying.png)

The figure the project turns on. A method that re-emits the query scores **high attribute accuracy**
— the query is, after all, an image of the target — while covering none of the class. On ColorMNIST
`poe-guided` beats us on accuracy (0.839 vs 0.539) **and** precision, and its recall is **0.001**. Any
evaluation reporting accuracy without coverage ranks it first.

### 2. Composition recovers combinations never observed

![composition](figures/f3_composition.png)
![ColorMNIST case study](figures/f9_case_colormnist.png)

ColorMNIST **F1 0.605 ± 0.043** against the companion method's **0.003 ± 0.003**, with every baseline
run in *its own* best configuration. On CelebA, where `Smiling` and `Big_Nose` never co-occur in
training, we reach `pair` **0.903 ± 0.094** — and MCMC-corrected score addition reaches **0.965**,
which we report rather than bury (see [docs/RESULTS.md §11c](docs/RESULTS.md)).

### 3. Preservation is structural, not learned

![preservation](figures/f5_preservation.png)

Eleven CL arms, two datasets, 3 seeds. Only arms that *remove* shared parameters reach bit-identity;
`ttt` also does but keeps all 768 stream images to manage it.

### 4. Cost scales with vocabulary, not experience

![growth](figures/f8_growth.png)

### 5. A consolidated concept behaves like a trained one

![native parity](figures/f6_native_parity.png)

### 6. It works on a model we did not train

![pretrained case study](figures/n25_pretrained_case_final.png)

`google/ddpm-celebahq-256`, 256×256. Two attributes learned as separate concepts, each blind to the
other, then composed. The last two rows are the **same bank read two ways** — score composition and
path union — which is the measurement behind contribution 6.

---

## Reproducing

```bash
bash reproduce.sh tests       # assertions, no backbone needed
bash reproduce.sh backbones   # every checkpoint the rest loads (~12 GPU-hours)
bash reproduce.sh colormnist  # ColorMNIST experiments, 3 seeds
bash reproduce.sh celeba      # CelebA experiments, 3 seeds
bash reproduce.sh pretrained  # the pretrained-backbone experiments
bash reproduce.sh analytic    # the analytic track (no gradient descent anywhere)
bash reproduce.sh supporting  # negative results and cost claims
bash reproduce.sh figures     # aggregate + every figure + tables   (CPU, seconds)
bash reproduce.sh all
```

## Documentation

| file | what it is |
|---|---|
| **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)** | each contribution in detail: claim, evidence, caveats |
| **[docs/SCOREBOARD.md](docs/SCOREBOARD.md)** | every axis vs every baseline, with the winner named |
| **[docs/](docs/)** | every result, in full |
| [docs/RESULTS.md](docs/RESULTS.md) | the complete record, including every refuted hypothesis |
| [docs/TABLES.md](docs/TABLES.md) | every number with error bars, regenerated from `results/` |
| [docs/RELATED.md](docs/RELATED.md) | positioning, which baselines are fair, citation hazards |
| [docs/ASSESSMENT.md](docs/ASSESSMENT.md) | publication readiness, assessed against the evidence |

## Layout

```
lifelong_ood/      the library: score field, adapters, router, composition, evals
experiments/       n1–n28, one file per question
tests/             assertions and the diagnostics that CHOOSE configurations
results/           per-seed JSON + the aggregate the figures are built from
figures/           every figure in this README
docs/              the full record
```

## A note on measurement

Several results in this repository are corrections of earlier results in this repository, and the
corrections are kept rather than quietly replaced. Three measurement bugs were found by *control
rows* rather than by the numbers they were supposed to support; one baseline was found to be running
in a configuration its authors would not recognise; one comparison was found to be varying two things
at once. [docs/RESULTS.md](docs/RESULTS.md) records each, because a repository that only shows what
survived is not evidence that anything was checked.
