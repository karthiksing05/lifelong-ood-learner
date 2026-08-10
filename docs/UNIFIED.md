# One operation: collapsing backbone training and lifelong absorption

This repository currently has two regimes. A backbone is trained ground-up, once, on the seen grid;
then it is frozen forever and every subsequent concept is an insert-only adapter over it. The split
is real in the code and real in the story, and it invites an obvious question: **is it necessary?**

This document argues it is not — that both phases are the same operation at different allocation
sizes — and states precisely what that unification costs and what remains unsolved. It is a design
document, not a result. Everything marked **[built]** exists and is measured; **[proposed]** is
design; **[open]** is unsolved.

---

## 1. The primitive

```
absorb(batch B):
    1. ROUTE      what can the model already express of B?
    2. ALLOCATE   new capacity, of a kind determined by what routing found
    3. FIT        optimize ONLY the newly allocated parameters on B
    4. SEAL       move them to frozen; never in an optimizer again
```

**Backbone training is the t = 0 case.** At t = 0 the router can express nothing, so allocation is
maximal: one always-on, full-rank block, fitted to the whole training set. Every later batch routes
mostly onto existing structure and allocates something small. There is no second procedure — only a
different answer at step 2.

The model then has one form throughout its life:

```
f(x ; c)  =  Σ_{n ∈ trunk} Δ_n(x)   +   Σ_{n ∈ path(c)} Δ_n(x)
```

The "backbone" is the first trunk block. Nothing about the read path changes as the model grows.

### The four allocation modes

Routing already returns a three-way decision; the unified view adds the degenerate case at the top.

| what routing finds | allocate | cost | status |
|---|---|---|---|
| nothing is expressible (t = 0) | a full-rank **trunk block** | the backbone | **[proposed]** |
| broadly novel regime, not one concept | a low-rank **trunk block** | one block, always-on | **[proposed]** |
| a genuinely new part | a **leaf** (+ group node if batched) | ~551K params | **[built]** §5, §15 |
| a new arrangement of known parts | a **basis direction** | 1024 bytes | **[built]** §6c |
| already in distribution | nothing | 0 | **[built]** §6c |

The bottom three are the current router, measured at AUROC 0.980 / 0.995 with its error one-sided in
the safe direction (0 unsafe routings in 120 queries).

---

## 2. How much of this already exists

More than the two-phase framing suggests.

| piece | where | status |
|---|---|---|
| route | `neural/router.py` | **[built]** |
| allocate → fit → seal | `add_concepts` → `consolidate_batch` → `_seal` | **[built]** |
| the batch form | `insert_batch` places K members against one snapshot | **[built]** §15 |
| a path selecting which blocks apply | `PathLoRA.forward` sums over `state["path"]` | **[built]** |
| an always-on trunk slot | the **root node** | **[built, and empty]** |

That last row is the tell. The tree allocates a root node on every path and it is measured at
**‖B‖ = 0.00000 exactly** — 551K parameters that never train, because the root exists before the
first insertion and so never enters `_open` (`tests/diag_ancestor_live.py`). It is dead weight
*precisely because the backbone is already playing the root's role from outside the tree.* The
unified architecture does not add a slot; it fills one that is already there.

---

## 3. The obstacle that resolves

An always-on trunk block added at time *t* changes every concept sealed before *t*. That appears to
destroy bit-identity and with it the central claim of the repository.

**Resolution: a concept's path pins the trunk version it was sealed against.** Concept *c* is
evaluated with the trunk prefix that existed when *c* was sealed, not the current one.

This requires no new mechanism, because it is already how the data structure behaves:
`concepts[name]["path"]` is a **stored list**, not recomputed at read time, so adding nodes to the
tree does not retroactively alter existing concepts. Versioned trunks are free.

**The price, stated plainly: no backward transfer.** A concept sealed at trunk version 1 never
benefits from trunk version 7. That is the same forward-only property §23 established when order
invariance was refuted — the model does not converge to an order-independent fixed point; it
accretes. The cost is already in the record, and this makes it structural rather than incidental.

---

## 4. What does not resolve

### 4a. Capacity allocation **[open]**

Step 2 currently allocates a fixed rank *r* chosen once and never revisited. A unified operation has
to *decide* how much a batch deserves. The natural rule — grow rank until the batch's residual loss
stops improving — is the adaptive-rank literature (AdaLoRA and relatives) and is tractable, but it is
unimplemented here and it is the piece most likely to hide a nasty surprise. Until it exists, "one
operation" is one operation with a hand-set hyperparameter in the middle of it.

The rank sweep (`make_rank_frontier.py`) bears on this from an unexpected direction: at one seed, ACC
*decreased* monotonically with rank (0.844 at r=1 down to 0.766 at r=16) while bit-identity stayed
1.000 at every rank. If that replicates at three seeds, "allocate more capacity" is not obviously the
right default and the allocation rule may need to be conservative.

### 4b. Early seals are permanently unrecoverable **[open]**

Sealing at time *t* restricts the reachable function class to `W_sealed + Σ(low-rank deltas)`. Sums
of low-rank updates are dense in the limit, so this is not a representational impossibility — but it
is **path-dependent**, and a bad early seal cannot be undone. This is the theory-side view of the
same fact §23 found empirically as order dependence.

Whether it *matters* is exactly the compensation question: can later blocks recover what an early
seal skipped? **N35** gives the cost of sealing early; **N37** gives the compensation curve.

### 4c. "Indefinitely" must mean bounded in vocabulary, not in time **[open]**

Monotone growth over an unbounded stream is unbounded parameters. The repository's answer is that
cost tracks *vocabulary* rather than *experience* — an arrangement costs 1024 bytes, so a
160-experience stream costs ~10× less than one adapter per query (§6c) — and the growth curve
flattens **iff the world's vocabulary is finite.** That is an assumption about the world, not a
property of the method, and it should be stated as one.

Both escape hatches are damaged:

- `merge_concept` folds a delta into the base, returning parameters to the base model's count — but
  §5r measured **repeated** merges destroying the model (ACC 0.837 → 0.073), because a delta trained
  while one concept's path was active becomes unconditional once merged. N17's orthogonal variant is
  the partial mitigation.
- `remove_concept` deletes exactly, but reclaims only the *private* part — median **0.250** of a
  path, and 0.000 for concepts whose leaf a later concept adopted as an ancestor (§22a).

---

## 5. Is it trainable at all?

Everything above is design. The prior empirical question is whether a model built *entirely* as a
chain of sealed blocks can learn in the first place, and that is what `n37_progressive_seal.py`
probes, with the control that separates the two ways it could fail:

| arm | what it is | isolates |
|---|---|---|
| `full` | ordinary training, all weights | the ceiling |
| `progressive` | frozen base + K sealed rank-*r* trunk blocks, in sequence | the unified regime |
| `one-block` | frozen base + ONE rank-*Kr* block, whole budget | low-rank-over-frozen vs. **sealing** |

`one-block` is the arm that makes this a test. If `progressive` trails `full`, that alone does not
say why — it could be that low-rank updates over a frozen base are simply weak, which would be a fact
about adapters and not about sealing. `one-block` has the same parameter budget and the same frozen
base but never seals, so:

```
progressive ≈ one-block  <  full     the constraint is LOW-RANK-OVER-FROZEN, not sealing
progressive  <  one-block ≈ full     SEALING is the cost, and the unified regime pays it
progressive ≈ one-block ≈ full       neither hurts; the unified regime is free
```

---

## 6. Summary

**One operation, four allocation modes, forward-only, bounded by vocabulary.**

The unification is coherent and most of it is already implemented; the versioned-trunk problem
resolves for free because paths are stored rather than recomputed. What stands between this document
and a method is capacity allocation (4a), and what stands between it and a *claim* is the
compensation curve (4b) — plus the honest admission that indefinite operation assumes a finite
vocabulary (4c).

The immediate experiments: **N37-brief** (is it trainable at all — §5 above), **N35** (what property
of a sealed unit predicts downstream success: raw quality or compositional structure), and **N37-full**
(can extension recover what an early seal skipped).
