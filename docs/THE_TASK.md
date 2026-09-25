# What the task is

Revised 2026-09-25 for the rework (`docs/PLAN_2026-09.md`). Read this before any result.

## The task, in one line

**Given a stream of instances, decompose each into the prototypes that explain it; store a code when
they do; fold a new prototype into the model's own manifold only where they do not; charge every byte;
damage nothing already learned.**

| | |
|---|---|
| **Input** | a stream of arrivals; each is ONE real image (the stream spec says which population it belongs to, the method is never told) |
| **Representation** | a pretrained diffusion model (the inherited prior) plus a PROTOTYPE TREE built from the model's own samples: nodes are modes of the noisy marginals, one level per noise level (`lifelong_ood/tree`) |
| **Operators** | DECOMPOSE / ADDRESS (a code over tree nodes, no weights) -- FOLD (new nodes at the levels where the arrival is unexplained, realised by a responsibility-gated minimal expert trained with anchoring on the model's own samples of the sibling nodes) -- CONSOLIDATE (distil an accrued node into the base weights, delete the expert) |
| **Output** | NEW samples of what arrived: of the composition an address names, or of the prototype a fold created, in contexts the fold never saw |

The arrival identifies *what* was seen. It is never a target to reproduce: a method that re-emits the
query is perfectly accurate and covers nothing, so coverage is a headline number, never a diagnostic.

## What it is not

*Not reconstruction.* Samples are scored against real held-out members of the arrival's class, with
the arrival itself excluded, in Inception space; pixel-space recall rewards diversity and is never
quoted (docs/REVIEW_2026-09.md, audit fault 7).

*Not test-time composition.* Zekun Wang et al.'s method discovers and composes prototypes for one query
and keeps nothing (its stated limitation is "detecting, consolidating and reusing genuinely new
primitives"). Here what is learned is KEPT, at a measured cost, and reused by later arrivals.

*Not addressing what the backbone already renders.* The C3 flagship showed 72 bytes of coordinates
reach combinations a backbone pretrained on those images can already draw. A fold is only claimed on
a value no training row contains (a withheld background, a white background outside the palette's
hull, a red foreground, an outline stroke), certified by `tests/test_stream_no_leakage.py`.

## The four axes, all load-bearing

1. **Identity**: does a sample show the arrival's composition (probe accuracy with certified ceilings)?
2. **Coverage**: does the set of samples cover the class (Inception-space coverage, KID, diversity)?
3. **Preservation**: per-old-class accuracy and coverage drift, eps-drift per noise band; a gate's
   false-positive rate on old images. Bit-identity is a construction property of a gated expert and
   is printed as such, never as a result.
4. **Cost**: every byte the method keeps, measured by `lifelong_ood.accounting.charge` (experts,
   nodes, codes, stored samples, teacher copies), plus the new quantity **depth of novelty**: the
   number of noise levels at which no node claims the arrival, which the fold's cost should track.

Axis 4 is the thesis. Axes 1-3 stop it from being trivially satisfiable: storing nothing is free and
copying the query is exact.

## Where it has been shown

Nothing yet under the rework. Every number enters this repository through `ledger/` and
`experiments/audit_numbers.py`; the README's ladder lists the rungs as their rows land. The
pre-rework numbers that survive the review (the label-gated per-value adapter, the C5 router, the
remeasured flagship) are backfilled rows and are quoted only as the reference the new mechanism must
reproduce.

## The translation to a diffusion policy

| here | in a diffusion policy |
|---|---|
| one image of an unseen combination | one demonstration of an unseen (object, container, verb) |
| new samples of that composition | new trajectories for that task, not the demo replayed |
| a prototype = a mode of the noisy image marginal at one noise level | a prototype = a mode of the noisy action-chunk marginal at one noise level (a motion scale) |
| a fold = a gated expert on the denoiser at the unexplained levels | the same, on the action denoiser |
| identity / coverage / preservation | rollout success / generalisation within the task / old tasks still succeed |
| bytes charged | what the new skill cost to keep |

The claim in the form it must take for policies: experiences accumulate as codes over prototypes and
as folded prototypes, so cost tracks the vocabulary of the world rather than the length of the
stream, and nothing already learned is overwritten. `docs/PLAN_2026-09.md` Section 7 states the
prerequisites and the benchmarks (LIBERO, THE COLOSSEUM).
