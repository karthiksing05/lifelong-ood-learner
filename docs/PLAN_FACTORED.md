# Plan D: factor every step into an object prototype and a goal prototype (2026-10-03)

The user's goal (docs/THE_TASK.md): one example in, new instances out, stored cheaply, nothing damaged, by composing prototypes,
then learning continually. First principles: COMPOSITIONALITY (a task is a combination of reusable parts, each learned once) and
PROTOTYPICALITY (each part is a prototype: the consolidated, shared summary of every example of that part).

## Why the current scheme falls short (measured)

- Against a per-task one-demo LoRA with the acceptance test, the stream reaches 400 vs 467 of 580 on fresh states (ledger
  f1_fresh); the gap sits on steps where the instance matters ("on the plate": 102 vs 141).
- The interpretation (ledger i3_behavior) shows why: a step primitive fuses two factors. "On top of the cabinet" changes the
  CARRY (the goal; z 0.36), but "on the plate" and "in the basket" change the APPROACH, i.e. which object the arm goes for, and the
  plate primitive is not specific to its step. Pooled over tasks with different objects, a step primitive overrides the policy's
  choice of object.
- Attention binding (the object factor) and a primitive conflict because both act during the approach (ledger c0_exec: 13 vs 63);
  training the primitive under binding helps (ledger q28_bindprim: 119 vs 82) but still lands below the policy (144).

## The design

A step is two factors in sequence, split at the POLICY'S OWN GRASP (its first sustained gripper close):

| phase | factor | prototype | learned from |
|---|---|---|---|
| approach (gripper open) | WHICH object | the object prototype (4 KB, SigLIP features at a grasp) + the qualifier grammar for the instance | a demo used as a query, or the library's type prototype |
| transport (grasp to release) | WHERE it goes | the GOAL primitive: a rank-4 LoRA fitted only on demonstrations' post-grasp frames, keyed by the goal (preposition, target), object omitted, consolidated over every task with that goal | the post-grasp segments of demonstrations sharing the goal |

At run time the factors never overlap: binding acts only before the grasp, the goal primitive only from the grasp to the release
(phase-gated hooks). The acceptance test keeps any of {policy, binding, goal primitive, binding + goal primitive} per task, so
nothing the policy already does is damaged.

## Reads

| | question | design |
|---|---|---|
| D0 (exploratory) | does a transport-only goal primitive keep the step's gain without overriding the object? | the 4 stream keys, leave-one-out goal primitives vs the earlier step primitives, with and without approach binding; stored states 0-9 |
| D1 (pre-registered) | the factored scheme against the per-task LoRA with the acceptance test | the stream's 30 tasks, fresh start states, new noise seed; kept arm vs zero-shot, vs per-task LoRA + acceptance, vs the step-primitive stream |
| D2 (pre-registered) | demonstrations saved | the stream with goal primitives: a demonstration is requested only when the library's goal primitive (with binding) fails acceptance |

Success = D1's kept arm level with or above the per-task LoRA + acceptance at a fraction of its bytes, and D2 requesting markedly
fewer demonstrations than the step-primitive stream (27 of 30).
