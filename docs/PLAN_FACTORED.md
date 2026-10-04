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

## D1's outcome (2026-10-04; ledger d1_factored)

H1 passes (kept 343 vs 271 of 600 zero-shot); H2 fails (324 vs 454 of 580 against the per-task LoRA + acceptance). The gap is
the APPROACH: on the 6 tasks where only the LoRA succeeds, the policy's first grasp takes the wrong object in 114 of 120 episodes
(milk for orange juice, alphabet soup for ketchup, the white bowl for the frying pan). The goal primitive acts after the grasp and
inherits the wrong object; attention binding turns the arm toward the named object but grasps nothing (59 of 120). The LoRA,
fitted on the task's own demonstration, grasps the named object first in 109 of 120.

## Plan E: the object factor gets its own prototype

The goal primitive is the transport's prototype; the approach has had only an attention bias, which says WHERE but not HOW to
reach and grasp. Plan E gives the approach its own consolidated primitive, symmetric to the goal's:

| phase | factor | key | prototype |
|---|---|---|---|
| approach (until the policy's own grasp) | the OBJECT | verb + object, the instance qualifier left out ("put the black bowl"; `primitives.object_key`) | a rank-4 LoRA fitted only on demonstrations' frames before their first sustained grasp, consolidated over EVERY LIBERO-90 demonstration with that object (the black bowl: 14 tasks, kitchens and living rooms, plates, drawers, cabinets, trays) |
| transport (grasp to release) | the GOAL | preposition + target | the goal primitive of plan D |

The step's prompt keeps its qualifier ("at the back", "middle"), so an object primitive fitted on several instances can read it;
binding stays available to the acceptance test. A task is the composition of its object's and its goal's prototypes; 30 tasks
need 15 object and 4 goal prototypes (19 x 4.7 MB = 89 MB) against 29 per-task LoRAs (545 MB).

| | question | design |
|---|---|---|
| E0 (exploratory) | does the object primitive fix the approach? | 30 stream tasks, stored states 0-9, noise seed 11 (paired with D1's acceptance runs): leave-one-out object + goal primitives (no demonstration of the task); all-in object + goal primitives (the task's one demonstration absorbed into both prototypes: the library after the stream); the all-in object primitive alone |
| E1 (pre-registered) | the factored library against the per-task LoRA | fresh start states, new noise seed; acceptance over {policy, leave-one-out pair, all-in pair}; kept vs zero-shot and non-inferiority to the per-task LoRA + acceptance |

Success: E1's kept arm non-inferior to the per-task LoRA + acceptance at a sixth of its bytes, every task composed from shared
object and goal prototypes.

## E0's outcome and a data defect (2026-10-04; ledgers e0_object, l90_scenes)

E0 (exploratory, stored states 0-9): the factored library with each task's demonstration absorbed (object + goal prototypes)
succeeds 201 of 300 against 137 zero-shot; the per-task LoRA 205 of 290. On D1's six wrong-object tasks: 39 of 60 (LoRA 42). The
object primitive fixes the approach. Leave-one-out (no demonstration of the task) gives 125 against 117 on 25 tasks: an object's
prototype from OTHER tasks transfers for some objects only.

While reading E0, identical fits for tasks 46 and 50 exposed a data defect: the IPEC port merges tasks by instruction, and the
first-episode rule gave 16 LIBERO-90 tasks another scene's demonstration (5 in the stream) and put 4 stream tasks' own
demonstration into their "leave-one-out" fits. Fixed at the data layer (Demos90.episodes_of_task, verified per scene block).
D1's verdicts hold without the 9 shared-instruction tasks. E1 is registered on the corrected data (m3_prereg tag e1_factored):
library refit, the 5 affected LoRAs refit, acceptance and test at noise seed 12.

## E1's outcome (2026-10-04; ledger e1_factored)

H1 passes (kept 439 vs 272 of 600). H2 fails, with half of D1's gap: 420 vs 478 of 580 against the scene-correct per-task LoRA
+ acceptance (-0.100 [-0.193, -0.010]). The library beats the LoRA in the basket (148 vs 134) and ties in the tray; 45 of the 58
missing successes sit on four tasks that NAME AN INSTANCE of the black bowl (middle, front, back), where the type prototype
(14 black-bowl demonstrations) does not pick the named bowl. Interpretation (i4): object prototypes are object-specific
directions; both factors act in the same layers and modules; a prototype is stable to leaving one demonstration out.

## Plan E2: instance children (Cobweb's hierarchy)

A type concept has children; an instance is served by the most specific concept that matches. A step whose object phrase
names an instance (primitives.instance_key = object key + qualifier) uses the CHILD prototype: consolidated over every
demonstration with that object and that qualifier, starting from the parent's prototype (c0_fit --init) and specialising on its
own demonstrations. The type prototype still serves unqualified instructions. Screen E2 (exploratory, the 4 instance tasks,
states 0-9, seed 12, paired with E1's acceptance runs): child-from-parent vs child-from-scratch vs the type prototype. If the
children close the instance gap, E3 (pre-registered, new seed) reads the hierarchical library on all 30 tasks; the parser fix
(step_key's rightmost preposition) adds task 60 to the tray goal prototype, refit for E3.

## E2 and E3 outcomes (2026-10-04; ledgers e2_instance, e3_hier)

E2 (exploratory): instance children 37 of 40 on the four instance tasks (type prototype 13, LoRA 35); starting a child from its
parent adds nothing. E3 (pre-registered, a second never-used set of start states, seed 13): the hierarchical library (22
prototypes, 102.9 MB) keeps 499 of 600 against 273 zero-shot; against the per-task LoRA + acceptance (545.4 MB) 480 vs 494 of 580,
-0.024 [-0.074, +0.024]: within a few points, interval holding zero, but non-inferiority at 0.05 NOT established (as the
registration's power projection expected; between-task differences dominate). Without acceptance, library 480 vs LoRA 485. The
library beats the LoRA in the basket and trails slightly on the other goals.
