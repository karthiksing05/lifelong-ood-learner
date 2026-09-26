# What pi0.5's agent prototypes are

Written for: collaborators reading the embodied line.  Numbers from `ledger/r2c_atlas` (`experiments/r2c_atlas.py`);
figures in `figures/robot/atlas_*.png`.

**Setup.** The library is the 40-task tree of `r2b_marginal` (modes of pi05_libero's action marginal over 10,249 demo frames of
every task it was fine-tuned on).  A prototype is a 10-step chunk of 7-D delta commands.  Each is described by the hand's
displacement and rotation over the chunk (measured scale: a unit command moves the hand about 1.1 cm and turns it about 0.1 rad
per step; signs checked against the demos' hand positions) and by its gripper behaviour, and named by its most typical member.

| noise level t | prototypes | motion words | members sharing the prototype's word | members sharing its gripper state |
|---|---|---|---|---|
| 0.2 | 6098 | 111 | 0.90 | 1.00 |
| 0.3 | 1936 | 109 | 0.57 | 1.00 |
| 0.4 | 242 | 73 | 0.31 | 0.99 |
| 0.55 | 3 | 3 | 0.13 | 1.00 |
| 0.7 | 2 | 2 | 0.07 | 0.99 |

**Findings.**

1. The first question the hierarchy asks is whether the hand is holding something.  At every level, 99-100% of a prototype's
   frames share its gripper state; the two coarsest prototypes are 'gripper open' (approach, reach, release: 55% of frames,
   all 40 tasks) and 'gripper closed' (carry, place: 45%, 38 tasks).
2. The finest level is a vocabulary of motion primitives: 90% of a prototype's members move the same way.  The most used, by
   share of frames: left + closed (7%, 25 tasks); hold + open (7%, 35 tasks); down + open (6%, 39 tasks); down-back + open (5%, 37 tasks); lift + closed (5%, 35 tasks); down + closed (4%, 35 tasks); forward + open (4%, 25 tasks); left + open (3%, 12 tasks); down-back + closed (3%, 34 tasks); lift-forward + closed (3%, 27 tasks).
3. Between them the prototypes group frames by phase and posture rather than by direction: at t = 0.4 only 31% of a
   prototype's members share its motion word, and the coarse modes are slow ('hold') because a mode of a blurred mixture sits
   where many motions overlap.  The mid levels are where tasks share prototypes (Section 10 of the plan: at t = 0.4, 242
   prototypes, about three tasks each).
4. Motions are shared across tasks, which is why a motion alone cannot name the task it came from (the prompt-program test).

Coarse-to-fine (the top of the tree, children by frames):

- t=0.7: **hold + open** (5630 frames, 40 tasks)
  - t=0.55: **hold + open** (5548 frames, 40 tasks)
    - t=0.4: 111 children; by frames: hold + open (3444), forward + open (481), left + open (469), forward-right + open (359), down-back + closing (286), opening (186), twist wrist + open (122), right-forward + open (26)
- t=0.7: **hold + closed** (4619 frames, 38 tasks)
  - t=0.55: **hold + closed** (3269 frames, 38 tasks)
    - t=0.4: 104 children; by frames: down-back + closing (143), opening (131), lift-right + closed (88), twist wrist + closed (69), down-back + closed (67), hold + closed (53), right + closed (45), right-forward + closed (32)
  - t=0.55: **left + closed** (1432 frames, 26 tasks)
    - t=0.4: 27 children; by frames: hold + closed (2339), left + closed (1322), lift + closed (260), down-left + closed (17), forward + closed (6), left-down + opening (4), forward-up + closed (4), down-back + closed (4)

![primitives](../figures/robot/atlas_prototypes.png)

![vocabulary by level](../figures/robot/atlas_vocabulary.png)

![tasks by primitive](../figures/robot/atlas_incidence.png)

## What a new task borrows

A held-out LIBERO-90 task is composed by a learned product of experts over its vocabulary of 32 library prototypes
(`lifelong_ood/policy/composer.py`; `experiments/r6_composer.py`, `r6_borrow_viz.py`).  Following the dominant prototype
frame by frame turns each task's single demo into a script written in the library's words (frame of the switch in brackets):

- **put the white bowl on the plate**: left-up + open [0] -> forward-right + open [7] -> down + open [43] -> down + closing [68] -> hold + closed [85] -> lift + closed [96] -> left + closed [102] -> down + closed [142] -> down + closed [172] -> back-down + opening [187] -> left-up + open [202]
- **put the white mug on the left plate**: right + open [0] -> down-back + open [8] -> down-back + closing [21] -> lift + closed [31] -> forward-right + closed [48] -> down-back + closed [61] -> back + open [70] -> hold + open [78]
- **turn off the stove**: forward + open [0] -> twist wrist + open [14] -> right-down + open [15] -> down-back + open [34] -> right-back + closed [69] -> hold + closed [70] -> down + open [86] -> right-back + closed [100]
- **open the microwave**: down + open [0] -> forward + open [26] -> forward + open [39] -> back-left + opening [53] -> right + open [75] -> forward + open [95] -> forward + open [101] -> back-left + opening [112] -> forward + open [125]
- **put the frying pan under the cabinet shelf**: forward-left + open [0] -> forward + open [7] -> forward-down + open [31] -> left-back + open [46] -> lift-back + open [83] -> down + open [88] -> hold + closed [102] -> right + closed [128] -> right + closed [169]
- **put the moka pot on the stove**: forward + open [0] -> forward + open [37] -> down-left + open [59] -> down + open [61] -> left-down + closed [75] -> down-back + open [111] -> opening [124]

The pick-and-place tasks read as reach, lower, close, lift, carry, lower, open, retreat, all borrowed from the
40 trained tasks; the new motions (the stove knob turned the other way, the microwave door pulled open) are assembled
from a wrist twist and closed-gripper right-back strokes, and from back-left strokes while the gripper opens.

![white bowl](../figures/robot/borrow_36.png)

![stove off](../figures/robot/borrow_39.png)

## Borrowing parts of prototypes

Two ways to compose a new task from the library by borrowing parts of prototype trajectories (numbers in `ledger/`):

- **Learned soft borrowing** (`lifelong_ood/policy/composer.py`): a softmax over 32 prototypes per step and dimension, from a
  small gate on the robot state and progress plus per-prototype time and dimension profiles.  Smooth and readable (the scripts
  above) but not precise enough for contact.
- **Dimension-wise borrowing** (`lifelong_ood/policy/dimcode.py`): each action dimension's whole 10-step trajectory from one of
  eight jointly chosen prototypes; the executed chunk is rebuilt exactly from the stored ids and choices, at half the demo's jerk
  and 1-8 KB per task.  With practice and a small fold where the demo grasps, it is non-inferior to a per-task LoRA at 74x less
  storage over 21 held-out tasks (pre-registered; README, "M3 FINAL"; docs/VALIDITY.md).

## Inspection: what composition actually uses (2026-09-26)

Numbers for this section are in `docs/VALIDITY.md` (generated from the ledger by `experiments/make_validity_doc.py`) and in
`ledger/r8_inspect`, `ledger/r9_library_ablation` and `ledger/r12_forced`.

- **The shared prototypes are the verbs of manipulation.**  The t = 0.4 prototypes that the most training tasks share are
  hold-open, hold-closed, descend-and-close (the grasp) and open (the release), each used by 34-40 of the 40 tasks.  The gallery
  shows the same phase of motion in different scenes, and forced execution in the simulator realises their commanded hand
  motion almost exactly from any scene (`experiments/r12_forced.py`).

  ![shared prototypes](../figures/robot/r8_gallery_shared_t0.4.png)

- **Composition does not borrow the shared prototypes; it borrows leaves.**  Dimension-wise borrowing takes almost all of its
  (frame, dimension) pairs from the finest level, from prototypes with one to three member frames of a single training task.  A
  confident prior can be steered only by a tight teacher (the precision-weighted product), so what a program can use is the
  prior's own precise motions: the policy's chunk means at the training frames, the components of its action marginal, which
  are the zero-noise level of the hierarchy.  A library of exactly those leaves rebuilds a held-out demo more closely than the
  tree's prototypes, and keeps their smoothness (the smoothness comes from the policy's denoising); the shared level alone is
  too coarse.  The hierarchy above the leaves organises them, measures novelty and names motions; composition reads the leaves.

  ![what a task borrows](../figures/robot/r8_gallery_39.png)

- **A new task is a mosaic of the prior's motions.**  Segment borrowing (`lifelong_ood/policy/segcode.py`) writes the demo's
  action sequence as 10-step pieces of leaves, one per action dimension: every piece of "turn off the stove" comes from some
  training task, the pieces come from all four suites, and the rebuilt trajectory traces the demo's in every dimension.  The
  borrowed pieces lean only slightly toward related training tasks (instruction-word overlap above a shuffled-pairing null,
  VALIDITY.md): the reuse is of motion, not of task.

  ![segments, stove](../figures/robot/segments_39.png)

  ![provenance](../figures/robot/r8_provenance.png)

- **The hierarchy's anatomy.**  Prototype counts fall from one per frame at t = 0.05 to two at t = 0.7, and the share of
  frames whose prototype several tasks share rises with t; precise composition draws its lenders from the bottom.

  ![anatomy](../figures/robot/hierarchy_anatomy.png)

- **Rollouts from the same start.**  On "turn off the stove" the prior, given the new instruction, does the trained moka-pot
  task instead; the composed programs turn the knob.  On "put the white bowl on the plate" replay drops the bowl beside the
  plate and borrowing + fold stalls at the cabinet, where the LoRA and borrowing at every fifth frame succeed.

  ![strip, stove](../figures/robot/strip_39_t30.png)

  ![strip, white bowl](../figures/robot/strip_36_t30.png)
