# Prototypes and composition for skill transfer and continual learning (frozen pi0.5, LIBERO-90)

Plan of 2026-09-28, written after the user's goal statement: "focus on how prototypes and composition can be used for skill
transfer and continual learning, leveraging the research we're already investigating". Decisions taken with the user: object
poses from the simulator are used as an ORACLE first, then replaced by a learned reader before any headline claim; the
continual-learning stream starts MINIMAL (about 14 tasks, two mirrored orders) and grows only if it shows falling marginal cost.
The history (M0-M3, every retraction) stays in [docs/PLAN_2026-09.md](PLAN_2026-09.md); every number below is in `ledger/` and audited.

## 0. Status (2026-09-29, midday): the ladder is run, then iterated (rules v2)

Every probe ran under its pre-registration (`ledger/m3_prereg` tags p0, p1, p1_amend1, p2, p3a, p3b, p4, g38, p6, then v2, v2_synth,
p3c, p3c_amend1, p3c_reader); the full reads are in [docs/VALIDITY.md](VALIDITY.md) ("Transfer plan probes") and the README's ladder.

| probe | read | verdict |
|---|---|---|
| P0a-c audits (`ledger/p0_audit`) | every held-out demo matched to its raw recording; 0.83 of 106 persistent movable-object failures are aim failures; the retargeted demo with no policy succeeds 58 of 180 | C2 survives; 3.8 viable |
| P1 bindings through the prior, rules v1 (`ledger/p1_bindings`) | 9 tasks: bindings alone 143 of 180, with the fold 97, control 66; all 21 tasks: bindings alone 314 of 420, with the fold 264, control 253 | PASSES for bindings alone; the fold arm meets its kill rule |
| P6 learned pose reader, v1 | 126 of 180 against 143 for the simulator's poses (lower bound -0.256) | above the control; not non-inferior |
| 3.8 data generator | 331 of 420 generated; LoRA on them 293 of 420 with no practice (one-demo LoRA 252, p 0.0014) | PASSES |
| P2, P4 | phase-localized negatives 284 vs 302; grown library 70 vs 74 of 120 | fail / killed |
| P3a, P3b cross-task bindings, v1 and no policy | P3a 39 vs 91 of 100; P3b 11 of 80 | failed; REVERSED by P3c |
| **rules v2** (a defect: the living-room demos' first frames catch the object still falling, up to 12.2 cm above rest) | generator dev check 386 vs 331 of 420 on the practice placements; through the prior, 21 tasks: **385 of 420** vs v1 314 and LoRA + failure negatives 302 (both p below 0.001), 54.9 KB for all 21 tasks; LoRA on v2-generated demos 327 vs 293 (p 0.003) | PASSES (V1, V2, V4) |
| **P6 again: readers** | first-frame reader on 30 placements: 362 (lower bound -0.121, fails); + 400 simulator-rendered placements per task (labels at training only): **376 of 420**, -0.021 vs the simulator's poses, lower bound -0.050, 9.3 MB for 21 tasks | **PASSES (V5): the headline condition holds** |
| **P3c: tasks composed from other tasks' phases** (grasp source by object type, place source by BDDL goal relation) | 6 targets: **114 of 120** vs 117 with their own demo (lower bound -0.067); sibling's whole demo 42 (p below 0.001); with a reader trained on rendered scenes only 92 (0 of 20 on the white bowl, whose carry comes from a pan held by the handle); with the in-hand pose predicted from the grasp binding **112** (lower bound -0.058) | **PASSES (C1, C2, R2): C3 holds under v2, with no demo, no practice and no simulator state at test**; R1 failed |
| stove (task 39) | 6 of 20 in every v2 arm: the knob starts anywhere from 0.53 to 2.09 rad; the demo's twist leaves most init states on | a joint-state binding, not built; the allocator routes it to prototype borrowing |
| **practice allocator** (3.6's rungs: v2 phases + reader, prototype borrowing + fold, v2-generated LoRA; the most practice successes on states 0-9) | **391 of 420** vs 376 for the phases alone (p 0.0015) and 302 for LoRA + failure negatives; 45.6 MB for 21 tasks; four tasks switch (the stove to borrowing: 17 vs 6) | **PASSES (AL1)**: the prototypes carry what bindings cannot |
| **C1 grammar from the prototype hierarchy** (`ledger/c1_grammar`) | 50 demos: at t = 0.55 (3 prototypes) the labels change 2.9 times per demo, 0.99 of the phase events at a change (random 0.20), 0.68 of the changes at an event (random 0.12) | **HOLDS at t = 0.55**: the hierarchy gives when phases start and end; the scene gives which object |
| perception across tasks by object type (`ledger/p6_reader` tag type_transfer) | leave-one-task-out type readers: fixtures 0.20 cm, movable objects 3.12 cm | negative for movable objects: readers stay per task (rendered scenes, no demo) |
| **P5 stream** (50 tasks, two mirrored orders; pre-registered p5_stream + amend1: validated bindings) | demos requested 35 (A) and 33 (B) of 50; 20 tasks compose only when late vs 4 only when early (p 0.0015); composed arrivals 252 vs 561 of 640 for their own demos | **S1 holds, S2 FAILS**: the vocabulary grows, but a place binding stored as a receptacle-relative path keeps the source's route; carry warping and a generic transport made it worse on practice states (82, 6 vs 123 of 320); validated composition (amend2, exploratory) under test |
| book yaw (pre-registered yaw) | the reader reads book yaw at 0.8 deg (fixed yaw 10.2 deg); success 173 vs 175 of 220 | fails: orientation is not what limits the book tasks |

What this means for the claims (section 2): C2 (binding) holds: the demo's phases re-anchored on the objects' poses guide the frozen
prior to 385 of 420 with no weights (376 with a learned reader, no simulator state at test). C3 (composition = transfer) holds under
v2 on the six tasks that have sources: a task with no demo is composed from the grasp of a task with the same object type and the
place of a task with the same goal relation, and scores as its own demo does; the grasp binding carries (another type's grasp: 42 of
120). The v1 negatives (P3a, P3b, and "bindings transfer across instances, not tasks") came from the falling-object defect and the
no-policy executor. C1 (grammar) was not tested separately: the phase cut from object events is the grammar. C5: the fold hurts once
bindings act. C4 (continual learning with falling cost) is the next step: P5's precondition now holds. Open: joint-state bindings (the
stove); a stream large enough to show demos per task falling with the binding vocabulary (the 21-task pool composes only 3-4 tasks
in either order: experiments/p5_stream.py plan).

## 1. Context: what the evidence says, and the reframe

What works: compose -> practise -> learn. A one-demo composition executed through the frozen prior is a demonstration engine;
its recorded successes and localized failures train a per-task LoRA to 302/420 (0.72), the best arm, above rank-64 and full
fine-tuning (267). Its cost is 18.7 MB per task and nothing is shared.

What failed, and why it matters for transfer:
- The composition program alone equals open-loop demo playback (168-176 vs 187/420): it copies the demo's path.
- 79 of the 123 never-rescued practice failures "stall" at the demo's last frames: the hand finished the demo's path and the task
  was undone; where the demo holds something at the stall, the robot's grip is > 5 mm tighter in 25 of 54 (it grasped less).
- Four tests of sharing WEIGHTS across tasks are negative (one fold for all tasks 146, the other half's fold 107, family pooling
  104/240 vs 137 random vs 153 one demo, family pretraining 148 vs 153). Pooled drawer siblings executed the SIBLING's instruction.
- The hierarchy's levels split cleanly: fine leaf sequences identify the task (a robot rollout matched to its task from human
  demos 0.97 vs 0.76 for raw trajectories); coarse sequences are near-invariant within a task (0.105 vs 0.433 between tasks) and
  group tasks into verb-like families without language.
- A sibling's WHOLE adapter reused unchanged scores 0 almost everywhere, with one telling exception: task 58 (ketchup -> tray) with
  task 48's adapter (ketchup -> basket) scores 12/20, above its own LoRA's 9/20. Reuse works where the BINDING coincides (the same
  object).

The reframe: a task = GRAMMAR x BINDINGS.
- GRAMMAR = the coarse phase sequence (reach-open, close, carry-closed, release): shared within a family, invariant across
  instances, already in the prior and the library. It transfers for free, which is why family pretraining added nothing.
- BINDINGS = which object, where it is, how to hold it, where it goes: task-specific, what held-out tasks lack, and what one
  adapter scrambles when siblings are pooled.
- LIBERO-90 is an explicit binding grid: the living-room scenes cross objects {ketchup, tomato sauce, milk, orange juice, salad
  dressing, butter, chocolate pudding, alphabet soup, cream cheese} with receptacles {basket, tray}; the kitchens reuse black/white
  bowls, frying pans, mugs and drawers across scenes. The prior already solves several cells (alphabet soup, cream cheese and butter
  into basket or tray; pudding into the tray), so its own rollouts can supply bindings.

So the plan makes the binding the unit of transfer and continual learning: store bindings as small object-relative records keyed
by object and receptacle, compose a new task as a family grammar x an object binding x a place binding, learn weights only for
the residual, and never share weights across siblings. This is the task definition's own form ("one demonstration of an unseen
(object, container, verb)"; `docs/THE_TASK.md`) and the thesis's cost claim restated where it can hold: cost tracks the binding
vocabulary, not the number of tasks.

## 2. Claims to test

- C1 GRAMMAR: the coarse prototype levels give a task's phase grammar; it is shared across a family and needs no new bytes.
- C2 BINDING: the persistent failures are binding failures (grasp pose, grip, placement); fixing the binding fixes them.
- C3 COMPOSITION = TRANSFER: a new (object, receptacle) combination, both seen before in other tasks (or in the prior's own
  successes), executes from the stored bindings with no new weights and fewer practice rollouts than learning it alone.
- C4 CONTINUAL LEARNING: a Cobweb loop over bindings (address, compose, practise, fold residual, grow, consolidate) accumulates
  tasks with falling marginal bytes and rollouts and nothing forgotten; bindings are shared by object/receptacle, weights never
  across siblings (the interference map).
- C5 RESIDUAL: weights are needed only where bindings fall short; a practice rule escalates from 0 B -> 2 KB -> 258 KB -> 18.7 MB.

## 3. Components

All paths under `/coc/flash10/ksingara3/lifelong-ood-learner/`.

3.1 Oracle state recording (`experiments/libero_client.py`, `--objects`). Per step: every movable object's `<o>_pos` / `<o>_quat`
(already in the LIBERO obs; `bddl_base_domain.py:470-490`), fixture body poses (`env.env.sim.data.body_xpos[env.env.obj_body_id[o]]`),
articulated joint positions (tuple-safe `sim.model.get_joint_qpos_addr`), `env.get_sim_state()`, `env.obj_of_interest`, and phase
predicates (grasp via robosuite `_check_grasp`, lifted = object z +2 cm, placed = the problem's `_eval_predicate`, drawer/door/knob
joint thresholds). Per query: `meta/objects`, `meta/pred` to the server. Caveat printed in every table: sim poses are privileged.

3.2 Demo identification (`experiments/p0_demo_init.py`, LIBERO venv, CPU). The IPEC demos carry no object state and demo k is NOT
init state k (IPEC keeps successful replays only; frame-0 states do not identify the init). Replay demo 0's actions from each of
the 50 init states with a fresh env per trial (the r18 protocol) and identify its placement by trajectory error + success;
the deterministic replay then yields object poses at every demo frame.

3.3 Phase programs (`lifelong_ood/policy/phase.py`: `Phase`, `PhaseProgram`, `PhaseHooks(ComposeHooks)`). Phases = maximal runs of
one t = 0.55 basin label (r19 `labels`/`collapse`), cut at every gripper event (`policy/data.py:gripper_events`); kind = contact
(contains an event or moves an object) or transport. Inside a phase the existing `ComposeHooks.align` runs on that phase's keys
expressed in the phase's anchor frame; phases advance on EVENTS (predicate true, or the prior itself flips the gripper), not on
hand proximity; exceeding 1.5x the demo's phase length is a logged PHASE FAILURE (`GuidedPolicy.infer` returns `"phase"`).

3.4 Bindings. Per contact phase, in the anchor object's frame: anchor role and object type (from `obj_of_interest`, language-free),
the eef pose at the gripper event `T_ee_event_obj = T_obj^-1 T_ee`, grip width (finger qpos), the approach offset, the predicate
id: about 42 B per phase, <= 210 B per task. LIBERO actions are OSC deltas, so an object translation changes WHERE a contact
phase runs, not its chunks: at phase entry the keys are moved to the object's current pose (yaw rotates the xy of the deltas), the
grasp fires when the eef reaches `T_ee_event_obj` (+-5 mm), the transport is re-timed to the transformed start (DemoGen-style).
Sources: the identified demo (3.2), the composition's practice successes (object-relative, so easy instances yield bindings valid on
hard ones), and the PRIOR's own successes on solved cells (seeding the binding table from the prior's competence). Several
sources pool into a binding node with `tree/update.py:incorporate` (Welford mean/variance of `T_ee_event_obj` and grip).

3.5 Binding table = the vocabulary. Keys: (role, object type) for grasp bindings, (role, receptacle type, relation) for place
bindings; values: binding nodes with provenance. A new task's program = its family grammar (or its own demo's) + the grasp
binding of its object + the place binding of its receptacle, each looked up or, if missing, extracted from its own demo/practice.
Weights are never looked up across tasks.

3.6 Residual rungs (reuse `adapt.train`, `adapt.train_nft`, `r4_fold` windows, `HybridHooks` masks). R0 program + bindings (0 B of
weights); R1 phase-local fold: rank 1 on `action_out_proj` (`adapt.SCOPES["out"]`, 2.1 KB fp16), on only in the failing phase;
R2 the 258 KB last-2-layer fold; R3 rank-16 LoRA (18.7 MB) with phase-localized negatives. Rule from practice on states 0-29:
a phase failing > 30% gets R1; escalate while practice success < 0.60 after 30 rollouts; stop at >= 0.80 or when a rung adds
< 2 successes in 30.

3.7 Phase-localized negatives (`experiments/r17_nft.py --negatives phase`): the executed chunks of the FAILED phase (by predicate
or finger gap) replace the hand-pointer stall, which by r18 mostly labels a hand that had finished the path.

3.8 Fallback: retargeting as a DATA GENERATOR. Transform the demo's contact segments to sampled placements, execute open-loop in
sim, keep the verified successes, and train the learner on them without practice rollouts (DemoGen's route). Used if the guide
route fails P1 but the generator ceiling (P0c) passes.

3.9 The loop (`experiments/r23_stream.py`), Cobweb operators on the binding table and the tree:
- ADDRESS: fine signature -> a known task (r19: 0.97 from robot rollouts; recall keys of `r15_consolidate`); coarse sequence -> a
  family (r19 `edit`, cut 0.5); object/receptacle types -> known bindings.
- COMPOSE: grammar + looked-up bindings; missing bindings extracted from the one demo.
- PRACTISE: 30 recorded rollouts (states 0-29) with `--objects`; phase outcomes.
- FOLD: rung escalation (3.6).
- GROW: phase-labelled success chunk means become leaves with provenance (`tree/update.py:create`, then `tree/build.py:link_parents`;
  r4/r5 left `parent = -1`); binding nodes `incorporate` new successes.
- CONSOLIDATE: `merge` binding nodes of the same object type across tasks; adapters are never merged (r13, r20).
- Storage per task (`m34_report.arm_bytes` extended): program (1.9-8 KB) + new bindings (<= 210 B) + rung (0 / 2 KB / 258 KB /
  18.7 MB) + grown leaves x 300 B.

3.10 Learned pose reader (the oracle's replacement): a regression head on the ResNet-18 features `RecallHooks` already computes,
trained on the recorded `<o>_to_robot0_eef_pos` of practice episodes; swapped in for a final row of every headline table.

## 4. Probe ladder (cheapest kill of the most expensive assumption first; each read pre-registered in `ledger/m3_prereg`)

| probe | question | arms and read | cost | kill / decision |
|---|---|---|---|---|
| P0a | can the demo's placement be identified? | replay demo 0 from all 50 init states, 21 tasks | CPU < 1 h | unique match (error ratio >= 3, replay succeeds) on < 15/21 tasks: bindings from practice/prior only |
| P0b | are the persistent failures binding failures? | bit-exact replay of the 123 with `--objects`; grasp offset at the close event vs the demo's; failure type of the ~557 recorded failures by finger gap (grasp-miss / drop / place-miss / fixture) | CPU < 1 h | < 50% of movable-object failures off by > 2 cm or grasp-miss: C2 fails, the guide route (P1) is dropped, P2 and 3.8 remain |
| P0c | ceiling of retargeting | open-loop retargeted playback of the demo at 20 sampled placements x 9 concentrated tasks (sim, no policy) | CPU ~1 h | generation success < 30%: retargeting as a guide dead; >= 30%: 3.8 viable |
| P1 | do bindings rescue failures? | 9 concentrated tasks (42, 32, 1, 35, 40, 36, 52, 62, 41), states 30-49, 180 paired: phase program + bindings + 258 KB fold vs the current `dimfold_s1_last2`; secondary: end-of-demo stalls | ~3 A40-h | lower bound of the paired difference < -0.05, or no fall in end-of-demo stalls |
| P2 | do phase-localized negatives beat pointer ones? | r17 rerun with phase negatives vs `r17bal` (302/420), 21 tasks, 420 paired | ~9 A40-h | < 331/420 (302 + MDE): negatives stay pointer-localized |
| P3 | does a binding transfer across tasks? (C3) | grid targets 58 (ketchup -> tray), 59 (tomato sauce -> tray), 32 (ketchup -> top drawer): own bindings vs bindings looked up from other tasks with the same object / receptacle (48, 49, 62, 55-57, 61, 3-5; prior successes for solved cells) vs MISMATCHED bindings (another object's grasp: the interference control) vs the whole-adapter reuse baseline (58 with 48's adapter 12/20); 20 states each, 60 paired per contrast; no own practice for the looked-up arm | ~4 A40-h + binding extraction | looked-up bindings not non-inferior to own at -0.10, or not above mismatched by >= 15 pts: C3 fails |
| P4 | does growth matter for success? | recompose siblings with the success-grown library vs the base library, 6 tasks x 20 | ~1 A40-h | < 5 pts: growth stays bookkeeping |
| P5 | the minimal stream (C4) | section 5 | ~30 A40-h | K2/K3 below |
| P6 | does it survive without the oracle? | P1/P3/P5's best arm with the learned pose reader vs the oracle | ~3 A40-h | report the gap; a headline claim needs the reader row within 0.10 |

P0a-c run first and in parallel (CPU); P1 and P2 are independent and can share the GPUs; P3 needs P0a or bindings from practice;
P5 only if P1 or 3.8 passes and P3 passes. If P1, P3 and 3.8 all fail, stop building mechanisms: the honest paper is the practice
loop plus the hierarchy's characterization (section 7).

## 5. The minimal stream (P5)

Tasks (14 + 3 repeated arrivals):
- pick-and-place grid, living room: 48 ketchup -> basket, 58 ketchup -> tray, 49 tomato sauce -> basket (new), 59 tomato sauce ->
  tray (new), 52 milk -> basket, 53 orange juice -> basket, 62 salad dressing -> tray;
- drawer family, kitchen: 3, 4, 5 (put in the top drawer and close); cross-scene composition 32 ketchup -> top drawer;
- unrelated controls: 42 frying pan under the shelf, 35 open the microwave, 36 white bowl on the plate;
- repeated arrivals: 3 earlier tasks re-presented late with their second demo (IPEC episode 1): ADDRESS should recognize them at
  near-zero new cost.
- binding seeds before the stream: the prior's recorded successes on the solved cells 46, 47, 50, 51, 55, 56, 57, 61 (10 episodes
  each with `--objects`, 80 episodes).

Per arrival (every arm, same states): one demo; 10 prior episodes (states 0-9; keep the prior at >= 0.5, as now); 30 recorded
practice rollouts (states 0-29) with updates after each block of 10; 20 scoring episodes on states 30-49, seed = order index.
Orders: A family-interleaved, `random.Random(0)`; B = A with each family's order reversed, so each combination target is first
in one order and last in the other (within-task pairing for transfer). Mid and end checkpoints re-score every earlier task
BLIND (no instruction: the agent must ADDRESS from its first frames and first chunk) and keyed.

Arms: A0 the loop; A1 the loop with the binding table and library frozen at the seeds (no growth, no cross-task lookup);
B1 per-task LoRA with failure negatives on A1's practice (the current best, the linear-cost reference); B2 sequential ER-LoRA and
B3 sequential LoRA without replay on order A only if budget remains (else cited: arXiv 2603.03818, TAIL, LOTUS, IsCiL, CLARE).

Metrics per task: success S_t (states 30-49), rollouts to the third practice success R_t, practice success in the first block
FWT_t, marginal bytes C_t (library + bindings + weights after minus before), binding-novelty depth d_t (how many of grammar /
object binding / place binding / grip were missing at arrival), blind and keyed BWT, AUC over checkpoints; frontier of cumulative
bytes vs cumulative successes per arm.

Pre-registered hypotheses (Holm over the sign tests; bounds one-sided 95%):
- H_ni: A0 non-inferior to B1 at -0.10 on the prior-fail tasks, pooled over orders.
- H_cost: (level) A0's total bytes <= 0.5x B1's; (slope) a combination target's marginal bytes as a later arrival < as a first
  arrival, paired within task across orders; descriptive: log bytes on d_t and position.
- H_transfer: R_t and FWT_t better as a later arrival than as a first, paired within task across orders.
- H_pres: blind end success non-inferior to acquisition at -0.10; blind ADDRESS accuracy >= 0.9; repeated arrivals add < 1 KB.
Kill rules: K2 after half of order A: A0 below A1 by more than the margin, or bytes not below half of B1's -> stop order B and the
baselines; K3: blind ADDRESS < 0.8 at the mid checkpoint -> drop the preservation claim.
Scale-up to the 36-task stream (all families, ~175 A40-h) only if H_cost(slope) and H_transfer hold.

## 6. Compute (about 120 episodes per A40-hour on the lab's 4 A40s; adapter fits 5-7 min)

| step | episodes | A40-h |
|---|---|---|
| P0a-c | 0 policy episodes | CPU ~3 h |
| P1 | 180 + 180 control | ~3 |
| P2 | 420 + 21 fits | ~9 |
| P3 | ~360 + extraction | ~4 |
| P4 | 120 | ~1 |
| binding seeds + prior episodes of new tasks | 80 + 20 | ~1 |
| P5 A0 x 2 orders, A1, B1, checkpoints | ~3,000 + ~60 fits | ~30 |
| P6 | ~360 | ~3 |
| total | ~4,700 | ~51 + CPU |

## 7. Honest outcomes and survival estimates

- Works (P1 or 3.8, P3, P5 pass): a frozen VLA learns new object-receptacle combinations from one demo by composing a shared
  grammar with object-relative bindings drawn from other tasks and from its own prior's successes, at bytes that fall with the
  binding vocabulary and without forgetting; the hierarchy is load-bearing (coarse levels = grammar and the interference map, fine
  levels = task identity and motion).
- Partial (typical: bindings help, the LoRA fallback still fires): claims per axis and per family; the frontier plot carries it;
  the allocator (compose, bind, escalate) is the method.
- Fails: the defensible paper is what the ledger supports now: the practice loop (0.72) with composition as a demonstration engine,
  the hierarchy as a language-free taxonomy and interference map, and a documented list of what does not transfer.
- Survival estimates (the skeptic's review): bindings as a guide 0.15-0.3 (placement variation in LIBERO-90 is small, about
  5 cm boxes and fixed yaw per the BDDL regions, to be confirmed in P0; contact phases may fail on closure, not aim); retargeting as a
  data generator 0.45; phase-localized negatives 0.35; cross-task binding lookup (C3) 0.3, raised by the 58 <- 48 reuse result;
  falling marginal cost 0.3 conditional on C3.

## 8. Files

New: `lifelong_ood/policy/phase.py` (PhaseProgram, PhaseHooks, Binding), `lifelong_ood/policy/bindings.py` (binding table, lookup,
pooling via `tree/update.py`), `experiments/p0_demo_init.py`, `experiments/p0_geometry.py` (P0b-c), `experiments/r22_phase.py`
(builds phase programs and bindings, ledger rows), `experiments/r23_stream.py` + `experiments/m5_stream_stats.py` (in the
`m3_stats.py` pattern), `tests/test_phase_program.py` (bytes; retarget then inverse = identity; a translation leaves chunks
unchanged and moves keys; event-driven advance on a synthetic recording).
Modified: `experiments/libero_client.py` (`--objects`, predicates, phase log), `experiments/serve_guided.py` (`phase:` mode, blind
`address:` mode), `experiments/r17_nft.py` (`--negatives phase`), `experiments/launch_pack.py` (arms `phasebind*`, `bindlookup`,
`bindmismatch`, stream arms), `experiments/m34_report.py` (marginal bytes, binding bytes), `lifelong_ood/stats.py` (family-cluster
bootstrap level), `experiments/audit_numbers.py` and `docs/` as results land.
Reused unchanged: `ComposeHooks.align/guide_at`, `HybridHooks` gating, `dimcode`/`segcode`, `r4_fold` windows, `adapt.train`,
`adapt.train_nft`, `adapt.PrefixBank` (host-memory store), `r15_consolidate` recall keys, `r18_branch` re-entry (fresh env + seed +
reset count), `r19_invariance` labels, `launch_pack`/`roll_pack`/`roll.sh`, `stats.py`, `m3_stats`, `make_validity_doc`.

## 9. Verification

- Unit tests (`tests/test_phase_program.py`) and the existing suite (`tests/test_audit.py`) pass before any rollout.
- Every recording with `--objects` replays bit-exactly through the r18 gates (G1 prefix, G3 recorded-noise continuation).
- A 4-episode smoke test per new server mode (as for r18): the knob changes actions from the phase boundary on and not before.
- Every probe is pre-registered (hypotheses, arms, kill rule) before its first rollout; reads use `stats.py` (Wilson, two-level
  paired bootstrap, sign test, Holm); every quoted number goes through `audit_numbers.py` and `audit_coverage.py`; README, plan and
  `docs/VALIDITY.md` updated per result; commits local only.
