# Method: composing a flow policy's own prototypes (Read -> Compose -> Fold -> Consolidate)

The method of the overhaul of 2026-10-01 (plan: `~/.claude/plans/i-m-interested-in-re-workshopping-quizzical-crane.md`). It
follows the thesis in [THE_TASK.md](THE_TASK.md): decompose an arrival into the model's prototypes, store a code when they
explain it, fold a small unit only where they do not, consolidate, charge every byte, damage nothing, and generate new
trajectories from the model's own conditional. Every result it rests on is a read in `ledger/`, quoted in the README; this page
carries no numbers. A new task arrives as its instruction and its scene, with at most ONE demonstration used once as a query
and never executed, replayed or fitted (the user's decisions of 2026-10-01: a demonstration to imitate is not a test of the
thesis; one used as a query is allowed when it makes learning rapid and reusable). Demo-executing work, including the earlier
one-demo program, is archived: [archive/2026-10-program/](../archive/2026-10-program/).

## What the frozen policy already holds

- **Prototypes.** Mode-finding on the policy's action marginal over its training tasks' frames gives a hierarchy of
  prototypes, noise level as depth (`experiments/r2b_marginal.py`; Wang et al.'s concept discovery, moved to the
  observation-marginal because one observation's conditional is a single narrow bump). Its coarse prototypes are verbs (reach
  down, close, carry, lower, release), its top split is the gripper's state, and its labels change at a task's phase events.
- **Task prototypes.** Every task the policy performs, stored as its factors and its prototype chain: the sequence of verbs its
  actions pass through (`lifelong_ood/policy/tasks.py`). Tasks group into families by their chains.
- **Readable factors.** The policy reads an instruction's factor (an object, a goal relation, a qualifier) where its training
  CONTRASTED that factor in a similar scene; elsewhere a scene keys a trained habit, whatever the instruction names. The user's
  ICLR paper found the same in image diffusion models: a model that never read several objects from one caption fails to,
  although it learned each object.

## 0 The method (2026-10-02): bind with a prototype, fold, accept by practice, reuse the library

- WHERE, from one query. The arrival's demonstration, if it has one, is used once: its first grasp, projected into its first
  frame, marks the moved object's image patch, and the policy's own SigLIP features there form a 4 KB OBJECT PROTOTYPE
  (`experiments/q4_template.py`). The prototype keeps the object's appearance and its place in the scene. Every prototype is kept
  in a LIBRARY, typed by the object.
- BIND (compose). While the policy's commanded gripper is open, the action expert's attention to the 4 image patches most like the
  prototype is raised to mass 0.7 at every layer and Euler step (`lifelong_ood/policy/bind_hooks.py`, `attn_bias.py`). The scene's
  trained routine supplies the motion and the prototype supplies which object; the policy generates every action.
- FOLD. Binding's successes on practice states fold into an 18.4 KB token that the task's own instruction then reads (section 3).
- ACCEPT. Before anything is kept, the learner tries the policy alone, the binding form and the token on fresh practice states and
  keeps the best, ties to the policy (`experiments/q13_read.py`). A unit that does not beat the policy on the learner's own
  attempts is not stored, so a task is never served worse than the frozen policy serves it on practice.
- REUSE. A later arrival whose object type is already in the library asks for no demonstration: its candidate forms use the
  library's prototypes of that type, and the acceptance test keeps one or none (`experiments/q16_read.py`). The kept unit is a
  library index.
- GRAMMAR. A qualifier word picks the instance: back / front / middle order the type's peaks by image row, right / left by
  column (`experiments/q22_grammar.py`); the type prototype is the mean of the library's prototypes of the type.
- MOTION PRIMITIVES (plan C, docs/PLAN_PRIMITIVES.md). An instruction splits into steps at its conjunction; a step's key is its
  verb, preposition and target. Where nothing the library holds passes the acceptance test, the arrival's one demonstration is
  cut into its steps at the first sustained release, and each step's primitive (a rank-4 LoRA on the action expert,
  `lifelong_ood/policy/primitives.py`, `experiments/c0_fit.py`) is REFIT on every demonstration of that step so far: one
  primitive per step key, shared by every task with that step. The policy switches primitives at its own release; the
  acceptance test keeps a primitive or the policy. A primitive fitted on one demonstration holds that object and room; fitted
  across the arrivals that share the step it transfers (docs/COBWEB_UPDATE.md, Proposition 7).
- NOT MECHANISMS. Attention binds the object slot only: raised on the goal region while the gripper is closed, it breaks the carry;
  added to a primitive it breaks the primitive; chaining clauses by switching the instruction needs clauses the policy performs.
  A token fitted across tasks binds the object more often but succeeds less often. Wang et al.'s product of experts over the
  policy family and motion prototypes (`lifelong_ood/policy/poe.py`) failed its pilot. Each is a README row.

Sections 1-4 below describe the role-code scheme of 2026-10-01 (no demonstration at all), kept as the no-demo reference.

## 1 Read: decompose the arrival

A task's factors are read by rule from its BDDL goal (`lifelong_ood/policy/factors.py`): the moved object's type, the goal
relation (predicate, target type, region), the qualifiers of the instruction, and the scene. The arrival is then decomposed into
TASK PROTOTYPES by a greedy cover of its factors (the selection step of Wang et al., moved from pixels to tasks): each selected
prototype supplies the factors it shares with the task, and its chain says in which phase that factor acts.

## 2 Compose: a task prototype with its object slot rebound

A new task that differs from a task the policy performs only in WHICH object fills a role is coded as that task prototype plus
a role map (`lifelong_ood/policy/roles.py`): the base is a performed task of the same scene whose goal equals the new goal with
the moved object replaced (same target, every other atom identical, so a trained sequence composes too). The code is the base's
id and the object pair, tens of bytes, and reads as a sentence: "task 50 with the milk in the alphabet soup's role". The policy
runs the base's own conditional; what composition must supply is the BINDING of the base's object slot to the new object, which
the policy cannot read from the word. Two injections bind it, both leaving every action to the policy:

- ROLE: the new object and the base's object exchange appearance in what the policy sees (the simulator's renderer exchanges
  their meshes: an oracle; `experiments/libero_client.py --swap`), while the policy's own commanded gripper is open. It works
  where the scene's routine keys on the object's appearance (in LIBERO, on its shape, where training contrasted shapes).
- ATTENTION: the action expert's attention to the image tokens covering the new object is raised to a target mass at every
  layer and Euler step (`lifelong_ood/policy/attn_bias.py`), while the gripper is open. It redirects a routine tied to a
  position, where the role swap cannot.

Practice decides which injection a task keeps. Prompt composition (the policy's velocities under several prompts, summed at
every Euler step by the ICLR paper's rule; `lifelong_ood/policy/conditionals.py`) remains available and amplifies a factor the
policy reads faintly; it cannot supply a binding the policy never learned.

## 3 Fold: from the policy's own attempts

A FACTOR TOKEN is a learned key and value per layer appended to the prompt's cache that the action expert attends to
(`lifelong_ood/policy/token_fold.py`; prefix tuning). Every policy weight stays frozen, so a task that does not address the token
is untouched by construction. It is fitted by flow matching on a composition's own successful attempts, recorded with the REAL
scene and relabelled with the task's own instruction, which makes the composition readable from that instruction alone
(self-distillation); no demonstration enters, and at test the token needs no injection and no simulator state. A second round
practises with the token alone and refits it on both rounds' successes. A token is a FACTOR only if it transfers: it is
reused by later tasks that share the factor. One arrival cannot tell which of its factors a token learned; two arrivals that share
the value and differ elsewhere can (docs/COBWEB_UPDATE.md, Proposition 7), so tokens of one factor are fitted on every arrival that
shares it.

## 4 Consolidate

Tokens of one factor are merged across arrivals; accrued tokens and experts are distilled into one shared adapter and deleted,
trading bytes for drift on old tasks. Designed; run after the zero-shot and fold reads.

## Known limits

- The binding injections need WHERE the new object is. The policy's own grounding and an open-vocabulary detector both fall short
  of the pre-registered pointing gate; WHERE comes from the simulator's segmentation (an oracle, during practice only), from one
  demonstration's object prototype, or from the library.
- The goal slot: where a scene's routine puts the object somewhere else (a drawer for "on top of the cabinet"), no injection here
  rebinds the goal.
- Coverage: binding gives a form on a minority of the failing LIBERO-90 tasks with a moved object (README, broad screens).
- A role swap binds only where the scene's routine keys on the object's appearance; a routine tied to a position, objects that
  differ only in colour or label, and objects whose grasps differ (a pan for a moka pot) stay out of reach.
- Role codes need a base: a task the policy performs in the same scene with the same goal up to the moved object.
- Simulation only (LIBERO-90), one policy so far (pi0.5).
