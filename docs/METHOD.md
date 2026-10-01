# Method: composing a flow policy's own prototypes (Read -> Compose -> Fold -> Consolidate)

The method of the overhaul of 2026-10-01 (plan: `~/.claude/plans/i-m-interested-in-re-workshopping-quizzical-crane.md`). It
follows the thesis in [THE_TASK.md](THE_TASK.md): decompose an arrival into the model's prototypes, store a code when they
explain it, fold a small unit only where they do not, consolidate, charge every byte, damage nothing, and generate new
trajectories from the model's own conditional. Every result it rests on is a read in `ledger/`, quoted in the README; this page
carries no numbers. The previous method (a one-demo object-relative program executed by the controller, with the policy
bypassed) is kept as a validated baseline: [archive/2026-10-program/METHOD_program.md](../archive/2026-10-program/METHOD_program.md).

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

## 1 Read: decompose the arrival

A task's factors are read by rule from its BDDL goal (`lifelong_ood/policy/factors.py`): the moved object's type, the goal
relation (predicate, target type, region), the qualifiers of the instruction, and the scene. The arrival is then decomposed into
TASK PROTOTYPES by a greedy cover of its factors (the selection step of Wang et al., moved from pixels to tasks): each selected
prototype supplies the factors it shares with the task, and its chain says in which phase that factor acts.

## 2 Compose: zero-shot, no demo, no weights

The policy's own velocities under several prompts are combined affinely at every Euler step of its sampler
(`lifelong_ood/policy/conditionals.py`; the user's ICLR paper restores composition by summing single-object predictions,
Liu et al.'s rule):

    v = v(o, base) + lambda * [ v(o, base with the task's object) - v(o, base) ]

The base is a task prototype the policy reads in this scene with the task's relation; the bracket is a minimal pair that differs
in the object only, so it isolates the object factor at this observation (the ICLR paper's d_j). The factor acts while the gripper
is open (the phase read from the policy's own gripper command, the hierarchy's top split) and is released after the grasp.
Nothing external is followed: every term is the policy's own conditional, so the policy keeps generating (a guide toward a target
chunk replaced the policy's prediction; this does not). The CODE a task keeps is its prompt ids, weights and phases
(`lifelong_ood/policy/factor_hooks.py`). Controls in every read: guidance on the task's instruction against the empty prompt,
and a placebo pair of unrelated training instructions at the same weight.

## 3 Fold: only where composition does not read the factor

A FACTOR TOKEN is a learned key and value per layer appended to the prompt's cache that the action expert attends to
(`lifelong_ood/policy/token_fold.py`; prefix tuning). Every policy weight stays frozen, so a task that does not address the token
is untouched by construction. It is fitted by flow matching on (a) a successful composition's own rollouts, which makes the
composition readable from one prompt (self-distillation, no demo), or (b) for a primitive no composition reaches, the arrival's
one demonstration as training data (never executed). A token is a FACTOR only if it transfers: it is reused, with no new demo, by
later tasks that share the factor. One arrival cannot tell which of its factors a token learned; two arrivals that share the
value and differ elsewhere can (docs/COBWEB_UPDATE.md, Proposition 7), so tokens of one factor are fitted on every arrival that
shares it. A token that cannot make the motion escalates to a keyed expert on the action expert.

## 4 Consolidate

Tokens of one factor are merged across arrivals; accrued tokens and experts are distilled into one shared adapter and deleted,
trading bytes for drift on old tasks. Designed; run after the zero-shot and fold reads.

## Known limits

- Composition reads only what the policy reads in some context: an object word the policy never contrasted anywhere is not
  readable by any composition, and needs a fold.
- The base must exist: a readable task prototype with the task's relation in the task's scene.
- Simulation only (LIBERO-90); pi0.5 so far. pi0 shares openpi's interface; SmolVLA runs through the same backbone interface for single prompts, and needs a batched observation method before composition runs on it.
