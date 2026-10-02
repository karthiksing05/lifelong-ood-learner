# Plan C: reusable motion primitives (2026-10-02, the user's decision "Go for C")

The scheme so far composes what the frozen policy holds (binding with object prototypes, the qualifier grammar, sequencing) and
keeps, by an acceptance test on the learner's own practice, whatever beats the policy. It reaches tasks the policy fails by
reading the wrong object; it does not reach a motion the policy lacks (README: "what binding cannot reach", "sequencing"). Plan C
lets the one demonstration TEACH such a motion, under three constraints that keep it compositional and continual:

1. **Fold only what composition fails.** An arrival first tries what the library already holds (the policy, binding forms,
   stored primitives for its steps); only when nothing beats the policy does it request the one demonstration.
2. **A primitive is a STEP, not a task.** The instruction splits into steps at its conjunction (`bind_hooks.split_conj`); each
   step has a key = (verb, preposition, target noun phrase), e.g. `close||the top drawer of the cabinet`,
   `put|in|the top drawer of the cabinet`, `put|on top of|the cabinet`. The demo is cut at the same event the policy switches
   steps on (its first sustained release), and one primitive is fitted per step on that segment, prompted with the step's clause.
   The moved object is NOT part of the key: binding supplies it, so `put|in|the top drawer` fitted on the butter serves the pudding.
3. **Reuse by key, keep by practice.** Primitives join the library under their key; a later arrival whose step has a stored
   primitive gets it as a candidate (with binding for its own object), and the acceptance test keeps it or not. A primitive is
   stored only if it was part of an arm the acceptance test kept.

Primitive = a keyed LoRA on pi0.5's action expert (`lifelong_ood/policy/adapt.py`; rank 16 over the expert, the archived one-demo
LoRA's capacity, 18.7 MB) trained by flow matching on the segment's frames. The policy stays frozen; at run time the server holds
the stored primitives as keyed banks and switches the active key when the policy's own gripper ends a step
(`lifelong_ood/policy/primitives.py`, PrimHooks). Smaller primitives (rank 4, the last two layers) are measured after the first read.

Measured so far: 38 of the 48 failing tasks with a moved object have every step shared with another LIBERO-90 task by key.

## Reads

| | question | design | kill |
|---|---|---|---|
| C0 | do step primitives fitted on a task's own demo execute the task? | the 13 pool tasks with an archived one-demo LoRA; arms: zero-shot, the LoRA (whole demo, one adapter), own-demo step primitives with step switching; practice states 0-9, 10 episodes (exploratory) | step primitives < half the LoRA's successes: the step cut loses what one adapter holds |
| C1 | does a primitive transfer to another task with the same step key? | pairs of pool tasks sharing a key (same scene and other scenes): the donor's primitive (+ binding for the recipient's object) vs zero-shot vs the recipient's own primitive; practice states (exploratory) | no gain over zero-shot on >= 2 of the pairs: the library cannot reuse |
| C2 | the stream (pre-registered) | the 48 tasks in q17's two orders: library candidates -> acceptance -> else one demo -> step primitives for the steps without a stored one -> acceptance; init states 30-49; zero-shot and the per-task LoRA (trained for the pool) as references | -- |

Reported in C2: success against zero-shot and against the per-task LoRA, demonstrations requested, primitives stored and reused,
bytes (prototypes 4 KB, primitives as stored), and success of earlier arrivals at the end (preservation, by construction: the
policy is frozen and units load only for the task that kept them).
