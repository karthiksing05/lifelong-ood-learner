# Plan W: Wang et al.'s selection on the factored library, Cobweb's operators for what it lacks (2026-10-04)

The user's goal: compositional OOD continual learning for an embodied agent, leveraging Wang et al. ("Test-time compositional
generalization in diffusion models via prototype discovery", arXiv 2605.07078). Wang et al. DISCOVER a frozen model's prototypes,
SELECT the few that best cover ONE out-of-distribution query, and COMPOSE them; their stated limitation is no mechanism to detect,
consolidate and reuse genuinely new primitives.

## Where the project stands

Plan E built the composition and the consolidation: a task is composed from an OBJECT prototype (approach, until the policy's own
grasp; an INSTANCE child when the instruction names one) and a GOAL prototype (transport), each a rank-4 LoRA consolidated over
every demonstration of its concept. On never-used start states the library performs within a few points of a per-task LoRA at a
fifth of its storage (ledger e3_hier). E4 runs it as a stream (ledger m3_prereg e4_stream).

What is not yet Wang's: the library SELECTS its prototypes by parsing the instruction's words (object_key, instance_key,
goal_key). Words miss equivalences ("the cream cheese box" and "the cream cheese" are one object) and cannot see a factor the
instruction does not name.

## The design

| step | Wang et al. | here |
|---|---|---|
| prototypes | modes of the frozen model's noisy marginals | consolidated object, instance and goal prototypes (the policy's own flow, adapted) |
| select | coverage of ONE query, per dimension | the arrival's one demonstration is the query: each prototype scored by how much it lowers the policy's flow-matching loss on the query's frames, per phase (approach: object prototypes; transport: goal prototypes) |
| compose | product of experts | phase-gated composition (object prototype until the grasp, goal prototype after) |
| new primitives | (missing) | Cobweb's operators decided by the query's gain: INCORPORATE the demonstration into the best-covering concept (refit it) when its gain clears the novelty threshold, CREATE a new concept from the demonstration when no concept covers it |

## Reads

| | question | design |
|---|---|---|
| W0 (exploratory, offline) | does the query select the right concept, and does a low gain flag a missing one? | a held-out demonstration of each stream task (never in any prototype) scored against E3's library; recognition = the top prototype is the instruction's concept; novelty AUC = own concept's gain against the best other concept's gain |
| W1 (pre-registered, if W0 passes) | the stream with query selection | E4's stream with two changes: (1) a demonstration is requested only when the best of {policy, language-keyed composition} fails practice (fixes E4's requests on tasks the policy already solves); (2) the requested demonstration first SELECTS among existing prototypes (incorporate) and creates a concept only when none covers it. Measures: demonstrations requested, prototypes created, success on fresh states, damage to earlier arrivals |

Gates for W1: object recognition (type level) on at least 24 of 30 queries and novelty AUC of at least 0.8 for both factors.

## W0's outcome (2026-10-05; ledger w0_select)

Both selection scores fail the gates. Flow-loss gain: the instruction's object concept on top for 12 of 29 held-out queries, its
goal concept for 13 (novelty AUC 0.49, 0.48). Sampled-chunk error: 12 and 17 (0.49, 0.52). One demonstration's motion varies
more between placements of the same task than between concepts, so a likelihood of one trajectory does not identify the concept,
although rollouts show the prototypes are sharply concept-specific. Query selection is dropped; W1 runs with the instruction's
keys and the thesis's request rule (pre-registered m3_prereg w1_stream). What Wang et al. contribute here is the program
(prototypes of a frozen generative policy, selected per OOD task and composed, the policy generating every action); what the
project adds is their missing mechanism: new primitives detected by practice, consolidated into concepts, reused by later tasks.
