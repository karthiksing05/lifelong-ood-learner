# lifelong-ood-learner

**Continual learning for diffusion policies by composing and folding agent prototypes** (the line since
2026-09-25, late: [docs/PLAN_2026-09.md](docs/PLAN_2026-09.md)). A trained diffusion policy holds a
hierarchy of *agent prototypes*, the modes of its noisy action-chunk marginals with the noise level as
depth (Zekun Wang et al.'s prototype discovery, applied to the action denoiser). A new task is first
COMPOSED from those prototypes; only a motion no prototype explains is FOLDED into the policy by a
responsibility-gated minimal expert; the library grows the way Cobweb's tree grows. Benchmark: LIBERO,
lifelong protocol. The image-based phases that built the machinery are paused; their plan is
[docs/PLAN_2026-09_images.md](docs/PLAN_2026-09_images.md) and their state is in the ladder below.

The review that retired the previous architecture is [docs/REVIEW_2026-09.md](docs/REVIEW_2026-09.md);
the theory the mechanism rests on (the ideal denoiser's one-instance update IS Cobweb's incremental
update, noise level = tree depth) is [docs/COBWEB_UPDATE.md](docs/COBWEB_UPDATE.md); the previous
repository is preserved under [archive/2026-09-pre-rework/](archive/2026-09-pre-rework/README.md).

## The thesis benchmark: tasks never seen, no demonstration

A new task arrives as its instruction and its scene. Nothing of the task itself is shown: no demonstration, no recorded
trajectory, no fitted adapter. The frozen policy (pi0.5) must do the task from what it already holds, its prototypes and the
task prototypes it performs, by composing them ([docs/METHOD.md](docs/METHOD.md)); later, what it learns from its own attempts is
kept as small units and reused by later arrivals. Results that read or execute a demonstration of the task (the one-demo program,
LoRAs, STRAP, DSRL on a LoRA, demo-fitted tokens) are not tests of this thesis; they are kept, audited, in
[archive/2026-10-program/README_demo_rows.md](archive/2026-10-program/README_demo_rows.md). Motivated by Wang et al.'s concept
discovery and the user's ICLR paper on caption coverage ("Seen but Unnamed"). Every row below is pre-registered
(`ledger/m3_prereg` tags zc_probe, zc_bind, zc_pilot, zc_main, b_bind_q1, b_bind_q2, b_bind_q2_amend1, b_scheme_main) unless marked exploratory.

| rung | result |
|---|---|
| why the policy fails new tasks (`ledger/f0_coverage`; lifelong_ood/policy/factors.py; exploratory, existing zero-shot runs) | over all 90 LIBERO-90 tasks, pi0.5 succeeds where its 40 training instructions named the task's factors in that scene family: the rule separates success 0.54 (29 tasks) from 0.14 (AUC 0.72); whether training contrasted the object with another in one scene predicts success with AUC 0.74. Every held-out failure is a task whose factors were named only elsewhere, or never |
| does the policy read object words? (`ledger/f1_binding` tags f1_bind_libero_90 and f1_bind_libero_object; experiments/f1_binding.py) | in living room 1 the policy picks up the alphabet soup first in 10 of 10 episodes whichever object the instruction names (alphabet soup, cream cheese, ketchup or tomato sauce): it runs the scene's trained habit, and its zero-shot successes there are coincidences of that habit. In LIBERO-Object's own layouts, an instruction naming another object present is followed 0.43 of the time |
| the first chunk as a readability probe (`ledger/f1_factor_probe`) | NOT VALID by its pre-registered gate: on LIBERO-Goal the first chunk points at the named object 0.43 of the time (gate 0.8), although the change between two prompts turns toward the newly named object (contrast cosine 0.72); the online binding read replaces it |
| zero-shot composition of the policy's own conditionals (`ledger/zc_pilot`, `ledger/p1_bindings` tag zc_main; lifelong_ood/policy/conditionals.py) | the pilot passed its gate (14 of 70 against 4 for guidance on the instruction and 8 for a placebo pair), but the main read on the 12 tasks with an object-factor code FAILS: 22 against 9 of 240 for zero-shot (lower bound -0.004 against the pre-registered +0.05), and 19 for guidance on the instruction at the same weight. The gain sits on one task, the ketchup in living room 1 (9 of 20 against 0) |
| task prototypes (`ledger/f0_task_library`; lifelong_ood/policy/tasks.py) | 49 tasks the policy performs, each with its verb chain; every held-out task is decomposed into the task prototypes that cover its factors (e.g. ketchup into the tray = the alphabet soup into the tray in that scene, plus the ketchup task of LIBERO-Object), and the factors no prototype covers are listed (turn off, close the top drawer) |
| where an unread object is, from the policy's own model (pre-registered Q1a; `ledger/b1_where` tags read_all and read_w3center, `ledger/m3_prereg` tag b_bind_q1; experiments/b1_where_probe.py) | FAILS its gates (pointing >= 0.8 on LIBERO-Goal and >= 0.6 on the class-a tasks). pi0.5's own PaliGemma attention from the object's words to the image points at the object 0.74 of the time on LIBERO-Goal at its best layer (layer 10) and 0.51 on class a; an open-vocabulary detector (OWLv2, box centre) 0.65 and 0.71. The detector finds kitchen objects (mugs, pans, bottles) and never the alphabet soup. Where-maps below come from the simulator's segmentation and are an oracle |
| binding by the action expert's attention, offline (pre-registered Q1b; `ledger/b2_steer` tag offline_class_a; lifelong_ood/policy/attn_bias.py) | raising the action expert's attention on an object's image tokens (every layer, every Euler step; zero bias is the policy bit for bit) turns the first chunk toward that object (contrast cosine 0.45 to 0.62 over the configurations on the class-a tasks) and, at target mass 0.7, points it at the object 0.43 of the time against 0.30 unbiased |
| binding by attention, online (pre-registered Q2-mini; `ledger/b2_q2` tags b_q2mini and b_q2full; practice states 0-9) | at target mass 0.7 the policy picks up the ketchup first in 7 of 10 episodes (zero-shot 0) and the milk in 7 (zero-shot 1), but grasps nothing in 10 of 10 tomato-sauce episodes. Over whole episodes on the 14 tasks with a role code it rarely completes a task (mean success 0.01); biased only while the gripper is open (EXPLORATORY) it puts the tomato sauce in the tray 10 of 10 times (zero-shot 1 of 10) |
| ROLE rebinding: a task prototype the policy performs, with the new object in its object's role (pre-registered amendment; `ledger/b2_q2` tag b_q2mini, `ledger/m3_prereg` tag b_bind_q2_amend1; lifelong_ood/policy/roles.py) | the code is the base task and an object pair (about 40 bytes, e.g. "task 50 with the milk in the alphabet soup's role"); the simulator's renderer gives each object the other's appearance (oracle) and the policy runs the base's instruction. In living room 2 the policy picks up the milk first 0.9 of the time and the orange juice 1.0 (zero-shot 0.1 and 0), succeeding 0.4 and 0.3; shown the real object once it holds it (EXPLORATORY), 0.7 and 0.7. It fails in living room 1, where the routine goes to the alphabet soup's position whatever stands there (ketchup 0.0, 7 of 10 grasping nothing), and on mugs that differ only in colour. Steerability: an unnamed can given the soup's appearance is never picked (0.0; same can shape), an unnamed carton is (1.0 and 0.8; EXPLORATORY): the scene's routine keys on shape where its training contrasted shapes |
| folding the policy's own successes into a token (EXPLORATORY; `ledger/f4_fold` tag token_rollouts_role, `ledger/b2_q2` tag b_q2mini) | the role episodes' successes on states 10-29, relabelled with the task's own instruction and recorded with the real scene, fit an 18.4 KB key-value token with the policy frozen. Given only its own instruction and the token, the policy picks up the orange juice in 10 of 10 episodes and succeeds 0.6 (zero-shot 0, 0); the milk 0.5 and 0.2. No demonstration and no simulator state at test |
| THE SCHEME'S MAIN READ (pre-registered; `ledger/b5_scheme` tag main, `ledger/m3_prereg` tag b_scheme_main; experiments/b5_scheme_read.py) | on the 14 failing tasks with a role code, practice on states 0-9 chose a form for 4 (milk and orange juice by role swap, tomato sauce into the tray by the attention pointer, the left bowl by role swap) and none for 10. On init states 30-49, which no step tuned on: the composed forms succeed 65 of 80 times against 21 for zero-shot (sign test p below 0.001); the tokens folded from the policy's own practice successes, given ONLY the task's instruction, succeed 51 of 80 times against 21 (Holm p below 0.001; milk 3 against 0, orange juice 13 against 2, tomato sauce 15 against 7, left bowl 20 against 12). The named object is picked up first in 0.75 of the token episodes against 0.21 zero-shot. Transfer FAILS: the tomato-sauce token from living room 3, used on the tomato-sauce task of living room 1, scores 0 of 20 (zero-shot 0). Over all 14 tasks the scheme scores 51 of 280 against 21; the 10 tasks without a form are untouched |
| binding split by phase (EXPLORATORY; `ledger/b2_q2` tag b_q2full) | the attention pointer while the gripper is open and the role swap while it is closed (so the routine carries an object it believes is its own) does not rescue the position-bound living room 1: ketchup 0 of 10, tomato sauce 1 |
| PROTOTYPE PoE PLANNING, the mechanics (pre-registered P0; `ledger/q0_share`, `ledger/m3_prereg` tag q_probes; lifelong_ood/policy/poe.py) | Wang et al.'s product of experts ported to the flow policy, the demonstration as the query (since 2026-10-02 one demo may serve as the query, used once). The policy's share is not set by the last Euler steps alone: a prototype weight applied only before them still moves the output by 0.38 of the gap (P0's kill fired), so weights are constant over the steps and the policy's share is a floor; at a constant weight of 0.5 the composed chunk keeps 0.33 of the policy's sensitivity to the observation, at weight 1 none |
| PoE planning, scene options (pre-registered P1; `ledger/q1_options`; experiments/q1_options.py) | the policy holds no hidden options at an observation: ascending its own score from 64 random starts under the empty prompt converges to one mode in every LIBERO-Goal frame and all but 0.06 of the class-a frames, so that generator is dropped. Raising its action expert's attention on one of 16 image blocks sends the first chunk toward 2.7 different objects per frame on the class-a tasks, and toward the object under the block 0.37 of the time (chance 0.25) |
| WHERE from the one demonstration (EXPLORATORY; `ledger/q4_template`; experiments/q4_template.py) | the gripper's position at the demo's first grasp, projected into its first frame, marks the moved object's patch; the policy's own image features there are a 4 KB object prototype. In new start states its best-matching patch lies on the object 0.60 of the time (27 tasks; 1.0 for the milk, the orange juice, the bowls; 0.0 to 0.2 for the tomato sauce in living room 1 and the salad dressing), and 0.55 when reused in other scenes holding the same type of object |
| PoE planning, offline explanation (pre-registered P2; `ledger/q2_explain`) | on the 21 held-out tasks, a code selected on demo 0 per gripper phase (the policy under its instruction, the policy under other prompts and with its attention on one of 16 image blocks, and the motion prototypes the demo's chunks fall into; Wang's coverage and product) explains demo 1 better than the policy alone in 21 of 21 tasks, and the policy's own options cut the fit's NLL by 0.57; but options recomputed at demo 1's observations beat frozen ones in only 10 of 21: the image-block options do not carry where the object is. Codes take 1.6 KB per task on average |
| PoE planning, pilot (pre-registered; `ledger/q3_pilot`, `ledger/m3_prereg` tag q_pilot) | FAILS its go rule: on 6 development tasks (10 practice episodes each) the phase codes succeed 3 of 60 times at policy share 0.5 and 0 at 0.3 and 0.7, against 1 zero-shot; the prototype product alone and the options alone 0. A product of motion prototypes held over a whole gripper phase is an average motion and carries no where; the main read did not run |
| binding with the demo's object prototype, practice states (EXPLORATORY; `ledger/b2_q2` tag q5_w4) | with no simulator state, raising the action expert's attention on the 4 patches most like the demo's object prototype while the gripper is open puts the white bowl on top of the cabinet in 8 of 10 episodes (zero-shot 0) and the milk in the basket in 2; over the 20 tasks with a prototype it succeeds 0.055 of the time against 0.009, often grasping nothing where the prototype points wrong |
| THE DEPLOYABLE SCHEME (pre-registered; `ledger/q6_main`, `ledger/m3_prereg` tag q6_main; experiments/q6_chain.py) | the demo supplies only the 4 KB object prototype; binding's successes on practice states fold an 18.4 KB token, the token practises alone and is refit. On init states 30-49, for the 2 tasks practice gave a form: binding 17 of 40 against 2 zero-shot (Holm p below 0.001), the token alone (instruction + token, nothing else) 10 of 40 (Holm p 0.039; white bowl 6 against 2, milk 4 against 0) |
| one round of the token's own practice (pre-registered; `ledger/q7_tokpractice`, `ledger/m3_prereg` tag q7_tokpractice; experiments/q7_read.py) | FAILS: each demo token practises alone on states 10-29 and is refit on the demo plus its own successes; on the same 420 paired episodes it succeeds 150 times against 140 for the demo token (sign test p 0.42) and stays far below the one-demo LoRA (252; lower bound -0.360). Self-practice does not add the motion a token lacks |
| broad screens of approach-only binding (EXPLORATORY; `ledger/b2_q2` tags q8_screen and q9_w4screen) | over 48 more failing LIBERO-90 tasks with a moved object, 10 practice episodes each: with the simulator's segmentation as where, the policy picks up the named object first in 17 tasks and practice success reaches 0.2 or more on 7 (the bowls onto the plate, the white bowl on the cabinet, the bowl into the drawer, the pudding beside the plate, a book into the caddy); with the demo's object prototype instead, and no simulator state, on 8 (mean success 0.09) |
| the broadened scheme (pre-registered; `ledger/q8_main`, `ledger/m3_prereg` tag q8_main) | on the 7 tasks the oracle-where screen gave a form, init states 30-49: the folded tokens alone succeed 77 of 140 times against 56 zero-shot (Holm p 0.013; the front bowl onto the plate 17 against 5); the binding form itself 70 (not significant). Folding does better than the composition it learned from |
| the broadened scheme, fully deployable (pre-registered; `ledger/q10_main`, `ledger/m3_prereg` tag q10_main) | on the 7 tasks the demo-prototype screen gave a form, with no simulator state at any step: the tokens alone 65 of 140 against 50 zero-shot and the binding form 66, both short of significance after correction (Holm p 0.088). The front bowl onto the plate rises from 5 to 18 and a book into the caddy from 4 to 10; the back bowl's token fails (0) where its form reached 12 |

## The three operators

| Cobweb | here | cost |
|---|---|---|
| categorise / best child | DECOMPOSE a query into the K prototypes (per noise level) that explain it | 0 weights; a code of node ids and weights |
| incorporate | ADDRESS: if the composition explains the query in the read band, store the code, update counts | 0 weights |
| create | FOLD a genuinely new prototype: new nodes at the levels where the query is unexplained, realised by a responsibility-gated minimal expert trained with anchoring on the model's own samples of the sibling nodes | the expert (the frontier experiment decides its size) |
| merge / promote | CONSOLIDATE: distil an accrued node into the base weights under the same gate, delete the expert | net zero or negative memory |

## What the frozen policy holds (pi0.5 on LIBERO; its training data only)

Every number quoted anywhere in this repository comes from a row in `ledger/` and is re-derived by
`experiments/audit_numbers.py` (`tests/test_audit.py` fails otherwise). Rungs are added as their rows land.

| rung | status |
|---|---|
| M0: LIBERO headless on the cluster (`ledger/r0_env_smoke`) | PASS: the ten LIBERO-Goal tasks render under EGL on an A40, 16.9 ms per simulation step |
| M1: pi0.5 (pi05_libero, openpi's PyTorch port) through `lifelong_ood/policy` (`ledger/r1_policy_smoke`, `ledger/r1_rollout`) | PASS: our sampler equals openpi's on the same noise (max abs diff 0.0); LIBERO-Goal success 0.95 with openpi's own policy and 0.98 with our sampler, ten episodes per task |
| M2: prototypes of the conditional p(a given o) (`ledger/r2_prototypes`) | KILL: the conditional is nearly deterministic. Ascending from each demo chunk under its own observation lands 0.043 from that chunk while chunks sit 0.788 apart; 942 of 960 chunks keep a mode of their own and no prototype spans two tasks |
| M2: prototypes of the action marginal over the demo observations (`ledger/r2b_marginal`) | a hierarchy with cross-task sharing: at t = 0.4, 130 prototypes used by 1.97 tasks each on average, 0.78 of frames in shared prototypes; at t = 0.7, two prototypes that split by gripper state. The policy's conditional is narrow (std 0.024 against 0.741 between frames), and the raw demonstrations give the same structure: the policy reproduces the data's hierarchy |
| the 40-task library (`ledger/r2b_marginal`, bench libero_40: every task pi05_libero was fine-tuned on) | at t = 0.4, 242 prototypes used by 2.94 tasks each, 0.94 of frames in shared prototypes; gripper events fall on a prototype boundary 0.99 of the time against 0.51 for random segmentations with as many boundaries |
| the prototype atlas (`ledger/r2c_atlas`, docs/PROTOTYPES.md) | the tree's first split is whether the hand holds something: at every level 0.99 or more of a prototype's frames share its gripper state; the finest level is a vocabulary of 111 motion words, 0.90 of a prototype's frames moving the same way |

## The image track (paused)

| rung | status |
|---|---|
| propositions 1-7 (numerical checks) | `tests/test_cobweb_correspondence.py` |
| the reference the fold must reproduce: C1's label-gated rank-1 LoRA on ColorMNIST (backfilled at af8f07a) | taught joint 0.716, never-taught joint 0.715, old-class drop 0.000, at 275,576 bytes; on Shapes3D 0.972 / 0.972 |
| the reference for storage: C5's router (backfilled) | tree store new-value joint 0.771, never-taught 0.795 at 277,601 bytes; flat store 0.646, 0.550 at 1,844,635 bytes |
| the reference for addressing: C3's flagship (backfilled; addressing what the backbone already renders) | 0.980 from one image at 72 bytes, 0.913 for a 1 KB inversion, 0.001 nearest trained class |
| sampler parity after the restructure | PASS: `sample(backbone)` equals the fixtures' `ddim_sample` bit for bit on the ColorMNIST UNet and the Shapes3D VE bridge, max abs diff 0.0 (`ledger/p0_sampler_parity`) |
| prototype tree, phase 0 (`ledger/p1_tree_build`, 104 nodes from 512 own samples at 6 levels) | KILL under the pre-registered criterion: no level reaches background purity 0.9 with digit purity 0.7 (best background 0.84 at u = 0.2, digit never above 0.25). The tree is a colour hierarchy: sharp digits only at the finest level, coloured blobs above. Retry pre-registered with k-means node selection and a coarser grid, judged per factor at its own level (background 0.9, foreground 0.8, digit 0.5) |
| fold frontier smoke (quick run, 2 tasks, 150-step cap; `ledger/p3_fold_frontier`, tag _smoke) | the label-gated reference arm reproduces the mechanism through the new stack (taught 0.778 / 0.736, never-taught 0.938 / 0.708, old classes exact); the analytic-gate arm failed (taught 0.083) because the new node lived at one level and the expert was off elsewhere; fixed by creating a branch down to the finest level, rerun pending |
| decompose (1), gate (2), fold frontier (3), compose-after-fold (4), consolidate (5) | pending |

## Reproduce

    PY=$HOME/flash/miniconda3/envs/chunks/bin/python
    $PY -m pytest tests/test_ledger.py tests/test_cobweb_correspondence.py -q     # numpy, login node
    for t in tests/test_keyed_lora.py tests/test_layout.py tests/test_field_contract.py \
             tests/test_sample_concept_path.py; do $PY $t; done                    # torch, seconds
    sbatch --wait --export=ALL,SCRIPT="-m pytest tests -q" slurm/jobs/cpu.sbatch  # off the login node
    sbatch --export=ALL,SCRIPT="experiments/legacy/c1_cobweb_update.py --bench colormnist --seed 0" slurm/jobs/c_gpu.sbatch
    $PY experiments/ledger_backfill.py --dry-run
    $PY experiments/audit_numbers.py && $PY experiments/audit_coverage.py && $PY experiments/make_numbers.py

## Layout

    lifelong_ood/backbones/   schedule, layout (one flat<->image pair), the Backbone handle, THE sampler, score ops
    lifelong_ood/neural/      UNet, DDPM training, VE bridge, pretrained shims, KeyedLoRA + anchoring (cobweb_update.py)
    lifelong_ood/data/        ColorMNIST, Shapes3D (rows="correct"), CelebA, the novel-value edits, multi-object scenes
    lifelong_ood/evals.py     probes, coverage, KID, vendi, memorisation;  instrument.py: ceilings and Wilson resolution
    lifelong_ood/ledger.py    the tracked results ledger
    lifelong_ood/tree/        the prototype tree: nodes at noise levels, build, responsibility, update
    lifelong_ood/experts/     the expert interface and the minimal experts (LoRA, band LoRA, FiLM, token, nonparametric)
    lifelong_ood/operators/   decompose / address, fold (gate, gated losses, stop rule), anchor; consolidate (pending)
    lifelong_ood/streams/     the benches and the arrival streams with the new primitives
    lifelong_ood/accounting.py  charge(): every byte a method keeps
    experiments/legacy/       c1 (fold reference arms), c5 (the label-gated router), c_report, diag_read_by_u
    experiments/p*.py         the probes: p0 sampler parity, p1 tree build, p3 fold frontier
    experiments/audit_*.py    the number audit and its coverage; make_numbers.py regenerates paper/numbers.json
    tests/                    the kept tests plus one per new module; everything else is in the archive
    ledger/                   one small JSON per run, tracked
    archive/2026-09-pre-rework/   the previous repository, with its claim map
