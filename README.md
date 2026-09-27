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

## The three operators

| Cobweb | here | cost |
|---|---|---|
| categorise / best child | DECOMPOSE a query into the K prototypes (per noise level) that explain it | 0 weights; a code of node ids and weights |
| incorporate | ADDRESS: if the composition explains the query in the read band, store the code, update counts | 0 weights |
| create | FOLD a genuinely new prototype: new nodes at the levels where the query is unexplained, realised by a responsibility-gated minimal expert trained with anchoring on the model's own samples of the sibling nodes | the expert (the frontier experiment decides its size) |
| merge / promote | CONSOLIDATE: distil an accrued node into the base weights under the same gate, delete the expert | net zero or negative memory |

## The embodied ladder (pi0.5 on LIBERO; the primary line)

Every number quoted anywhere in this repository comes from a row in `ledger/` and is re-derived by
`experiments/audit_numbers.py` (`tests/test_audit.py` fails otherwise). Rungs are added as their rows land.

| rung | status |
|---|---|
| M0: LIBERO headless on the cluster (`ledger/r0_env_smoke`) | PASS: the ten LIBERO-Goal tasks render under EGL on an A40, 16.9 ms per simulation step |
| M1: pi0.5 (pi05_libero, openpi's PyTorch port) through `lifelong_ood/policy` (`ledger/r1_policy_smoke`, `ledger/r1_rollout`) | PASS: our sampler equals openpi's on the same noise (max abs diff 0.0); LIBERO-Goal success 0.95 with openpi's own policy and 0.98 with our sampler, ten episodes per task |
| M2: prototypes of the conditional p(a given o) (`ledger/r2_prototypes`) | KILL: the conditional is nearly deterministic. Ascending from each demo chunk under its own observation lands 0.043 from that chunk while chunks sit 0.788 apart; 942 of 960 chunks keep a mode of their own and no prototype spans two tasks |
| M2: prototypes of the action marginal over the demo observations (`ledger/r2b_marginal`) | a hierarchy with cross-task sharing: at t = 0.4, 130 prototypes used by 1.97 tasks each on average, 0.78 of frames in shared prototypes; at t = 0.7, two prototypes that split by gripper state. The policy's conditional is narrow (std 0.024 against 0.741 between frames), and the raw demonstrations give the same structure: the policy reproduces the data's hierarchy |
| ADDRESS on held-out LIBERO-90 tasks against the Goal library (`ledger/r3_prepare`) | fraction of demo frames the library does not explain: turn off the stove 0.38, open the microwave 0.28, put the white mug on the left plate 0.06 |
| library growth without rollouts (`ledger/r5_growth`) | after one frying-pan task is folded in, another frying-pan task drops from 0.31 to 0.07 unexplained, while open the microwave stays at 0.17 |
| the 40-task library (`ledger/r2b_marginal`, bench libero_40: every task pi05_libero was fine-tuned on) | at t = 0.4, 242 prototypes used by 2.94 tasks each, 0.94 of frames in shared prototypes; gripper events fall on a prototype boundary 0.99 of the time against 0.51 for random segmentations with as many boundaries |
| M3 pilot: can a guide steer pi0.5? (`ledger/r3_compose`; turn off the stove, ONE demo) | only a constant one: Zekun's variance-aware weight, strong at high noise and fading, succeeds 0.0 of the time because the policy's tight conditional overwrites it in the last steps; weight 1 with a tight teacher, aligned to the demo by the robot state, succeeds 0.4 of the time; a stall fallback that hands control back to the prior for two calls lifts it further (`ledger/m3_ablation`) |
| M3 compose on held-out LIBERO-90 tasks (`ledger/m3_heldout`, `ledger/m3_summary`; ONE demo per task, init states the demo and the practice never saw, twenty episodes per arm) | on the four tasks pi0.5 fails zero-shot (1 of 80 episodes), following the demo's program of LIBRARY codes succeeds 0.45 of the time (36 of 80) with no trained weights; a rank-16 LoRA fine-tuned on the same demo succeeds 0.40 of the time (32 of 80) and stores 18.7 MB. RETRACTED storage claim: the executed code teacher mixes the prototypes with per-dimension weights fitted to the demo chunk, so a program needs as much per frame as the raw chunk (earlier rows charged only the node ids and one scalar weight each, which rebuild a teacher far from the demo); the codes behave like replaying the demo and store no less. Turn off the stove goes from 0.0 to 0.9. On the two tasks the prior already solves, forcing a program halves them, so the rule is to practise first: keep the prior where a few practice episodes succeed, compose elsewhere; over all six tasks that succeeds 0.625 of the time (75 of 120) against 0.333 zero-shot (40 of 120) and 0.408 for always composing (49 of 120) |
| M3 FINAL, pre-registered (`ledger/m3_final`, `ledger/m3_prereg`, derived `ledger/m3_stats` tag final_23tasks; docs/VALIDITY.md; 23 held-out tasks: the six of development plus seventeen drawn at random from the unique-language tasks the prior fails zero-shot; ONE demo each; init states 30-49, twenty episodes per arm; every method gets the same practice step, keeping pi0.5 where it already succeeds; bytes recomputed from the artifacts) | over the 21 tasks where pi0.5 fails its practice (420 paired episodes per method): borrowing from the library plus a 258 KB fold where the demo grasps succeeds 253 of 420 times at 5.3 MB, the per-task rank-16 LoRA 252 of 420 at 392.9 MB. NON-INFERIOR as pre-registered (pooled paired difference +0.002, one-sided 95% lower bound -0.091 against the margin -0.10), at 74x less storage, WITH THIS DEMO ONLY (RETRACTED across demos: see the second-demonstration row). It beats replaying the demo, 171 of 420 at 106.2 KB (difference 0.195, sign test p below 0.001), and zero-shot, 6 of 420. The fold carries it: borrowing alone matches replay (176 of 420 at every fifth frame, 23.6 KB; difference 0.012) and trails the LoRA by 0.181; the fold adds 0.202. The pre-registered practice allocation (keep the program where its ten practice episodes succeed, else train the LoRA) is the most successful option, 270 of 420 at 97.4 MB, a quarter of the LoRA's storage (0.043 above the LoRA, p 0.11). Replayed demos store their action sequence (CORRECTED 2026-09-26: they were charged a full chunk per frame, about five times too much, and the earlier '22x less than replay' was wrong) |
| validity checks (`ledger/m3_stats`, `ledger/r12_forced`; docs/VALIDITY.md) | test-retest: the same arm on two disjoint init-state sets, Pearson r 0.958 over 22 cells; a second demonstration is in its own row below; forced execution of the most shared prototypes (M2 validation iv, run after M3) realises a median 1.05 of the commanded hand motion, cosine 0.998; the library's novelty read does not predict which tasks compose (exploratory) |
| what composition borrows (`ledger/r8_inspect`, `ledger/r9_library_ablation`; docs/PROTOTYPES.md) | the LEAVES of the hierarchy, not its shared prototypes: 0.89 of the borrowed pairs come from the finest level, from prototypes of one to three frames of a single training task; a library of the policy's own chunk means rebuilds the held-out demos closest (0.025 from the demo against 0.033 for the tree's prototypes and 0.080 for the shared level alone) and keeps their smoothness; borrowing from the shared prototypes alone succeeds 65 of 240 times, 0.133 below borrowing from the whole band. The pieces come from about thirty training tasks each and lean only slightly toward related ones (instruction-word overlap 0.122 against 0.106 for shuffled pairings). The shared prototypes are the verbs of manipulation: hold, grasp, release |
| a second demonstration (pre-registered; `ledger/m3_stats` second_demo, docs/VALIDITY.md) | every arm rebuilt from another demo of each task: the per-task LoRA scores 298 of 420 against 261 for borrowing + fold; over both demos pooled the LoRA leads by 0.043 (sign test p 0.037), so NON-INFERIORITY IS RETRACTED; borrowing + fold still beats replaying the demo by 0.110 pooled. Borrowing follows the demo's path, so a better demo helps it less than it helps a LoRA |
| baselines (pre-registered; `ledger/m3_stats`, docs/VALIDITY.md) | playing the demo's actions back open-loop succeeds 187 of 420 times, above every program without a fold; DemoDiffusion 36 (the confident prior pulls the lightly noised demo chunk back to its own motion); VINN 167; an equal-bytes adapter trained on the whole demo 184 (0.164 below borrowing + fold); LoRA rank 64 and full fine-tuning of the action expert 267 each |
| learning from practice (pre-registered; `ledger/m3_stats`, `ledger/m5_learn_stats`, reports/RL from practice for composition.md) | a LoRA on the demo + the composition's practice successes scores 274 of 420; two more rounds of practice bring it to 281 and 294 (0.100 above the one-demo LoRA); adding the FAILED rollouts as negatives at the composition's stall (negative-aware flow matching) reaches 302 of 420 on the same data as the second round (0.050 above success-only training, sign test p 0.035). The 258 KB learners fall to 198 and, with negatives, 175. On practice instances the composition failed, consolidation solves 52 of 164 against 36 for a fresh-noise retry (rerunning with the same noise replays the episode exactly) |
| one fold for many tasks (pre-registered; `ledger/m3_stats`, `ledger/r13_shared_fold`) | NEGATIVE: one 258 KB fold trained on the contact windows of all grasping tasks succeeds 146 of 420 times and a fold trained on the other half of the tasks 107 (both below borrowing with no fold, 168 of 420): folds are task-specific |
| learned soft borrowing (`ledger/r6_composer`, `ledger/m3_dev`) | a small model that learns per step and dimension how much to borrow from each of 32 library prototypes (4.3 KB) composes smooth, interpretable scripts (docs/PROTOTYPES.md) but loses precision: turn off the stove 0.3 against 1.0 for dimension-wise borrowing on the development states. Adding pi0.5's own chunk as an expert, the model learns to trust the prior where it is competent (weight 0.47 on the white mug, 0.10 on the stove) |
| the prototype atlas (`ledger/r2c_atlas`, docs/PROTOTYPES.md) | the tree's first split is whether the hand holds something: at every level 0.99 or more of a prototype's frames share its gripper state; the finest level is a vocabulary of 111 motion words, 0.90 of a prototype's frames moving the same way |
| M4 fold (`ledger/r4_fold`, `ledger/m4_fold`; development states, ten episodes) | NEGATIVE: a rank-4 expert trained on the unexplained frames and switched on there by the program adds nothing over the codes (open the microwave 0.6 with or without it, turn off the stove 0.5 against 0.6); switched on everywhere it succeeds 0.0 of the time on all three tasks |
| gates that detect a competent prior from ONE demo (`ledger/m3_compose`, tag code_agree20) | NEGATIVE: guiding only where the prior's own chunk departs from the program restores put the white mug on the left plate (0.8) but not put the moka pot on the stove (0.0; the prior solves it along another path) and erases the gains where the prior fails (turn off the stove 0.0) |
| M5 stream, M6 consolidate | next |

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
