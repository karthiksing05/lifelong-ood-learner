# Benchmarks and evaluation protocols for zero-shot compositional generalization, robustness, and continual learning of robot manipulation policies (VLAs / diffusion policies), as of 2026-10-01

Scope note: these notes cover benchmarks only, not methods. Each section uses Takeaway / Cited Findings / Inferences / Gaps. Every number comes from a page fetched in this session. Where a fetched page gave garbled or inconsistent numbers, the notes say so. Numbers are success rates in % unless stated otherwise.

---

## Q0. Catalog: what each benchmark measures, scale, simulator, protocol/metrics, checkpoints, integration cost

### Takeaway
The benchmarks fall into five groups:
- **LIBERO-derived OOD/robustness suites:** LIBERO-Plus, LIBERO-PRO, LIBERO-X, LIBERO-OOD, LIBERO-Para, VLA-Arena, RoboCerebra. They reuse LIBERO's MuJoCo/robosuite stack, so they are close to drop-in for a pi0.5-LIBERO checkpoint.
- **RLBench-derived generalization suites:** COLOSSEUM, GemBench, AGNOSTOS. These run on CoppeliaSim, which is CPU-bound, and require re-fine-tuning on their seen tasks.
- **Long-horizon chain benchmarks:** CALVIN, RoboCerebra, BEHAVIOR-1K.
- **Large task-diversity suites:** RoboCasa365, VLABench, RoboTwin 2.0, ManiSkill3-based suites (Colosseum V2, RL4VLA, SimplerEnv/INT-ACT).
- **Older compositional RL / imitation benchmarks:** VIMA-Bench, CompoSuite, KitchenShift.

Only LIBERO's original lifelong protocol and RoboCasa365 ship official continual-learning protocols.

### Cited Findings

**LIBERO (base benchmark and lifelong protocol)**
- 130 tasks in four suites. LIBERO-Spatial, -Object and -Goal have 10 tasks each. LIBERO-100 has 90 short-horizon pretraining tasks (LIBERO-90) and 10 long-horizon evaluation tasks (LIBERO-10/Long). Each task has 50 human-teleoperated demos — [LIBERO paper HTML](https://arxiv.org/html/2306.03310)
- Lifelong protocol: tasks arrive sequentially, 50 epochs per task, evaluation every 5 epochs on 20 test trajectories. Metrics are FWT (forward transfer), NBT (negative backward transfer) and AUC. Baselines are SeqL, ER, EWC, PackNet and MTL — [LIBERO paper HTML](https://arxiv.org/html/2306.03310)
- Original finding: "Sequential finetuning outperforms existing lifelong learning methods in forward transfer", and "naive supervised pretraining can hinder agents' performance". FWT with ResNet-T on LIBERO-Long: SeqL 0.54, ER 0.48, EWC 0.13, PackNet 0.22 — [LIBERO arXiv](https://arxiv.org/abs/2306.03310); [LIBERO HTML](https://arxiv.org/html/2306.03310)
- Simulator: MuJoCo + robosuite with a Franka Panda. RoboCerebra's LeRobot docs describe "LIBERO's simulator stack (MuJoCo + robosuite, Franka Panda 7-DOF)" — [LeRobot RoboCerebra docs](https://huggingface.co/docs/lerobot/en/robocerebra)
- Official pi0.5 LIBERO checkpoint (`pi05_libero`, 30k steps): Spatial 98.8, Object 98.2, Goal 98.0, LIBERO-10 92.4, average 96.85 — [openpi LIBERO README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
- openpi releases base checkpoints for pi0, pi0-FAST and pi0.5, plus pi0.5-LIBERO and pi0.5-DROID. Inference needs >8 GB of GPU memory, LoRA fine-tuning >22.5 GB and full fine-tuning >70 GB. The PyTorch port is validated on LIBERO — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)

**LIBERO-Plus (robustness; CVPR 2026)**
- 10,030 perturbed test tasks across 7 dimensions: camera viewpoint, robot initial state, language, lighting, background texture, sensor noise, object layout. Difficulty is stratified into levels L1–L5 — [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626); [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- Integration: "Simply replace the originally installed LIBERO repository with our repository without modifying your code." Assets come from HF, and `num_trials_per_task` is changed from 50 to 1. Training datasets are released in RLDS and LeRobot formats — [LIBERO-Plus GitHub](https://github.com/sylvestf/LIBERO-plus)

**LIBERO-PRO (memorization audit)**
- Perturbs objects (color, texture, scale), initial positions, instructions (semantic paraphrase plus task-level changes) and environments, across the four LIBERO suites — [LIBERO-PRO HTML](https://arxiv.org/html/2510.03827); [GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO)
- Protocol: 50 episodes per task. OpenVLA uses its four per-suite official checkpoints. "pi0 and pi0.5 are evaluated with a single official checkpoint" — [LIBERO-PRO PDF](https://arxiv.org/pdf/2510.03827)
- Integration: bddl and init files are downloaded from HF and moved into the `libero/libero/` subdirectories. Task variants are registered with suffixes `_object`, `_swap`, `_task`, `_env` — [LIBERO-PRO GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO)

**LIBERO-X (arXiv 2602.06556, Feb 2026)**
- MuJoCo. 100 scenes, 600 tasks, 2,520 VR-teleoperated demos. Five cumulative levels:
  - L1: local spatial perturbation
  - L2: extended spatial perturbation
  - L3: scene topology reconstruction (altered object relations)
  - L4: visual attributes and unseen objects
  - L5: semantic-equivalent instruction reformulation
- Protocol: 10 trials per task, time limit 1.1× the average human completion time — [LIBERO-X HTML](https://arxiv.org/html/2602.06556)
- Code and dataset: [github.com/zackhxn/LIBERO-X](https://github.com/zackhxn/LIBERO-X) — [project page](https://zackhxn.github.io/LIBERO-X/)
- This is a new training set, not the LIBERO-90 data, so a pi0.5-LIBERO checkpoint must be re-fine-tuned on it. Results are "behavior cloning via supervised fine-tuning" — [LIBERO-X HTML](https://arxiv.org/html/2602.06556)

**LIBERO-OOD (compositional; Quanyi Li, IJCAI)**
- Two suites, LIBERO-Spatial-OOD and LIBERO-Goal-OOD, with 20 novel compositional tasks — [AC-VLA HTML](https://arxiv.org/html/2607.15714); [Li, arXiv 2505.03500](https://arxiv.org/abs/2505.03500)
- Construction: "both the grasping and placement locations have individually appeared in the training data … However, the specific trajectory connecting these two locations has not been demonstrated." Example: cream cheese → top of cabinet — [Li HTML](https://arxiv.org/html/2505.03500)
- Code: [github.com/QuanyiLi/pi0-text-latent](https://github.com/QuanyiLi/pi0-text-latent) — [Li HTML](https://arxiv.org/html/2505.03500)
- Discrepancy: Li's abstract page describes "20 tasks extrapolated from standard LIBERO suites". AC-VLA describes 20 tasks per suite (Spatial-OOD and Goal-OOD). The exact count per suite is unresolved — [Li abs](https://arxiv.org/abs/2505.03500); [AC-VLA](https://arxiv.org/html/2607.15714)

**LIBERO-Para (paraphrase robustness, arXiv 2603.28301, Mar 2026)**
- 4,092 meaning-preserving paraphrases across 43 type combinations, built on two axes: action expression (10 types, lexical/structural/pragmatic) and object reference (3 types). Introduces the PRIDE metric — [LIBERO-Para HTML](https://arxiv.org/html/2603.28301v1)
- Code: [GitHub](https://github.com/cau-hai-lab/LIBERO-Para); data: [HF dataset](https://huggingface.co/datasets/HAI-Lab/LIBERO-Para)

**ReGuide compositional cells (LIBERO-based, arXiv 2609.38616, posted 2026-09-29)**
- 16 cells, each changing one training factor (Target Region, Target Object, Unseen Object, Task Scene), plus 10 unmodified retention tasks. 50 replayed initial states per cell, 1,300 paired trials in total. Base VLAs: OpenVLA-OFT, pi0, pi0.5, X-VLA, GR00T N1.7. No code link — [ReGuide HTML](https://arxiv.org/html/2609.38616)

**VLA-Arena (arXiv 2512.22539)**
- 170 tasks in 11 suites over 4 dimensions: Safety, Distractor, Extrapolation (preposition combinations, task workflows, unseen objects), Long-Horizon. Difficulty levels L0 (in-distribution), L1 (near) and L2 (far, "compositional reasoning"). Orthogonal language (W0–W4, WordNet substitution) and visual (V0–V4) perturbation axes. MuJoCo/RoboSuite with Constrained BDDL. Code is CC BY-NC-SA 4.0 — [VLA-Arena HTML](https://arxiv.org/html/2512.22539v4); [site](https://vla-arena.github.io)

**RoboCerebra (NeurIPS 2025)**
- Long-horizon "System 2" benchmark covering planning, reflection and memory, with a VLM-planner + VLA-controller framework. 6,660 episodes, 571,116 frames, 1,728 language-grounded sub-tasks. Runs on LIBERO's simulator, with evaluation on `libero_10` at 10 episodes per task — [arXiv](https://arxiv.org/abs/2506.06677); [LeRobot docs](https://huggingface.co/docs/lerobot/en/robocerebra)
- Released checkpoint: `lerobot/smolvla_robocerebra` — [LeRobot docs](https://huggingface.co/docs/lerobot/en/robocerebra)

**THE COLOSSEUM (RLBench)**
- 20 tasks and 14 perturbation factors: color, texture and size of objects/table/background, lighting, distractors, physical properties (friction, mass), and camera pose — [arXiv 2402.08191](https://arxiv.org/abs/2402.08191); [project](https://robot-colosseum.github.io/)
- Single perturbations degrade success by 30–50%. Combined perturbations degrade it by ≥75%. Sim–real correlation R² = 0.614 — [arXiv](https://arxiv.org/abs/2402.08191)
- Code: [GitHub](https://github.com/robot-colosseum/robot-colosseum); challenge data: [HF](https://huggingface.co/datasets/colosseum/colosseum-challenge) — [project](https://robot-colosseum.github.io/)

**Colosseum V2 (arXiv 2605.27759; accepted 2026-08-30)**
- ManiSkill GPU-parallel (5,000+ parallel envs). 28 tasks: 14 single-arm and 14 bimanual. 100 demos per task, 200 evaluation episodes per task-perturbation pair.
- Perturbations: visual (11), language (4: paraphrase, unrelated instruction, random words, no instruction), action (3: object sizes, pose randomization), plus "All" — [Colosseum V2 HTML](https://arxiv.org/html/2605.27759); [site](https://colosseum-v2.github.io/)

**GemBench (RLBench; ICRA 2025)**
- Training: 16 tasks / 31 variations, 7 action primitives (press, pick, push, screw, close, open, put), 100 demos per variation.
- Test levels:

| Level | Content | Tasks / variations |
|---|---|---|
| L1 | novel placements | 16 / 31 |
| L2 | novel rigid objects | 15 / 28 |
| L3 | novel articulated objects | 18 / 21 |
| L4 | long-horizon | 6 / 12 |

- Protocol: 20 episodes × 5 seeds per variation (9,200 episodes), at most 25 steps — [GemBench HTML](https://arxiv.org/html/2410.01345)
- Code: [github.com/vlc-robot/robot-3dlotus](https://github.com/vlc-robot/robot-3dlotus); data: [HF](https://huggingface.co/datasets/rjgpinel/GEMBench) — [project page](https://www.di.ens.fr/willow/research/gembench/)

**AGNOSTOS (RLBench; arXiv 2505.15660, v3 Oct 2025)**
- Train on 18 standard RLBench tasks with 200 demos each (3,600 total).
- Test zero-shot on 23 unseen tasks:
  - Level-1 (13 tasks): share object types or motion primitives with seen tasks.
  - Level-2 (10 tasks): "no overlap in either object categories or motion types".
- Protocol: 3 seeds × 25 rollouts. All foundation VLAs were fine-tuned on the 18 seen tasks — [AGNOSTOS HTML](https://arxiv.org/html/2505.15660v3); [project](https://jiaming-zhou.github.io/AGNOSTOS)

**CALVIN**
- 34 tasks, 4 environments (A–D). ABC→D trains on A, B and C and evaluates zero-shot in D on 1,000 chains of 5 consecutive instructions. The metric is average chain length (0–5) — [OpenRAL issue summarizing CALVIN](https://github.com/OpenRAL/openral/issues/52); [CALVIN paper](http://ais.informatik.uni-freiburg.de/publications/papers/mees22ral_a.pdf)

**SimplerEnv**
- Google Robot and WidowX (Bridge). "Visual Matching" overlays real images onto the simulation. "Variant Aggregation" averages over randomized backgrounds, lighting, distractors and textures. Built on ManiSkill2, with a GPU ManiSkill3 branch that is 10–15× faster. Real-to-sim fidelity is measured with MMRV and Pearson correlation — [SimplerEnv GitHub](https://github.com/simpler-env/SimplerEnv)
- INT-ACT extends SimplerEnv-Bridge from 4 to 50 tasks to probe object diversity, language complexity and vision-language "thinking". It adds an "Intention Correct Rate" metric — [INT-ACT project](https://ai4ce.github.io/INT-ACT/); [arXiv](https://arxiv.org/abs/2506.09930)

**ManiSkill3**
- GPU-parallel simulation and rendering, "10–1000x faster", 30,000+ FPS, 12 task domains — [arXiv 2410.00425](https://arxiv.org/abs/2410.00425)
- Used as the base for the RL4VLA OOD suite. That suite uses a WidowX-250S arm, 16 tables and 16 objects for training. OOD splits:
  - Vision: novel tables, dynamic textures, noise
  - Semantics: 9 unseen objects, 16 novel receptacles, paraphrases, multi-object
  - Execution: OOD positions, robot init, mid-episode moves
  
  — [RL4VLA HTML](https://arxiv.org/html/2505.19789v1)

**RoboCasa365 (ICLR 2026)**
- MuJoCo. 365 tasks (65 atomic + 300 composite), 2,500 pretraining kitchens (50 layouts × 50 styles), 10 target evaluation scenes. Around 2,500 objects in 153 categories plus 57 new categories.
- Demos: 30k human pretraining demos (404 h), 600k MimicGen demos (1,615 h), 25k human target demos (208 h) — [RoboCasa365 HTML](https://arxiv.org/html/2603.04356); [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/a05003fdb1e9562ab0c0a9719ea4de10-Paper-Conference.pdf)
- Target evaluation: 50 tasks — Atomic-Seen (18), Composite-Seen (16), Composite-Unseen (16, "novel for zero-shot evaluation") — [RoboCasa365 HTML](https://arxiv.org/html/2603.04356)
- Public leaderboard at robocasa.ai, updated 2026-10-01, 15 models — [RoboCasa leaderboard](https://robocasa.ai/leaderboard.html)
- Submission rules: the innovation must act at inference time, and code, checkpoint and paper links are required — [leaderboard GitHub](https://github.com/robocasa-benchmark/leaderboard)
- GR00T ships RoboCasa evaluation scripts — [Isaac-GR00T RoboCasa README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa/README.md)

**VLABench (ICCV 2025)**
- MuJoCo/dm_control. 100 task categories (60 primitive, 40 composite), 2,164 objects in 163 categories. Composite tasks average more than 500 steps, primitives about 120 — [VLABench HTML](https://arxiv.org/html/2412.18194)
- Six evaluation tracks:
  1. in-distribution
  2. cross-category
  3. common sense
  4. semantic instruction
  5. cross-task
  6. unseen texture
- The released fine-tuning set covers 10 primitive tasks with 500 samples each. There is openpi integration — [VLABench GitHub](https://github.com/OpenMOSS/VLABench)

**RoboTwin 2.0 (ICML 2026)**
- SAPIEN. 50 bimanual tasks, 5 embodiments, RoboTwin-OD with 731 objects in 147 categories. More than 100k trajectories (100 clean + 400 randomized per task-embodiment pair).
- Protocol: Easy (clean) vs Hard (clutter, lighting, texture, table height), 50 clean demos per task, 100 rollouts — [RoboTwin 2.0 HTML](https://arxiv.org/html/2506.18088); [search summary](https://huggingface.co/docs/lerobot/robotwin)
- LeRobot integration exists — [LeRobot RoboTwin docs](https://huggingface.co/docs/lerobot/robotwin)

**BEHAVIOR-1K / 2025 BEHAVIOR Challenge**
- OmniGibson on Isaac Sim. The challenge uses 50 long-horizon household tasks (average 6.6 min, up to 14 min) with 10,000 teleoperated demos (1,000+ h). Each task is evaluated over 10 randomized episodes. The q-score combines success with partial subtask credit — [challenge call](https://behavior.stanford.edu/challenge/call_for_participation.html); [RLinf BEHAVIOR guide](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/behavior.html)

**RoboArena (real-world)**
- Distributed double-blind pairwise evaluations on DROID Franka setups at 7 institutions. More than 600 pairwise episodes and 4,284 rollouts. Ranking uses a task-aware Bradley–Terry model.
- The paper evaluated 7 DROID policies (pi0-flow/FAST-DROID and PaliGemma variants). New policies need a hosted policy server and safety validation — [RoboArena HTML](https://arxiv.org/html/2506.18123); [site](https://robo-arena.github.io)

**Open X-Embodiment evaluation**
- 22 robots, 21 institutions, 527 skills (160,266 tasks). RT-1-X and RT-2-X were evaluated in the real world on in-distribution vs "emergent" skills: 50% improvement and 3× more emergent skills — [OXE arXiv](https://arxiv.org/abs/2310.08864)

**VIMA-Bench**
- Ravens simulator. 17 task types; 13 used for training with 50K trajectories each, 4 held out for L4. Levels:
  - L1: placement
  - L2: combinatorial — "recombines seen adjectives and nouns"
  - L3: novel object
  - L4: novel task
  
  — [MIDAS HTML](https://arxiv.org/html/2310.09676v2); [VIMA](https://vimalabs.github.io/)

**CompoSuite**
- 256 tasks = 4 robot arms × 4 objects × 4 obstacles × 4 objectives (pick-place, push, shelf, trash). Designed for compositional multi-task/lifelong RL — [CompoSuite GitHub](https://github.com/Lifelong-ML/CompoSuite); [PMLR paper](https://proceedings.mlr.press/v199/mendez22a/mendez22a.pdf)
- Offline datasets: expert, medium, warmstart and replay, 256M transitions each — [Hussing et al. 2023](https://arxiv.org/abs/2307.07091)

**KitchenShift**
- NeurIPS 2021 DistShift workshop (Xing, Gupta, Powers, Dean). Domain shifts on a Franka Kitchen-derived environment (lighting, camera view, object orientation and others), applied to multi-stage imitation from RGB — [NeurIPS virtual page](https://neurips.cc/virtual/2021/35477); [mlanthology](https://mlanthology.org/neuripsw/2021/xing2021neuripsw-kitchenshift/)

**MultiNet v1.0 (Dec 2025)**
- Generality across 6 regimes, including continuous robot control. Tested GPT-5, pi0 and Magma: "no model demonstrates consistent generality" — [arXiv 2512.11315](https://arxiv.org/abs/2512.11315)

**Checkpoints of flow/diffusion VLAs that ship benchmark evaluations**
- X-VLA-0.9B (flow matching, soft prompts) releases per-benchmark checkpoints:

| Checkpoint | Score |
|---|---|
| X-VLA-Libero | 98.1 |
| X-VLA-Calvin-ABC_D | 4.43 |
| X-VLA-Google-Robot | 83.5 VM / 76.4 VA |
| X-VLA-WidowX | 95.8 |
| X-VLA-VLABench | 51.1 score |
| X-VLA-RoboTwin2 | 70 |

  These are native to LeRobot. LoRA-adapter LIBERO results are lower (Long 83.2) — [X-VLA GitHub](https://github.com/2toINF/X-VLA)
- FLOWER: 950M parameters, flow matching, pretrained in 200 H100 hours. CALVIN ABC 4.53. Weights are released — [FLOWER arXiv](https://arxiv.org/abs/2509.04996)
- SmolVLA: 450M parameters. LIBERO 90/96/92/71 (average 87.3), Meta-World average 57.3 — [SmolVLA HTML](https://arxiv.org/html/2506.01844v1)
- GR00T N1.6: fine-tuned checkpoints for Bridge/WidowX and Fractal/Google, and scripts for LIBERO, SimplerEnv and RoboCasa — [GR00T N1.6 RoboCasa leaderboard entry](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/gr00t_n1.6_2026_05_14.md); [Isaac-GR00T RoboCasa README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa/README.md)

### Inferences
- **Integration cost tiers for a pi0.5-LIBERO stack:**
  - Lowest: LIBERO-Plus (replace the repo), LIBERO-PRO (bddl/init files), LIBERO-OOD (code public), LIBERO-Para (HF data), and RoboCerebra (libero_10, same simulator). All reuse the `pi05_libero` checkpoint without retraining.
  - Moderate: LIBERO-X and VLA-Arena. Same MuJoCo/robosuite family but new training sets, so pi0.5 must be fine-tuned.
  - High: RoboCasa365. Same simulator family, but it needs fine-tuning on 30k demos with a mobile base.
  - High: VLABench. dm_control, openpi integration exists.
  - Higher: RLBench suites (COLOSSEUM, GemBench, AGNOSTOS). Different simulator (CoppeliaSim), keyframe-style 3D baselines, re-fine-tuning on 3.1k–3.6k demos.
  - Higher: RoboTwin 2.0 (bimanual, SAPIEN).
  - Highest: BEHAVIOR (Isaac Sim, mobile, 50 long tasks) and RoboArena (real DROID hardware; pi0.5-DROID exists).
- CALVIN ABC→D is an environment-shift benchmark: an unseen table and textures with the same 34 skills. It is not a test of new object–relation combinations.

### Gaps
- vlaleaderboard.com returned only a title, so no cross-benchmark leaderboard snapshot could be verified.
- The KitchenShift paper (OpenReview) was behind a login. The exact list and count of its shift axes was not verified.
- The X-VLA arXiv HTML extraction returned scrambled numbers. The GitHub README values are used instead: Google VM/VA 83.5/76.4. The paper may report 80.4/75.7; that figure appeared in the garbled extraction and is unverified.
- No public code link was found for ReGuide's 16-cell LIBERO compositional benchmark.

---

## Q1. Which benchmarks explicitly test new combinations of seen objects/relations/verbs (compositional splits), and which test language following?

### Takeaway
Explicit compositional splits (seen parts, unseen combination) exist in the following benchmarks:
- **LIBERO-OOD** (Spatial-OOD / Goal-OOD: a seen pick plus a seen place location, never demonstrated together)
- **ReGuide's 16 LIBERO cells** (one factor changed at a time)
- **VIMA-Bench L2** (seen adjectives × nouns recombined)
- **CompoSuite** (4×4×4×4 factorial)
- **AGNOSTOS Level-1** (shares objects or primitives)
- **VLA-Arena Extrapolation** (preposition combinations, task workflows)
- **LIBERO-X L3** (altered object relations)
- **RoboCasa365 Composite-Unseen** (new composite tasks built from known atomic skills)
- **GemBench L4** (new long-horizon compositions of the 7 primitives)
- **LIBERO-PRO task perturbation** ("constructed from training components")

Language following is tested directly by LIBERO-Para, LIBERO-Plus (language dimension, blank-instruction and goal-replacement probes), LIBERO-PRO (instruction corruption), LIBERO-X L5, Colosseum V2 language perturbations, VLA-Arena W0–W4, VLABench Track 4, INT-ACT and RL4VLA semantics.

### Cited Findings
- LIBERO-OOD tasks combine "the object from one original task with the target location from another". Both locations appear in training, but the connecting trajectory does not — [AC-VLA](https://arxiv.org/html/2607.15714); [Li HTML](https://arxiv.org/html/2505.03500)
- The Spatial-OOD suite includes new pick targets with the familiar plate, and the familiar black bowl placed via unseen spatial relations (on cookie box, next to plate, table center, onto stove or cabinet) — [search summary of AC-VLA / Unleashing More Actions](https://arxiv.org/html/2607.00351)
- ReGuide cells vary exactly one factor (Target Region, Target Object, Unseen Object, Task Scene) relative to training — [ReGuide HTML](https://arxiv.org/html/2609.38616)
- VIMA-Bench L2 "recombines seen adjectives and nouns". L3 introduces unseen adjectives or nouns. L4 introduces new meta-tasks — [search summary citing VIMA](https://arxiv.org/pdf/2210.03094); [MIDAS HTML](https://arxiv.org/html/2310.09676v2)
- CompoSuite builds each of its 256 tasks from one element per axis (arm, object, obstacle, objective). It is designed to test whether methods "discover the decomposition … into simpler subproblems" — [CompoSuite GitHub](https://github.com/Lifelong-ML/CompoSuite)
- Offline RL on CompoSuite is "unable to extract the compositional structure to generalize to unseen tasks" — [Hussing et al.](https://arxiv.org/abs/2307.07091)
- AGNOSTOS defines its levels as follows. Level-1 unseen tasks share object types (e.g., cups) or motion primitives (e.g., stacking) with seen tasks. Level-2 tasks share neither, "requiring broader compositional reasoning" — [AGNOSTOS HTML](https://arxiv.org/html/2505.15660v3)
- VLA-Arena's Extrapolation dimension covers "preposition combinations, task workflows, unseen objects". L2 is "far-distribution … requiring compositional reasoning" — [VLA-Arena HTML](https://arxiv.org/html/2512.22539v4)
- LIBERO-X L3 is "scene topology reconstruction: altered object relationships", and L5 is semantic-equivalent instruction reformulation. Labels include spatial relation types (surface, container, left/right, ordinal, distance) and attributes — [LIBERO-X HTML](https://arxiv.org/html/2602.06556)
- RoboCasa365 Composite-Unseen consists of 16 composite tasks "novel for zero-shot evaluation" — [RoboCasa365 HTML](https://arxiv.org/html/2603.04356)
- GemBench L4 is 6 novel long-horizon tasks built from its 7 primitives — [GemBench HTML](https://arxiv.org/html/2410.01345)
- LIBERO-PRO models "nearly collapse under changes to object positions or minor task modifications, even when constructed from training components". The paper lists "compositional failure: successfully learned subtasks cannot be combined into multi-step behaviors" — [LIBERO-PRO PDF](https://arxiv.org/pdf/2510.03827); [LIBERO-PRO HTML](https://arxiv.org/html/2510.03827)
- **Language-following probes:**
  - LIBERO-Plus: with a blank instruction, performance "remained largely unchanged" (models act as "Vision-Action"). When the goal object was replaced, success dropped "nearly to zero" — [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626)
  - LIBERO-PRO: "The model still executes nearly the same action trajectory" under corrupted instructions — [LIBERO-PRO HTML](https://arxiv.org/html/2510.03827)
  - MINERVA: a task-ID permutation probe shows that "standard LIBERO instruction conditioning primarily selects among memorized tasks" — [MINERVA arXiv](https://arxiv.org/abs/2609.03715)
  - LIBERO-Para: object-reference paraphrases cause 19.8–51.0 pp drops, and 80–96% of failures are trajectory divergence, i.e. wrong task identification — [LIBERO-Para HTML](https://arxiv.org/html/2603.28301v1)
  - INT-ACT: intention is correct 80–100% of the time, but grasp and task success drop sharply. Fine-tuning erodes paraphrase, negation and appearance-description handling — [INT-ACT summary](https://pith.science/paper/2506.09930); [project](https://ai4ce.github.io/INT-ACT/)
  - Colosseum V2 language perturbations: paraphrase, unrelated instruction, random words, instruction removal — [Colosseum V2 HTML](https://arxiv.org/html/2605.27759)
  - VLABench Track 4: semantic instruction understanding — [VLABench GitHub](https://github.com/OpenMOSS/VLABench)
- Benchmarks that are robustness-only, with no compositional split:
  - COLOSSEUM: visual and physical perturbations only — [arXiv](https://arxiv.org/abs/2402.08191)
  - RoboTwin 2.0: Easy/Hard domain randomization, though it does run an unseen-task transfer experiment — [RoboTwin 2.0 HTML](https://arxiv.org/html/2506.18088)
  - SimplerEnv: visual matching and variant aggregation — [SimplerEnv GitHub](https://github.com/simpler-env/SimplerEnv)
  - KitchenShift — [NeurIPS page](https://neurips.cc/virtual/2021/35477)
  - CALVIN ABC→D: environment shift — [OpenRAL CALVIN summary](https://github.com/OpenRAL/openral/issues/52)

### Inferences
- For "new combinations of seen object × relation × verb", the cleanest splits on a pi0.5-LIBERO stack are LIBERO-OOD and ReGuide's cells. Both hold the skills fixed and recombine object with target, which isolates binding from motor learning.
- Language-following results on LIBERO are confounded: MINERVA, LIBERO-Plus and LIBERO-PRO all show instructions acting mainly as task selectors. A language-following claim therefore needs LIBERO-Para-style paraphrases or goal-replacement controls, not standard LIBERO success.

### Gaps
- I found no benchmark that crosses verb × object × relation factorially for VLAs at LIBERO scale. VIMA-Bench and CompoSuite do this in older, non-VLA settings.
- The exact per-task composition list of LIBERO-OOD (which object/target pairs) was not retrieved beyond the Spatial-OOD summary.

---

## Q2. Reported numbers for frozen or fine-tuned VLAs on held-out splits, and the gap to in-distribution

### Takeaway
Fine-tuned VLAs that score 93–99% on LIBERO in-distribution fall to roughly 0–50% on compositional LIBERO splits:
- LIBERO-OOD: pi0.5 35.5 / 46.6; OpenVLA-OFT 0 / 1; pi0 about 9 overall.
- ReGuide cells: pi0.5 94.0 → 39.0.

They also fall to roughly 18–65% on LIBERO-X (pi0.5 L1 65.2 → L5 18.0), 1–4% on RoboCasa365 Composite-Unseen, and 12–22% on AGNOSTOS.

Methods that keep the policy frozen close much of the compositional gap:
- ReGuide with frozen pi0.5: 94.5
- Text-latent interpolation with pi0: 83
- AC-VLA with retrained pi0.5: 64.2 / 73.3

pi0.5 is consistently the most robust open VLA in these tables, but it is not robust in absolute terms.

### Cited Findings

**LIBERO-OOD**
- Li's table, overall success (goal-ood / spatial-ood):

| Model | Overall | goal-ood | spatial-ood |
|---|---|---|---|
| UniVLA | 0.21 | 0.32 | 0.11 |
| OpenVLA-OFT | 0.01 | 0.01 | 0 |
| pi0-FAST | 0.15 | — | — |
| pi0 | 0.09 | 0.02 | 0.16 |
| pi0 + text-latent interpolation (no retraining) | 0.83 | 0.85 | 0.81 |

  Most models score about 95% on standard LIBERO — [Li HTML](https://arxiv.org/html/2505.03500)
- AC-VLA's table (ID Spatial/Goal/Object/Long → Spatial-OOD/Goal-OOD):

| Model | Spatial | Goal | Object | Long | Spatial-OOD | Goal-OOD |
|---|---|---|---|---|---|---|
| pi0.5 | 98.8 | 98.5 | 99.3 | 92.9 | 35.5 | 46.6 |
| OpenVLA-OFT | 97.6 | 97.9 | 98.4 | 94.5 | 0.0 | 1.0 |
| UniVLA | 96.5 | 95.6 | 96.8 | 92.0 | 11.0 | 32.0 |
| AC-VLA (pi0.5) | — | — | — | — | 64.2 | 73.3 |
| AC-VLA (GR00T-N1) | — | — | — | — | 36.4 | 44.0 |

  — [AC-VLA HTML](https://arxiv.org/html/2607.15714)
  - The OFT and UniVLA OOD numbers match Li's table. The pi0.5 OOD baseline appears only in AC-VLA.

**ReGuide 16-cell LIBERO compositional benchmark**
- pi0.5: in-distribution 94.0, composition average 39.0.
- ReGuide (frozen pi0.5, "without retraining"): 94.5. ReGuide on OpenVLA-OFT: 70.8.
- Real xArm6 (15 cells): bare pi0.5 15.0 vs ReGuide 90.0 — [ReGuide HTML](https://arxiv.org/html/2609.38616)

**LIBERO-X (levels L1→L5, fine-tuned on the LIBERO-X set)**

| Model | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| OpenVLA-OFT | 29.0 | 17.6 | 8.8 | 6.4 | 4.2 |
| pi0 | 29.4 | 21.9 | 11.0 | 7.6 | 5.1 |
| X-VLA | 30.1 | 22.6 | 10.3 | 6.0 | 4.1 |
| GR00T-N1.5 | 43.3 | 32.9 | 18.7 | 13.3 | 9.7 |
| pi0.5 | 65.2 | 53.2 | 36.0 | 24.1 | 18.0 |

— [LIBERO-X HTML](https://arxiv.org/html/2602.06556)

**LIBERO-Plus (README leaderboard; dimensions are Camera / Robot / Language / Light / Background / Noise / Layout)**

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| pi0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| pi0-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| UniVLA | — | — | — | — | — | — | — | 43.9 |
| WorldVLA | — | — | — | — | — | — | — | 25.0 |
| NORA | — | — | — | — | — | — | — | 39.0 |
| RIPT-VLA | — | — | — | — | — | — | — | 68.4 |
| OpenVLA-OFT+ (perturbation-augmented training) | — | — | — | — | — | — | — | 79.6 |

- No pi0.5, SmolVLA, GR00T or X-VLA rows — [LIBERO-Plus GitHub](https://github.com/sylvestf/LIBERO-plus)
- The arXiv HTML version differs slightly (e.g., OpenVLA-OFT camera 59.7 vs 56.4; enhanced model total 79.5) — [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626)
- Headline: "performance dropping from 95% to below 30% under modest perturbations" of camera and robot init — [Emergent Mind summary](https://www.emergentmind.com/topics/libero-plus)
- MINERVA's 0.54M-parameter policy scores 46–56% on LIBERO-Plus, with "near-zero robustness to photometric shifts" — [MINERVA](https://arxiv.org/abs/2609.03715)

**LIBERO-PRO**
- All models exceed 90% on original LIBERO and collapse under position and task perturbations. "Pi0.5 achieves a 0.38 success rate in the libero-goal task under position changes, compared to 0 for OpenVLA and Pi0" — [LIBERO-PRO PDF](https://arxiv.org/pdf/2510.03827)
- The README reports a "Total" of OpenVLA 0.52, pi0 0.44, pi0.5 0.53, MolmoAct 0.41, NORA 0.40, X-VLA 0.46. The per-suite columns appear to be original-LIBERO scores. The README table's exact semantics could not be verified because the paper's main table is an image — [LIBERO-PRO GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO)

**LIBERO-Para**
- Paraphrasing drops success by 22.8–51.9 pp across 7 configurations (OpenVLA-OFT, pi0.5, pi0.5 expert-only, Xiaomi-Robotics-0, X-VLA, VLA-Adapter, OFT-mixed). Examples: Xiaomi-Robotics-0 98.8 → 76.0, VLA-Adapter 98.2 → 46.3 — [LIBERO-Para HTML](https://arxiv.org/html/2603.28301v1)

**VLA-Arena**
- L0 60–97% → L2 0–33% across pi0, pi0-FAST, OpenVLA-OFT, UniVLA, SmolVLA and OpenVLA. A 50.5% drop under visual noise — [VLA-Arena HTML](https://arxiv.org/html/2512.22539v4)

**RoboCasa365 (multi-task, 30k human demos; Atomic-Seen / Composite-Seen / Composite-Unseen / Avg)**

| Model (paper) | Atomic-Seen | Composite-Seen | Composite-Unseen | Avg |
|---|---|---|---|---|
| Diffusion Policy | 15.7 | 0.2 | 1.25 | 6.1 |
| pi0 | 36.3 | 5.2 | 0.7 | 15.0 |
| pi0.5 | 39.6 | 7.1 | 1.2 | 16.9 |
| GR00T N1.5 | 43.0 | 9.6 | 4.4 | 20.0 |

— [RoboCasa365 HTML](https://arxiv.org/html/2603.04356)

| Leaderboard entry (2026-10-01) | Atomic-Seen | Composite-Seen | Composite-Unseen | Avg |
|---|---|---|---|---|
| Paimon-0 (top) | 79.3 | 55.8 | 36.7 | 58.1 |
| Xiaomi-Robotics-1 | 80.2 | 57.1 | 32.1 | 57.4 |
| GR00T N1.6 | 51.1 | 9.4 | 1.7 | 21.9 |
| GR00T N1.5 | 50.7 | 14.8 | 2.7 | 23.9 |
| pi0 | 34.6 | 6.1 | 1.1 | 14.8 |
| pi0.5 | 39.6 | 7.1 | 1.2 | 16.9 |

— [RoboCasa leaderboard](https://robocasa.ai/leaderboard.html)

- Discrepancy: the leaderboard's pi0 and GR00T N1.5 rows differ from the paper's table.
- With GR00T N1.5, pretraining on RoboCasa365 data raises Composite-Unseen with 10% target data to 23.8–32.3 (Human50 / Human300), vs 11.2 with no pretraining — [RoboCasa365 HTML](https://arxiv.org/html/2603.04356)

**AGNOSTOS (cross-task zero-shot; L1 / L2 / overall)**

| Model | Level-1 | Level-2 | Overall |
|---|---|---|---|
| pi0 | 21.7 | 12.0 | 17.5 |
| OpenVLA | 16.9 | 9.2 | 13.6 |
| RDT | 10.4 | 14.4 | 12.1 |
| VoxPoser | 18.1 | 12.1 | 15.6 |
| SAM2Act | 13.1 | 15.9 | 14.0 |
| LLARVA | 4.5 | 2.4 | 3.6 |
| X-ICM 7B (in-context LLM) | 28.6 | 16.9 | 23.5 |
| X-ICM 72B (in-context LLM) | 37.6 | 20.3 | 30.1 |

— [AGNOSTOS HTML](https://arxiv.org/html/2505.15660v3)

**GemBench (L1 / L2 / L3 / L4)**

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| 3D-LOTUS | 94.3 | 49.9 | 38.1 | 0.3 |
| 3D-LOTUS++ | 68.7 | 64.5 | 41.5 | 17.4 |
| 3D Diffuser Actor | 91.9 | 43.4 | 37.0 | 0.0 |
| RVT-2 | 89.1 | 51.0 | 36.0 | 0.0 |

— [GemBench HTML](https://arxiv.org/html/2410.01345)

- BridgeVLA: 91.1 / 65.0 / 43.8 / 0.0 (average 50.0). No 2D VLA (pi0/OpenVLA) is evaluated on GemBench in that paper — [BridgeVLA HTML](https://arxiv.org/html/2506.07961)

**COLOSSEUM**
- BridgeVLA average 64.0 vs RVT-2 56.7 across 14 perturbations — [BridgeVLA HTML](https://arxiv.org/html/2506.07961)
- Colosseum V2 (no-perturbation success; then drops):

| Model | Single-arm | Bimanual | Visual-perturbation drop | Language-perturbation change |
|---|---|---|---|---|
| pi0.5 | 21.2 | 12.6 | 7.73% (robust) | −19.16% |
| ACT | 29.2 | 44.5 | 28.57% | −0.28% |

  — [Colosseum V2 HTML](https://arxiv.org/html/2605.27759)

**VIMA-Bench (L1 / L2 / L3 / L4)**

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| VIMA | 87.2 | 87.0 | 84.0 | 49.6 |
| Gato | 57.0 | 53.9 | 45.6 | 13.5 |
| MIDAS (w/ pretraining) | 97.8 | 97.9 | 93.4 | 59.1 |

- The L1→L2 combinatorial gap is near zero. The real gap is at L4 (novel task) — [MIDAS HTML](https://arxiv.org/html/2310.09676v2)

**RoboTwin 2.0 (Easy → Hard)**

| Policy | Easy | Hard |
|---|---|---|
| pi0 | 46.4 | 16.3 |
| RDT | 34.5 | 13.7 |
| DP3 | 55.2 | 5.0 |
| ACT | 29.7 | 1.7 |
| DP | 28.0 | 0.6 |

- Unseen-task transfer (32 training tasks → 5 unseen) with randomized pretraining data: pi0 improves 29.3% relative — [RoboTwin 2.0 HTML](https://arxiv.org/html/2506.18088)
- X-VLA reports 70 on RoboTwin 2.0 (split unspecified in the README) — [X-VLA GitHub](https://github.com/2toINF/X-VLA)

**VLABench Track 1 (in-distribution)**
- pi0-ft 47%, pi0.5-ft 40.6%, pi0-FAST delta-chunk 51.2% — [VLABench GitHub](https://github.com/OpenMOSS/VLABench)
- X-VLA: 51.1 (score) — [X-VLA GitHub](https://github.com/2toINF/X-VLA)
- Numbers for the cross-category and unseen-texture tracks were not found for pi0/pi0.5.

**CALVIN ABC→D (average chain length out of 5)**

| Model | ABC→D | ABCD→D |
|---|---|---|
| Xiaomi-Robotics-0 | 4.75 | 4.80 |
| FLOWER | 4.53 | — |

  — [Xiaomi-Robotics-0 HTML](https://arxiv.org/html/2602.12684)
- X-VLA: 4.43 — [X-VLA GitHub](https://github.com/2toINF/X-VLA)
- FLOWER: 4.53 — [FLOWER arXiv](https://arxiv.org/abs/2509.04996)
- The Xiaomi table's pi0.5 row (ABC→D 4.41) is identical to its UniVLA row, which suggests an extraction error. Treat pi0.5 CALVIN numbers as unverified.

**SimplerEnv**
- Xiaomi-Robotics-0: Google VM 85.5, VA 74.7, WidowX 79.2. It lists pi0 at 71.4 VM and 69.2 WidowX — [Xiaomi-Robotics-0 HTML](https://arxiv.org/html/2602.12684)
- LatBot reports 78.0 Google VM, "+25.3 over pi0" (implying pi0 ≈ 52.7), and WidowX 87.5, "+32.3 over pi0.5" — [LatBot](https://arxiv.org/pdf/2511.23034)
- The two papers' pi0 baselines differ by about 19 pp.

**RL4VLA (ManiSkill3, OpenVLA)**
- In-distribution: SFT 78.1 vs PPO 93.8.
- OOD averages, SFT vs PPO: execution ~40 vs ~78, semantics ~55 vs ~75, vision ~68 vs ~70 — [RL4VLA HTML](https://arxiv.org/html/2505.19789v1)

**BEHAVIOR Challenge 2025**
- Winner: q-score 0.26 with task success 0.124 (private leaderboard). Second (Comet, an openpi-based solution): 0.2514 — [1st-place paper](https://arxiv.org/html/2512.06951v2); [Comet](https://arxiv.org/pdf/2512.10071); [winning write-up](https://robot-learning-collective.github.io/winning-behavior-1k-challenge)

**Standard LIBERO in-distribution (saturation reference)**
- pi0.5 96.85 (openpi) — [openpi](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
- pi0.5 96.9, OpenVLA-OFT 97.1, GR00T-N1 93.9, Xiaomi-Robotics-0 98.7 — [Xiaomi-Robotics-0](https://arxiv.org/html/2602.12684)
- X-VLA 98.1 — [X-VLA GitHub](https://github.com/2toINF/X-VLA)
- SmolVLA 87.3 — [SmolVLA](https://arxiv.org/html/2506.01844v1)
- A 0.54M-parameter MINERVA policy reaches 95.1 — [MINERVA](https://arxiv.org/abs/2609.03715)

### Inferences
- **Gap sizes for pi0.5 specifically:**

| Benchmark | In-distribution | Held-out | Gap |
|---|---|---|---|
| ReGuide cells | 94 | 39 | ≈ −55 pp |
| LIBERO-OOD | ≈ 96–99 | 35.5 / 46.6 | ≈ −50 to −60 pp |
| LIBERO-X | L1 65.2 | L5 18.0 | ≈ −47 pp |
| RoboCasa365 | Atomic-Seen 39.6 | Composite-Unseen 1.2 | ≈ −38 pp |

  These are the most directly comparable published pi0.5 baselines for a compositional claim.
- The published frozen-policy competitors on LIBERO compositional splits are ReGuide (94.5 frozen pi0.5) and text-latent interpolation (0.83 with pi0). A new frozen-pi0.5 method on LIBERO-OOD or the ReGuide cells would be compared against these numbers. Both papers are recent (TLI v-latest 2026-05; ReGuide 2026-09-29).
- Baselines are not reproducible across papers. Examples: pi0 on SimplerEnv (52.7 vs 71.4), RoboCasa365 paper vs leaderboard rows, LIBERO-Plus arXiv vs README. Credible claims should re-run the baseline in the same harness, as LIBERO-PRO does with official checkpoints.

### Gaps
- No published pi0.5 numbers were found on AGNOSTOS, GemBench, original COLOSSEUM, LIBERO-Plus (README) or VLABench beyond Track 1.
- No SmolVLA, GR00T or FLOWER numbers were found on any compositional LIBERO split (LIBERO-OOD, ReGuide cells).
- The LIBERO-PRO per-perturbation table is an image in the PDF and could not be extracted. Only the pi0.5 goal-position 0.38 figure is confirmed from text.

---

## Q3. Which benchmarks have lifelong / continual-learning protocols?

### Takeaway
- **Official continual protocols:** LIBERO (sequential suites; FWT/NBT/AUC) and RoboCasa365 (a 4-phase lifelong protocol with increasing horizon).
- **Designed for compositional lifelong RL, not VLAs:** CompoSuite.
- **Recent continual-VLA papers** reuse LIBERO suites: Liu et al. 2026 (pi0, GR00T) and CRL-VLA (OpenVLA-OFT + PPO).
- **Real-world:** a 10-task real-robot continual benchmark (2605.26820).

The key 2026 result: pretrained VLAs barely forget on LIBERO with even 2% replay. A continual-learning claim on LIBERO therefore needs to beat replay baselines, not naive fine-tuning.

### Cited Findings
- LIBERO lifelong protocol and metrics (FWT, NBT, AUC). Sequential fine-tuning beats ER, EWC and PackNet on FWT — [LIBERO HTML](https://arxiv.org/html/2306.03310)
- **Liu, Kim, Liu, Liu, Zhu (UT Austin), arXiv 2603.03818 (v2, Mar 2026):**
  - Setup: pi0 and GR00T N1.5 vs BC-Transformer, BC-Diffusion and BC-ViT on LIBERO-Spatial/Object/Goal/10, 10 sequential tasks each. Replay buffers of 10/100/1000 transitions per task (0.2/2/20%), 10k steps per task.
  - Average NBT: pi0 −0.016, GR00T 0.027, BC-Transformer 0.245.
  - VLAs "recover peak performance with fewer than 10% of the original training steps".
  
  — [HTML](https://arxiv.org/html/2603.03818); [project](https://ut-austin-rpl.github.io/continual-vla); [project page title](https://continual-vlas.github.io/forget-me-not/)
- **CRL-VLA (arXiv 2602.03445, Feb 2026):**
  - Setup: OpenVLA-OFT with PPO, LIBERO task subsets, no replay. Baselines: SL, LwF, ER, MTL. Metrics: FAR, BWT, Forgetting, FT.
  - Results: CRL-VLA(Q) reaches FAR 0.98 with forgetting 0.03 vs 0.62 for baselines (single-task scenario); FAR 0.74, BWT 0.17 (multi-task scenario).
  
  — [CRL-VLA HTML](https://arxiv.org/html/2602.03445v1)
- **Real-world continual VLA (arXiv 2605.26820, May/Aug 2026):** 10 sequential real tasks, single-arm and bimanual. Sequential fine-tuning gives "severe catastrophic forgetting". Well-configured experience replay can "outperform joint multi-task training under equivalent computational budgets". The abstract does not name the VLAs — [arXiv](https://arxiv.org/abs/2605.26820)
- **RoboCasa365 lifelong protocol:**
  - Phases: Phase 1 is 65 atomic tasks; Phase 2 is 20 tasks with 2–3 subtasks; Phase 3 is 20 with 4–5; Phase 4 is 20 with 6+.
  - Atomic success degrades 41.5 → 13.9 → 13.9 → 10.6. Phase-2 tasks degrade 24.5 → 4.8 → 1.7. The paper calls this catastrophic forgetting.
  
  — [RoboCasa365 HTML](https://arxiv.org/html/2603.04356)
- **Compositional continual world models (Oxford, arXiv 2609.22055, Sep 2026):** six suites on Meta-World assets, with each curriculum ending in a composition task. DreamerV3, TD-MPC2 and a modular PWM are compared under FT, ER, EWC and PackNet. PWM nearly eliminates forgetting (BWT 3.85 vs 34.97), but only with a frozen pretrained encoder — [HTML](https://arxiv.org/html/2609.22055)
- CompoSuite was built for compositional multi-task and lifelong RL — [CompoSuite GitHub](https://github.com/Lifelong-ML/CompoSuite); [PMLR](https://proceedings.mlr.press/v199/mendez22a/mendez22a.pdf)
- No continual protocol exists in COLOSSEUM, GemBench, AGNOSTOS, CALVIN, SimplerEnv, VLABench, RoboTwin 2.0, LIBERO-Plus, LIBERO-PRO, LIBERO-X or LIBERO-OOD as published. Their papers describe only single-stage train-then-test protocols — see the Q0 sources.

### Inferences
- For a frozen-backbone method that stores per-task state (like the lifelong-ood-learner "bindings"), forgetting is zero by construction. The credible comparison is therefore forward transfer, storage per task and success vs (a) ER at 2% replay on pi0/pi0.5 (Liu et al.) and (b) per-task LoRA. Showing NBT ≈ 0 alone is weak, because pi0 already has NBT ≈ −0.016 with replay.
- RoboCasa365's lifelong protocol is the only official one where tasks grow in compositional length. pi0.5's starting point there is very low (Atomic 39.6), so absolute numbers will be small.

### Gaps
- No public continual-learning results for pi0.5 specifically were found. Liu et al. use pi0 and GR00T N1.5.
- No continual protocol was found that interleaves new compositional tasks (e.g., LIBERO-OOD tasks) as a stream over a frozen VLA.

---

## Q4. Recommendation: most credible benchmarks for zero-shot compositional generalization and continual learning with a frozen pi0.5-style policy

### Takeaway
1. **Primary benchmark: LIBERO-OOD (Spatial-OOD + Goal-OOD).** It is purpose-built for seen-skill / unseen-combination tasks, its code is public, and it has published pi0 (0.09) and pi0.5 (35.5 / 46.6) baselines plus a frozen-policy competitor (TLI 0.83).
2. **Second compositional benchmark: ReGuide's 16-cell LIBERO protocol** (frozen pi0.5: 94.0 → 39.0; ReGuide 94.5). Code is not yet public.
3. **Memorization and language controls:** report LIBERO-PRO (position/task) and LIBERO-Para (paraphrase) or LIBERO-Plus (language/layout) to show the method is not exploiting memorized layouts or task-ID instruction selection.
4. **Continual learning:** run the LIBERO sequential protocol (FWT/NBT/AUC) against ER at 2% replay and per-task LoRA, following Liu et al. 2026. Optionally run RoboCasa365 Composite-Unseen and its 4-phase lifelong protocol as a harder, leaderboard-backed external test (pi0.5 baseline 1.2% Composite-Unseen).
5. **Cross-simulator checks:** AGNOSTOS and GemBench are credible cross-task / long-horizon zero-shot tests, but integration is expensive (RLBench re-fine-tuning) and no pi0.5 baseline exists.

Lower credibility for this specific claim:
- **Standard LIBERO:** saturated, and solvable by a 0.54M policy via task-ID memorization.
- **CALVIN ABC→D:** environment shift, near ceiling at 4.75 of 5.
- **SimplerEnv:** inconsistent pi0 baselines.
- **COLOSSEUM:** visual robustness only, RLBench 3D policies dominate.

### Cited Findings
- LIBERO saturation: 0.54M parameters reach 95.1%. Task-ID permutation brings success "to near chance" — [MINERVA](https://arxiv.org/abs/2609.03715)
- LIBERO's test tasks are "nearly identical to training tasks, differing only in minor initial state perturbations" — [LIBERO-PRO HF page](https://huggingface.co/papers/2510.03827)
- LIBERO-OOD baselines, construction and code — [Li HTML](https://arxiv.org/html/2505.03500); [AC-VLA HTML](https://arxiv.org/html/2607.15714)
- ReGuide frozen-pi0.5 compositional baseline and method — [ReGuide HTML](https://arxiv.org/html/2609.38616)
- Replay-based continual VLA baseline — [Liu et al. 2026](https://arxiv.org/html/2603.03818)
- RoboCasa365 lifelong protocol, Composite-Unseen split, and leaderboard — [RoboCasa365 HTML](https://arxiv.org/html/2603.04356); [leaderboard](https://robocasa.ai/leaderboard.html)
- CALVIN ABC→D near ceiling: 4.75 / 5 — [Xiaomi-Robotics-0](https://arxiv.org/html/2602.12684)
- SimplerEnv baseline inconsistency — [Xiaomi-Robotics-0](https://arxiv.org/html/2602.12684); [LatBot](https://arxiv.org/pdf/2511.23034)
- AGNOSTOS requires fine-tuning on 18 seen RLBench tasks with 200 demos each — [AGNOSTOS HTML](https://arxiv.org/html/2505.15660v3)
- GemBench requires 3,100 training demos and has 4 levels — [GemBench HTML](https://arxiv.org/html/2410.01345)
- pi0.5 is the most robust open VLA on LIBERO-X: L5 18.0 vs ≤ 9.7 for the others — [LIBERO-X HTML](https://arxiv.org/html/2602.06556)
- pi0.5 is the only model above 0 on LIBERO-PRO goal-position (0.38) — [LIBERO-PRO PDF](https://arxiv.org/pdf/2510.03827)

### Inferences
- **Credibility ranking for "zero-shot compositional generalization with a frozen pi0.5":**
  1. LIBERO-OOD
  2. ReGuide cells
  3. LIBERO-X L3 (relation changes; needs re-fine-tuning on the LIBERO-X data, so not frozen relative to `pi05_libero`)
  4. RoboCasa365 Composite-Unseen
  5. AGNOSTOS L1/L2
  6. GemBench L2–L4
  7. VIMA-Bench L2/L4 (classic, but not a VLA embodiment)
- **"Frozen" caveat:** `pi05_libero` is itself fine-tuned on all four LIBERO suites — [LIBERO-PRO PDF](https://arxiv.org/pdf/2510.03827); [openpi](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md). "Frozen" therefore means no further weight updates after that checkpoint. That is the same setting as ReGuide and TLI.
- **Baseline integrity:** re-run pi0.5 on LIBERO-OOD in the user's own harness. The published 35.5 / 46.6 comes from one paper (AC-VLA), and published cross-paper baselines vary by up to about 19 pp elsewhere.
- **Control for the policy-bypass finding:** if a method's outputs can bypass the policy (as the user's 2026-10-01 bindings finding suggests), a credible report should include a no-policy / program-only control on the same splits. No benchmark enforces this; it is a reporting choice.

### Gaps
- No benchmark currently combines compositional OOD tasks with a continual stream over a frozen VLA. Such a protocol would need to be defined (e.g., LIBERO-OOD tasks presented sequentially, with FWT/NBT/AUC and storage per task).
- No RoboArena leaderboard entry for pi0.5-DROID was verified in this session. A search snippet attributing a 47.6% pi0.5 score to RoboArena actually came from ArmnetBench ([ArmnetBench](https://arxiv.org/html/2607.24481v1)).
- RoboCerebra's numerical results for VLM+VLA systems were not retrieved.
