# Architecture and representation routes to zero-shot generalization in robot manipulation (2023-2026), and what each would do for a frozen VLA that fails new LIBERO-90 tasks

Scope note: I collected these notes on 2026-10-01. Most numbers come from arXiv abstracts or HTML pages, fetched through a summarizing tool. I checked the Knowledge Insulation numbers against the PDF text myself. The tool's first summary of that PDF made up LIBERO-90 numbers ("46.2%"), so I discarded it and quote only the PDF Table 1. Numbers marked "~" were read off figures and are approximate. Several 2026 papers are fresh v1 preprints without peer review; I flag them where they appear.

## Q1. Which approaches achieve zero-shot success on genuinely new tasks (new object-relation combinations, new scenes), and how much?

### Takeaway
Only a few methods report measured zero-shot success on new object-relation combinations inside LIBERO:
- **ReGuide** keeps the VLA frozen and corrects the target referent from outside it. It raises frozen pi0.5 from 39.0% to 94.5% on LIBERO composition cells.
- **PrimitiveVLA** fine-tunes the VLA on primitive-segmented data and adds a VLM planner. It raises pi0.5 from 56.0% to 75.5% on 8 unseen LIBERO-90 tasks.

Elsewhere, cross-task zero-shot success stays low. On AGNOSTOS (RLBench), pi0 scores 21.7% on Level-1 and 12.0% on Level-2, and the best method, an LLM in-context approach, reaches 37.6% and 20.3%. Most other families (co-training, world models, flow and keypoint methods) report gains on new objects, scenes or backgrounds, not on new tasks.

### Cited Findings

#### Baseline evidence: how badly current VLAs do on new tasks and perturbed LIBERO
- AGNOSTOS is built on RLBench. It has 18 seen tasks and 23 unseen ones. Level-1 has 13 unseen tasks that share partial semantics with seen tasks. Level-2 has 10 tasks with no overlapping objects or actions. — [AGNOSTOS paper](https://arxiv.org/html/2505.15660v3); [project page](https://jiaming-zhou.github.io/AGNOSTOS/)
- Zero-shot success on AGNOSTOS:

  | Method | Level-1 | Level-2 |
  |---|---|---|
  | OpenVLA | 16.9% | 9.2% |
  | RDT | 10.4% | 14.4% |
  | pi0 | 21.7% | 12.0% |
  | VoxPoser | 18.1% | 12.1% |
  | X-ICM (7B) | 28.6% | 16.9% |
  | X-ICM (72B) | 37.6% | 20.3% |

  The authors report that "all prior models completely fail on at least eight of the 23 tasks." X-ICM conditions an LLM on in-context demonstrations from seen tasks. Benchmark: RLBench-based AGNOSTOS only. — [AGNOSTOS / X-ICM, NeurIPS 2025](https://arxiv.org/html/2505.15660v3)
- **LIBERO-PRO.** Models score over 90% on standard LIBERO, but "their performance collapses to 0.0% under our generalized setting." Models keep grasping when the target object is swapped for an irrelevant item. Their outputs barely change under corrupted instructions or "messy tokens." The evaluated models include OpenVLA, pi0 and pi0.5. Under position shifts, OpenVLA and pi0 drop to zero once displacement exceeds about 0.2 units; pi0.5 holds out to about 0.4. — [LIBERO-PRO abstract](https://arxiv.org/abs/2510.03827); [LIBERO-PRO HTML](https://arxiv.org/html/2510.03827)
- **LIBERO-Plus.** Ten models were tested: OpenVLA, OpenVLA-OFT (3 variants), pi0, pi0-fast, Nora, WorldVLA, UniVLA and RIPT-VLA.
  - With the instruction left blank, "the performance of OpenVLA-OFT on the object suite remained largely unchanged."
  - When the target object named in the instruction is replaced, "the model's success rate dropped nearly to zero."
  - Language perturbations cause the second-smallest average drop of the seven dimensions (-25.3 pp; range -9.6 to -49.7). The authors read this apparent robustness as "language ignorance."
  - Source: [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626); [abstract](https://arxiv.org/abs/2510.13626)
- **LangForce, vision-only control.** A vision-only policy, given no instruction, reaches 44.6% on RoboCasa against 47.8% for the language-conditioned baseline. On LIBERO-Goal, where several tasks share one scene, vision-only success "plummets to 9.8% (97.4% in Baseline)." — [LangForce](https://arxiv.org/html/2601.15197v6)
- **Robust Skills, Brittle Grounding** (Feb 2026). In controlled multi-object picking with SmolVLA and pi0.5 on held-out object-location pairings, "manipulation skill acquisition is decoupled from instruction following": grasp execution stays reliable while instruction-conditioned success falls. — [Emukpere et al. 2026](https://arxiv.org/abs/2602.24143)
- **RoboSemanticBench** (Jun 2026). After controlling for grasp success, many policies "select the semantically correct block at near-random or below-random rates." — [RoboSemanticBench](https://arxiv.org/abs/2606.02277)

#### Family 1: co-training with web/VLM data and keeping language grounding
**Core idea.** Keep or restore the VLM's language-vision knowledge while learning actions. Methods do this by:
- co-training on web data,
- representing actions as text,
- stopping action-expert gradients from reaching the VLM backbone,
- adding reasoning supervision.

**Works and results:**
- **RT-2** (2023) co-fine-tunes VLMs on web VQA plus robot trajectories, with actions as text tokens. In about 6k real trials it scores 62% on unseen objects, backgrounds and environments, against 32% for RT-1. It also shows emergent semantic commands ("place on the number 5", smallest/largest object). Results are on Google's real-robot evaluation only. — [RT-2 arXiv](https://arxiv.org/abs/2307.15818); [62% vs 32% summary (alphaXiv)](https://www.alphaxiv.org/abs/2307.15818v1)
- **OpenVLA** (2024) is a 7B model trained on 970k real demonstrations. It beats RT-2-X (55B) by 16.5% absolute across 29 tasks and Diffusion Policy by 20.4%. — [OpenVLA](https://arxiv.org/abs/2406.09246)
- **ECoT** (Embodied Chain-of-Thought, 2024) has the policy reason about plans, sub-tasks, motions, bounding boxes and gripper positions before acting. It raises OpenVLA's absolute success "by 28% across challenging generalization tasks, without any additional robot training data." It uses synthetic reasoning labels on existing datasets. — [ECoT](https://arxiv.org/abs/2407.08693)
- **ERVLA** ("Revisiting ECoT", Jun 2026) finds that "high-level reasoning alone brings only marginal gains." What helps is grounded action guidance: end-effector movement descriptions and image-space trajectories. It uses chain-of-thought as training-time supervision with reasoning dropout, built from 978,743 trajectories (2,592.5 h). It reports 86.9% on LIBERO-Plus and 53.2% on VLABench. — [ERVLA](https://arxiv.org/abs/2606.03784)
- **pi0.5** (2025) co-trains on heterogeneous sources; in the first phase, 97.6% of examples do not come from mobile manipulators. Removing web data "causes significantly worse performance on out-of-distribution (OOD) objects."
  - Figure-read values for OOD success: full model ~60%, no web data ~45%.
  - Two sources disagree on the exact in-distribution numbers: the search snippet gives follow rate 86% (full) vs 86% (no web data), success 83% vs 82%; my figure reading differs.
  - Treat all of these as approximate. Training used about 400 h of mobile-manipulation data, 280k pre-training steps and 80k post-training steps.
  - Source: [pi0.5 HTML](https://arxiv.org/html/2504.16054v1); [PI blog](https://www.physicalintelligence.company/blog/pi05)
- **Knowledge Insulation** (KI, 2025) stops the action expert's gradient from flowing into the backbone and co-trains with FAST tokens and VLM data.
  - Without it, "naive training with such a randomly initialized action expert harms the models' ability to follow language commands." In one example, pi0 grasps trash when told to bus a spoon.
  - "Co-training on VLM data is particularly important" for the OOD follow rate on unseen objects in a mobile-manipulator drawer task.
  - pi0 needs "7.5 times as many training steps" to match.
  - "Freezing the backbone is not a viable option."
  - Limitation: language following "is still far from perfect, likely because correlations in the training data still cause the model to sometimes ignore language instructions."
  - LIBERO Table 1 (in-distribution; KI fine-tuned from the generalist): Spatial 98.0, Object 97.8, Goal 95.6, Long 85.8, LIBERO-90 96.0. From scratch, LIBERO-90 is 92.7. These are LIBERO-90 training-task scores, not new-task scores.
  - Source: [KI arXiv](https://arxiv.org/abs/2505.23705); [KI PDF](https://arxiv.org/pdf/2505.23705)
- **ChatVLA** (2025) uses phased alignment training plus mixture-of-experts. It reports 47.2% on MMStar, 6x higher MMMU than baselines, and beats OpenVLA on 25 real tasks. — [ChatVLA](https://arxiv.org/abs/2502.14420)
- **VLM2VLA** ("Actions as Language", ICLR 2026) relabels low-level actions as natural language and trains only LoRA.
  - It keeps VQA ability: MMStar 48.0, against 0 for OpenVLA and ECoT and 46.3 for the base Gemma-3-12B.
  - It reports zero-shot generalization to novel tasks needing open-world semantic reasoning and multilingual instructions (Hindi, Mandarin) across 800+ real experiments.
  - I did not find exact success percentages on the project page.
  - Source: [VLM2VLA arXiv](https://arxiv.org/abs/2509.22195); [project page](https://vlm2vla.github.io/)
- **LangForce** (Jan 2026) trains a two-branch model: a vision-only prior p(a|v) and a language posterior pi(a|v,l), maximizing the log-likelihood ratio (PMI between action and instruction). Base: Qwen3-VL-4B with a DiT head.
  - SimplerEnv: 66.5% vs 55.2% (QwenGR00T).
  - RoboCasa: 52.6% vs 47.8%.
  - LIBERO: 99.2 / 99.6 / 99.4 / 95.2.
  - Real-world OOD unseen red block: 9/30 vs 2/30.
  - Source: [LangForce](https://arxiv.org/html/2601.15197v6); [code](https://github.com/ZGC-EmbodyAI/BayesianVLA)
- **In-Context VLA / VLA-Talker** (Aug 2026) feeds detector, depth and gripper-projection evidence (GroundingDINO, DepthAnything, Qwen2.5-VL fallback) to OpenVLA-OFT as read-only context, then trains it with SFT and GRPO.
  - Language-generalization evaluation: seen objects 97.4%, unseen objects 85.1%, unseen objects plus distractors 80.3%. BC baseline with distractors: 47.6%.
  - SimplerEnv: 72.4% vs 54.7% for generative chain-of-thought.
  - The VLA is fine-tuned, not frozen.
  - Source: [In-Context VLA](https://arxiv.org/html/2608.05738v1)

#### Family 2: hierarchical policies (VLM planner over a low-level policy)
**Core idea.** A high-level model emits an intermediate representation that a low-level policy is trained to follow. The representation can be:
- subtask text,
- "language motions",
- 2D paths, trajectory sketches or keypoints,
- goal images.

**Works and results:**
- **RT-H** (2024) inserts "language motions" ("move arm forward") between task and action.
  - It averages 15% above RT-2 on 8 diverse tasks. On novel objects it scores 65% vs 55%.
  - Training on language-motion corrections lifts success from 40% to 63% with 30 correction episodes per task.
  - Failure mode: systematic language-motion prediction errors cause oscillations.
  - Source: [RT-H HTML](https://arxiv.org/html/2403.01823)
- **RT-Trajectory** (2023) conditions the policy on coarse 2D trajectory sketches drawn by humans or produced by image generators or waypoint generators. With the same training data, it performs "a wider range of tasks compared to language-conditioned and goal-conditioned policies." I did not retrieve exact numbers. — [RT-Trajectory](https://arxiv.org/abs/2311.01977)
- **HAMSTER** (2025) fine-tunes a VLM on off-domain data (action-free video, hand-drawn sketches, simulation) to predict 2D end-effector paths. A 3D-aware low-level policy follows the path. It averages "20% improvement in success rate across seven different axes of generalization" over OpenVLA, a 50% relative gain. — [HAMSTER](https://arxiv.org/abs/2502.05485)
- **Hi Robot** (2025) puts a VLM high-level policy, trained on segmented demos plus VLM-generated synthetic user interactions, over pi0.
  - It averages "over 40% higher instruction accuracy than GPT-4o" and beats a flat VLA on table bussing, sandwich making and grocery shopping.
  - Limitation relevant here: the low-level policy sometimes ends up "temporarily ignoring instructions" because of training bias.
  - Source: [Hi Robot HTML](https://arxiv.org/html/2502.19417)
- **pi0.5 high-level inference.** The same model predicts the subtask in text, then the actions. On mock-home task progress (figure-read), the full model scores ~70%, implicit high-level ~65%, no high-level data ~50%, GPT-4 high-level ~45% and human high-level oracle ~72%. The authors say the full model "outperforms even the human HL 'oracle' baseline." — [pi0.5 HTML](https://arxiv.org/html/2504.16054v1)
- **Gemini Robotics 1.5 / ER 1.5** (2025) pairs an embodied-reasoning model (planning, spatial reasoning, progress estimation) with a VLA that "interleaves actions with a multi-level internal reasoning process in natural language." Motion Transfer lets it learn from multi-embodiment data. Its generalization benchmark has 230 tasks, including a new-task category; I could not verify per-axis numbers from primary text. — [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342)
- **VLA-OS** (2025) is a controlled comparison: "visually grounded planning representations are generally better than language planning representations." Hierarchical VLAs give "superior or comparable performance ... generalization ability ... continual learning ability, albeit at the cost of slower training and inference speeds." — [VLA-OS](https://arxiv.org/abs/2506.17561)
- **SayCan lineage** (2022, pre-window): an LLM supplies procedures and skill value functions supply grounding. — [SayCan](https://arxiv.org/abs/2204.01691)

#### Family 3: VLM-planned constraints, keypoints, and flow / object-centric interfaces
**Core idea.** A VLM or LLM writes costs, constraints or keypoints in 3D or image space, and a planner or optimizer produces motion. Alternatively, a model trained on video predicts point or object flow, and a small policy maps flow to actions.

**Works and results:**
- **VoxPoser** (2023) has an LLM write code that composes 3D value maps with VLMs, then MPC synthesizes trajectories zero-shot. — [VoxPoser](https://arxiv.org/abs/2307.05973)
- **ReKep** (2024) uses VLM-generated relational keypoint constraints (Python cost functions) and solves them with hierarchical optimization, "without task-specific data or environment models." — [ReKep](https://arxiv.org/abs/2409.01652)
- **MOKA** (2024) uses mark-based visual prompting for point affordances; it can also distill its own experience into a policy. — [MOKA](https://arxiv.org/abs/2403.03174)
- **CoPa** (2024) uses part-level spatial constraints and is training-free. — [CoPa](https://arxiv.org/abs/2403.08248)
- **OmniManip** (2025) uses object-centric canonical interaction primitives with dual closed loops and is training-free. Zero-shot results on 12 real tasks:

  | Category | OmniManip closed-loop | OmniManip open-loop | VoxPoser | CoPa | ReKep |
  |---|---|---|---|---|---|
  | Rigid objects | 68.3% | 51.7% | 15.0% | 30.0% | 45.0% |
  | Articulated objects | 61.7% | 45.0% | 16.7% | 26.7% | — |

  Stated limits: no deformable objects; depends on 3D-generated mesh quality; multiple VLM calls are costly. — [OmniManip HTML](https://arxiv.org/html/2501.03841)
- **ATM** (Any-point Trajectory Model, 2024) is pre-trained on action-free video to predict point tracks; the policy is conditioned on tracks.
  - On 130+ LIBERO tasks with 10 action-labeled demos plus 50 action-free videos per task, it averages 63% against 37% for the best prior method.
  - Limitation: it still needs action-labeled demos per task.
  - Source: [ATM](https://arxiv.org/html/2401.00025)
- **Im2Flow2Act** (2024) generates object flow from human videos; a flow-conditioned policy trained on simulated play transfers to real. Absolute numbers were not in the abstract. — [Im2Flow2Act](https://arxiv.org/abs/2407.15208)
- **General Flow** (2024) is a language-conditioned 3D flow model trained on RGBD human video. It reports "81% success rate in zero-shot human-to-robot skill transfer, covering 18 tasks in 6 scenes." — [General Flow](https://arxiv.org/abs/2401.11439)
- **Track2Act** (2024) predicts tracks from web video plus a residual policy trained on a few demos. It claims zero-shot manipulation of unseen objects in novel scenes; the abstract gives no numbers. — [Track2Act](https://arxiv.org/abs/2405.01527)

#### Family 4: world models and video-prediction policies
**Core idea.** Pre-train on video to predict the future (pixels or latents), then decode actions with inverse dynamics, or use the model to plan or generate data.

**Works and results:**
- **UniPi** (2023) treats a text-conditioned video plan plus inverse dynamics as the policy. It claims combinatorial generalization to novel goals; the abstract gives no numbers. — [UniPi](https://arxiv.org/abs/2302.00111)
- **GR-1** (2023): CALVIN success 88.9% → 94.9%; zero-shot unseen-scene CALVIN success 53.3% → 85.4%. — [GR-1](https://arxiv.org/abs/2312.13139)
- **GR-2** (2024) pre-trains on 38M video clips (50B tokens) and averages 97.7% across 100+ tasks. It claims generalization to unseen backgrounds, environments, objects and tasks; the abstract gives no out-of-distribution numbers. — [GR-2](https://arxiv.org/abs/2410.06158)
- **VPP** (2024/ICML 2025) gives an 18.6% relative gain on CALVIN ABC-D and +31.6% on real dexterous tasks. — [VPP](https://arxiv.org/abs/2412.14803)
- **Seer** (2024) gains +13% on LIBERO-Long, +21% on CALVIN ABC-D (average length 4.28) and +43% in real tasks. — [Seer](https://arxiv.org/abs/2412.15109)
- **UniSim** (2023) trains vision-language and RL policies "purely in simulation" inside the learned simulator and deploys them in the real world zero-shot. — [UniSim](https://arxiv.org/abs/2310.06114)
- **Genie Envisioner** (2025) combines a video world model (GE-Base), a flow-matching action decoder (GE-Act) and a neural simulator (GE-Sim). The abstract gives no numbers. — [Genie Envisioner](https://arxiv.org/abs/2508.05635)
- **Cosmos Policy** (Jan 2026) fine-tunes a video model to output actions, future states and values. It scores 98.5% on LIBERO and 67.1% on RoboCasa with 50 demos per task, and 93.6% on 4 ALOHA tasks. All of these are in-distribution numbers. — [Cosmos Policy](https://arxiv.org/abs/2601.16163)
- **DreamGen** (2025) fine-tunes a video model and labels its "neural trajectories" with latent actions or an inverse-dynamics model. A humanoid learns "22 new behaviors in both seen and unseen environments" from teleop data for one pick-and-place task. — [DreamGen](https://arxiv.org/abs/2505.12705)
- **V-JEPA 2-AC** (2025) pre-trains on 1M+ hours of video and post-trains on under 62 h of unlabeled DROID video. It does zero-shot pick-and-place on Franka arms in two labs by planning with image goals. — [V-JEPA 2](https://arxiv.org/abs/2506.09985)

#### Family 5: skill libraries, motion primitives, prompt conditioning
- **PrimitiveVLA** (May 2026, v2 Jul 2026) segments demos into 11 primitives (grasp, place, lift, move, push, pull, insert, press, twist, tilt, rotate), fine-tunes the VLA on primitive-level supervision, and uses a Qwen3-VL planner plus an LLM-written switch module at inference.
  - On LIBERO-90-Novel (8 unseen tasks drawn from LIBERO-90 scenes, for example "Open top drawer, place bowl in, close" and "Pick butter, place in basket"):

    | Base model | Baseline | PrimitiveVLA |
    |---|---|---|
    | pi0.5 | 56.0% | 75.5% |
    | OpenVLA-OFT | 13.5% | 71.0% |
    | OpenVLA | 7.38% | 45.5% |

  - LIBERO-Long with pi0.5: 30.5% → 80.25%.
  - Real UR5e, unseen objects: 80% vs 10%.
  - The base VLAs are fine-tuned, not frozen.
  - Source: [PrimitiveVLA HTML](https://arxiv.org/html/2605.28634v1); [abstract](https://arxiv.org/abs/2605.28634)
- **LOTUS** (ICRA 2024) discovers skills continually on LIBERO, using open-vocabulary vision models and a meta-controller. It gains "over 11% in success rate" over the prior state of the art. — [LOTUS](https://arxiv.org/abs/2311.02058)
- **Decompose and Recompose** (May 2026) decomposes demos into atomic skill-action pairs and retrieves them from a dynamic library, guided by a planning agent's skill sequence. It is evaluated on AGNOSTOS and real robots; the abstract gives no numbers. — [arXiv 2605.01448](https://arxiv.org/abs/2605.01448)
- **X-VLA** (2025) adds learnable soft-prompt embeddings per data source or embodiment to a 0.9B flow-matching transformer. It is tested on 6 simulators and 3 real robots. Soft prompts here address cross-embodiment heterogeneity, not task semantics. — [X-VLA](https://arxiv.org/abs/2510.10274)

#### Family 6: compositional generalization via policy composition or factorization
- **PoCo** (2024) composes diffusion policies trained on different domains (simulation, human video, real robot) and cost functions at inference. It beats single-source baselines; the abstract gives no numbers. — [PoCo](https://arxiv.org/abs/2402.02511)
- **GPC** (General Policy Composition, ICLR 2026) makes training-free convex combinations of the scores of pre-trained diffusion or flow policies, including vision-action and VLA policies, with a test-time search over weights. The abstract gives no numbers. — [GPC](https://arxiv.org/abs/2510.01068); [code](https://github.com/SageCao1125/GPC)
- **Modality-Composable Diffusion Policy** (2025) does inference-time distribution-level composition across input modalities. I saw it only in search results and did not fetch it. — [arXiv 2503.12466](https://arxiv.org/pdf/2503.12466)
- **Factored Diffusion Policies** (May 2026) train one score network with per-factor null-token dropout and compose additively at test time.
  - Drone racing, held-out factor pairs: 90% vs 17% for the unfactored model.
  - Composing K separately trained networks with the same formula collapses to 3%. The authors conclude that "parameter sharing rather than the additive formula is the mechanism."
  - Simulation only, no manipulation.
  - Source: [Factored Diffusion Policies](https://arxiv.org/html/2605.22596v1)
- **ReGuide** ("Correcting WHERE, Preserving HOW", arXiv v1 dated 2026-09-29) keeps the VLA frozen.
  - **What it adds.** No neural training. It adds a perception-based detector with depth and statistics drawn from the demonstrations. "Semantic rebinding" detects when the policy's predicted endpoint heads for the wrong referent. "Geometric rebinding" moves the end-effector into a pre-contact set derived from demos, then hands control back to the frozen policy.
  - **LIBERO composition cells.** There are 16 cells (4 tasks × target region / target object / unseen object / task scene), with 50 initial states each.

    | Policy | Bare | With ReGuide |
    |---|---|---|
    | pi0.5 | 39.0% | 94.5% |
    | OpenVLA-OFT | — | 70.8% (+56.8 pp) |
    | GR00T N1.7 | 13.0% | 69.1% |

  - **Real xArm6, recombined tasks.** pi0.5 goes from 15.0% to 90.0%.
  - **Diagnosis.** "Failures usually stem from incorrect global grounding rather than a loss of local manipulation skills." If the frozen policy's actions after hand-back are replaced with generic commands, success drops from 94.5% to 28.0–38.1%.
  - Source: [ReGuide](https://arxiv.org/html/2609.38616)
- **VLA Grounder** (Jul 2026) keeps the VLA "fully frozen." An RL-trained language-conditioning policy rewrites human instructions into "VLA-grounded" commands. It is tested on RL4VLA and VL-Think; the abstract gives no numbers. — [VLA Grounder](https://arxiv.org/abs/2607.04517)

### Inferences
- **Large gains on new LIBERO combinations come from adding grounding, not motor skill.** ReGuide's frozen-VLA result (+55.5 pp) and PrimitiveVLA's fine-tuned result (+19.5 pp for pi0.5) both add an external source of "which object / where":
  - a detector plus a demo-derived pre-contact pose in ReGuide,
  - a VLM planner that chooses primitive-plus-object in PrimitiveVLA.

  Both then reuse the VLA's existing local motor skill. This matches the diagnosis that VLAs keep their skills but misbind referents. It is consistent across LIBERO-Plus, LIBERO-PRO, LangForce, "Robust Skills, Brittle Grounding" and RoboSemanticBench.
- **Most families do not report new-task success on LIBERO.**
  - World-model and video policies (GR-1/2, VPP, Seer, Cosmos Policy) report in-distribution or new-scene gains on LIBERO and CALVIN.
  - Co-training families report new-object and new-scene gains in the real world.
  - Their headline numbers therefore do not show that they would solve unseen object-relation instructions.
- **Training-free VLM-constraint methods do succeed zero-shot on new tasks, but they bypass the learned policy.** OmniManip reaches 62–68% on 12 real tasks; VoxPoser and ReKep are lower. They swap a learned visuomotor policy for an optimizer, so they give up the frozen VLA's dexterity instead of fixing its grounding.

### Gaps
- No primary-text numbers verified for RT-Trajectory, Gemini Robotics 1.5 generalization axes, Hi Robot (beyond ">40% over GPT-4o"), GPC, PoCo, ReKep, MOKA, CoPa (beyond OmniManip's table), UniPi, Genie Envisioner, or Decompose-and-Recompose. I fetched abstracts only, and they gave no figures.
- pi0.5 ablation numbers are figure-read and conflict between two sources. Treat them as approximate.
- ReGuide (2 days old), PrimitiveVLA, LangForce, In-Context VLA and Factored Diffusion Policies are unreviewed preprints. PrimitiveVLA does not say clearly what instruction text the baselines receive on LIBERO-90-Novel.
- I found no paper that evaluates co-trained or KI models on unseen LIBERO-90 object-relation combinations specifically.

## Q2. Which address language/instruction grounding versus motor generalization?

### Takeaway
Methods fall into three groups:
- **Language grounding:** co-training/KI, LangForce, ECoT/ERVLA, VLM2VLA, In-Context VLA, VLA Grounder and ReGuide's semantic rebinding.
- **Motor/visual generalization:** world models, flow/track models, X-VLA soft prompts, KI's training-speed gains.
- **Both:** hierarchical methods and VLM-constraint methods, which split the problem. The high level grounds the words and the low level executes.

Evidence from several sources: language-only intermediate representations (subtask text, chain-of-thought) help less than visually grounded ones (points, paths, boxes) when the low level does not read words reliably.

### Cited Findings
- **Language failure comes from the data, not model capacity.** Goal-driven datasets make the instruction "highly predictable from visual observations alone," which causes "the vision shortcut." — [LangForce](https://arxiv.org/html/2601.15197v6)
- **KI's limitation says the same.** "Correlations in the training data still cause the model to sometimes ignore language instructions." — [KI PDF](https://arxiv.org/pdf/2505.23705)
- **Gradient interference from a randomly initialized action expert degrades language following.** Stop-gradient or VLM co-training restores it, but "Freezing the backbone is not a viable option" for training the VLA. — [KI PDF](https://arxiv.org/pdf/2505.23705)
- **Web data matters most for OOD objects.** It "makes the biggest difference for generalizing to out-of-distribution objects," while cross-embodiment and multi-environment robot data matter across all conditions. — [pi0.5 blog](https://www.physicalintelligence.company/blog/pi05); [pi0.5 HTML](https://arxiv.org/html/2504.16054v1)
- **Grounded reasoning beats abstract reasoning.** "High-level reasoning alone brings only marginal gains"; grounding into end-effector motion descriptions and image-space trajectories helps. — [ERVLA](https://arxiv.org/abs/2606.03784)
- **Visual plans beat language plans.** "Visually grounded planning representations are generally better than language planning representations." — [VLA-OS](https://arxiv.org/abs/2506.17561)
- **A hierarchical system still fails if the low level ignores the subtask text.** Hi Robot lists a low-level policy "temporarily ignoring instructions" among its failure modes. — [Hi Robot HTML](https://arxiv.org/html/2502.19417)
- **Generative chain-of-thought has a "grounding gap."** The rationale comes from the same features the action head sees. In-Context VLA instead injects external detector and depth evidence. — [In-Context VLA](https://arxiv.org/html/2608.05738v1)
- **ReGuide separates WHERE (referent) from HOW (execution).** It corrects WHERE externally and keeps HOW from the frozen policy. — [ReGuide](https://arxiv.org/html/2609.38616)
- **Motor-side families:**
  - Flow and track interfaces target cross-embodiment and human-to-robot transfer of motion: [ATM](https://arxiv.org/html/2401.00025), [General Flow](https://arxiv.org/abs/2401.11439), [Im2Flow2Act](https://arxiv.org/abs/2407.15208).
  - X-VLA soft prompts target embodiment heterogeneity: [X-VLA](https://arxiv.org/abs/2510.10274).
  - Video world models target dynamics and visual generalization: [GR-1](https://arxiv.org/abs/2312.13139), [Seer](https://arxiv.org/abs/2412.15109), [VPP](https://arxiv.org/abs/2412.14803).

### Inferences
- **Some routes cannot reach a policy that ignores words.** If a frozen VLA does not read object and relation words it never had to tell apart within a scene, then:
  - rewriting the instruction (VLA Grounder) can only steer it through phrasings it already reads;
  - a language-only high-level planner (Hi Robot style, pi0.5 subtask text) passes the same unread words downstream.
- **The routes that fit this failure mode supply grounding in a non-linguistic channel the frozen policy already responds to.** Options include image-space points or paths, a pre-contact pose (ReGuide), or a different visual context. Alternatively, retrain so that words carry information the image does not (LangForce PMI, VLM co-training, KI).
- **Motor-side families (world models, flow, soft prompts) do not target the reported failure mode.** The LIBERO evidence says motor skill is largely intact and the binding is wrong.

### Gaps
- No study I found measures, for a frozen pi0.5 on LIBERO-90, whether grounding is restored by subtask text, by 2D path conditioning, or by pre-contact pose. ReGuide tests only the last of these.
- No quantitative LIBERO results for VLA Grounder were available in the abstract.

## Q3. What do they require (extra models, extra data, retraining), and would each help a frozen VLA failing new LIBERO-90 tasks?

### Takeaway
Nearly every family requires retraining the policy or training a new low-level policy:
- **Co-training/KI:** full VLA training with web data.
- **Hierarchical:** a fine-tuned VLM plus a low-level policy trained on the intermediate representation.
- **World models:** video pre-training and action decoders.
- **PrimitiveVLA:** primitive-segmented fine-tuning.
- **Factored diffusion:** shared-network training.

Only these keep the VLA frozen:
- **Training-free VLM-constraint methods.** They replace the policy with an optimizer.
- **GPC.** It needs several policies to compose.
- **VLA Grounder.** It needs an RL-trained instruction rewriter.
- **ReGuide.** It needs a detector with depth and demo statistics. ReGuide is the only one with a published LIBERO composition result on frozen pi0.5: 39.0% → 94.5%.

### Cited Findings

| Family / method | What must be trained | Extra models at inference | Extra data | Keeps VLA frozen? | Evidence on new LIBERO tasks |
|---|---|---|---|---|---|
| RT-2 / OpenVLA / pi0.5 co-training | Whole VLA | none | web VQA + large robot data (pi0.5 ~400 h target data; 97.6% non-target examples in phase 1) | No | none (real-world OOD objects only) — [RT-2](https://arxiv.org/abs/2307.15818), [pi0.5](https://arxiv.org/html/2504.16054v1) |
| Knowledge Insulation | Whole VLA (stop-grad + FAST + VLM data) | none | VLM data | No ("freezing ... not viable") | LIBERO-90 96.0 in-distribution only — [KI](https://arxiv.org/pdf/2505.23705) |
| ECoT / ERVLA | VLA on reasoning labels | none (reasoning internal) | synthetic CoT labels; ERVLA 978,743 traj | No | ERVLA 86.9% LIBERO-Plus — [ECoT](https://arxiv.org/abs/2407.08693), [ERVLA](https://arxiv.org/abs/2606.03784) |
| VLM2VLA | LoRA on VLM | none | actions relabeled as language | Backbone mostly kept via LoRA, but trains a new VLA | real-world only — [VLM2VLA](https://arxiv.org/abs/2509.22195) |
| LangForce | VLA (dual-branch PMI) | none | none extra | No | LIBERO in-distribution 95–99.6 — [LangForce](https://arxiv.org/html/2601.15197v6) |
| In-Context VLA | OpenVLA-OFT SFT + GRPO | detector, depth, VLM | none extra | No | unseen-object 85.1% (eval details unclear) — [In-Context VLA](https://arxiv.org/html/2608.05738v1) |
| RT-H / HAMSTER / RT-Trajectory / Hi Robot | low-level policy conditioned on new interface; VLM fine-tune (HAMSTER, Hi Robot) | VLM planner | off-domain path data (HAMSTER), synthetic dialogue (Hi Robot) | No (low level retrained for the interface) | none on LIBERO — [RT-H](https://arxiv.org/html/2403.01823), [HAMSTER](https://arxiv.org/abs/2502.05485), [Hi Robot](https://arxiv.org/html/2502.19417) |
| VoxPoser / ReKep / CoPa / OmniManip | nothing | VLM/LLM + perception + motion planner | none | Bypasses policy | none on LIBERO; OmniManip 61.7–68.3% real — [OmniManip](https://arxiv.org/html/2501.03841) |
| ATM / Im2Flow2Act / General Flow / Track2Act | track/flow model + flow-conditioned policy | flow model | action-free video + few demos | No | ATM 63% vs 37% on LIBERO with 10 demos/task (seen tasks) — [ATM](https://arxiv.org/html/2401.00025) |
| GR-1/2, VPP, Seer, Cosmos Policy, DreamGen, V-JEPA 2 | video model + action decoder (or IDM) | video model | large video corpora (GR-2 38M clips; V-JEPA 2 1M+ h) | No | in-distribution LIBERO (Cosmos 98.5; Seer +13% Long) — [Cosmos Policy](https://arxiv.org/abs/2601.16163), [Seer](https://arxiv.org/abs/2412.15109) |
| PrimitiveVLA | VLA fine-tune on primitive-segmented data | Qwen3-VL planner + LLM switch | automatic segmentation | No | LIBERO-90-Novel pi0.5 56.0 → 75.5 — [PrimitiveVLA](https://arxiv.org/html/2605.28634v1) |
| LOTUS | skill library + meta-controller | open-vocab vision model | continual LIBERO demos | No | +11% lifelong LIBERO — [LOTUS](https://arxiv.org/abs/2311.02058) |
| X-VLA soft prompts | soft prompts + model | none | multi-source robot data | No | — [X-VLA](https://arxiv.org/abs/2510.10274) |
| GPC | nothing | ≥2 pre-trained diffusion/flow policies | none | Yes (composes) | none reported on LIBERO — [GPC](https://arxiv.org/abs/2510.01068) |
| Factored Diffusion Policies | single shared score net with factor dropout | none | O(Σ\|Z_i\|) tasks | No | drone sim only — [FDP](https://arxiv.org/html/2605.22596v1) |
| VLA Grounder | RL instruction rewriter | rewriter | RL rollouts | Yes | no LIBERO numbers — [VLA Grounder](https://arxiv.org/abs/2607.04517) |
| ReGuide | nothing (no NN training) | detector + depth; demo statistics | none | Yes | LIBERO composition cells, frozen pi0.5 39.0 → 94.5 — [ReGuide](https://arxiv.org/html/2609.38616) |

- **The frozen policy's local skill still carries much of ReGuide's success.** Replacing its post-hand-back actions with generic commands drops success to 28.0–38.1%. — [ReGuide](https://arxiv.org/html/2609.38616)
- **Composing separately trained networks failed in the factored-diffusion study** (3% held-out), while one shared network trained with factor dropout succeeded (90%). — [Factored Diffusion Policies](https://arxiv.org/html/2605.22596v1)
- **X-ICM beats VLAs cross-task with in-context demonstrations and an LLM.** Absolute success stays at or below 37.6% on Level-1 and 20.3% on Level-2. — [AGNOSTOS / X-ICM](https://arxiv.org/html/2505.15660v3)

### Inferences
- **For a frozen VLA that fails new LIBERO-90 tasks because it does not read contrasting object/relation words:**
  - The published approach closest to that setting is external referent correction (ReGuide): detect the instructed object, steer to a demo-derived pre-contact pose, hand back. It claims the largest measured gain on LIBERO composition cells.
  - Training-free VLM-constraint planners also work zero-shot, but they replace the policy instead of using it.
  - Every route that makes the policy itself read the words requires retraining or fine-tuning: co-training, KI, LangForce, ECoT/ERVLA, VLM2VLA, PrimitiveVLA.
- **For "composing a frozen policy's own conditionals," the Factored Diffusion Policies result is a cautionary data point.** Additive score composition generalized only when one network was trained with per-factor dropout. If a frozen VLA was never trained to contrast the words, its conditionals on those words may carry no usable signal (the LIBERO-Plus blank-instruction result). That limits any composition over them. This is an inference across papers, not a tested claim.
- **Hierarchical routes help only if the intermediate representation is visual.** A high-level VLM emitting subtask text does not fix a low-level policy that ignores words. VLA-OS and ERVLA favor visually grounded intermediate representations, but using them needs a low-level policy trained to follow points or paths (HAMSTER, RT-Trajectory). Pre-trained pi0.5 is not trained that way.

### Gaps
- No paper reports cost (GPU-hours) for ReGuide's detector/depth stack, or a comparison against a VLM-planner-plus-primitive approach on the same LIBERO cells.
- I found no evaluation of GPC or PoCo on LIBERO new-task splits, or on policies that ignore language.
- I did not survey object-centric diffusion policies specifically (for example GROOT-style object-centric BC). No numbers are included for that subfamily.
