# Language grounding failures in VLA robot policies: diagnosis and fixes (2024-2026)

Context for the reader: our frozen pi0.5 (fine-tuned on LIBERO's 40 suite tasks), placed in a LIBERO-90 living room where it was trained on one two-object instruction, picks up that instruction's first object in 10/10 episodes, whatever object the new instruction names. In LIBERO-Object layouts it follows a swapped object name 43% of the time. The notes below set out how the field measures, explains and fixes this. All dates are arXiv submission dates unless stated.

## Q1. Which papers show VLAs ignore language on LIBERO, and with what numbers?

### Takeaway
At least seven independent 2025-2026 studies show that LIBERO-trained VLAs (OpenVLA, OpenVLA-OFT, pi0, pi0-FAST, pi0.5, X-VLA, SmolVLA) mostly ignore the instruction. Language perturbations barely move success, and blank or garbage prompts leave behaviour unchanged. A swapped target in a familiar scene drops success to near zero, and on target-location swaps it drops to exactly 0%. pi0.5 is the best of the models tested and still follows a changed object name only about 30-38% of the time on LIBERO. Our 43% on LIBERO-Object and 10/10 "memorized first object" fall squarely within these published numbers.

### Cited Findings

**LIBERO-PRO (Zhou et al., Oct 2025; v2 May 2026): garbage prompts and object swaps**
- LIBERO-PRO evaluates OpenVLA (4 official checkpoints), pi0 and pi0.5 under object, position, instruction and environment perturbations. All three models score over 90% on standard LIBERO and "collapse to 0.0%" under generalized settings. — [LIBERO-PRO arXiv 2510.03827](https://arxiv.org/abs/2510.03827); [HTML v2](https://arxiv.org/html/2510.03827v2)
- Instruction corruption: the instruction is replaced with meaningless strings ("fdsgfdsgsd", "xxxx") and the models "still execute nearly the same action trajectory." When the target object is replaced with an irrelevant item, the models keep grasping. The paper attributes this to "rote memorization of action sequences and environment layouts," with "actions … driven by trajectory recall rather than language grounding." — [LIBERO-PRO HTML](https://arxiv.org/html/2510.03827)
- Position: OpenVLA and pi0 drop to zero once object displacement exceeds 0.2 units. pi0.5 keeps non-trivial success up to about 0.4 units. When the target object is removed, the models "grasp the air" at the memorized location. — [LIBERO-PRO HTML](https://arxiv.org/html/2510.03827)
- Caveat: LIBERO-PRO shows the corrupted-instruction and object-replacement cases qualitatively (figures) and gives no success rates for them. — [LIBERO-PRO HTML v2](https://arxiv.org/html/2510.03827v2)

**LIBERO-Plus (Fei et al., Oct 2025; CVPR 2026): "apparent language robustness is language ignorance"**
- LIBERO-Plus has 10,030 task instances across 7 perturbation factors and 21 sub-dimensions. — [LIBERO-Plus arXiv 2510.13626](https://arxiv.org/abs/2510.13626); [project page](https://sylvestf.github.io/LIBERO-plus/); [CVPR 2026 version](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- Table 1, original success vs success under language perturbation:
  - OpenVLA 76.5 → 26.8
  - OpenVLA-OFT 97.1 → 81.5
  - OpenVLA-OFT_w 95.3 → 73.2
  - OpenVLA-OFT_m 97.6 → 83.6
  - pi0 94.2 → 61.0
  - pi0-FAST 85.5 → 63.3
  - Nora 87.9 → 67.0
  - WorldVLA 79.1 → 44.2
  - UniVLA 95.2 → 71.8
  - RIPT-VLA 97.5 → 80.1

  For comparison, camera perturbation drops pi0 to 15.8 and robot initial-state perturbation drops it to 6.6. — [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626v1)
- Blank-instruction test (Sec. 4.1): the language input is "entirely replaced with an empty value." OpenVLA-OFT's performance "on the object suite remained largely unchanged, with significant degradation observed only on the long suite." — [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626v1)
- Goal-replacement test (Sec. 4.2): the target object in the instruction is swapped for another object in the same scene (e.g., "pick up alphabet soup" → "pick up tomato sauce"). Success "dropped nearly to zero," worst for OpenVLA-OFT. The models persist with "fixed vision–action mappings." — [LIBERO-Plus HTML](https://arxiv.org/html/2510.13626v1)
- LangGap's later summary of LIBERO-Plus: language perturbations cause the second-smallest drop of the 7 factors (−25.3%), which indicates the models "simply do not attend to language." — [LangGap arXiv 2603.00592](https://arxiv.org/pdf/2603.00592)

**LangGap (Hou & Zhao, NUS, Feb 2026): the closest match to our setup (pi0.5, same scene, changed instruction)**
- The benchmark keeps the tabletop layout fixed and changes the instruction semantics along four dimensions: Change Object, Change Target, Spatial Description and Drawer Action. It has 99 tasks (40 original + 59 extended), 20 episodes per task, 1,980 episodes for pi0.5. — [LangGap PDF](https://arxiv.org/pdf/2603.00592)
- pi0.5 results:
  - Original tasks: 93.8%. Extended same-scene tasks: 21.4% (−72.4 pts).
  - Change Object: 29.3% (38 tasks). Per-task results range from 0% to 80%, and success comes "when the modified instruction happens to match training patterns."
  - Change Target: 0.0% on all 13 tasks (260 episodes). "The model never redirects to an alternative target location."
  - Spatial Description: 11.0%. Drawer Action: 31.7%.
  — [LangGap PDF](https://arxiv.org/pdf/2603.00592)
- Per suite, original vs extended:

  | Model | Spatial | Goal | Object | All extended |
  |---|---|---|---|---|
  | pi0.5 | 97.0 → 5.9 | 97.0 → 30.0 | 100.0 → 37.7 | 21.4 |
  | pi0 | — | — | — | 8.6 |
  | pi0-FAST | — | — | — | 2.7 |
  | SmolVLA | — | — | — | 6.4 |

  pi0.5's 37.7% on LIBERO-Object is the closest published analogue of our 43%. — [LangGap PDF](https://arxiv.org/pdf/2603.00592)
- Diagnosis in the paper: LIBERO "assign[s] only one task per layout … the same visual input always corresponds to the same action output." In a same-scene benchmark with k tasks per scene, a language-blind policy can reach at most 1/k. — [LangGap PDF](https://arxiv.org/pdf/2603.00592)

**LIBERO-CF, "When Vision Overrides Language" (Fang et al., UNC, Feb 2026; v2 Jul 2026)**
- The benchmark assigns alternative feasible instructions to existing LIBERO layouts. It has four suites: CF-Spatial (15 tasks), CF-Object (10), CF-Long (10) and CF-OOD (15, with unseen objects). Scenes are visually identical to training; only the instruction differs. — [arXiv 2602.17659](https://arxiv.org/html/2602.17659)
- Grounding rate / success on counterfactual instructions:
  - OpenVLA-OFT: 4.7% / 0.9%. It executes the trained task instead 83.6% of the time ("biased execution").
  - pi0: 28.5% / 9.6%.
  - pi0.5: 30.8% / 13.2%.

  Vision-only inputs preserved performance, while language-only inputs collapsed to near zero. — [arXiv 2602.17659](https://arxiv.org/html/2602.17659)

**Mechanistic study, "Not All Features Are Created Equal" (Grant, Zhao, Wang, Mar 2026)**
- Six VLAs (80M-7B, including pi0.5, SmolVLA, GR00T and X-VLA), over 394k episodes, analysed with activation injection, SAEs and linear probes.
- Injecting baseline activations into null-prompt episodes "recovers near-identical behavior."
- X-VLA on libero_goal drops from 94% to 10% under wrong prompts (multiple goals share the scene). On libero_object it succeeds 60-100% regardless of the prompt (the layout uniquely specifies the task).
- Under cross-task injection, 99.8% of X-VLA episodes follow the source trajectory ("spatially bound motor programs").
— [arXiv 2603.19233](https://arxiv.org/abs/2603.19233)

**LIBERO-Para (Kim et al., Mar 2026): paraphrase robustness**
- 4,092 paraphrased LIBERO-Goal instructions along an action axis and an object axis. Original → paraphrased success:
  - OpenVLA-OFT 97.9 → 64.7
  - pi0.5 97.6 → 71.4
  - Xiaomi-Robotics-0 98.8 → 76.0
  - VLA-Adapter 98.2 → 46.3
- Object synonym substitution (e.g., "stove" → "range") costs 19.8-51.0 pts. The authors read this as "surface-level keyword matching."
- 79.5-95.5% of failures are planning-level: the model picks the wrong task, not a poorly executed right one.
- 4× more task diversity did not improve robustness.
— [LIBERO-Para HTML](https://arxiv.org/html/2603.28301v1)

**LIBERO-X (Wang et al., Feb 2026; RSS 2026)**
- Five-level hierarchy: Level 3 swaps target and distractor positions, Level 4 changes visual attributes and adds unseen objects, Level 5 reformulates instructions.
- Success by level, Level 1 → 5:
  - pi0.5: 65.2 / 53.2 / 36.0 / 24.1 / 18.0
  - pi0: 29.4 → 5.1
  - OpenVLA-OFT: 29.0 → 4.2
  - X-VLA: 30.1 → 4.1
  - GR00T-N1.5: 43.3 → 9.7
- pi0.5 reaches 28.9% on confounder objects and 7.2% on unseen objects.
- LIBERO-X training data averages 6 tasks per scene, against 2.6 in original LIBERO (2,520 demonstrations, 600 tasks, 100 scenes).
— [LIBERO-X HTML](https://arxiv.org/html/2602.06556v1); [RSS 2026 listing](https://roboticsconference.org/program/papers/97/)

**Other diagnostics**
- "Robust Skills, Brittle Grounding" (Emukpere et al., Feb 2026): multi-object picking with SmolVLA and pi0.5. "Execution of the manipulation primitive remains substantially more reliable than instruction-conditioned task success." The authors propose decomposed metrics: primitive success vs grounded success. — [arXiv 2602.24143](https://arxiv.org/abs/2602.24143)
- DeLock / "Breaking Lock-In" (Huang et al., Stanford, Apr 2026): after low-data post-training of pi0.5-BASE, policies show "concept lock-in" (they fixate on trained objects) and "spatial lock-in" (they stay anchored to trained targets). The RETAIN baseline gets 0-6/20 on novel-prompt tasks. — [arXiv 2604.23121](https://arxiv.org/html/2604.23121)
- ReSteer (Chen et al., Mar 2026): LIBERO multitask policies are poorly steerable by new instructions mid-execution. Steerability correlates with "limited overlap among training task trajectory distributions." — [arXiv 2603.17300](https://arxiv.org/abs/2603.17300)
- LangForce (Jan 2026): a vision-only policy reaches 44.6% on RoboCasa against 47.8% for the language-conditioned policy. On LIBERO-Goal, where scenes are ambiguous, vision-only falls to 9.8% against 97.4%. — [LangForce HTML](https://arxiv.org/html/2601.15197)
- Knowledge Insulation (Driess et al., Physical Intelligence, May 2025): "π0 … ignores the command and grasps a piece of trash instead." The paper states that "a common limitation of many robot policies is that they pay much more attention to images than the language input." — [arXiv 2505.23705](https://arxiv.org/pdf/2505.23705)

### Inferences
- Our two results line up with the literature. "10/10 picks the trained instruction's first object" matches LangGap's Change Target = 0% and LIBERO-Plus's goal replacement "nearly zero." "43% on swapped LIBERO-Object names" sits between LangGap's pi0.5 Object-extended result (37.7%) and the X-VLA libero_object "60-100% regardless of prompt."
- The split between the two follows a single rule, stated in the mechanistic study and in LangGap: when the scene uniquely identifies the trained task, language is ignored. pi0.5 keeps partial object-noun grounding from its VLM pretraining, which is why it scores above 0% on object swaps but exactly 0% on target swaps.
- The "Robust Skills, Brittle Grounding" decomposition is directly useful for our evaluation. Report "picked the named object" separately from "completed the task," as our 10/10 first-object metric already partly does.

### Gaps
- LIBERO-PRO reports no success rates for the "xxxx" prompt or object-replacement cases; they appear only as qualitative figures.
- I found no paper that runs our exact protocol: a LIBERO-90 scene, a frozen 40-task pi0.5, and a new instruction in a scene trained on one instruction. LangGap and LIBERO-CF are the nearest matches but use the 40-task suites' own layouts.
- I could not locate a paper literally titled "Do VLAs follow instructions?". The closest are LIBERO-CF ("When Vision Overrides Language") and the mechanistic study above.

## Q1b. Why it happens: explanations (dataset structure, information collapse, gradient interference)

### Takeaway
The dominant explanation is information-theoretic. Goal-driven collection pairs each scene with essentially one instruction, so the image predicts the instruction (H(ℓ|v) ≈ 0). Language then carries no extra information about the action, and maximum-likelihood training learns p(a|v). This has been named "information collapse" (LangForce), "posterior collapse" (CAST), "modality imbalance" (BayesVLA), "modality collapse" (RSS), "shortcut learning" (Xing et al.) and "lock-in" (DeLock). Low linguistic diversity in the datasets and gradient interference from a freshly initialized action expert make it worse.

### Cited Findings
- **LangForce:** "In goal-driven datasets, the deterministic mapping v→ℓ implies H(ℓ|v)≈0," so the conditional mutual information between instruction and action vanishes and the model "degenerate[s] into vision-only policies." — [LangForce arXiv 2601.15197](https://arxiv.org/abs/2601.15197); [HTML](https://arxiv.org/html/2601.15197)
- **CAST:** "When language is vague or not diverse, there is often a stronger correlation between observations and actions … the policy learns to ignore language, instead inferring the task and actions purely from observations" (posterior collapse). — [CAST arXiv 2508.13446](https://arxiv.org/html/2508.13446); [project page](https://cast-vla.github.io/)
- **BayesVLA:** "language diversity is much lower than visual and action diversity," which produces visual shortcuts and forgotten language understanding. — [BayesVLA arXiv 2512.11218](https://arxiv.org/abs/2512.11218)
- **RSS:** "strong visual priors overwhelm sparse linguistic signals, causing agents to overfit to specific instruction phrasings while ignoring the underlying semantic intent" (modality collapse). — [arXiv 2601.04052](https://arxiv.org/abs/2601.04052)
- **Shortcut learning in generalist policies** (Xing et al., Aug 2025): shortcut learning (reliance on task-irrelevant features such as viewpoint or background) is driven by (1) limited diversity within sub-datasets and (2) distribution gaps across sub-datasets (fragmentation). Targeted augmentation reduces it. — [arXiv 2508.06426](https://arxiv.org/abs/2508.06426)
- **Dataset audit** (Wanna et al., ACL 2026): VLA corpora contain "highly repetitive, template-like commands with limited structural variation." — [arXiv 2601.03136](https://arxiv.org/abs/2601.03136). LangGap cites this audit as finding that "fewer than 2% of instructions are linguistically unique" ([LangGap PDF](https://arxiv.org/pdf/2603.00592)). The abstract I read did not state the 2% figure itself.
- **LIBERO structure:** LIBERO "assign[s] only one task per layout" ([LangGap](https://arxiv.org/pdf/2603.00592)). It averages 2.6 tasks per scene against LIBERO-X's 6 ([LIBERO-X](https://arxiv.org/html/2602.06556v1)).
- **Gradient interference** (Knowledge Insulation, PI): "naive training with such a randomly initialized action expert harms the models' ability to follow language commands (presumably due to gradient interference)." Stop-gradient and/or co-training with VLM data restores language following. Even then it is "still far from perfect, likely because correlations in the training data still cause the model to sometimes ignore language instructions." — [arXiv 2505.23705](https://arxiv.org/pdf/2505.23705)
- **Representation drift** (Anchor-Align, Jul 2026): standard behaviour cloning "progressively overwrites pretrained representations during finetuning." — [arXiv 2607.13429](https://arxiv.org/abs/2607.13429)
- **Secondary source:** an industry blog summarizes the same information-collapse argument ("if the instruction is predictable from the image … language carries no information"). — [Label Studio blog](https://labelstud.io/blog/why-your-vla-model-ignores-language-and-the-cheapest-fix-in-robotics/)

### Inferences
- Our LIBERO-90 living room is the extreme case: k = 1 trained instruction in that scene, so H(ℓ|v) = 0 for that layout, and the policy runs the memorized program. The LIBERO-X and LangGap numbers on tasks per scene work as a direct "instruction coverage per scene" statistic, the robot counterpart of caption coverage.

### Gaps
- No paper I found measures how instruction following scales with the number of distinct instructions per scene in a controlled sweep. LangGap's 1 / 6 / 16 / 45 / 56-task series is the nearest, and it confounds scene count with task count.

## Q2. Fixes: which work without retraining (inference-time), and which need retraining or new data?

### Takeaway
Inference-time fixes exist, but their gains are modest. CFG-style contrast of a language-conditioned against a null-prompt (or trained-prompt) prediction gives +3.6 to +12 pts, and visual-input editing (BYOVLA) also helps. The largest gains come from retraining:
- an explicit vision-only prior with a likelihood-ratio objective (LangForce, BayesVLA, CAG-VA);
- counterfactual or same-scene instruction data (CAST, CounterAlign, LangGap, DIAL);
- object-centric conditioning such as boxes, traces or masks (Point-VLA, MolmoAct, OBEYED-VLA).

Same-scene augmentation helps at small scale but gets diluted at scale (LangGap).

### Cited Findings

**A. Inference-time, no weight updates**

- **Counterfactual Action Guidance, CAG** (LIBERO-CF paper, 2026)
  - Mechanism: π_CAG = π(a|o,∅) + ω·(π(a|o,l) − π(a|o,∅)), i.e., the posterior is reweighted as P(a|o)·P(l|a,o)^ω. The paper says it "adapts classifier-free guidance principles from diffusion models."
  - Training-free variant (TF, the same VLA with language dropped for the unconditional branch), on pi0.5: grounding 30.8 → 40.5% (+9.7), success 13.2 → 16.8% (+3.6).
  - Variant with a separately trained vision-action prior (VA, 30-50k steps): grounding 46.3% (+15.5), success 21.7% (+8.5).
  - Real robot: 9.4% fewer counterfactual failures, +17.2% task success.
  — [arXiv 2602.17659](https://arxiv.org/html/2602.17659)
- **Residual Affordance Steering, RAS** (part of RSS, Zhan et al., Jan 2026)
  - Mechanism: π̃ ∝ exp(s(a|o,∅) + γ·[s(a|o,l) − s(a|o,∅)]) with γ ≈ 1.25-1.5. It needs two forward passes, the same weights and no retraining.
  - pi0 on LIBERO "destructive instruction overwriting": 52.37 → 64.46% with RAS alone.
  - With MCSI added (which does require training on LLM paraphrases): pi0 82.22%; pi0.5 75.90 → 86.98%; pi0.5 on LIBERO-Plus 81.4 → 90.0%.
  — [RSS HTML](https://arxiv.org/html/2601.04052); [abs](https://arxiv.org/abs/2601.04052)
- **Contrastive Prompt Guidance, CPG** (DeLock, Stanford, Apr 2026)
  - Mechanism: v = v(o,τ⁻) + w·(v(o,τ⁺) − v(o,τ⁻)). The negative prompt τ⁻ is the trained prompt and τ⁺ is the novel prompt. It needs no language dropout and no auxiliary model.
  - Full DeLock (CPG plus visual-encoder weight-drift L2 regularization during post-training) on pi0.5-BASE: 11-19/20 on novel-prompt tasks against 0-6/20 for RETAIN, matching or beating pi0.5-DROID.
  — [arXiv 2604.23121](https://arxiv.org/html/2604.23121)
- **BYOVLA** (Princeton, Oct 2024)
  - Mechanism: a run-time observation intervention. A VLM flags task-irrelevant regions, sensitivity probing finds which regions the VLA reacts to, and those regions are inpainted.
  - No fine-tuning or weight access needed; +20-40% task success under distractors.
  - It targets visual distractors, not language directly.
  — [arXiv 2410.01971](https://arxiv.org/abs/2410.01971); [project page](https://aasherh.github.io/byovla/)
- **Activation steering** (Häon et al., Sep 2025)
  - Mechanism: FFN activations are projected onto the token-embedding basis to find sparse semantic directions (e.g., speed, direction), which are injected at inference. Applied to pi0 and OpenVLA on LIBERO and a UR5.
  - It steers motion attributes, not object choice.
  — [arXiv 2509.00328](https://arxiv.org/abs/2509.00328)
  - A related SAE study (Swann et al., Mar 2026) finds that amplifying semantic SAE features "induces behaviors consistent with their meanings." It proposes a metric separating "general transferable primitives" from "episode-specific memorizations." — [arXiv 2603.19183](https://arxiv.org/abs/2603.19183)
- **Hierarchical high-level planners** (inference-time on top of the policy, though the high level is usually trained)
  - Hi Robot (PI, Feb 2025): a VLM reasoner emits subgoal language for a pi0 executor. It averages over 40% higher instruction accuracy than a GPT-4o high level and beats flat pi0. Its high level is trained on VLM-synthesized instructions. — [arXiv 2502.19417](https://arxiv.org/html/2502.19417v1)
  - pi0.5's own explicit high-level subtask inference beats both implicit and no high-level inference. GPT-4 zero-shot as the high level performs worst. — [pi0.5 arXiv 2504.16054](https://arxiv.org/html/2504.16054v1)

**B. Retraining or new data required**

- **LangForce** (Jan 2026; formerly "BayesianVLA")
  - Mechanism: a dual-branch model with 64 learnable latent action queries. The same VLM computes a vision-only prior p(a|v) (causal masking hides ℓ) and a posterior π(a|v,ℓ). It maximizes the log-likelihood ratio log p(ℓ|a,v) − log p(ℓ|v), with no new data.
  - Results vs the QwenGR00T baseline:
    - SimplerEnv: 66.5% vs 55.2% (+11.3)
    - RoboCasa: 52.6% vs 47.8%
    - LIBERO: 98.4% vs 96.5%
    - Real robot, OOD: 9/30 vs 2/30
  - The paper does not discuss CFG.
  — [HTML](https://arxiv.org/html/2601.15197); [code](https://github.com/ZGC-EmbodyAI/BayesianVLA)
- **BayesVLA** (Dec 2025)
  - Mechanism: a visual-action prior ("seeing-to-act") plus a language-conditioned likelihood ("prompting-to-specify"), with pre- and post-contact phases.
  - Claims better generalization to unseen instructions, objects and environments. The abstract gives no numbers.
  — [arXiv 2512.11218](https://arxiv.org/abs/2512.11218)
- **CAST** (Glossop, Chen, Bhorkar, Shah, Levine, Aug 2025)
  - Mechanism: Gemini 2.5 Pro proposes counterfactual instructions at points along existing trajectories, and matching atomic-command action chunks are synthesized and appended.
  - Manipulation with distractors: 35/60 vs 19/60 ("doubling").
  - Navigation (27 tasks): +27% over a standard VLA, e.g., continuous navigation 3.1/5 vs 0.2/5.
  — [HTML](https://arxiv.org/html/2508.13446); [project page](https://cast-vla.github.io/)
- **CounterAlign** (Kondoh, Ota, Kanezaki, Wu, Aug 2026)
  - Mechanism: counterfactual supervision on pi0.5. Instructions and/or actions are relabelled into mismatched tuples, a GAIL-style discriminator is combined with a PU-learning relabeling discriminator, and the result serves as reward for offline RL.
  - LIBERO-PRO: Object-Pos 0.19 → 0.51; Goal-Task 0.17 → 0.46; Spatial-Pos 0.53 → 0.60.
  - Real robot: e.g., 0.05 → 0.20.
  — [arXiv 2608.21740](https://arxiv.org/html/2608.21740)
- **CAG-VA:** CAG with a separately trained vision-only prior (numbers in Section A above). — [arXiv 2602.17659](https://arxiv.org/html/2602.17659)
- **LangGap same-scene augmentation** (pi0.5 + LoRA, about 150 scripted demos per new task)
  - Single task: 3.75 → 90%.
  - 6 tasks (1 original + 5 extended): 0 → 28% on the extended tasks. Adding the 40 original tasks dilutes this to 4%.
  - 16 extended tasks only: 26.2 → 6.2%. With 40 original added (56 tasks): 27.5%, and original-task success falls 93.8 → 85.5%.
  - Change Target only reaches about 5-6% after training.
  - Authors' conclusion: "merely adding multi-semantic-perturbation data under the same visual layout has limited effectiveness."
  — [LangGap PDF](https://arxiv.org/pdf/2603.00592)
- **DIAL**, the classic VLM relabeling method (Xiao et al., RSS 2023, pre-2024 but foundational)
  - Mechanism: a CLIP model fine-tuned on robot data proposes hindsight instructions for 80k episodes, of which only 3.5% had crowd-sourced labels, yielding 38,516 relabels.
  - 60 novel instructions: 60.0% vs 47.5% for the best baseline (RT-1 + IL); semantic tasks 30.8% vs 15.4%.
  — [RSS 2023 PDF](https://www.roboticsproceedings.org/rss19/p029.pdf)
- **"See Less, Specify More", S2** (Wu, Matsushima, Ota, Jun 2026)
  - Mechanism: an explicit visual-evidence budget during training forces the executor to act from task-sufficient evidence, and trajectories are relabelled with refined trajectory- and subtask-level language.
  - Real robot (8 tasks, TX-G2 and HSR): mean subtask success 54.2 → 79.0% over pi0.5.
  — [arXiv 2606.02735](https://arxiv.org/abs/2606.02735)
- **Dense re-annotation, DeMiAn** (May 2026)
  - Mechanism: Qwen3-VL relabels demonstrations along motion, scene, pose and reasoning aspects, and a 2B instructor selects annotations at inference.
  - RoboCasa365: 44 → 49% (oracle 52%). On MolmoSpaces it matches the baseline with about 62% less compute.
  — [arXiv 2605.17077](https://arxiv.org/html/2605.17077)
- **LIBERO-Plus perturbation training** (20k+ trajectories): +11.5 overall, mostly on visual axes (camera 55.6 → 92.8). It does not target language. — [LIBERO-Plus](https://arxiv.org/html/2510.13626v1)
- **Co-training and knowledge insulation** (PI)
  - Stopping gradients from the action expert, or co-training with VLM data, improves language following over pi0 and joint training. Freezing the backbone is "not viable."
  - VLM data matters most for following instructions on OOD objects. Exact rates appear only in figures. A secondary search summary quoted 85% vs 69%, which I could not verify in the text.
  — [arXiv 2505.23705](https://arxiv.org/pdf/2505.23705)
  - pi0.5: language-following rate (how often the robot picks the commanded object) is about 85% in-distribution and about 70% OOD with 104 training locations. Removing web data mainly hurts OOD objects. — [pi0.5 arXiv 2504.16054](https://arxiv.org/html/2504.16054v1)
- **Anchor-Align** (Jul 2026)
  - Mechanism: layer-wise distillation from a frozen VLM copy, plus language-action alignment that converts actions into discrete motion-direction labels trained on the same observations.
  - Consistent gains over behaviour cloning on LIBERO-PRO, LIBERO-Plus and CALVIN. The abstract does not give numbers.
  — [arXiv 2607.13429](https://arxiv.org/abs/2607.13429)
- **Object-centric conditioning**
  - **Point-VLA** (Dec 2025): a bounding box drawn on the first frame, co-trained 1:1 with text-only data, on a pi0.5 backbone. Six real tasks average 92.5% against 32.4% for text-only and 40.0% for Interleave-VLA; OOD objects 92.5% vs 57.5%. — [arXiv 2512.18933](https://arxiv.org/html/2512.18933v1)
  - **OBEYED-VLA** (Dec 2025): a frozen Qwen3-VL set-of-mark stage selects instruction-relevant regions and masks distractors, and the VLA is fine-tuned on the result. With 4 distractors, pi0 goes from about 10% to about 85%. On absent-target rejection it reaches about 95% against 10-15%. — [arXiv 2512.22519](https://arxiv.org/html/2512.22519)
  - **MolmoAct** (Ai2, Aug 2025): steering with visual traces gives 75% success, 33% better than steering with open-ended language, and 29% better than pi0-FAST on language steering. — [arXiv 2508.07917](https://arxiv.org/pdf/2508.07917)
- **Paraphrase-specific fixes:** GSR/ParaVLA (Aug 2026) extracts task semantics separately and retrains the action expert, gaining up to +44.6% on LIBERO-Para. — [arXiv 2608.02497](https://arxiv.org/abs/2608.02497)
- **ReSteer:** a steerability estimator, synthesized motion segments and self-refinement give +11% steerability over 18k LIBERO rollouts. — [arXiv 2603.17300](https://arxiv.org/abs/2603.17300)

### Inferences
- For a frozen pi0.5, three cheap options exist: CAG-TF, RAS and CPG. All three compute two velocity fields per denoising step and extrapolate. CPG fits our setting best because the negative prompt is the trained instruction for that scene, which we know.
- The expected gains from these are modest: pi0.5 goes 30.8 → 40.5% grounding with CAG-TF, and DeLock's 0-6 → 11-19/20 includes training-time regularization. They are unlikely to turn 0% target-redirection into reliable following when the policy never saw the new object-goal pairing in that scene.
- Retraining methods that add a vision-only prior or counterfactual data give larger gains. LangGap shows that brute-force same-scene data saturates and dilutes.
- Object-centric conditioning (boxes, traces) sidesteps the language channel entirely. It amounts to replacing the instruction with a visual target, which matters for our bindings and program approach.

### Gaps
- None of the CFG-style methods report results on a frozen pi0.5 fine-tuned without language dropout specifically in LIBERO-90 scenes. Whether the null-prompt branch of the openpi pi05_libero checkpoint is meaningful is untested in the literature I found. CAG-TF reports it works on "standard VLAs."
- DeLock's ablation separating CPG alone from weight-drift regularization was not visible in what I read.
- RAS-alone numbers for pi0.5 were not captured (only pi0's +12.1).
- I found no robotics paper that applies Attend-and-Excite-style attention interventions on language tokens to fix object choice.
- An MDPI Entropy 2026 paper, "Restoring the Right Stream: Training-Free OOD Robustness for VLA Policies" ([link](https://www.mdpi.com/1099-4300/28/9/975)), returned HTTP 403 and was not read.

## Q3. Link to text-to-image prompt-following and compositional failures (attention, caption quality and coverage, recaptioning)

### Takeaway
The analogy is close in mechanism, but robotics papers rarely state it explicitly. In T2I, captions that omit objects, positions and counts produce models that ignore prompt words. Recaptioning with dense synthetic captions fixes much of this (DALL-E 3; SD3's GenEval two-objects 41.4 → 52.5 and position 6.5 → 18.0). Inference-time attention fixes such as Attend-and-Excite address "catastrophic neglect." The robot counterparts are VLM relabeling (DIAL, CAST, DeMiAn, S2) and CFG on language (CAG, RAS, CPG). Of the papers I read, only CAG explicitly cites the diffusion CFG lineage.

### Cited Findings
- **DALL-E 3** (Betker et al., OpenAI, 2023, pre-2024 but foundational)
  - Hypothesis: "Existing text-to-image models … often ignore words or confuse the meaning of prompts. We hypothesize that this issue stems from noisy and inaccurate image captions." Human captions "omit background details," notably "the presence of objects," "the position of objects … and the number," and colors and sizes.
  - Recipe: recaption with a bespoke captioner, train on 95% descriptive synthetic captions and 5% ground truth. A 65% blend was "far behind" the others, and higher blends "always improved" CLIP score. At inference, GPT-4 "upsamples" short prompts.
  - DALL-E 3 vs DALL-E 2 vs SDXL:

    | Metric | DALL-E 3 | DALL-E 2 | SDXL |
    |---|---|---|---|
    | DrawBench long | 81.0% | 52.4% | 51.1% |
    | T2I-CompBench color binding | 81.1% | 59.2% | 61.9% |

    The authors caution that DALL-E 3 has other improvements, so this is not a pure caption ablation.
  — [DALL-E 3 paper PDF](https://cdn.openai.com/papers/dall-e-3.pdf)
- **SD3** (Esser et al., Mar 2024), 50% original + 50% CogVLM captions vs original only:

  | GenEval | Original only | 50/50 mix |
  |---|---|---|
  | Overall | 43.27 | 49.78 |
  | Two objects | 41.41 | 52.53 |
  | Position | 6.50 | 18.00 |
  | Color attribution | 11.75 | 24.75 |
  | Counting | 33.44 | 41.56 |

  — [arXiv 2403.03206](https://arxiv.org/html/2403.03206)
- **Attend-and-Excite** (Chefer et al., SIGGRAPH 2023): "catastrophic neglect," where the model "fails to generate one or more of the subjects," plus attribute mis-binding. It is fixed at inference by optimizing cross-attention so that every subject token gets attention (Generative Semantic Nursing), with no retraining. — [arXiv 2301.13826](https://arxiv.org/abs/2301.13826)
- **Robot-side counterparts**
  - CAG explicitly "adapts classifier-free guidance principles from diffusion models." — [arXiv 2602.17659](https://arxiv.org/html/2602.17659)
  - RAS's γ-extrapolation is CFG-shaped. — [arXiv 2601.04052](https://arxiv.org/html/2601.04052)
  - DeLock's CPG is negative-prompt guidance. — [arXiv 2604.23121](https://arxiv.org/html/2604.23121)
  - CAST does not mention DALL-E 3 or T2I. — [CAST HTML](https://arxiv.org/html/2508.13446)
  - DeMiAn makes no explicit DALL-E 3 analogy. — [arXiv 2605.17077](https://arxiv.org/html/2605.17077)
- **"Coverage" in robotics:** LangGap's "one task per layout" and LIBERO-X's 2.6 vs 6 tasks per scene are the robot statistics that correspond to how many objects or relations a caption names per image. — [LangGap](https://arxiv.org/pdf/2603.00592); [LIBERO-X](https://arxiv.org/html/2602.06556v1)

### Inferences
- The mapping is direct:
  - A LIBERO scene with one instruction is like an image whose caption names one object.
  - The VLA learning p(a|v) is like a T2I model learning to ignore unnamed or rarely varied words.
  - Recaptioning and coverage correspond to counterfactual or same-scene relabeling (DIAL, CAST, LangGap, LIBERO-X).
  - Attend-and-Excite and CFG correspond to CAG, RAS and CPG.
- One difference matters. In T2I, recaptioning only needs new text for existing images. In robotics, a new instruction for the same scene also needs a new action trajectory. That is why CAST and LangGap must synthesize actions or script demonstrations, and why relabeling alone (DIAL-style hindsight) only adds coverage where the recorded trajectory actually accomplishes the relabelled goal.
- LangGap's dilution result (6-task 28% → 45-task 4%) has a T2I parallel in DALL-E 3's finding that low synthetic-caption blends (65%) underperform: the share of covered examples matters, not just their presence.

### Gaps
- I found no primary robotics paper that explicitly frames VLA language neglect as a caption-coverage problem in the DALL-E 3 sense, and none that runs a controlled "instructions per scene" sweep analogous to DALL-E 3's blending-ratio experiment. The DALL-E 3 parallel to robot relabeling appeared only in a search-engine summary, not in a primary paper.
- Recap-DataComp-1B and other 2024 recaptioning-at-scale papers were not fetched. SD3 is the 2024 primary data point used here.
