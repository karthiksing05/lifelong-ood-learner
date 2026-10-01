# Industry mechanisms for zero-shot generalization and continual improvement of robot foundation models (2024-2026)

Scope note: compiled 2026-10-01 from company blogs, arXiv technical reports and press coverage. "CLAIM (unverified)" marks a company statement with no independent replication. Where the numbers came from a figure that a summarizer read off, they are marked "approx." Each finding carries its URL.

---

## Physical Intelligence (pi0, pi0-FAST, Hi Robot, pi0.5, Knowledge Insulation, RTC, pi0.6 / pi*0.6 Recap, pi0.7): what is their recipe for generalizing to new tasks and for continual improvement?

### Takeaway
PI's recipe has four parts. (1) Pretrain one VLA on very diverse cross-embodiment robot data plus web vision-language data. (2) Protect the VLM's semantics while adding a continuous action expert (Knowledge Insulation). (3) Make the policy hierarchical and promptable: subtask text in pi0.5; in pi0.7, subtask text plus quality/speed metadata, control mode and generated subgoal images, so a new task can be "coached" step by step. (4) Improve from the robot's own experience with advantage-conditioned RL (Recap), then fold that mixed-quality autonomous data back into the next generalist. pi0.7 claims that one generalist matches the earlier per-task RL specialists. Unseen tasks still succeed only 60-80% of the time, against >90% on seen tasks.

### Cited Findings
**pi0 (Oct 31 2024)**
- The pi0 VLA uses a flow-matching action expert on a pretrained VLM. It is trained on single-arm, dual-arm and mobile-manipulator data, performs tasks zero-shot after pretraining, follows language "from people and from a high-level VLM policy", and acquires new skills via fine-tuning — [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)
- The open-source base checkpoints (pi0, pi0-FAST, pi0.5) are "pre-trained on 10k+ hours of robot data". The repo is Apache-2.0, plus the Gemma license — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- pi0-FAST is an autoregressive VLA built on the FAST action tokenizer — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)

**Hi Robot (Feb 26 2025, ICML 2025): hierarchical instruction following**
- A high-level VLM "reasons over complex prompts and user feedback to deduce the most appropriate next step", then issues low-level commands to the VLA. It handles situated feedback such as "that's not trash" mid-task, and was tested on single-arm, dual-arm and mobile robots (table cleaning, sandwich making, grocery shopping). The abstract page gives no exact numbers — [arXiv 2502.19417](https://arxiv.org/abs/2502.19417)

**pi0.5 (Apr 2025): co-training and hierarchical subtask prediction, open-world generalization to new homes**
- Data mix:
  - ~400 h of mobile-manipulator (MM) data in ~100 homes.
  - Multi-environment non-mobile arm data (ME).
  - Cross-embodiment lab data (CE), including Open X-Embodiment.
  - High-level subtask annotations (HL).
  - Web data (WD): captioning (CapsFusion, COCO), VQA (Cambrian-7M, PixMo, VQAv2) and object localization.
  - Verbal-instruction data (VI) in post-training.
  - 97.6% of first-phase training data is NOT target mobile-manipulator data — [arXiv 2504.16054](https://arxiv.org/html/2504.16054)
- Architecture: the model first predicts a high-level subtask in text (e.g. "pick up the plate"), then low-level actions conditioned on it. Pretraining uses FAST discrete action tokens. Post-training adds a flow-matching action expert, trained jointly with next-token prediction — [arXiv 2504.16054](https://arxiv.org/html/2504.16054)
- Evaluated in 3 real homes not seen in training. Performance rises steadily with the number of training locations (3, 12, 22, 53, 82, 104) — [arXiv 2504.16054](https://arxiv.org/html/2504.16054)
- "Language following rate" is defined as "how often the robot selects the object indicated in the language command". The "no WD" ablation is "not statistically significant" on standard tasks but "significantly worse" on out-of-distribution objects. Removing ME or CE data causes a "large degradation". Exact values appear only in figures — [arXiv 2504.16054](https://arxiv.org/html/2504.16054)
- The full pi0.5 high-level inference "outperforms even the human HL oracle baseline" (figure only) — [arXiv 2504.16054](https://arxiv.org/html/2504.16054)

**Knowledge Insulation (KI, May 29 2025)**
- Problem: naively adding a continuous action expert to a VLM "significantly harms both training speed and knowledge transfer". The authors attribute this to action-expert gradients corrupting the backbone. Fix:
  - stop gradient from the action expert into the VLM;
  - train the backbone on FAST tokens;
  - co-train with VLM data.
  - [arXiv 2505.23705](https://arxiv.org/abs/2505.23705)
- Language-following rate on the "items in drawer" task (approx., read from Fig. 4b): KI ~90%, pi0 ~40%, pi0-FAST ~85%, joint training without stop-gradient ~55%. pi0 needs "7.5 times as many training steps" to reach similar performance. Co-training on VLM data "is particularly important" for the novel-object "OOD follow rate" — [arXiv 2505.23705 HTML](https://arxiv.org/html/2505.23705)
- LIBERO results (Table 1), KI / pi0 / pi0-FAST:

  | Suite | KI | pi0 | pi0-FAST |
  |---|---|---|---|
  | LIBERO-90 | 96.0 | n/a | n/a |
  | Spatial | 98.0 | 96.8 | 96.4 |
  | Object | 97.8 | 98.8 | 96.8 |
  | Goal | 95.6 | 95.8 | 88.6 |
  | LIBERO-10 | 85.8 | 85.2 | 60.2 |

  These are in-distribution (train-on-suite) numbers — [arXiv 2505.23705 HTML](https://arxiv.org/html/2505.23705)
- The openpi pi0.5 is described as "upgraded π0 with improved generalization via knowledge insulation". The repo releases pi0.5 base, pi0.5-LIBERO and pi0.5-DROID checkpoints (PyTorch and pi0.5 release noted Sept 2025) — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)

**Real-Time Chunking (RTC, NeurIPS 2025): a test-time method**
- RTC is an inference-time algorithm and needs no retraining. It generates the next action chunk while the current one executes, freezes the actions guaranteed to execute and inpaints the rest. Tested on 12 simulated and 6 real bimanual tasks. It is "uniquely robust to inference delay" (e.g. lighting a match) — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339)

**pi0.6 / pi*0.6 with Recap: RL from experience (Nov 2025)**
- pi0.6 base model: Gemma 3 4B VLM backbone plus an 860M action expert, 50 Hz action chunks. Pretraining covers robot data from more platforms plus web VL co-training, trained end to end with the KI procedure — [arXiv 2511.14759](https://arxiv.org/html/2511.14759v1)
- Recap ("RL with Experience & Corrections via Advantage-conditioned Policies"):
  - Value function: a multi-task distributional value function (201 bins, smaller 670M VLM) predicts the negative number of steps to success.
  - Advantage conditioning: the policy gets the text input "Advantage: positive/negative", with the threshold at the 30th percentile, and the indicator is randomly dropped so classifier-free guidance can be applied.
  - Data: demonstrations, on-policy rollouts and teleoperated corrections; corrections are forced to "positive".
  - [arXiv 2511.14759](https://arxiv.org/html/2511.14759v1)
- Results:
  - Throughput "more than doubles" on diverse laundry and espresso, and failure rate "roughly halves" on the hardest tasks.
  - Success is in the 90%+ range except diverse laundry.
  - Recap beats PPO and AWR on throughput.
  - Two iterations give +50% throughput on t-shirts and 2x on box assembly.
  - A strict laundry variant reaches 97% with 600 trajectories per iteration.
  - [arXiv 2511.14759](https://arxiv.org/html/2511.14759v1)
- Specialists and forgetting: pi*0.6 is first fine-tuned on demonstrations for the target task, so the reported systems are per-task specialists. "Both the value function and policy are finetuned from the pre-trained checkpoint, rather than the policy and value function from the last iteration", which was "useful for avoiding drift over multiple iterations" — [arXiv 2511.14759](https://arxiv.org/html/2511.14759v1)
- CLAIM: 18 hours of espresso making, 50 novel laundry items folded in a new home, 59 factory boxes assembled — [PI blog via search snippet / X summary](https://x.com/TheHumanoidHub/status/1990585956269965743); primary page [pi.website/blog/pistar06](https://www.pi.website/blog/pistar06) returned HTTP 403 to the fetcher

**pi0.7 (Apr 16 2026, v2 Apr 24): a steerable generalist with compositional generalization**
- Architecture: a 5B VLA built from a 4B Gemma3 VLM (400M vision encoder), a MEM-style video-history encoder (up to 6 history frames per camera, up to 4 cameras at 448x448) and an 860M flow-matching action expert — [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- The prompt/context has four parts, each randomly dropped out during training:
  - (1) subtask instructions in language, from a high-level policy or a human coach;
  - (2) episode metadata: quality 1-5, speed binned in 500-step intervals, mistake true/false;
  - (3) multi-view subgoal images generated by a lightweight world model initialized from BAGEL (a 14B mixture-of-transformers image understanding/editing/generation model);
  - (4) control mode, joint or end-effector.
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- Data: demonstrations from many static and mobile, single-arm and bimanual platforms, plus:
  - autonomous data from "a large amount of policy evaluations";
  - human interventions;
  - lower-quality and failure episodes, and data collected by prior model versions;
  - open-source robot datasets and egocentric human video;
  - web data (object localization/attributes, VQA, text-only).
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- New tasks at test time:
  - Short-horizon unseen tasks run zero-shot from the prompt (scooping rice into a rice cooker, wiping headphones).
  - Long-horizon unseen tasks (air fryer, toasting a bagel) use "language coaching", where a human gives step-by-step verbal subtask commands.
  - Coaching episodes can then train a high-level policy that "can roughly match the performance of the coaching episodes...without collecting any additional teleoperation data".
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- Results:
  - Matches or exceeds the task-specific pi*0.6 RL specialists on espresso, box building and laundry, with higher throughput on laundry and boxes.
  - UR5e zero-shot cross-embodiment shirt folding: 85.6% task progress and 80% success, versus expert teleoperators (n=10, mean 375 h experience) at 90.9% and 80.6%.
  - 14 instruction-following scenarios in 4 unseen kitchens and 2 unseen bedrooms: "high absolute success", significantly above pi0.5 and pi0.6.
  - Complex referring instructions ("object I would use to eat soup"): significantly better, and better still with subgoal images.
  - "Reverse Bussing", a task contradicting training-set correlations, improves significantly. "Reverse Fridge to Microwave" needed generated subgoals.
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- Data scaling and diversity:
  - Without metadata, the model "can actually get worse when trained on larger, mixed-quality datasets". With metadata it "is able to continuously improve as we train on more data".
  - Removing the 20% of data with the highest task diversity substantially hurts unseen tasks.
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- Limitations PI states:
  - Seen tasks often exceed 90% success, while "unseen tasks or unseen task-robot combinations have success rates in the 60-80% range".
  - It is "practically difficult...to definitively determine which tasks are truly 'seen' or 'unseen'", and "the model may well be achieving generalization primarily by 'remixing' skills".
  - Dexterous tasks need classifier-free guidance on metadata (beta in {1.3, 1.7, 2.2}).
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- Independent scrutiny: The Decoder found that the air-fryer task had "two episodes in the training data in which a robot closes an air fryer" plus a structurally similar DROID air-fryer-drawer task. It argues the novelty framing is questionable — [The Decoder](https://the-decoder.com/physical-intelligence-shows-robot-model-with-llm-like-generalization-flaws-included/)
- Press coverage of the "figure out tasks it was never taught" claim: [TechCrunch Apr 16 2026](https://techcrunch.com/2026/04/16/physical-intelligence-a-hot-robotics-startup-says-its-new-robot-brain-can-figure-out-tasks-it-was-never-taught/); Levine's framing ("learn new tasks just from instructions"): [X](https://x.com/svlevine/status/2044840590261796895)

### Inferences
- PI's continual-improvement loop, as published, is batch and offline, not online:
  - deploy;
  - collect autonomous episodes and corrections;
  - label them (value or advantage in pi*0.6, quality/speed/mistake metadata in pi0.7);
  - retrain from the pretrained checkpoint (pi*0.6), or retrain the next generalist on everything (pi0.7).
  - The metadata and advantage tokens are what make it safe to train on mixed-quality and failure data without degrading the policy.
- The new-task mechanism moved over time: pi0 fine-tuned on demonstrations; pi0.5 added hierarchical subtask text; pi0.7 adds prompt steering (subtask coaching plus subgoal images), with no weight update needed for short tasks.
- The "coaching to high-level policy" result is the closest public industry analogue to learning a new composite task from language instructions without new motor data.
- Gemma3-4B / Gemma 3 4B is the backbone for both pi0.6 and pi0.7 according to the papers. Openpi's pi0.5 uses a PaliGemma-lineage backbone; the README fetch did not confirm this, so treat it as unverified.

### Gaps
- PI's own blog pages (pi.website) returned 403. Primary blog-level claims for pi*0.6 (18 h espresso etc.) are therefore via secondary snippets.
- No public per-object numbers for pi0.5 language following: they appear only in figures.
- The fetches did not establish whether pi0.6 or pi0.7 weights are public. The openpi README fetched lists only pi0, pi0-FAST and pi0.5.
- Not found: whether pi0.7's match to the RL specialists comes specifically from training on the specialists' rollouts.

---

## Google DeepMind (RT-2/RT-X lineage, AutoRT/SARA-RT/RT-Trajectory, Gemini Robotics 1.0, ER, On-Device, 1.5, ER 1.6, Gemini Robotics 2): how do they get instruction following and generalization?

### Takeaway
DeepMind builds the action model directly on Gemini, so web-scale semantics and multilinguality carry over. Generalization is benchmarked explicitly along visual, instruction, action and task axes. Since 1.5 they split work between a reasoning orchestrator (Gemini Robotics-ER) that decomposes tasks into short language steps and a "thinking" VLA that interleaves language reasoning with actions. Motion Transfer pools multi-embodiment data. New tasks and robots are handled by small-data fine-tuning: ≤100 demos (1.0), 50-100 demos (On-Device), <200 examples / a few hours for new embodiments (On-Device 2). Everything is closed; ER models are exposed via API and VLAs via trusted-tester programs.

### Cited Findings
**Pre-Gemini lineage (Jan 2024)**
- AutoRT: an LLM+VLM orchestrator over RT-1/RT-2 controllers with a safety "Robot Constitution". It ran up to 52 robots (20 at once) for 7 months and collected 77,000 trials across 6,650 unique tasks — [DeepMind blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)
- SARA-RT is "10.6% more accurate and 14% faster than RT-2" via linear attention — [DeepMind blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)
- RT-Trajectory conditions on a 2D gripper-trajectory sketch. It reached 63% on 41 unseen tasks against 29% for RT-2, a non-language task specification that beats language on novel tasks — [DeepMind blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)

**Gemini Robotics 1.0 and ER (Mar 2025)**
- Architecture: a cloud VLA backbone (<160 ms) plus a local action decoder, with ~250 ms end-to-end latency and 50 Hz effective control — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- Data: "thousands of hours" of ALOHA 2 teleoperation over 12 months, plus non-action data (web documents, code, image/audio/video) — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- Generalization benchmark: 85 tasks, of which 20% in-distribution, 28% visual, 28% instruction and 24% action generalization. Gemini Robotics outperformed baselines (pi0, multi-task diffusion) and is "non-zero where baselines fail catastrophically" — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- Instruction following: 25 instructions across 5 scenes, strong on "novel objects and fine-grained instructions (e.g., 'Place the toothpaste in the bottom compartment of the caddy')". It still works with instructions in a new language, where baselines fail catastrophically — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- Fast adaptation: ">70% success on 7 out of 8 tasks with at most 100 demonstrations", 100% on two tasks, and 100% on long-horizon lunch-box packing after specialization. Adaptation to bi-arm Franka gives 63% average — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- Gemini Robotics-ER as a controller: zero-shot ALOHA sim 53% vs 27% for Gemini 2.0 Flash, and 65% with 10-demo in-context learning — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)

**Gemini Robotics On-Device (June 2025)**
- The first Gemini VLA available for fine-tuning. It adapts "with as few as 50 to 100 demonstrations", ships an SDK with MuJoCo sim through a trusted-tester program, and has been adapted to Apptronik Apollo and Franka FR3 — [DeepMind blog](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

**Gemini Robotics 1.5 and ER 1.5 (Sept/Oct 2025)**
- The VLA "interleaves actions with a multi-level internal reasoning process in natural language" ("think before acting"). Motion Transfer (MT) is a training recipe for learning from heterogeneous multi-embodiment data (ALOHA, Bi-arm Franka, Apollo; "thousands of diverse tasks"). Exact MT details are not disclosed — [arXiv 2510.03342](https://arxiv.org/html/2510.03342)
- It consistently beats 1.0 across visual, instruction, action and task generalization, with "substantial gains in instruction, action, and task generalization" (figures only). It shows zero-shot transfer of skills seen only on another embodiment — [arXiv 2510.03342](https://arxiv.org/html/2510.03342)
- Agentic stack (ER 1.5 orchestrator + GR 1.5 VLA): ~80% progress on long-horizon tasks vs 44% for the thinking VLA alone. Planning errors fall from 25.5% to 9% and total failures from 44.5% to 22% — [arXiv 2510.03342](https://arxiv.org/html/2510.03342)
- The tech report does not discuss fast adaptation or N-demo fine-tuning — [arXiv 2510.03342](https://arxiv.org/html/2510.03342)

**Gemini Robotics-ER 1.6 (Apr 14 2026)**
- Better spatial reasoning, multi-view success detection and instrument reading. Instrument reading: ER 1.5 23%, Gemini 3.0 Flash 67%, ER 1.6 86%, ER 1.6 with agentic vision 93% — [MarkTechPost](https://www.marktechpost.com/2026/04/15/google-deepmind-releases-gemini-robotics-er-1-6-bringing-enhanced-embodied-reasoning-and-instrument-reading-to-physical-ai/) / [DeepMind blog](https://deepmind.google/blog/gemini-robotics-er-1-6/). This conflicts with [ResultSense](https://www.resultsense.com/news/2026-04-16-gemini-robotics-er-instrument-reading), which reports 98%.
- Integrated into Boston Dynamics Spot (Orbit AIVI-Learning) for gauge reading — [WinBuzzer](https://winbuzzer.com/2026/04/16/google-deepmind-gemini-robotics-er-1-6-autonomous-industrial-inspections-xcxwbn/)

**Gemini Robotics 2 family (July 2026)**
- The family has three models: a whole-body humanoid VLA, ER 2 (agent for multi-minute tasks) and On-Device 2. On-Device 2 adapts to new bi-arm embodiments "with just a few hours of adaptation time, typically with less than 200 examples". One checkpoint controls Apollo 2 (two hand types) and Franka Duo — [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- CLAIM numbers:
  - Apollo with Inspire hands: pick from table 68.4%, floor 45.7%, shelf 76.3%.
  - SharpaWave hands: unscrew bulb 92%, screw bulb 36%.
  - Franka Duo: precise insertion 89.6%, general pick/place 74.2%.
  - [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- Access: ER 2 in Google AI Studio and Gemini Enterprise Agent Platform (private preview); the VLA and On-Device to early-access partners via trusted testers — [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/). Release date: the fetched blog page says July 30 2026; a search snippet said July 28 — [SiliconANGLE Jul 30](https://siliconangle.com/2026/07/30/google-deepmind-debuts-gemini-robotics-2-model-series-humanoid-robots/)

### Inferences
- DeepMind's answer to instruction grounding is mainly "inherit it from Gemini" plus hierarchical decomposition: ER turns long or abstract goals into short, concrete VLA commands. RT-Trajectory's 63% vs 29% shows that a geometric (non-language) task spec helps on unseen tasks.
- Continual improvement is not described publicly as RL from deployment. The public mechanism is versioned retraining and small-demo fine-tuning.

### Gaps
- No public parameter counts, data hours for 1.5/2, or Motion Transfer details.
- No published forgetting analysis for On-Device fine-tuning.
- RT-2/RT-X numbers were not fetched in this pass.

---

## NVIDIA (GR00T N1 / N1.5 / N1.6, DreamGen / neural trajectories, Cosmos, data pyramid, Isaac)

### Takeaway
GR00T is an open dual-system VLA: a VLM "System 2" plus a flow-matching DiT "System 1". It is trained on a "data pyramid":
- real robot teleoperation at the top;
- simulation and video-model-generated "neural trajectories" in the middle;
- human video with latent-action pseudo-labels at the base.

Language following improved most when the VLM was frozen (N1.5: 46.6% to 93.3% on GR-1). NVIDIA itself states that multi-task language following and OOD task generalization "continue to be challenging". New tasks are handled by post-training (10K-30K steps), co-trained with pretraining data and DAgger when needed.

### Cited Findings
**GR00T N1 (Mar 2025)**
- Size: 2.2B parameters (1.34B in the VLM).
- System 2: Eagle-2 VLM using 12th-layer representations, with the text component frozen, at 10 Hz.
- System 1: flow-matching DiT producing 16-step action chunks at 120 Hz, 4 inference steps.
- [arXiv 2503.14734](https://arxiv.org/html/2503.14734)
- Data pyramid:
  - 88 h of in-house humanoid teleoperation;
  - human video (Ego4D, Ego-Exo4D, Assembly-101, EPIC-KITCHENS, HOI4D, HoloAssist, RH20T-Human) with VQ-VAE latent actions (LAPA-style);
  - 780,000 sim trajectories (≈6,500 h);
  - 827 h of neural (video-generated) trajectories from ~105k L40 GPU-hours (~10x augmentation of real data).
  - [arXiv 2503.14734](https://arxiv.org/html/2503.14734)
- Sim results at 100 demos (GR00T N1 vs Diffusion Policy): RoboCasa 32.1 vs 25.6, DexMimicGen 66.5 vs 56.1, GR-1 Tabletop 50.0 vs 32.7. On the real GR-1, N1 with 10% of the data reaches 42.6% vs Diffusion Policy's 10.2%, and is "only 3.8% lower than Diffusion Policy trained on the full dataset" — [arXiv 2503.14734](https://arxiv.org/html/2503.14734)

**GR00T N1.5 (2025)**
- Eagle 2.5 VLM, "frozen during both pretraining and finetuning", with better grounding (GR-1 grounding IoU 40.4, RefCOCOg-val 89.6, vs Qwen2.5-VL-3B 35.5 and 85.2). Adds a FLARE loss (future latent alignment, coefficient 0.2) that "unlocks the ability to learn from human videos" — [NVIDIA GEAR N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
- Real GR-1 results, N1 to N1.5: language following rate 46.6% to 93.3%, overall success 43.3% to 83.0% — [NVIDIA GEAR N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
- Sim results, N1 to N1.5: RoboCasa (30 demos) 17.4 to 47.5, Sim GR-1 0-shot 39.6 to 43.9. Novel-object zero-shot 0% to 15.0%, and 55.0% after post-training with human videos. DreamGen on 12 new-verb tasks: 13.1% to 38.3% — [NVIDIA GEAR N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)

**DreamGen (May 2025)**
- A 4-stage pipeline:
  - fine-tune a video world model on the embodiment;
  - prompt it to generate rollouts;
  - recover pseudo-actions via latent-action or inverse-dynamics models;
  - train the policy.
- A humanoid performs "22 new behaviors in both seen and unseen environments" with teleop data from only one pick-and-place task in one environment. DreamGen Bench correlates video-model quality with downstream policy success — [arXiv 2505.12705](https://arxiv.org/abs/2505.12705)

**GR00T N1.6 (released Dec 15 2025 per GitHub/HF snippets)**
- Internal Cosmos-2B VLM variant; DiT twice as large (32 layers vs 16). Drops N1.5's 4-layer adapter and instead unfreezes the top 4 VLM layers in pretraining. Predicts state-relative action chunks — [NVIDIA GEAR N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- Pretraining adds "several thousand hours" of teleop data (YAM, AGIBot Genie-1, simulated Galaxea R1 Pro, Unitree G1 loco-manipulation). Post-training: 10K-30K steps at batch ≤1K, with "stronger state regularization, additional data augmentations, and co-training with pretraining data", plus DAgger when needed. No success percentages are published on the page — [NVIDIA GEAR N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- NVIDIA's own caveat: "Multi-task language following and out-of-distribution task generalization continue to be challenging for current VLA models" — [NVIDIA GEAR N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- Open weights on Hugging Face and code on GitHub — [Isaac-GR00T releases](https://github.com/NVIDIA/Isaac-GR00T/releases)

**2026**
- The Isaac GR00T Reference Humanoid was announced June 1 2026 at GTC Taipei: Unitree H2 Plus chassis, Sharpa Wave hands, Jetson AGX Thor T5000, available late 2026 — [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-open-humanoid-robot-reference-design)

### Inferences
- NVIDIA's strongest documented lever for language following is VLM preservation: frozen Eagle 2.5 in N1.5, then partial unfreezing in N1.6. This is the same lever as PI's KI. For new behaviors, their lever is synthetic data from world models, not prompting.
- Their forgetting control during task adaptation is co-training with pretraining data in post-training (N1.6).

### Gaps
- N1.6 success rates were not given on the page.
- No public evidence of fleet or online RL in GR00T.
- Isaac Lab specifics were not fetched.
- No search confirmed a "GR00T N2" model.

---

## Figure, Tesla, TRI, Boston Dynamics, 1X, Skild AI, Agility, Covariant, Dyna, Generalist, ByteDance Seed

### Takeaway
- Hierarchy is common: Figure Helix S2/S1/S0 and Gemini ER+VLA, while BD/TRI use one flat language-conditioned diffusion policy.
- The new data trend is human-centric: egocentric human video (Figure Go-Big, Tesla, 1X, Skild) and wearables (Generalist: 500k h, no robot data in pretraining).
- New tasks are handled by short task-specific post-training: ~1 h of robot data (Generalist, Skild), 3-5x fewer demos than single-task (TRI), 0.7% of data in <2 weeks (Dyna).
- Two notable exceptions: Skild S1 does in-context imitation from one human video with frozen weights, and 1X derives actions from a text-conditioned video world model.
- Continual improvement is mostly "deploy, collect, then retrain": reward-model-in-the-loop (Dyna), teleop "Expert" sessions as training data (1X), RL fine-tuning for speed (Generalist).
- Nearly all of this is closed and company-claimed. TRI's LBM paper is the rigorous exception, and it finds that the non-finetuned generalist is NOT good zero-shot.

### Cited Findings
**Figure (Helix)**
- Helix (Feb 2025):
  - System 2 is a "7B-parameter open-source, open-weight VLM" at 7-9 Hz; System 1 is an "80M parameter cross-attention encoder-decoder" at 200 Hz over a 35-DoF upper body.
  - Trained on "~500 hours" of teleop, "<5%" of prior VLA datasets.
  - An auto-labeling VLM writes hindsight instructions ("What instruction would you have given the robot to get the action seen in this video?").
  - One set of weights with no task-specific heads. CLAIM: handled "thousands of novel items in clutter".
  - [Figure Helix](https://www.figure.ai/news/helix)
- Helix 02 (Jan 27 2026): adds System 0, a 10M-parameter network at 1 kHz for balance and contact, trained in sim over 200,000+ parallel environments with "over 1,000 hours of joint-level retargeted human motion data". Adds palm cameras and fingertip tactile sensing (3 g). CLAIM: a 4-minute dishwasher sequence of 61 loco-manipulation actions without resets — [Figure Helix 02](https://www.figure.ai/news/helix-02). The CLAIM of 8-hour autonomous shifts is via [Interesting Engineering](https://interestingengineering.com/ai-robotics/figure-helix02-humanoid-robots-8-hour-shifts).
- Project Go-Big (Sept 2025). CLAIM: "zero-shot human video-to-robot transfer". Navigation from language ("go to the fridge") was trained on 100% egocentric human video from Brookfield homes, with no robot demos — [Figure Go-Big](https://www.figure.ai/news/project-go-big)

**Tesla Optimus (closed; no technical publications)**
- May 2025, Milan Kovac: Optimus "can now transfer a big chunk of the learning directly from human videos to the bots (1st person views for now)". Skills are triggered by natural language via one multitask network. Next steps are third-person internet video and RL — [TeslaNorth](https://teslanorth.com/2025/05/21/tesla-shows-off-new-optimus-robot-skills-learned-from-human-videos/); [Quartz](https://qz.com/elon-musk-tesla-optimus-robot-learns-from-videos-1851781684)
- Late June 2025: Tesla shifted from motion-capture suits and teleoperation to primarily video of workers, who wear five-camera helmets. Kovac departed and Ashok Elluswamy took over — [Business Insider via Yahoo](https://finance.yahoo.com/news/tesla-shifted-optimus-training-strategy-092502233.html)
- No success rates, data hours or architecture details are public. All Optimus capability statements are company claims.

**Toyota Research Institute: Large Behavior Models (Jul 2025)**
- Architecture: a diffusion transformer (8 layers, 768-d) with fine-tuned CLIP ViT-B/16 vision and a frozen CLIP text encoder. It predicts 16 x 20-D actions and executes 8 at 10 Hz — [arXiv 2507.05331](https://arxiv.org/html/2507.05331)
- Data: ~1,695 h in total. TRI-Ramen is ~545 h across 532 tasks and 64,262 demos (468 h real, 45 h sim, 32 h UMI); OXE-Ramen is ~1,150 h — [arXiv 2507.05331](https://arxiv.org/html/2507.05331)
- Findings:
  - Fine-tuned LBMs need "less than 30% of the data" of from-scratch training, and "with only 15% of the data" statistically beat the single-task baseline. Press summary: 3-5x less data.
  - The pretrained (non-finetuned) LBM is "statistically indistinguishable from single-task in simulation" and "performs worse than single-task baseline" in real nominal conditions, and was not expected to solve unseen tasks zero-shot.
  - Negative findings: one fine-tuned task got worse (0.335 success), a data-normalization bug occurred during pretraining, and low-motion filtering helped single-task but hurt the LBM.
  - [arXiv 2507.05331](https://arxiv.org/html/2507.05331); [The Robot Report](https://www.therobotreport.com/tri-pretrained-large-behavior-models-accelerate-robot-learning/)
- Eval rigor: 50 real rollouts per task per policy per condition, 200 sim rollouts, blind evaluators, Compact Letter Display at 95%, Bayesian Beta/Dirichlet posteriors. A rubric QA pass on ~27% of ~2,700 real rollouts found 6.25% discrepancies — [arXiv 2507.05331](https://arxiv.org/html/2507.05331)
- Language: scenarios B and K test language conditioning, but there is "no explicit analysis of whether model actually follows language" (summarizer reading of the paper) — [arXiv 2507.05331](https://arxiv.org/html/2507.05331)

**Boston Dynamics + TRI: Atlas LBM (2025)**
- A 450M DiT trained with a flow-matching loss at 30 Hz with 48-step (1.6 s) chunks. Inputs are images, proprioception and language. One multi-task policy is trained on Atlas, Atlas MTS and TRI-Ramen data.
- New reactive behaviors are added by demonstrating recoveries and retraining, with "no algorithmic changes".
- Policies were sped up 1.5x-2x at inference without significant performance loss.
- [Boston Dynamics blog](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)

**1X (NEO; Redwood; 1X World Model)**
- Redwood (June 10 2025): a 160M-parameter vision-language transformer with a diffusion-policy decoder, running onboard at ~5 Hz. It uses pretrained language embeddings and auxiliary "cognitive" targets (hand location, object localization). It is trained on teleop and autonomous episodes, learning "from both successful and failure rollouts" — [1X Redwood](https://www.1x.tech/discover/redwood-ai)
- World model for evaluation (June 16 2025): it predicts future latents plus final-state value. "A World Model with 70% accuracy can accurately predict the better policy with 90% success" given a 15% true performance gap — [1X World Model](https://www.1x.tech/discover/redwood-ai-world-model); [tech report PDF](https://www.1x.tech/1x-world-model.pdf)
- 1XWM as policy (Jan 12 2026):
  - Model: a 14B text-conditioned video diffusion backbone mid-trained on 900 h of egocentric human video and fine-tuned on 70 h of NEO data. An inverse-dynamics model trained on 400 h of robot data converts generated frames to actions.
  - Speed: 11 s per generation for 5 s of video.
  - Success (30 trials each): grab chips ~80%; pull tissue 30% with 1 generation and 45% with 8 generations (best-of-N test-time compute); scrub dish ~20%.
  - Limitations they state: weak 3D grounding, dexterous tasks, physics-violating generations. Research stage.
  - [1X World Model self-learning](https://www.1x.tech/discover/world-model-self-learning)
- NEO consumer product: autonomous by default. For unknown chores a "1X Expert" teleoperates the robot, and these sessions help NEO learn — [The Robot Report](https://www.therobotreport.com/1x-announces-pre-order-launch-neo-humanoid-robot/). Secondary reviews: "every teleoperated task is recorded, labeled, and fed into" Redwood, and users can opt out — [review blog, secondary](https://www.firgellirobots.com/blogs/robot-reviews/1x-technologies-neo-review)

**Skild AI (Skild Brain, S1)**
- "Learning by watching human videos" (Jan 12 2026). CLAIM: new skills by fine-tuning on human videos plus "<1 hr" of robot data — [Skild blog](https://www.skild.ai/blogs/learning-by-watching)
- S1 (Aug 18 2026):
  - An "in-context learner": the task is specified by a single human video in the prompt, with "no fine-tuning, no post-training", the same weights for every task, and tasks up to ~10 min.
  - Skild's scaling study on unseen tasks at 100k h: language-prompted VLA 9% vs S1 in-context 66%, a gap that "widens exponentially" with data.
  - "A single demonstration in context is worth roughly 380 post-training examples".
  - VLAs eventually win with 2,000 post-training demos (86% vs 66%).
  - Stated reasons language fails: novel primitives lack "grounding in actions", and "language is often too coarse to specify a novel chaining of known primitives".
  - All of this is company-reported. The architecture is not disclosed.
  - [Skild S1 blog](https://www.skild.ai/blogs/s1); [The Robot Report Aug 31 2026](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/)
- Data: teleop, human video, simulation and data-capture gloves. "Omni-bodied" across quadrupeds, humanoids and arms — [The Robot Report](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/); [Skild blog index](https://www.skild.ai/blogs)

**Agility Robotics**
- A whole-body "motor cortex" model: an LSTM under 1M parameters trained in Isaac Sim (decades of sim time in 3-4 days) that transfers zero-shot to Digit. It is prompted with arm and torso poses. Manipulation skills and LLM orchestration are layered on top — [Agility](https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model); [The Robot Report](https://www.therobotreport.com/agility-robotics-explains-train-whole-body-control-foundation-model/)

**Covariant (RFM-1)**
- RFM-1 (Mar 11 2024): an 8B any-to-any autoregressive model over text, images, video, actions and physical measurements. It predicts future frames, e.g. a bin image after removing an item. CLAIM: new behavior can be programmed "in minutes" via language — [BusinessWire](https://www.businesswire.com/news/home/20240311948570/en/Covariant-Introduces-RFM-1-to-Give-Robots-the-Human-like-Ability-to-Reason)
- Aug 2024: Amazon hired the founders (Abbeel, Chen, Duan) and ~25% of staff and took a non-exclusive license to the models ("reverse acquihire") — [TechCrunch](https://techcrunch.com/2024/08/31/amazon-hires-the-founders-of-robotics-ai-startup-covariant/); [GeekWire](https://www.geekwire.com/2024/amazon-hires-covariant-founders-inks-licensing-deal-with-robotics-ai-startup-in-latest-reverse-acquihire-deal/)

**Dyna Robotics (DYNA-1)**
- CLAIM: 99.4% success across a 24-h autonomous run, 850+ napkins at ~60% of human speed with zero interventions, and 75% at production quality (grade 4-5). Training is "RM-in-the-loop" with "the first scalable foundation reward model for robotics", used for exploration, error recovery and dataset curation. Conventional approaches "stall at ~80%" — [Dyna research](https://www.dyna.co/research/dyna-1)
- New tasks (laundry, cup filling) were reached with "just 0.7% of our full training dataset under 2 weeks" — [Dyna research](https://www.dyna.co/research/dyna-1)
- Secondary reports differ on the napkin count: "over 800" — [Robotics & Automation News](https://roboticsandautomationnews.com/2025/05/01/dyna-robotics-unveils-breakthrough-in-robust-real-world-embodied-ai/90152/). Secondary reports also describe the reward model as a 0-1 progress score with dips on mistakes and "nightly model updates", details not on the primary page fetched — [Robotics & Automation News](https://roboticsandautomationnews.com/2025/05/01/dyna-robotics-unveils-breakthrough-in-robust-real-world-embodied-ai/90152/)

**Generalist AI (GEN-0, GEN-1)**
- GEN-0 (Nov 4 2025):
  - Data: 270,000 h of real-world manipulation data, growing by 10,000 h per week.
  - "Harmonic Reasoning": asynchronous sensing and acting token streams, with no System1/System2 split.
  - Scaling: 1B models "struggle" ("ossification"), 6B models start to benefit, and 7B+ models transfer "with only a few thousand steps of post-training".
  - Downstream performance improves predictably with pretraining scale across 16 task sets. Post-training data ranges from 5.6 h up to 550+ h of task-specific data.
  - Closed.
  - [Generalist GEN-0](https://generalistai.com/blog/nov-04-2025-GEN-0)
  - The power law L(D) ∝ D^-0.5 is reported only by a secondary source — [36kr](https://eu.36kr.com/en/p/3551715251804547)
- GEN-1 (Apr 2 2026):
  - Data: "half a million hours" collected with "low-cost wearable devices on humans", and pretraining contains "no robot data".
  - New tasks are built with "approximately one hour of robot data", matching GEN-0 with 10x less task-specific data.
  - CLAIM results: 99% on several tasks vs 64% for GEN-0 and 19% from scratch, ~3x faster (box folding 12.1 s vs ~34 s).
  - Uses RL ("learn from experience") for speed. Early-access partners only.
  - [Generalist GEN-1](https://generalistai.com/blog/gen-1); [SiliconANGLE](https://siliconangle.com/2026/04/06/generalist-releases-gen-1-highly-capable-robotic-intelligence-ai-foundation-model/)

**ByteDance Seed (GR-3, Jul 2025)**
- Recipe: co-training with web-scale vision-language data, few-shot fine-tuning from VR-collected human trajectories, and robot imitation. Generalizes to novel objects and to "instructions involving abstract concepts", surpassing pi0 on several tasks. The abstract gives no numbers — [arXiv 2507.15493](https://arxiv.org/abs/2507.15493); [Seed blog](https://seed.bytedance.com/en/blog/seed-research-gr-3-released-a-generalist-robot-model-for-generalization-long-horizon-tasks-and-bi-manual-deformable-object-manipulation)

### Inferences
- TRI's rigorous study is the counterweight to startup claims. Without task-specific fine-tuning, a multitask diffusion LBM with a frozen CLIP text encoder did not beat single-task policies. Pretraining helped mostly through data efficiency, not zero-shot.
- "Continual improvement" at Dyna, 1X, Generalist and PI uses the same pattern: an automated judge (reward or value model, world-model evaluator, metadata labels) decides what deployment data to train on next. Updates are periodic retrains, not online weight updates.
- The industry is splitting on task specification. Language plus hierarchy (PI, DeepMind, Figure) competes with demonstration video in context (Skild S1) and with generated video or subgoal images (1X, pi0.7 subgoals, NVIDIA DreamGen for data).

### Gaps
- Tesla, Figure, Skild and Generalist publish no peer-reviewed results or independent evals.
- Not found: Figure's numbers on new-task success, Tesla's data hours, or the S1 architecture.
- Agility's VLA-level work was not found.
- TRI LBM weights were not established as open in this pass.

---

## For each company: how is a NEW task handled at deployment, and how is forgetting avoided?

### Takeaway
At deployment, zero-shot prompting covers only short or recombined tasks:
- pi0.7 (60-80% on unseen tasks);
- Gemini via ER decomposition;
- Helix and Redwood on novel objects.

Real new skills come from a few demos plus fine-tuning, or from in-context video (Skild S1). Forgetting is avoided structurally, not with continual-learning algorithms. The structural fixes are:
- re-finetuning from the pretrained checkpoint (PI Recap);
- freezing or insulating the VLM (NVIDIA N1.5, PI KI);
- co-training on pretraining data during post-training (NVIDIA N1.6);
- one retrained generalist on all data with metadata (pi0.7);
- per-task specialists that leave the base untouched (TRI, Dyna, Gemini fine-tunes, Generalist).

### Cited Findings
| Company / model | New task at deployment | Forgetting / update strategy | Source |
|---|---|---|---|
| PI pi0.7 | Zero-shot prompt (short tasks); step-by-step language coaching (long); coaching data trains a high-level policy | Single generalist retrained on all data; metadata lets mixed or failure data help, not hurt | [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2) |
| PI pi*0.6 Recap | Fine-tune on task demos, then RL iterations with corrections | Each iteration fine-tunes from the pretrained checkpoint to avoid drift | [arXiv 2511.14759](https://arxiv.org/html/2511.14759v1) |
| PI KI / pi0.5 | Fine-tune (e.g. openpi LIBERO, DROID checkpoints) | Stop-gradient from action expert into the VLM, plus VLM-data co-training | [arXiv 2505.23705](https://arxiv.org/abs/2505.23705); [openpi](https://github.com/Physical-Intelligence/openpi) |
| Gemini Robotics 1.0 / On-Device / On-Device 2 | ≤100 demos (1.0); 50-100 demos (On-Device); <200 examples, a few hours, for a new embodiment (OD2); ER 10-shot in-context control 65% | Not disclosed | [arXiv 2503.20020](https://arxiv.org/html/2503.20020); [On-Device blog](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/); [GR2 blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) |
| Gemini Robotics 1.5 | ER 1.5 orchestrator decomposes into VLA-executable steps | Not disclosed | [arXiv 2510.03342](https://arxiv.org/html/2510.03342) |
| NVIDIA GR00T N1.5 / N1.6 | Post-train 10K-30K steps; DreamGen synthetic trajectories for new verbs | N1.5 frozen VLM; N1.6 co-trains with pretraining data; DAgger | [N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/); [N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) |
| TRI LBM | Fine-tune; ~15-30% of from-scratch data suffices | Per-task fine-tuned specialists from the pretrained LBM | [arXiv 2507.05331](https://arxiv.org/html/2507.05331) |
| BD + TRI Atlas | Add demos (e.g. recoveries) and retrain the single policy | Retrain on all data across tasks | [BD blog](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/) |
| Figure Helix | Claims novel objects with no new demos; new behaviors via more data (human video in Go-Big) | Single set of weights, single training stage | [Helix](https://www.figure.ai/news/helix); [Go-Big](https://www.figure.ai/news/project-go-big) |
| 1X | Redwood plus Expert teleop sessions as training data; 1XWM generates video and an IDM maps it to actions; best-of-8 generations raise tissue pulling 30% to 45% | Not disclosed | [Redwood](https://www.1x.tech/discover/redwood-ai); [1XWM](https://www.1x.tech/discover/world-model-self-learning); [Robot Report](https://www.therobotreport.com/1x-announces-pre-order-launch-neo-humanoid-robot/) |
| Skild S1 | One human video in context, frozen weights | No weight update, so no forgetting by construction | [Skild S1](https://www.skild.ai/blogs/s1) |
| Generalist GEN-1 | ~1 h of robot data post-training, plus RL for speed | Not disclosed | [GEN-1](https://generalistai.com/blog/gen-1) |
| Dyna DYNA-1 | Fine-tune from base with targeted data (0.7% of dataset, <2 weeks) and a reward model in the loop | Not disclosed | [Dyna](https://www.dyna.co/research/dyna-1) |
| Tesla Optimus | Human video (1st-person, aiming for 3rd-person) plus a multitask net triggered by language | Not disclosed | [TeslaNorth](https://teslanorth.com/2025/05/21/tesla-shows-off-new-optimus-robot-skills-learned-from-human-videos/) |

### Inferences
- No company describes online, sequential, task-by-task continual learning with a forgetting analysis in a single deployed model. "Lifelong" improvement in industry means growing the dataset and periodically retraining, or keeping per-task specialists. This leaves the research gap of cheap, non-destructive addition of new tasks largely open in public industry work.
- In-context task specification (Skild S1 video, pi0.7 subgoal images, Gemini-ER 10-shot) is the only industry route that adds tasks with no weight change. It is also the most clearly company-claimed and least verified.

### Gaps
- No public fine-tuning compute or cost (GPU-hours) per new task from any company, beyond step counts (NVIDIA 10K-30K steps; Generalist "a few thousand steps").
- No published measurement of forgetting after fine-tuning for Gemini On-Device or GR00T.

---

## What do companies do that addresses instruction grounding / language-following specifically? (For a frozen pi0.5 fine-tuned on LIBERO's 40 tasks that fails held-out LIBERO-90 tasks because it does not read object/relation words it was never trained to contrast within a scene)

### Takeaway
Companies attack language-following failures with five levers:
1. Keep the VLM's semantics intact while learning actions (PI KI stop-gradient; NVIDIA frozen or partially unfrozen VLM). This took language following from ~40% to ~90% (PI) and from 46.6% to 93.3% (NVIDIA N1.5).
2. Co-train on web VQA, captioning and object-localization data, which mainly helps out-of-distribution objects (pi0.5, KI, GR-3).
3. Diversify language over the same behaviors via hindsight relabeling (Figure) and dense subtask annotation (pi0.5, pi0.7).
4. Decompose into short, concrete subtask commands with a high-level model (pi0.5 HL, Hi Robot, Gemini-ER, pi0.7 coaching).
5. Bypass language with grounded task specs: subgoal images (pi0.7), trajectory sketches (RT-Trajectory), video demos in context (Skild S1), to break dataset priors that language alone cannot override.

### Cited Findings
- PI KI: stop-gradient plus VLM-data co-training gives ~90% language following on the "items in drawer" task, vs ~40% for pi0 and ~55% for joint training without stop-gradient (approx., from figure). VLM co-training is "particularly important" for the novel-object OOD follow rate — [arXiv 2505.23705](https://arxiv.org/html/2505.23705)
- pi0.5 defines language following as selecting the commanded object. Removing web data significantly worsens OOD-object performance, and language following improves with more training environments — [arXiv 2504.16054](https://arxiv.org/html/2504.16054)
- pi0.7:
  - complex referring instructions ("object I would use to eat soup") are significantly better than pi0.5/pi0.6;
  - "Reverse Bussing", against training correlations, is significantly improved;
  - "Reverse Fridge to Microwave" needed generated subgoal images;
  - prompt components are randomly dropped in training so any subset can steer.
  - [arXiv 2604.15483](https://arxiv.org/html/2604.15483v2)
- NVIDIA N1.5: a frozen Eagle 2.5 VLM with better referring-expression grounding (RefCOCOg 89.6). Real GR-1 language following went from 46.6% to 93.3% — [NVIDIA N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/). N1 used middle-layer (12th) VLM embeddings with the text side frozen — [arXiv 2503.14734](https://arxiv.org/html/2503.14734). N1.6: language following remains "challenging" — [NVIDIA N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- Figure Helix: a VLM auto-labels clips with hindsight instructions to produce language-conditioned training pairs at scale — [Figure Helix](https://www.figure.ai/news/helix)
- Gemini Robotics 1.0 measures "instruction generalization" as its own axis (28% of 85 tasks). It succeeds on fine-grained relational instructions ("bottom compartment of the caddy") and on new languages — [arXiv 2503.20020](https://arxiv.org/html/2503.20020). Gemini Robotics 1.5 shows the largest gains in instruction generalization over 1.0 — [arXiv 2510.03342](https://arxiv.org/html/2510.03342)
- Hi Robot: a high-level VLM converts open-ended prompts and corrections into low-level commands for pi0 — [arXiv 2502.19417](https://arxiv.org/abs/2502.19417)
- RT-Trajectory: trajectory-sketch conditioning reaches 63% vs RT-2's 29% on 41 unseen tasks — [DeepMind blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)
- Skild S1 (company-reported) argues language prompting underspecifies novel tasks: 9% for a language-prompted VLA vs 66% for one-video in-context at 100k h — [Skild S1](https://www.skild.ai/blogs/s1)
- TRI LBM uses a frozen CLIP text encoder, and the paper does not analyze whether the policy actually uses language — [arXiv 2507.05331](https://arxiv.org/html/2507.05331)
- GR-3: web VL co-training gives strict language following, including abstract concepts (no numbers retrieved) — [arXiv 2507.15493](https://arxiv.org/abs/2507.15493)

### Inferences
- The researcher's frozen pi0.5 is the openpi KI version, so lever 1 was already applied in pretraining. The KI paper's LIBERO-90 96.0% is a train-on-LIBERO-90 number and says nothing about held-out LIBERO-90 tasks. Companies counteract fine-tuning-induced language collapse in two ways during adaptation: co-training with pretraining/VLM data (NVIDIA N1.6, PI KI) and not updating the VLM (N1.5). Fine-tuning only on LIBERO's 40 tasks, where few scenes contrast object or relation words, is exactly the regime those levers target.
- The levers most directly matched to "doesn't read contrastive words within a scene" are:
  - in-scene contrastive language data: Figure-style hindsight relabeling and pi0.5/pi0.7-style dense subtask annotation, generating multiple instructions per scene;
  - grounding-oriented web co-training: object localization and referring expressions (pi0.5 WD, N1.5 RefCOCOg);
  - a high-level model that converts the instruction into explicit grounded intermediates such as subtask text, points (Gemini-ER) or subgoal images (pi0.7).
- Industry evidence (pi0.7 reverse tasks, RT-Trajectory, Skild S1) suggests that language alone often cannot override strong dataset correlations. A grounded visual intermediate (subgoal image, trajectory, point) is the common workaround.

### Gaps
- No company reports held-out LIBERO-90 (train on 40, test on unseen) results.
- Exact pi0.5 per-condition language-following numbers exist only in figures.
- No public company ablation isolates "relation-word" (spatial preposition) following from object-word following, except Gemini's qualitative "bottom compartment" example and pi0.7's complex-reference category.
