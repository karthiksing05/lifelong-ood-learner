# RL fine-tuning, test-time scaling, and inference-time steering for VLA and diffusion/flow robot policies (2024-2026), judged against a frozen pi0.5 that fails held-out LIBERO-90 tasks on grounding

Scope and verification note (2026-10-01).
- Sources: arXiv abstract or HTML pages, project pages, and docs, fetched on 2026-10-01. Some facts come from the sibling notes in `research_notes/RL from practice for composition/` (frozen_policy_steering.md, vla_rl_finetuning.md, failure_aware_learning.md, diffusion_flow_policy_rl.md, compositional_search.md). Those were checked against arXiv on 2026-09-27, and each such fact keeps its original primary-source URL here.
- Summarizer extractions: numbers marked "(summarizer)" came through a summarizing fetch tool, not from a table I read myself.
- Venues: given only where an arXiv comment, a proceedings page, or a project page states them.

Our setting, for reference:
- The policy is a frozen pi0.5 (`pi05_libero`, flow matching). It fails held-out LIBERO-90 tasks mainly because it does not read object or relation words it never had to contrast within one scene.
- Our DSRL (noise-space RL) implementation barely improved on the base.
- Composing the policy's own conditionals under different prompts (classifier-free-style score sums) worked only where a factor was already faintly readable.

---

## Q0 (objective). Per family: core idea, representative works, reported gains, cost, failure modes, and relation to our setting

### Takeaway
The work falls into four families:
- **Weight-changing RL** (pi-RL/RLinf, RIPT-VLA, SimpleVLA-RL, VLA-RL, GRAPE, RECAP, RL-100, ConRFT, PA-RL, DPPO, residual RL). It reaches 90-99% on LIBERO tasks the policy was already SFT'd on, at 10^3-10^5 rollouts per suite.
- **Frozen-base noise or latent steering** (DSRL, Golden Ticket, QPILOTS). Most of its gain comes from selecting a better mode the base already produces. One fixed noise vector recovers much of DSRL's gain.
- **Test-time sample-and-verify** (V-GPS, RoboMonkey, CoVer, Q-Planning, VGAS). It follows power-law scaling in the number of samples, but it is bounded by pass@k, and its verifiers transfer poorly across policies.
- **Inference-time guidance and composition** (ITPS, DynaGuide, PoCo, CFG/CAG, DiMaS, VLS). It works when the guided factor already exists in the model, either as a readable conditioning signal or as a represented behavior. Several 2026 papers attack language-grounding failures directly (CAG, IGAR, SARL, VLA Grounder, CoVer), and their gains on grounding benchmarks are modest: tens of points at best, from low baselines.

### Cited Findings

#### Family 1: RL fine-tuning of VLAs (base weights change)

- **pi-RL / piRL (RLinf).**
  - It runs online PPO on flow VLAs, made stochastic by Flow-SDE or Flow-Noise.
  - pi0.5 from a 40-trajectory SFT reaches 77.1 → 97.9 (Flow-SDE) / 98.3 (Flow-Noise) on LIBERO. pi0 reaches 57.6 → 96.1 / 97.6.
  - The pi0.5 VLM is frozen and the ~300M action expert is trained.
  - These are the same LIBERO tasks the SFT used, not held-out tasks.
  - Sources: [arXiv HTML v3](https://arxiv.org/html/2510.25889v3); sibling notes vla_rl_finetuning.md.
- **RLinf** is the open-source framework. It ships pi0.5 PyTorch recipes for RECAP (advantage-conditioned CFG), STEAM and DSRL. Its DSRL-for-pi0 recipe has about 500K trainable parameters. — [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html); [RLinf DSRL docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html); [RLinf STEAM docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html)
- **RIPT-VLA** (arXiv 2505.17016).
  - RLOO/PPO ("LOOP") with sparse success. Dynamic rollout sampling throws away any init-state context whose K rollouts all get the same reward.
  - Cross-scenario with 1 demo: the best pair went 3.5% → 97.2%, and an "unworkable SFT model (4%)" reached 97% "within 15 iterations" (K=16, B=160 rollouts per iteration).
  - Cross-goal with 3 demos: SFT 0.7% → RIPT 59.7%.
  - Sources: [arXiv HTML](https://arxiv.org/html/2505.17016); [GitHub](https://github.com/Ariostgx/ript-vla)
- **SimpleVLA-RL** (arXiv 2509.09674).
  - GRPO-style RL on OpenVLA-OFT. One-trajectory SFT goes 48.9 → 96.9 LIBERO average, and LIBERO-Long goes 17.3 → 91.7.
  - Cost: 8× A800, with 500 scenarios per suite.
  - Emergent "pushcut": the policy pushes the can instead of grasp-move-place.
  - Source: [arXiv HTML](https://arxiv.org/html/2509.09674v1)
- **VLA-RL** (arXiv 2505.18719). PPO with LoRA on OpenVLA-7B and a learned process reward model. LIBERO average 81.0 (+4.5 over SFT) using "only 48 GPU hours". — [arXiv HTML](https://arxiv.org/html/2505.18719)
- **RL4VLA, "What Can RL Bring to VLA Generalization?"** (arXiv 2505.19789, NeurIPS 2025 per arXiv comment).
  - OpenVLA on ManiSkill pick-and-place.
  - PPO is "comparable to SFT" on visual OOD, better on semantic OOD (unseen objects, multi-object scenes, unseen instruction phrasings and receptacles), and better on execution OOD.
  - About 42 h on one A100. RL overtakes SFT-16k on OOD "after roughly 0.4 M environment steps". PPO beats GRPO and DPO.
  - Source: [arXiv HTML](https://arxiv.org/html/2505.19789) (summarizer)
- **GRAPE** (arXiv 2411.19309; ICLR 2025 workshop per [ML Anthology](https://mlanthology.org/iclrw/2025/zhang2025iclrw-grape/)).
  - Trajectory-wise preference optimization on OpenVLA.
  - The abstract claims success-rate gains of 51.79% in-domain and 58.20% on unseen tasks, 37.44% fewer collisions, and 11.15% shorter rollouts.
  - Budget: 5 rollouts per task per iteration, 3 iterations.
  - Sources: [arXiv abs](https://arxiv.org/abs/2411.19309); [arXiv HTML](https://arxiv.org/html/2411.19309)
- **pi\*0.6 / RECAP** (Physical Intelligence; arXiv 2511.14759).
  - Mechanism: advantage-conditioned training ("Advantage: positive/negative" text token). A 670M distributional value model scores progress, and CFG is optional at inference.
  - Results: on the hardest tasks it "more than doubles task throughput and roughly halves the task failure rate".
  - Data per iteration: 300 trajectories (laundry), or 600 autonomous plus 360 intervention trials (box assembly).
  - It is evaluated only on the tasks it trains on.
  - Sources: [arXiv abs](https://arxiv.org/abs/2511.14759); [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- **ConRFT** (arXiv 2502.05450): 96.3% average on 8 real tasks "within 45-90 minutes of online fine-tuning", with human interventions. — [arXiv](https://arxiv.org/abs/2502.05450)
- **iRe-VLA** (arXiv 2501.16664; "Accepted to ICRA 2025"). It alternates RL on the action head and critic (VLM frozen) with supervised learning on successes. The abstract gives no unseen-task numbers. — [arXiv abs](https://arxiv.org/abs/2501.16664)
- **ReinboT** (ICML 2025) predicts dense returns to guide actions on mixed-quality CALVIN data. — [ICML poster page](https://icml.cc/virtual/2025/poster/45523); [arXiv 2505.07395](https://arxiv.org/abs/2505.07395)
- **RL-100** (arXiv 2510.14830). Clipped PPO inside the denoising process (offline then online), then consistency distillation. It reports 1000/1000 real successes over 8 tasks. — [arXiv](https://arxiv.org/abs/2510.14830)
- **PA-RL** (arXiv 2412.06685).
  - Sample actions, rerank with Q, refine locally, then distill into the policy.
  - OpenVLA 40% → 70% in 40 min of online RL. On Cup-to-Rack, iterated filtered BC stayed at 50% while PA-RL went 55% → 90%.
  - Source: [arXiv HTML](https://arxiv.org/html/2412.06685)
- **DPPO** (arXiv 2409.00588, ICLR 2025). PPO through the denoising MDP; 80% (16/20) zero-shot sim-to-real on One-leg, against 0% for a Gaussian policy. FPO++ notes that DPPO baselines "often failed when base policies had low stochastic sampling success rates" (Can at about 10%). — [DPPO arXiv](https://arxiv.org/abs/2409.00588); [FPO++ arXiv 2602.02481](https://arxiv.org/abs/2602.02481)
- **DiffusionNFT** (arXiv 2509.16117, ICLR 2026 Oral per arXiv comment).
  - Likelihood-free negative-aware flow-matching loss: GenEval 0.24 → 0.98 in 1k steps, against FlowGRPO's 0.95 in more than 5k steps.
  - It has no robotics experiments. A robot variant, pi-StepNFT, is arXiv 2603.02083.
  - Sources: [arXiv HTML](https://arxiv.org/html/2509.16117); [pi-StepNFT](https://arxiv.org/abs/2603.02083)
- **Residual RL over a frozen base** (bounded corrective actor).
  - ResiP needs ~75M env steps for a one-step residual. Policy Decorator uses H = 1M-4M steps.
  - PLD uses 250k steps per LIBERO-90 task and reaches ">95%", then distills by SFT. ResFiT needs 134-343 real episodes.
  - DSRL reports that ResiP "[was not] able to significantly improve the performance of pi0" on LIBERO.
  - Sources: [ResiP](https://arxiv.org/html/2407.16677v4); [Policy Decorator](https://arxiv.org/html/2412.13630v1); [PLD](https://arxiv.org/html/2511.00091v1); [ResFiT](https://arxiv.org/html/2509.19301v2); [DSRL HTML](https://arxiv.org/html/2506.15799v2)
- **TT-VLA, test-time RL** (arXiv 2601.06748).
  - It runs value-free PPO with LoRA within a single episode, rewarded by the change in progress from a frozen VLAC progress model.
  - Gains are small: on ManiSkill, OpenVLA semantics 38.63 → 41.51, execution 36.39 → 39.83, vision 45.42 → 47.08.
  - Source: [arXiv HTML](https://arxiv.org/html/2601.06748v1)

#### Family 1b: Noise-space / latent RL on a frozen base

- **DSRL** (arXiv 2506.15799, CoRL 2025).
  - SAC trains a small actor whose action is the sampler's initial noise; the pi0 code tiles one 32-dim noise across the chunk.
  - LIBERO: one unseen LIBERO-90 task with `pi0_libero` goes ≈20% → ≈100% in ≈10k online samples.
  - Real Franka: 5/20 → 18/20 in 80 episodes, and 15/20 → 19/20 in 65 episodes.
  - Training ran on an RTX 3070 (8 GB).
  - Sources: [DSRL HTML](https://arxiv.org/html/2506.15799v2); [dsrl_pi0 code](https://github.com/nakamotoo/dsrl_pi0)
- **QPILOTS** (arXiv 2606.14801) runs on the frozen pi0.5-LIBERO checkpoint over 6 held-out LIBERO-90 tasks, at 5×10^5 env steps. Mean success:
  - zero-shot: 26
  - DSRL: 58
  - best-of-N (N=5, highest-Q): 40
  - EXPO: 39
  - critic-gradient guidance at each Euler step (QPILOTS-U): 76
  - Source: [QPILOTS HTML](https://arxiv.org/html/2606.14801)
- **Golden Ticket, "You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector"** (Patil et al., arXiv 2603.15757, v1 2026-03-16, v3 2026-06-19; venue not stated).
  - Mechanism: it replaces Gaussian initial noise with one constant noise vector found by random search, CEM or zeroth-order search on sparse episode reward. The policy is frozen, and "no new networks" are trained.
  - Results: better than Gaussian noise on 46/51 tasks, and at least equal on 49.
  - LIBERO with SmolVLA (30 tasks): +13% (Spatial), +12.8% (Goal), +8% (Object).
  - SimplerEnv with GR00T N1.5 (7 tasks): +26.5% average, e.g., put_eggplant_in_sink 21% → 76.3%.
  - Real Franka: cup push 40% → 100%, piston assembly 60% → 88%. It reports "28% within 60 search episodes for real-world tasks".
  - Multi-task: one ticket gives +14% over 7 SimplerEnv tasks and regresses on 2/7.
  - Comparison to DSRL: "a single noise vector is able to recover most of the gains of DSRL" on robomimic Lift/Can.
  - Sources: [arXiv abs](https://arxiv.org/abs/2603.15757); [arXiv HTML](https://arxiv.org/html/2603.15757) (summarizer)

#### Family 2: Test-time scaling (sample, then verify or rerank)

- **V-GPS** (arXiv 2410.13816, CoRL 2024).
  - It samples K actions and picks by a 25.6M Cal-QL value shared across Octo, RT1-X and OpenVLA.
  - Gains: SIMPLER +0.07 to +0.26, real WidowX +82.8%. K=50 is 1.59× slower.
  - DSRL reports that V-GPS did not significantly improve pi0 on its LIBERO task.
  - Sources: [V-GPS HTML](https://arxiv.org/html/2410.13816v2); [DSRL HTML](https://arxiv.org/html/2506.15799v2)
- **RoboMonkey** (arXiv 2506.17811, CoRL 2025 / PMLR v305).
  - OpenVLA with a LLaVA-7B verifier trained on 20M synthetic preference comparisons.
  - Gains: SIMPLER in-distribution +9%, real OOD 35% → 60%, LIBERO-Long +6.7%.
  - Cost: 16 candidates take ~650 ms on an H100.
  - Scaling law: action error versus number of samples "follows an exponentiated power law across a range of VLAs".
  - Sources: [arXiv abs](https://arxiv.org/abs/2506.17811); [HTML](https://arxiv.org/html/2506.17811v2); [PMLR](https://proceedings.mlr.press/v305/kwok25a.html)
- **CoVer-VLA, "Scaling Verification Can Be More Effective than Scaling Policy Learning for VLA Alignment"** (Kwok, ..., Finn, Pavone; arXiv 2602.12281).
  - Target: the "intention-action gap": "semantically equivalent rephrases can cause sharp drops in success".
  - Mechanism: 8 VLM-generated rephrases × 5 actions = 40 candidates, scored by a 1B contrastive verifier (ensemble of 3).
  - Verifier training: 20M Bridge samples with 16× synthetic instructions, 2,000 steps on 8× H200.
  - SIMPLER with red-teaming instructions, in-distribution / OOD: pi0 41.5 / 29.7, pi0+CoVer 57.0 / 61.0, pi0 (rephrase)+CoVer 65.5 / 62.0.
  - PolaRiS with pi0.5: progress 40.0 → 53.9, success 3.8 → 13.1. Real WidowX: +45% absolute.
  - Rephrasing yields a broader candidate set than Gaussian perturbation.
  - Transfer failure: RoboMonkey's verifier "fails catastrophically on most tasks" on pi0, because it was trained on a different policy's distribution.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.12281v2); [project](https://cover-vla.github.io)
- **Q-Planning** (arXiv 2608.21204).
  - A frozen BC policy plus a ~500M Q retrained on 20 episodes per iteration; it executes a Q-weighted average of N=32-64 samples.
  - LIBERO-10: 93 → 99. Real: stack-cups 40 → 90 and insert-wallet 25 → 80, while "SFT on successful rollouts alone stalls at 55% and 30%".
  - Source: [HTML](https://arxiv.org/html/2608.21204)
- **VGAS** (Xu, Yu, Xuan, Lu; arXiv 2602.07399).
  - Best-of-N chunk selection by a geometry-regularized critic, trained from 5 demos per task, on a frozen SmolVLA-0.5B proposer.
  - LIBERO average 49.0 vs 39.8 for BC (Spatial 56.2, Object 59.0, Goal 60.8, Long 20.0).
  - It targets "near-miss" failures, "semantically correct but spatially imprecise" actions. It assumes the proposal keeps "local support around expert demonstrations", and it has no real-robot evaluation.
  - Sources: [arXiv HTML](https://arxiv.org/html/2602.07399); [code](https://github.com/Jyugo-15/VGAS)
- **SVA** (arXiv 2607.03751): frozen VLAs rise "from 33% at pass@1 to 92% at pass@32". It uses MCTS in simulation and distills the result into a light Q model. — [arXiv abs](https://arxiv.org/abs/2607.03751)
- **MG-Select** (arXiv 2510.05681, ICLR 2026 per arXiv comment) is verifier-free. It uses KL to a masked-condition reference from the same VLA, which requires condition-dropout training. — [arXiv abs](https://arxiv.org/abs/2510.05681)
- **FOREWARN** (arXiv 2502.01828, RSS 2025): 100 samples are clustered to 6 plans, imagined with a world model, and chosen by a VLM. Real tasks: Cup 0.30 → 0.80, Fork 0.10 → 0.70. Selection takes 3.7 s. — [HTML](https://arxiv.org/html/2502.01828v3)
- **ELASTIC** (arXiv 2606.31132): SAC learns denoising stride and sample count for a frozen pi0.5, using 10 offline + 20 online rollouts per LIBERO task. On a real Franka it matches best-of-10 with 34% lower latency. — [HTML](https://arxiv.org/html/2606.31132)
- **BOKBO** (arXiv 2605.30660) is a caution: K-sample disagreement correlates 0.98 with the noise hyperparameter and "at the noise floor with actual safety violations". — [arXiv abs](https://arxiv.org/abs/2605.30660)

**Generalist reward models and verifiers**
- **RoboReward** (Lee, Wagenmaker, Pertsch, Liang, Levine, Finn; arXiv 2601.00675).
  - Models: 4B and 8B VLM reward models. Hard negatives come from "counterfactual relabeling of successful episodes and temporal clipping".
  - Benchmark: "no model excels across tasks".
  - Downstream: in real-robot RL the 8B model "improves policy learning over Gemini Robotics-ER 1.5 while narrowing the gap to RL training with human-provided rewards".
  - A secondary source reports MAE 0.665 for RoboReward-8B against 0.691 for GPT-5 mini.
  - Sources: [arXiv abs](https://arxiv.org/abs/2601.00675); [The Batch (secondary)](https://www.deeplearning.ai/the-batch/better-reward-models-for-robots)
- **GVL** (ICLR 2025): ALOHA median value-order correlation 0.12 zero-shot and 0.37 one-shot. — [HTML](https://arxiv.org/html/2411.04549v1)
- **VLAC**: 30% → 90% in 200 real PPO episodes. — [HTML](https://arxiv.org/html/2509.15937v1)
- **ROBORMBENCH** (arXiv 2609.05401): paraphrasing the instruction "can even flip identical robot behavior between failure and success". — [arXiv abs](https://arxiv.org/abs/2609.05401)
- **"No Free Checker"** survey (arXiv 2609.09250): "credibility falls as availability rises". — [arXiv abs](https://arxiv.org/abs/2609.09250)

#### Family 3: Inference-time steering, guidance and composition (frozen policy)

- **ITPS** (arXiv 2411.16627, ICRA 2025). It compares six ways to steer a frozen Diffusion Policy toward user inputs.
  - Results, alignment / constraint satisfaction: stochastic sampling (MCMC product) 71% / 73%, guided diffusion (sum) 37% / 82%, output perturbation 89% / 37%, random 38% / 90%.
  - Alignment trades against staying in distribution.
  - Source: [HTML](https://arxiv.org/html/2411.16627v2)
- **DynaGuide** (Du, Song; arXiv 2506.13922, NeurIPS 2025).
  - It guides denoising with a separately trained dynamics model toward goal or avoid objectives, using log-sum-exp guidance plus stochastic sampling.
  - Results: 70% average steering success on articulated CALVIN tasks, and 5.4× over goal-conditioning with low-quality objectives. It can "enhance underrepresented base policy behaviors".
  - Sources: [arXiv abs](https://arxiv.org/abs/2506.13922); [search summary](https://www.emergentmind.com/papers/2506.13922)
- **PoCo** (Wang et al.; arXiv 2402.02511; RSS 2024 is not stated on the arXiv page, venue unverified).
  - Mechanism: score-sum composition at task level (CFG weight α), behavior level (cost gradients) and domain level (sum of policies).
  - Real tool-use multitask with α=2.0: 60% ± 18 vs 50% ± 8 unconditioned. Domain composition on hammering: 5/5 vs 1/5 for human-data-only.
  - Weights were hand-tuned; "the composition hyperparameter needs to stay within a range to be effective and stable".
  - Assumptions and costs: policies must be aligned in horizon and scale, and two composed policies run at only 5 Hz.
  - Source: [arXiv HTML](https://arxiv.org/html/2402.02511v3)
- **Factored Diffusion Policies** (arXiv 2605.22596, revised 2026-09-28). One network is trained with per-factor null-token dropout, so its score "decomposes additively across factors". It reaches 90% of held-out drone gates (matching an oracle) vs 3% for a K-network composition baseline. It assumes approximate conditional independence of the factors. — [arXiv abs](https://arxiv.org/abs/2605.22596)
- **GPC** (arXiv 2510.01068): convex score combinations, including pi0 + Flow Policy, with the weight grid-searched at 200 rollouts per configuration. Gains are +2.22 to +7.55. — [arXiv](https://arxiv.org/abs/2510.01068) (from sibling compositional_search.md)
- **CAG, Counterfactual Action Guidance, "When Vision Overrides Language"** (Fang et al.; arXiv 2602.17659).
  - Benchmark: LIBERO-CF, 50 counterfactual tasks.
  - Base rates, grounding / success: pi0.5 30.8% / 13.2%, pi0 28.8% / 9.6%, OpenVLA-OFT 4.7% / 0.4%. The same models keep 65.6-97.1% on their training tasks.
  - Mechanism: π_CAG = π_uncond + ω(π_cond − π_uncond), with ω=1.5 for pi0 and pi0.5.
  - pi0.5 results: training-free (language dropped from the same model) gives grounding 40.5 and success 16.8. A separately trained vision-only model (30k steps) gives 46.3 and 21.7.
  - Language-dropout fine-tuning "performed poorly".
  - Real pi0.5: grounding +13-17%, success +17.2%.
  - Limitations: CAG assumes language signals "exist but are underweighted", over-guidance hurts manipulation, and long horizons compound the shortcut.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.17659v1) (summarizer)
- **IGAR** (arXiv 2603.06001). Training-free attention recalibration that restores language influence against "linguistic blindness". It was tested on pi0, pi0.5 and OpenVLA-OFT over 30 LIBERO tasks of its ICBench, and reduces execution under contradictory instructions. The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2603.06001)
- **SARL** (Bhatia, Wagenmaker, Chen, Levine; arXiv 2606.31958).
  - Mechanism: TD Q-learning over which language prompt to send to a frozen pi0.5 fine-tuned on LIBERO-90. A VLM proposes atomic commands such as "move down and grasp", cached at 32-100 per task.
  - Results: held-out LIBERO-10 and 4 WidowX tasks go from near 0% to ~80% in 60-100 episodes. It "successfully adapts a policy on five tasks", and "four tasks remain unsolved by any method".
  - On DSRL and residual RL, the paper says they are "fundamentally constrained by the base policy's performance under the task's original prompt", and that DSRL "can filter to find good actions ... but cannot synthesize fundamentally new ones".
  - Stated limitations: VLM latency, and "the prerequisite of having VLAs that produce diverse behaviors when prompted with diverse instructions".
  - Source: [arXiv HTML](https://arxiv.org/html/2606.31958)
- **VLA Grounder** (arXiv 2607.04517): RL with sparse task-completion reward over a policy that rewrites the human instruction into "a short VLA-grounded command" (object appearance, spatial relations, target cues) for black-box VLAs. The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2607.04517)
- **DiMaS** (Khayatan, Meziane, Parekh, Cord; arXiv 2607.14280, 2026-07-15).
  - Mechanism: an optimal-transport map between distributions of action-expert residual-stream representations with and without a feature, gated by a classifier and an interpolation knob α. Tested on SmolVLA and pi0.5 on LIBERO.
  - It controls HOW a task is done (speed, and vertical displacement on pi0.5). It does not control which task is done.
  - Linear (mean-difference) steering and prompt baselines ("Do this quickly.") "perform inconsistently".
  - Success is preserved for speed but drops for vertical displacement.
  - Transfer: control persists on held-out tasks within a suite, and weakens across suites (pi0.5 cross-suite to Goal: p=0.032, not significant at p<0.01).
  - "Behavioral features are linearly decodable but not linearly steerable".
  - Sources: [arXiv abs](https://arxiv.org/abs/2607.14280); [PDF](https://arxiv.org/pdf/2607.14280) (text read directly)
- **VLS** (arXiv 2602.03973). A VLM synthesizes "trajectory-differentiable reward functions" that guide a frozen diffusion or flow policy at inference. Gains: CALVIN +31%, LIBERO-PRO +13%, with a real Franka deployment. — [arXiv abs](https://arxiv.org/abs/2602.03973)
- **ACG** (arXiv 2510.22201, ICRA 2026). Training-free guidance against "jerks, pauses, and jitter" in flow VLAs. It targets motor coherence, not instruction following. — [arXiv abs](https://arxiv.org/abs/2510.22201)
- **Guided Action Flow** (arXiv 2607.02092): critic-gradient guidance on a frozen SmolVLA, trained offline on 100 real rollouts; 47.5% → 85.0%. — [arXiv abs](https://arxiv.org/abs/2607.02092)
- **RL^2-VLA** (arXiv 2607.26991): a failure-gated latent velocity steer, up to +17.3% OOD on SIMPLER and PolaRiS. — [arXiv abs](https://arxiv.org/abs/2607.26991)
- **CFGRL** (arXiv 2505.23458): guidance weight as a policy-improvement knob, π ∝ π̂ · p(o|s,a)^w. It needs optimality-conditioned training. — [HTML](https://arxiv.org/html/2505.23458v1)

#### Family 4: Learning from autonomous practice without human labels

- **Self-Improving Embodied Foundation Models** (arXiv 2509.15155, NeurIPS 2025). The reward is the change in predicted steps-to-go, plus a success detector. 10% extra robot time moves success 45% → 75%, while 8× more imitation data moves it 45% → 60%. — [arXiv HTML](https://arxiv.org/html/2509.15155)
- **SOAR** (arXiv 2407.20635, CoRL 2024). Goal relabeling turns failures into goal-conditioned data: 0.65 vs 0.28 for language-conditioned filtered BC. Its VLM success detector had precision 0.63 and recall 1.0. — [arXiv HTML](https://arxiv.org/html/2407.20635)
- **PLD** (ICLR 2026 per project page). Residual RL probing collects data that is distilled back into the VLA. In one-shot LIBERO-10 evaluation, PLD data beat self-bootstrapped data but "still falls short" of human demos. — [HTML](https://arxiv.org/html/2511.00091v1); [project](https://wenlixiao.com/self-improve-VLA-PLD)
- **SEES** (arXiv 2609.32698, Sept 2026). It restores failure states, has an LLM write success criteria, and runs online RL on adapters shared within skill families. It reports "continued improvement on unseen tasks". The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2609.32698)
- **FIND** (arXiv 2609.32069). A VLM agent chooses feasible tasks with low recent success and judges outcomes from before/after images. Human-assessed success over 8 real tasks rises 55% → 71.9%. — [arXiv abs](https://arxiv.org/abs/2609.32069)
- **EXIMO** (Sukhija, ..., Riedmiller; arXiv 2608.19891). A VLM planner decomposes long-horizon tasks into shorter commands for the VLA, then the method imitates the collected data and runs residual off-policy RL. The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2608.19891)

### Inferences
- Across families, our two observations reproduce published patterns:
  - "DSRL barely improved" matches SARL's account: noise RL selects within what the base does under the given prompt.
  - "Prompt-composition worked only where a factor was faintly readable" matches CAG's stated assumption that the language signal "exist[s] but [is] underweighted".
- Golden Ticket implies that much of DSRL's benefit is observation-independent mode selection. A per-task constant noise found by CEM is a cheap control that tests whether our DSRL's failure is a credit-assignment problem or a "no good mode exists" problem.

### Gaps
- Per-task numbers were not extracted for IGAR, VLA Grounder, SEES and EXIMO (abstract-only). DynaGuide's real-robot numbers were not extracted.
- PoCo's venue (commonly cited as RSS 2024) is not stated on the arXiv page I read.
- I found no paper that applies Golden Ticket search to pi0.5 or to held-out LIBERO-90 tasks.

---

## Q1. How much do these methods improve on NEW tasks (as opposed to training tasks), and do they require the base policy to already succeed sometimes?

### Takeaway
Most headline RL numbers are measured on the same tasks the policy was SFT'd on (LIBERO 4 suites). On genuinely new tasks there are two kinds of evidence:
- **Per-task RL or search on the new task itself.** DSRL (≈20% → ≈100% on one LIBERO-90 task), QPILOTS on pi0.5 (26 → 58 DSRL / 76, 6 LIBERO-90 tasks, 5×10^5 steps), SARL (≈0% → 80% on 5 of 10 LIBERO-10 tasks), and RIPT (4% → 97% cross-scenario). Every one of these needs the base to produce successful behavior sometimes, under the original prompt (DSRL, residual, best-of-N) or under some prompt (SARL).
- **Zero-shot transfer from RL on other tasks.** Modest and uneven: SimpleVLA-RL unseen tasks 0 → 7.1 and 0 → 13.3 (but 43.3 → 71.8 on one), RL4VLA better than SFT on semantic OOD, and Golden Ticket tickets matching or beating the base on only 5-7 of 10 other tasks.

The base-success requirement is explicit in several papers:
- SimpleVLA-RL: 0% stays 0%.
- RIPT discards groups where every rollout gets the same reward.
- SARL: DSRL "cannot synthesize fundamentally new" actions.
- TT-VLA: weak base gives limited gains.
- FPO++: DPPO fails at about 10% base success.

### Cited Findings

**Gains on held-out or new tasks (per-task adaptation)**
- DSRL on an unseen LIBERO-90 task ("pick up the cream cheese and put it in the tray"). `pi0_libero` had been fine-tuned on the 40 tasks of the 4 standard suites. It went ≈20% → ≈100% after ≈10k online samples, and ResiP and V-GPS "[were not] able to significantly improve" it. — [DSRL HTML](https://arxiv.org/html/2506.15799v2)
- QPILOTS on 6 held-out LIBERO-90 tasks (26/31/38/59/60/64) with frozen pi0.5-LIBERO, mean success: zero-shot 26, DSRL 58, BoN 40, EXPO 39, QPILOTS 76, after 5×10^5 env steps. — [QPILOTS HTML](https://arxiv.org/html/2606.14801)
- SARL with pi0.5 (fine-tuned on LIBERO-90) on held-out LIBERO-10: near 0% → ~80% in 60-100 episodes on the five tasks it adapts. "Four tasks remain unsolved by any method". — [arXiv HTML](https://arxiv.org/html/2606.31958)
- RIPT-VLA, 1 demo in a new scenario: SFT "around 5%", best gain 3.5% → 97.2%. Cross-goal with 3 demos: 0.7% → 59.7%. — [arXiv HTML](https://arxiv.org/html/2505.17016)
- Golden Ticket: SmolVLA LIBERO +8 to +13 points per suite. A failure case is LIBERO-Goal task 3 (base 52%, best ticket 40%). — [arXiv HTML](https://arxiv.org/html/2603.15757) (summarizer)
- VGAS, 5-shot LIBERO with SmolVLA: 49.0 vs 39.8. Evaluation is on held-out initial configurations of the trained tasks, not on new tasks. — [arXiv HTML](https://arxiv.org/html/2602.07399)
- Test-time verification on OOD:
  - RoboMonkey real OOD 35% → 60%. — [HTML](https://arxiv.org/html/2506.17811v2)
  - CoVer SIMPLER OOD 29.7 → 61.0 (pi0), and PolaRiS pi0.5 success 3.8 → 13.1. — [HTML](https://arxiv.org/html/2602.12281v2)
  - VLS LIBERO-PRO +13%. — [arXiv abs](https://arxiv.org/abs/2602.03973)

**Zero-shot transfer of RL gains to tasks not trained on**
- SimpleVLA-RL (9 seen / 1 unseen per suite), unseen-task success SFT → RL:
  - LIBERO-Spatial: 43.3 → 71.8, 0 → 7.1, 0 → 13.3.
  - LIBERO-Object: 38.1 → 74.6, 57.8 → 74.2.
  - LIBERO-Goal: SFT "catastrophic forgetting ... to 0%" on unseen tasks, while RL keeps small gains.
  - Source: [arXiv HTML](https://arxiv.org/html/2509.09674v1) (summarizer)
- RL4VLA: RL beats SFT on semantic OOD (unseen objects, instruction phrasings, receptacles) and execution OOD, and is comparable on visual OOD. — [arXiv HTML](https://arxiv.org/html/2505.19789) (summarizer)
- GRAPE's abstract claims +58.20% on unseen manipulation tasks. — [arXiv abs](https://arxiv.org/abs/2411.19309)
- Golden Ticket cross-task: a ticket optimized on one LIBERO task "matches or exceeds the base policy" on 7/10 (Goal), 7/10 (Object) and 5/10 (Spatial) tasks. A multi-task ticket improves 7 SimplerEnv tasks by 14% on average and regresses 2/7. — [arXiv HTML](https://arxiv.org/html/2603.15757) (summarizer)
- DiMaS: behavioral steering (speed) persists on held-out tasks within a suite and weakens across suites. — [PDF](https://arxiv.org/pdf/2607.14280)
- Simple Recipe (continual RL with LoRA): zero-shot on held-out tasks 57.1 / 25.4 / 86.6 (Spatial / Object / Long), against 51.2 / 27.6 / 85.2 for the multitask oracle. — [arXiv HTML v3](https://arxiv.org/html/2603.11653v3)
- SEES claims "continued improvement on unseen tasks" (no numbers in the abstract). — [arXiv abs](https://arxiv.org/abs/2609.32698)

**Evidence on the base-success requirement**
- SimpleVLA-RL: "The base model (0-trajectory SFT) achieves a 0% success rate across all tasks... RL is unable to improve performance, which remains at 0%." It adds that "a minimal level of task competence is essential for effective RL. Below this threshold, exploration is ineffective." With a 100-trajectory SFT (7.3%), RL reaches 25.4%; with 1,000 trajectories (28.2%), it reaches 50.4%. — [arXiv HTML](https://arxiv.org/html/2509.09674v1) (summarizer)
- RIPT-VLA's dynamic sampling discards any context whose K rollouts all get the same reward, because those groups carry no RLOO signal. — [arXiv HTML](https://arxiv.org/html/2505.17016)
- SARL: DSRL and residual RL "are fundamentally constrained by the base policy's performance under the task's original prompt", and DSRL "can filter to find good actions from the base policy's distribution, but cannot synthesize fundamentally new ones". — [arXiv HTML](https://arxiv.org/html/2606.31958)
- Simple Recipe: on-policy RL "only reweight[s] probability mass where πθ already has support". — [arXiv HTML v3](https://arxiv.org/html/2603.11653v3)
- TT-VLA: "when the base policy is weak or poorly aligned with the unseen task, TT-VLA may yield limited improvements". — [arXiv HTML](https://arxiv.org/html/2601.06748v1)
- FPO++: DPPO baselines "often failed when base policies had low stochastic sampling success rates" (Can at about 10%). — [arXiv](https://arxiv.org/abs/2602.02481)
- Golden Ticket: "reaching perfect performance may demand stronger search methods ... particularly when the base policy itself is weak". — [arXiv HTML](https://arxiv.org/html/2603.15757) (summarizer)
- Test-time selection is capped by pass@k. SVA: frozen VLAs go "from 33% at pass@1 to 92% at pass@32". — [arXiv abs](https://arxiv.org/abs/2607.03751)
- Counter-evidence (exploration can find new strategies): SimpleVLA-RL's "pushcut", where RL found pushing instead of the demonstrated grasp-move-place. — [arXiv HTML](https://arxiv.org/html/2509.09674v1)

### Inferences
- For our held-out LIBERO-90 failures, the relevant budget-matched evidence is QPILOTS. On our checkpoint family, DSRL more than doubles mean success (26 → 58), but only at 5×10^5 steps per task. "Our DSRL barely improved" is therefore consistent with either too small a budget or tasks where the base rarely succeeds. The two are distinguishable: our failing tasks with ~0% base success should show no DSRL gain, while tasks at 10-40% should show some.
- Tasks where pi0.5 never picks the named object under any noise are outside the reach of DSRL, Golden Ticket, best-of-N and residual RL. The literature's only demonstrated escape from a ~0% start without new demos is changing the conditioning: SARL's prompt search, or RIPT and SimpleVLA-RL with at least one demo in the SFT. SARL itself left 4 of 10 LIBERO-10 tasks unsolved.
- Zero-shot transfer of RL gains to new tasks is small and inconsistent (SimpleVLA-RL 0 → 7-13 points on two unseen tasks). Per-task practice remains necessary.

### Gaps
- No paper reports RL or steering on held-out LIBERO-90 tasks broken down by failure cause (grounding vs motor), so no source shows which failure type the gains came from.
- Which benchmark SimpleVLA-RL's Table 7 (0/100/1000 trajectories) uses was not confirmed (summarizer extraction).
- GRAPE's "+58.20% unseen" is an abstract claim; the definition of unseen tasks and the absolute numbers were not extracted.
- SARL's per-task LIBERO-10 numbers, and its explanation of why 4 tasks stayed unsolved, are not in the text I could extract.

---

## Q2. Which methods can fix instruction-grounding failures versus motor failures?

### Takeaway
- **Motor-side methods.** These act on actions, noise or action-expert representations: DSRL, Golden Ticket, residual RL, V-GPS, RoboMonkey, VGAS, Q-Planning, ACG and DiMaS. They fix motor, precision and mode-selection failures. VGAS explicitly targets "semantically correct but spatially imprecise" chunks, and DiMaS controls how a task is done, not which task. They help grounding only when the correct-object behavior already sits in the policy's conditional distribution for that prompt.
- **Language-side methods.** These act on the language channel: CAG (language CFG), IGAR (attention recalibration), SARL and VLA Grounder (RL over prompts), and CoVer (rephrase plus verify). They target grounding directly, but each assumes either that the word signal is present but underweighted, or that some other prompt the policy does read produces the right behavior.
- **Size of grounding gains.** On counterfactual-grounding benchmarks the best published gains for pi0.5 are modest: LIBERO-CF grounding 30.8 → 46.3, success 13.2 → 21.7.
- **Weight-changing RL with object diversity.** RL4VLA improves semantic OOD (unseen objects and instructions) more than SFT does. RL with contrasting objects in the scene is the main published route to actually strengthening grounding.
- **External-model guidance.** VLS and DynaGuide inject the task semantics from an external model, the VLM or the dynamics model, at the cost of extra models.

### Cited Findings

**Diagnosis: LIBERO-trained VLAs often ignore language**
- LIBERO-CF: VLAs "disregard the given instructions and default to executing well-learned tasks" under "fixed scenes with limited instruction diversity". pi0.5 scores 30.8% grounding and 13.2% success on counterfactual tasks, against 65.6-97.1% on training tasks. — [arXiv HTML](https://arxiv.org/html/2602.17659v1)
- LIBERO-Plus: "models tend to ignore language instructions completely". This is from the abstract via search extraction; I did not read a table. — [arXiv HTML](https://arxiv.org/html/2510.13626v1); [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- IGAR names the same failure "linguistic blindness" on pi0, pi0.5 and OpenVLA-OFT. — [arXiv abs](https://arxiv.org/abs/2603.06001)
- CoVer: "semantically equivalent rephrases can cause sharp drops in success". Color words caused misidentification in its "Redbull on Plate" example. — [HTML](https://arxiv.org/html/2602.12281v2)
- DiMaS: prompt-level modifiers ("Do this quickly.") moved pi0.5 and SmolVLA behavior "inconsistently", often in the same direction for opposite instructions. — [PDF](https://arxiv.org/pdf/2607.14280)

**Methods that target grounding, and their limits**
- CAG (language CFG), pi0.5 on LIBERO-CF:
  - Training-free: grounding 30.8 → 40.5, success 13.2 → 16.8.
  - With a separately trained vision-only prior: 46.3 / 21.7.
  - Real tasks: object recognition (Coke/Sprite/Fanta), left/right spatial reasoning (20-30% → 36.6%) and goal targeting improved without degrading in-domain tasks.
  - Limitation: it "assumes language signals exist but are underweighted; it may not address cases where language is entirely absent from training".
  - Source: [arXiv HTML](https://arxiv.org/html/2602.17659v1) (summarizer)
- SARL (RL over prompts) needs "VLAs that produce diverse behaviors when prompted with diverse instructions". Its prompts are atomic commands the policy already follows ("move down and grasp", "open the fridge"). — [arXiv HTML](https://arxiv.org/html/2606.31958)
- VLA Grounder learns to rewrite instructions into "short VLA-grounded command[s]" using "object appearance, spatial relations, and target-grounding cues", with sparse-reward RL. — [arXiv abs](https://arxiv.org/abs/2607.04517)
- CoVer: joint rephrasing plus verification raises OOD success under red-teaming instructions (pi0 29.7 → 62.0). The verifier is a 1B contrastive model trained on 20M samples. — [HTML](https://arxiv.org/html/2602.12281v2)
- IGAR is training-free attention recalibration and is reported to cut erroneous execution under contradictory instructions (no numbers in the abstract). — [arXiv abs](https://arxiv.org/abs/2603.06001)
- RL4VLA: weight-changing PPO improves unseen objects, unseen instruction phrasings and multi-object scenes more than SFT. Its explanation: "RL is able to learn the skill of 'grasp' in a manner that is less dependent on object type". — [arXiv HTML](https://arxiv.org/html/2505.19789) (summarizer)
- External-model steering:
  - VLS: a VLM writes differentiable rewards for a frozen policy; LIBERO-PRO +13%, CALVIN +31%. — [arXiv abs](https://arxiv.org/abs/2602.03973)
  - DynaGuide: a dynamics model steers toward goal objects and "enhance[s] underrepresented base policy behaviors". — [arXiv abs](https://arxiv.org/abs/2506.13922)
  - PoCo: task-level CFG composition helps only within a tuned weight range. — [arXiv HTML](https://arxiv.org/html/2402.02511v3)
- Factored Diffusion Policies get compositional generalization from additive per-factor scores only when they are trained with per-factor null-token dropout, under approximate conditional independence. — [arXiv abs](https://arxiv.org/abs/2605.22596)

**Methods that target motor or execution failures**
- VGAS targets "near-miss" failures ("semantically correct but spatially imprecise actions") and assumes the proposal keeps support near the expert. — [arXiv HTML](https://arxiv.org/html/2602.07399)
- ACG targets "jerks, pauses, and jitter". — [arXiv abs](https://arxiv.org/abs/2510.22201)
- DiMaS controls speed and height ("how a robot performs a task"). — [PDF](https://arxiv.org/pdf/2607.14280)
- Golden Ticket's tickets are "observation-independent" and do not condition on language. — [arXiv HTML](https://arxiv.org/html/2603.15757) (summarizer)
- DSRL "cannot synthesize fundamentally new" actions beyond the base's distribution under the original prompt (SARL's characterization). — [SARL HTML](https://arxiv.org/html/2606.31958)
- Residual RL learns bounded corrections around the base action (PLD residual scale ξ=0.5 on LIBERO). — [PLD HTML](https://arxiv.org/html/2511.00091v1)

### Inferences
- Our diagnosis (pi0.5 does not read object or relation words it never had to contrast in one scene) is the same mechanism LIBERO-CF names: fixed scenes with low instruction diversity produce visual shortcuts. CAG's limitation sentence describes our prompt-composition result: score sums amplify a weak language signal but cannot create one.
- For a grounding failure, every noise-, residual- or verifier-based method can help only if the right-object behavior already appears at some rate under the prompt. A cheap diagnostic per failing task is pass@k under the original prompt (SVA-style): count how many of k noise samples touch the named object. If none do, DSRL, Golden Ticket and best-of-N are ruled out for that task.
- Of the published options, the ones that do not need the policy to read the missing word all route around it:
  - SARL and VLA Grounder rewrite the instruction into words the policy does read.
  - VLS and DynaGuide supply the object or goal through an external model.
  - CoVer and RoboMonkey supply a verifier that reads the word.
  - Strengthening the word itself requires training with contrast: RL4VLA multi-object RL, or CAG's trained vision-only prior.
- Verifiers transfer poorly across policies. RoboMonkey's OpenVLA-trained verifier "fails catastrophically" on pi0 (per CoVer), so a verifier for pi0.5 would have to be trained on pi0.5 samples.

### Gaps
- No paper found applies CAG, IGAR or prompt RL to held-out LIBERO-90 tasks with pi05_libero specifically.
- I found no paper that decomposes RL or steering gains into "fixed a grounding error" versus "fixed a motor error" on the same task set.
- The CAG numbers came through a summarizer; LIBERO-CF per-suite tables were not checked directly.

---

## Q3. Costs: episodes, wall-clock, and extra models

### Takeaway
- **Cheapest per task:** Golden Ticket search (no new networks; real gains within 60 episodes, while simulated searches used up to 250-5,000 candidate tickets × 5-100 environments), training-free guidance (CAG-TF, IGAR, ACG, ITPS), and SARL / DSRL at 60-100 real episodes.
- **Test-time verification.** Its main cost is a large verifier (1B-7B, trained on ~20M samples) and 0.45-0.65 s per decision for 16 candidates.
- **Weight-changing RL.** Thousands to hundreds of thousands of rollouts per suite on 1-8 large GPUs (RIPT ≈2,400 rollouts for one task; VLA-RL 48 GPU-hours; SimpleVLA-RL 8× A800; QPILOTS 5×10^5 steps per task).
- **Residual RL.** The most sample-hungry family: 10^5 to 10^7 steps.

### Cited Findings

| Method | Base frozen? | Per-task / per-run budget | Wall-clock / hardware | Extra models | Source |
|---|---|---|---|---|---|
| DSRL | yes | ≈10k env steps (LIBERO task); 65-80 real episodes | RTX 3070 8 GB | SAC noise actor + critics (~500K params in RLinf recipe) | [DSRL](https://arxiv.org/html/2506.15799v2); [RLinf DSRL](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html) |
| QPILOTS | yes (pi0.5) | 5×10^5 env steps per task | n/a | task critic | [QPILOTS](https://arxiv.org/html/2606.14801) |
| Golden Ticket | yes | <250 tickets × 5 envs (SimplerEnv) up to 5000 tickets × 100 envs (robomimic); +28% within 60 real search episodes | n/a | none | [arXiv abs](https://arxiv.org/abs/2603.15757); [HTML](https://arxiv.org/html/2603.15757) |
| SARL | yes (pi0.5) | 60-100 episodes | VLM in the loop is a stated speed limitation | VLM prompt proposer + Q-function | [SARL](https://arxiv.org/html/2606.31958) |
| RIPT-VLA | no (LoRA) | K=16, B=160 rollouts/iteration; 15 iterations to 97% (≈2,400 rollouts, my arithmetic) | 1 GPU (single task) | none (critic-free) | [RIPT](https://arxiv.org/html/2505.17016) |
| SimpleVLA-RL | no (full) | 500 scenarios per LIBERO suite | 8× A800 80 GB | none | [SimpleVLA-RL](https://arxiv.org/html/2509.09674v1) |
| VLA-RL | no (LoRA) | n/a | 48 GPU-hours | process reward model + critic | [VLA-RL](https://arxiv.org/html/2505.18719) |
| RL4VLA | no | ~0.4M env steps to beat SFT on OOD | ~42 h on 1 A100 | critic | [RL4VLA](https://arxiv.org/html/2505.19789) |
| piRL | no (action expert) | (not extracted) | n/a | critic (PPO) | [piRL](https://arxiv.org/html/2510.25889v3) |
| GRAPE | no (LoRA) | 5 rollouts/task/iteration × 3 iterations | n/a | VLM-generated stage costs | [GRAPE](https://arxiv.org/html/2411.19309) |
| RECAP (pi\*0.6) | no | 300 traj/iteration (laundry); 600 autonomous + 360 interventions (box) | n/a | 670M value model | [RECAP](https://arxiv.org/html/2511.14759v1) |
| ConRFT | no | 45-90 min real, with interventions | n/a | critic | [ConRFT](https://arxiv.org/abs/2502.05450) |
| PA-RL | no (distilled) | 40 min real (OpenVLA); 10-70 episodes | n/a | Q-function | [PA-RL](https://arxiv.org/html/2412.06685) |
| PLD | yes, then SFT | 250k steps per LIBERO-90 task | n/a | residual actors + Cal-QL critic | [PLD](https://arxiv.org/html/2511.00091v1) |
| ResiP / Policy Decorator / ResFiT | yes | ~75M steps / 1M-4M steps / 134-343 real episodes | n/a | residual actor + critic | [ResiP](https://arxiv.org/html/2407.16677v4); [PD](https://arxiv.org/html/2412.13630v1); [ResFiT](https://arxiv.org/html/2509.19301v2) |
| TT-VLA | no (LoRA, in-episode) | within one 160-step episode, update every 8 steps | n/a | frozen VLAC progress model | [TT-VLA](https://arxiv.org/html/2601.06748v1) |
| V-GPS | yes | offline value on Bridge + Fractal | K=50 → 1.59× slower | 25.6M Cal-QL value | [V-GPS](https://arxiv.org/html/2410.13816v2) |
| RoboMonkey | yes | 20M synthetic comparisons (verifier) | ~650 ms / 16 candidates on H100 | LLaVA-7B verifier | [RoboMonkey](https://arxiv.org/html/2506.17811v2) |
| CoVer | yes | 20M Bridge samples × 16 synthetic instructions; 2,000 steps | 8× H200 training; ~453 ms / 16 candidates | 1B verifier × 3 + VLM (boot-time rephrasing) | [CoVer](https://arxiv.org/html/2602.12281v2) |
| Q-Planning | yes | 20 episodes/iteration × 5 (real) | n/a | ~500M Q-decoder | [Q-Planning](https://arxiv.org/html/2608.21204) |
| VGAS | yes | 5 demos/task (critic) | best-of-N latency (stated limitation) | Q-Chunk-Former critic | [VGAS](https://arxiv.org/html/2602.07399) |
| ELASTIC | yes (pi0.5) | 10 offline + 20 online rollouts/task | 34% lower latency than best-of-10 | SAC meta-policy + V-GPS critic | [ELASTIC](https://arxiv.org/html/2606.31132) |
| FOREWARN | yes | 100 demos/task (base) | 3.7 s per selection | world model + fine-tuned Llama-3.2 VLM | [FOREWARN](https://arxiv.org/html/2502.01828v3) |
| CAG | yes | training-free variant; or vision-only prior trained 30k steps | two forward passes per step (conditional + unconditional) | optional vision-only model | [CAG](https://arxiv.org/html/2602.17659v1) |
| ITPS | yes | zero training | M=4 MCMC steps per diffusion step | objective only | [ITPS](https://arxiv.org/html/2411.16627v2) |
| PoCo | yes (composition) | hand-tuned weights | 5 Hz with 2 composed policies | the other policies | [PoCo](https://arxiv.org/html/2402.02511v3) |
| GPC | yes | 200 rollouts per weight configuration (~2.5 h full grid on Robomimic) | n/a | second policy | [GPC](https://arxiv.org/abs/2510.01068) |
| DynaGuide | yes | dynamics model trained separately | n/a | dynamics model | [DynaGuide](https://arxiv.org/abs/2506.13922) |
| DiMaS | yes | representations from rollouts per feature | n/a | OT map + gating classifier | [DiMaS PDF](https://arxiv.org/pdf/2607.14280) |
| VLS | yes | training-free | VLM call | VLM | [VLS](https://arxiv.org/abs/2602.03973) |
| Self-Improving EFMs | no | +10% robot time → 45% → 75% | n/a | steps-to-go model + success detector | [arXiv HTML](https://arxiv.org/html/2509.15155) |
| RoboReward | n/a (reward model) | n/a | n/a | 4B / 8B VLM | [arXiv abs](https://arxiv.org/abs/2601.00675) |

### Inferences
- For tens of rollouts per held-out task, the budget-compatible candidates are:
  - **Golden Ticket-style CEM over a constant noise.** Zero networks, and its search episodes double as practice data.
  - **SARL-style choice among a few cached prompts.** 60-100 episodes, and it can address grounding by routing to words pi0.5 reads.
  - **Training-free language CFG (CAG-TF).** Zero training, 2× inference. Our composition results already suggest a ceiling on this route.
  - **A small verifier or Q head trained on practice rollouts (Q-Planning style).** Needs ~20 episodes per iteration.
- Large shared verifiers (RoboMonkey, CoVer) are expensive to train and do not transfer across policies, so a per-task scorer for pi0.5 is the cheaper choice.
- Policy-gradient RL (RIPT, SimpleVLA-RL, piRL) and residual RL exceed the budget by 1-4 orders of magnitude, and on ~0%-success tasks they would not start (SimpleVLA-RL 0% → 0%).

### Gaps
- Parameter counts are not reported for DSRL's actor and critic in the paper, for the SARL Q-function, or for the DiMaS OT map.
- Wall-clock times are missing for piRL, Golden Ticket simulated searches, QPILOTS, SARL and DynaGuide.
- I found no paper that reports cost per unit of gain on held-out tasks for any of these methods, so cross-method efficiency comparisons above are my own and assume equal per-episode cost.
- Not read (seen in search listings only): ReGuide ("From Test-Time Guidance to Self-Improving Diffusion Policies", [arXiv 2606.28939](https://arxiv.org/html/2606.28939)), "Feedback World Model Enables Precise Guidance of Diffusion Policy" ([arXiv 2605.15705](https://arxiv.org/pdf/2605.15705)), ProGAL-VLA ([arXiv 2604.09824](https://arxiv.org/html/2604.09824)), TOPReward, Robometer, and RynnValue.
