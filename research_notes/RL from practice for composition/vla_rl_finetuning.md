# RL fine-tuning of vision-language-action models (VLAs), with a focus on LIBERO (state of the field, September 2026)

Scope note. These notes cover 2025-2026 VLA RL work and judge it against our setting. In our setting the prior is frozen pi0.5 (openpi `pi05_libero`, PyTorch port, 10-step Euler flow sampling). Each held-out LIBERO-90 task gets one demo. That demo is composed from the prior's own chunks, guided through the flow sampler, and a small gated LoRA "fold" is added. Practice gives sparse binary success on about 20-30 init states per task, and the budget is tens of rollouts per task. Zero-training composition solves about 61% of practice instances. A rank-16 LoRA trained on the demo plus composition successes reaches about 0.66 on unseen init states, against 0.61 demo-only. A second round of success-only self-imitation plateaus.

Verification method. I opened each arXiv ID on its arXiv abstract page and read the title, authors, dates and Comments field. Most numbers come from the arXiv HTML full text, parsed table by table (the page is cited next to each number). Numbers that came only from a summarizer, or only from a secondary page, are marked as such. "Unverified" means I could not confirm the item from a primary source.

## Q1. For each method: algorithm, trained parameters, likelihood for flow/diffusion heads, reward, rollout budget, LIBERO numbers (including one-demo SFT starting points)

### Takeaway
The field splits into three families:
- **Likelihood-based on-policy RL.** PPO, GRPO or RLOO applied to autoregressive VLAs, or to flow VLAs made stochastic with Flow-SDE or Flow-Noise. Examples: RIPT-VLA, SimpleVLA-RL, VLA-RL, TGRPO, piRL/RLinf, Simple Recipe, SRPO, pi-StepNFT.
- **Likelihood-free improvement.** Advantage- or outcome-conditioning with classifier-free guidance (RECAP/pi*0.6, CFGRL), noise-space or prompt-space steering of a frozen policy (DSRL, SARL), and sample-rank-distill or residual methods (PA-RL, PLD, RLDG, iRe-VLA).
- **World-model RL.** WMPO, VLA-RFT, WISE.

On standard LIBERO, likelihood-based RL takes one-demo SFT checkpoints to about 92-99% per suite. It does so with thousands to hundreds of thousands of rollouts per suite. Only piRL/RLinf runs this on pi0.5, and in that work the pi0.5 VLM is frozen and the 300M action expert is fully trained.

### Cited Findings

#### Likelihood-based on-policy RL

- **RIPT-VLA** (Tan, Dou, Zhao, Krähenbühl; arXiv 2505.17016, submitted 22 May 2025).
  - The algorithm is "LOOP": RLOO leave-one-out advantages, b_k = mean of the other K-1 rewards in a group sharing one initial-state "context", plus PPO clipping.
  - Dynamic rollout sampling discards any context whose K rollouts all get the same reward and resamples a new one.
  - Reward is sparse binary success; there is no value network — [arXiv HTML](https://arxiv.org/html/2505.17016); [GitHub README](https://github.com/Ariostgx/ript-vla)
- **RIPT-VLA trained parameters and likelihood.**
  - OpenVLA-OFT: LoRA rank 32, with learning rate 1e-4 on LoRA and 1e-5 on the action head.
  - Likelihood for OFT's continuous head: they first train a separate Laplace "scale header" with NLL for 500 steps. Rollouts sample from a factorized Laplace distribution, and evaluation uses the mean.
  - QueST: the whole ~80 MB model is trained, with its discrete skill-token logits used as log-probs.
  - Hyperparameters: OFT uses K=8, B=192, N=1, ε=0.1. QueST multitask uses K=16, B=2880, N=20, ε=0.2 on 3 GPUs. QueST single-task uses K=16, B=160 on 1 GPU.
  - OFT training ran on 4× RTX A5000 (24 GB) — [arXiv HTML](https://arxiv.org/html/2505.17016); [GitHub README](https://github.com/Ariostgx/ript-vla)
- **RIPT-VLA LIBERO results.**
  - QueST four-suite average rises from 82.7 to 93.6. Goal 80.8→92.7, Spatial 87.4→95.6, Object 93.6→98.4, Long 68.8→87.5.
  - OpenVLA-OFT rises from 96.7 to 97.5.
  - LIBERO-90 (QueST, full data) rises from 88.6 to 94.3, and MetaWorld ML45 from 91.0 to 92.2.
  - 5-shot LIBERO-LONG rises from 50.2 to 71.4, and 5-shot ML45 from 63.6 to 76.0 — [arXiv HTML, Tables 1-2](https://arxiv.org/html/2505.17016)
- **SimpleVLA-RL** (Li et al.; arXiv 2509.09674, submitted 11 Sep 2025; ICLR 2026 per the repo README).
  - The algorithm is GRPO with group size 8 and dynamic sampling that keeps only groups with 0 < #successes < G.
  - Clip-higher uses ε_low 0.2 and ε_high 0.28. The KL term is removed, and rollout temperature is 1.6.
  - The model is OpenVLA-OFT with tokenized actions. All parameters are trained (not LoRA), with lr 5e-6.
  - Hardware is 8× A800 80 GB. RL uses 500 simulation scenarios per LIBERO suite — [arXiv HTML](https://arxiv.org/html/2509.09674v1); [GitHub](https://github.com/PRIME-RL/SimpleVLA-RL)
- **SimpleVLA-RL LIBERO results** (I parsed the table cell by cell, because a summarizer garbled it).

  | Starting point | Spatial | Object | Goal | Long | Avg |
  |---|---|---|---|---|---|
  | One-trajectory SFT (1 demo per task, 10 per suite) | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
  | After RL | 98.2 | 98.7 | 98.8 | 91.7 | 96.9 |
  | Full-trajectory SFT | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
  | After RL | 99.4 | 99.1 | 99.2 | 98.5 | 99.1 |

  — [arXiv HTML](https://arxiv.org/html/2509.09674v1)
- **VLA-RL** (Lu et al.; arXiv 2505.18719, submitted 24 May 2025; venue unverified).
  - The algorithm is PPO with GAE on OpenVLA-7B with LoRA. Log-probs are summed over the 7 action tokens.
  - The reward is sparse success plus a robotic process reward model, a VLM fine-tuned on pseudo-labeled task segments.
  - Other components: a critic warmup, and a curriculum that samples tasks with P ∝ exp((0.5 − s_j)/τ).
  - LIBERO average is 81.0 (Spatial 90.2, Object 91.8, Goal 82.2, Long 59.8), which is +4.5 over the SFT baseline. This took "only 48 GPU hours of RL training" — [arXiv HTML](https://arxiv.org/html/2505.18719); [GitHub](https://github.com/GuanxingLu/vlarl)
- **TGRPO** (Chen et al.; arXiv 2506.08440, v3 27 Sep 2025; venue unverified).
  - The algorithm is trajectory-wise GRPO. It fuses step-level and trajectory-level advantages with α1 = 0.3 and α2 = 0.7.
  - Dense stage rewards are produced by Claude 3.7 Sonnet task decomposition.
  - The model is OpenVLA with LoRA, lr 1e-5, trained on a single A100.
  - LIBERO average rises from 76.5 to 80.7 (Spatial 84.7→90.4, Object 88.4→92.2, Goal 79.2→81.0, Long 51.1→59.2). No code URL is given — [arXiv HTML](https://arxiv.org/html/2506.08440) (summarizer extraction)
- **piRL** (Chen et al.; arXiv 2510.25889, v1 29 Oct 2025, v3 29 Jan 2026; venue unverified). This is RL for the flow VLAs pi0 and pi0.5.
  - Flow-Noise adds a learnable noise network, which makes the K-step denoising chain a discrete-time MDP with an exact log-likelihood. The noise net is discarded after training.
  - Flow-SDE converts the ODE into a marginal-preserving SDE with σ_τ = a·sqrt(τ/(1−τ)). It forms a two-layer MDP and uses hybrid ODE-SDE sampling, where one random denoising step is stochastic.
  - The algorithm is PPO with GAE. For pi0.5 the critic is attached after the VLM.
  - RL freezes the VLM and fully fine-tunes the 300M action expert. SFT beforehand tunes the full 3.3B model.
  - Hardware is 8× H100 — [arXiv HTML v3](https://arxiv.org/html/2510.25889v3)
- **piRL few-shot LIBERO results.**
  - pi0 was SFT'd on 58 of 1,692 LIBERO demos (208 for Long). pi0.5 used "only 40 trajectories to provide a unified few-shot SFT checkpoint across all task suites". The paper says 40 is the minimum that gives every one of the 40 tasks a positive success rate.

  | Model | Spatial | Object | Goal | Long | Avg |
  |---|---|---|---|---|---|
  | pi0 SFT | 65.3 | 64.4 | 49.8 | 51.2 | 57.6 |
  | pi0 + Flow-SDE | 98.4 | 99.4 | 96.2 | 90.2 | 96.1 |
  | pi0 + Flow-Noise | 99.0 | 99.2 | 98.2 | 93.8 | 97.6 |
  | pi0.5 SFT | 84.6 | 95.4 | 84.6 | 43.9 | 77.1 |
  | pi0.5 + Flow-SDE | 99.6 | 100 | 98.8 | 93.0 | 97.9 |
  | pi0.5 + Flow-Noise | 99.6 | 100 | 99.6 | 94.0 | 98.3 |

  — [arXiv HTML v3, Table 3 and App. C.1](https://arxiv.org/html/2510.25889v3)
- **piRL LIBERO hyperparameters** (Table 11).
  - Common: 500 train epochs, global batch 2048, 64 parallel envs, 8 rollout epochs, clip ε 0.2, actor lr 5e-6.
  - pi0.5 specifics: action horizon 10, replan horizon 5 (10 for Long), 3-5 denoise steps, Flow-SDE noise level 0.3-0.5 — [arXiv HTML v3](https://arxiv.org/html/2510.25889v3)
- **GRPO vs PPO on pi0.5 (RLinf docs).**
  - pi0.5 few-shot 77.1 → +GRPO 91.5 (Spatial 97.4, Object 99.8, Goal 91.2, Long 77.6) vs +PPO 97.9.
  - pi0 few-shot 57.6 → +GRPO 90.0 vs +PPO 96.0 — [RLinf pi0 docs (repo source)](https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/pi0.rst)
- **RLinf-VLA** (Zang et al.; arXiv 2510.06710, v3 12 Aug 2026; "Accepted to RSS 2026" in the arXiv comments).
  - It is a unified framework: OpenVLA, OpenVLA-OFT, pi0 and pi0.5, with PPO and GRPO.
  - One OpenVLA-OFT model was trained on all 130 LIBERO tasks with GRPO. It went from a LoRA-base SFT model to: Object 50.20→99.60, Spatial 51.61→98.69, Goal 49.40→98.09, Long 11.90→93.45, LIBERO-90 42.67→98.02, LIBERO-130 42.09→97.85. RL rollouts used temperature 1.6 — [arXiv abs](https://arxiv.org/abs/2510.06710); [RLinf LIBERO docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)
- **Simple Recipe** (Hu, Shim, Tang, Sung, Liu, Stone, Martín-Martín; arXiv 2603.11653, v1 12 Mar 2026, v3 11 Jul 2026; "Accepted at RLC 2026; Best paper award at ICRA26 RL4IL Workshop").
  - It does continual RL: tasks are learned in sequence with GRPO plus a LoRA (rank 32), and the sequential LoRA fine-tuning is called "Seq. FT".
  - Hyperparameters: group size 8, KL coefficient 0.0, clip 0.20/0.28, global batch 8192, 16 rollout epochs, lr 2e-5.
  - The model is OpenVLA-OFT, starting from 10-demo (1 per task) SFT checkpoints for Spatial and Object, with initial success 56.9 and 55.6. Long uses 432 demos and starts at 83.0.
  - The budget is 10,240 episodes per task for Spatial and Object, and 5,120 for Long, with 5 training tasks per sequence. Pi-0 was run on RoboCasa using Flow-SDE — [arXiv HTML v3, Tables 1, 5, 7](https://arxiv.org/html/2603.11653v3)
- **SRPO** (Fei et al.; arXiv 2511.15605, v2 30 Nov 2025; appears in CVPR 2026 open access).
  - It is GRPO-style. Failed trajectories get progress rewards: their distance, in world-model latent space, to the successful trajectories in the same batch.
  - LIBERO rises from 48.9 to 99.2 in 200 RL steps. The abstract also claims a 167% improvement on LIBERO-Plus — [arXiv abs](https://arxiv.org/abs/2511.15605); [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_SRPO_Self-Referential_Policy_Optimization_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf)
- **pi-StepNFT** (Wang et al.; arXiv 2603.02083, v2 9 Mar 2026). This is critic-free and likelihood-free "step-wise negative-aware fine-tuning" for flow VLAs.
  - It regresses the next denoising step and uses a logistic contrastive ranking loss. It needs one forward pass per step.
  - The abstract claims LIBERO gains "with competitive few-shot robustness". It extends RLinf — [arXiv abs](https://arxiv.org/abs/2603.02083); [GitHub](https://github.com/wangst0181/pi-StepNFT)
- **RL4VLA, "What Can RL Bring to VLA Generalization?"** (Liu et al.; arXiv 2505.19789; "Accepted by NeurIPS 2025"). It finds "PPO as a more effective RL algorithm for VLAs than LLM-derived methods like DPO and GRPO" — [arXiv abs](https://arxiv.org/abs/2505.19789)

#### Likelihood-free improvement: advantage conditioning, steering, sample-and-distill

- **pi*0.6 / RECAP** (Physical Intelligence; arXiv 2511.14759, v1 18 Nov 2025).
  - The policy is conditioned on a text token, "Advantage: positive/negative". The token is a binarized improvement indicator, thresholded per task at the 30th percentile of predicted values.
  - The value function is distributional over B bins. It predicts negative steps-to-completion normalized to (−1, 0) and uses a 670M Gemma-3 backbone.
  - At inference, CFG with β > 1 is optional.
  - Training data are demos, autonomous rollouts and expert interventions. Per iteration: 300 trajectories for t-shirts; 600 autonomous plus 360 intervention trials for box assembly.
  - Results: task throughput more than doubles and the failure rate roughly halves on the hardest tasks. The paper says an AWR baseline "leads to much slower policies with lower throughput" and PPO "fall[s] far short".
  - Motivation stated in the paper: policy gradients are hard to apply "to flow matching models, which do not readily provide a tractable log-likelihood" — [arXiv HTML](https://arxiv.org/html/2511.14759) (summarizer extraction with quotes)
- **RECAP in RLinf (open-source, offline, pi0.5).** It runs in four stages: returns, value-model SFT (SigLIP2 + Gemma3-270M, 201 bins), N-step advantages labelled positive for the top 30%, then CFG training of **pi0.5 PyTorch**.
  - Only positive samples serve as conditional inputs, and 10% of them are dropped to unconditional.
  - `cfgrl_guidance_scale` sets the inference strength.
  - Successful SFT data use r = −1 per step; failed rollouts get a terminal penalty (e.g., −300) — [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- **STEAM** (in RLinf). It is the same CFG training as RECAP, but the value signal is an ensemble of pairwise temporal-progress critics with a worst-of-N aggregation, which suppresses overestimation on OOD rollouts — [RLinf STEAM docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html)
- **CFGRL, "Diffusion Guidance Is a Controllable Policy Improvement Operator"** (Frans, Park, Abbeel, Levine; arXiv 2505.23458, 29 May 2025; venue unverified).
  - The policy is written as π ∝ π̂(a|s)·f(A)^w. Guidance weight w > 1 gives further improvement. The model is trained by plain conditional flow/diffusion loss on an optimality indicator, with no value function required.
  - Increasing w increases performance. AWR saturates near 1/β = 10, while guidance keeps improving.
  - CFGRL beats AWR on most ExORL and OGBench tasks. Code: github.com/kvfrans/cfgrl — [arXiv HTML](https://arxiv.org/html/2505.23458)
- **DSRL, "Steering Your Diffusion Policy with Latent Space RL"** (Wagenmaker et al.; arXiv 2506.15799, v2 25 Jun 2025; CoRL 2025 per RLinf docs, not checked against proceedings).
  - SAC learns a small latent-noise policy. The base diffusion/flow policy stays frozen and is used as a black box, with no backprop through the chain.
  - pi0 on an **unseen LIBERO-90 task** ("pick up the cream cheese and put it in the tray"): the base was `pi0_libero`, fine-tuned on the 40 tasks of Spatial/Object/Goal/10. It went from ≈20% to ≈100% after ≈10,000 online samples, with chunk 50, 20 actions executed per chunk and 10 flow steps. Residual RL (ResiP) and V-GPS baselines did not improve pi0.
  - Real world: toaster 5/20→18/20 in ~80 episodes; spoon 15/20→19/20 in ~65 episodes. DSRL training ran on an RTX 3070 (8 GB) — [arXiv HTML](https://arxiv.org/html/2506.15799v2)
- **DSRL for pi0 in RLinf.** About 500K trainable params (actor plus a 10-head Q ensemble). It produces 32-dim noise per action-horizon step, and the example config is LIBERO-Spatial — [RLinf DSRL docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)
- **SARL, "Adapting Generalist Robot Policies with Semantic RL"** (Bhatia, Wagenmaker, Chen, Levine; arXiv 2606.31958, 30 Jun 2026).
  - The base VLA is **pi0.5, fine-tuned on LIBERO-90** (plus DROID/Bridge), and it is frozen.
  - RL (TD Q-learning with softmax selection) acts over language prompts. The prompts are proposed by a VLM and cached (32 for LIBERO), and a "reset-to-home" command is added.
  - On held-out LIBERO-10 tasks and 4 WidowX tasks, success goes from near 0% "up to 80% after only 60-100 online episodes". SARL solved 5 LIBERO-10 tasks.
  - DSRL and residual RL "only learn effectively when the base policy already exhibits close to successful behaviors" — [arXiv HTML](https://arxiv.org/html/2606.31958)
- **PA-RL, Policy-Agnostic RL** (Mark et al.; arXiv 2412.06685, 9 Dec 2024; venue unverified).
  - It samples action candidates from the base policy, ranks them globally with a Q-function (Cal-QL/RLPD), refines them locally with gradients, then distills the result into the policy by supervised learning.
  - It improved OpenVLA 40%→70% in 40 minutes of real online RL, distilling through LoRA rank 32.
  - On real robots it gave 75-100% gains in 40-110 minutes (10-70 episodes). On Cup-to-Rack, **iterated filtered BC stayed at 50%** while PA-RL went 55%→90% — [arXiv HTML](https://arxiv.org/html/2412.06685)
- **iRe-VLA** (Guo et al.; arXiv 2501.16664; "Accepted to ICRA 2025").
  - It alternates two stages. RL trains only the lightweight action head and critic with the VLM frozen. SL trains the whole model (with LoRA) on successful trajectories.
  - Full-model RL fine-tuning caused performance drops. Freezing the VLM in both stages lowered the final performance.
  - Benchmarks are MetaWorld, Franka Kitchen and real robots, with a BLIP-2 3B backbone — [arXiv HTML](https://arxiv.org/html/2501.16664)
- **RLDG** (Xu, Li, Luo, Levine; arXiv 2412.09858; RSS 2025). Task-specific RL experts generate data, which is distilled into a generalist by SFT. It reports "up to 40% higher success rates" than training on human demos — [arXiv abs](https://arxiv.org/abs/2412.09858); [RSS 2025 proceedings](https://www.roboticsproceedings.org/rss21/p028.pdf)
- **PLD, "Self-Improving VLA Models with Data Generation via Residual RL"** (arXiv 2511.00091; ICLR 2026).
  - Residual actors are trained off-policy on a frozen VLA. A hybrid rollout biases data toward the base policy's state distribution. The data are then distilled by SFT, which works for flow and autoregressive heads.
  - About 99% on LIBERO — [arXiv abs](https://arxiv.org/abs/2511.00091); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008318)
- **ConRFT** (Chen et al.; arXiv 2502.05450; RSS 2025).
  - Offline stage: BC plus Q-learning on a small demo set. Online stage: a consistency-policy head with human interventions.
  - 96.3% average on 8 real tasks within 45-90 minutes of online fine-tuning — [arXiv abs](https://arxiv.org/abs/2502.05450); [RSS 2025 proceedings](https://www.roboticsproceedings.org/rss21/p019.pdf); [GitHub](https://github.com/cccedric/conrft)
- **RL-100** (Lei et al.; arXiv 2510.14830, v4 10 Mar 2026; venue unverified).
  - A diffusion visuomotor policy goes through IL, then iterative offline RL, then online RL. A clipped PPO surrogate is applied across the denoising steps, and consistency distillation gives one-step control.
  - Real world only: 1000/1000 successful episodes — [arXiv abs](https://arxiv.org/abs/2510.14830)
- **VLAC** (Zhai et al.; arXiv 2509.15937, 19 Sep 2025; venue unverified).
  - One model emits both a pairwise "progress delta and done signal" (a dense reward) and actions.
  - In real-world RL, success goes from about 30% to about 90% within 200 episodes, and human-in-the-loop adds 50% sample efficiency — [arXiv abs](https://arxiv.org/abs/2509.15937)
- **Hi-ORS** (Lu et al.; arXiv 2510.26406, 30 Oct 2025).
  - It filters out negatively rewarded samples (online rejection sampling), uses a reward-weighted supervised objective, and adds human corrections.
  - It fine-tunes a "pi-base policy" in 1.5 h of real training — [arXiv abs](https://arxiv.org/abs/2510.26406)

#### World-model RL

- **WMPO** (Zhu et al.; arXiv 2511.09515; ICLR 2026).
  - GRPO with no KL runs entirely inside a pixel video world model (OpenSora-based, OXE-pretrained, then fine-tuned on policy rollouts). A VideoMAE success classifier (F1 > 0.95) gives the reward, and the policy is OpenVLA-OFT.
  - Mimicgen, 4 tasks: with 128 real trajectories, WMPO 47.1 vs GRPO 33.2 and DPO 37.3; with 1,280, WMPO 57.6 vs 37.1 and 42.4.
  - Compute is 32 H100s — [arXiv HTML](https://arxiv.org/html/2511.09515); [ICLR 2026 (ML Anthology)](https://mlanthology.org/iclr/2026/zhu2026iclr-wmpo/); [GitHub](https://github.com/WM-PO/WMPO)
- **VLA-RFT** (Li et al.; arXiv 2510.00406, 1 Oct 2025; venue unverified).
  - A 138M autoregressive world model serves as the simulator. The "verified reward" is the L1 plus LPIPS distance to reference trajectories.
  - The base is VLA-Adapter with a flow head, made stochastic through an SDE "Sigma Net". Training is GRPO on 4× A800 with veRL.
  - LIBERO average rises from 86.6 to 91.1 in 400 steps (Spatial 88.4→94.4, Object 88.0→94.4, Goal 92.8→95.4, Long 77.2→80.2) — [arXiv HTML](https://arxiv.org/html/2510.00406). The summarizer turned the base's SFT budget "15w" into "15 weeks"; it is most likely 150K steps (w = 10^4), which is my inference.
- **WISE** (arXiv 2609.03681, 3 Sep 2026). It schedules world-model imagination only at interaction-relevant states. It reports gains with **pi0 and pi0.5** and about 80% less GPU time than full imagination — [arXiv abs](https://arxiv.org/abs/2609.03681)
- **Ctrl-World** (Guo, Shi, Chen, Finn; arXiv 2510.10125). It is a controllable DROID-trained world model. SFT on successes synthesized in imagination raises policy success by 44.7% — [arXiv abs](https://arxiv.org/abs/2510.10125)

#### Flow-policy RL building blocks
These are the IDs verified on arXiv.
- Flow-GRPO, the ODE→SDE source for Flow-SDE: arXiv 2505.05470 — [arXiv](https://arxiv.org/abs/2505.05470)
- ReinFlow, learnable noise injection, the source for Flow-Noise: arXiv 2505.22094 — [arXiv](https://arxiv.org/abs/2505.22094)
- Flow Matching Policy Gradients (FPO): arXiv 2507.21053 — [arXiv](https://arxiv.org/abs/2507.21053)
- SAC Flow: arXiv 2509.25756 — [arXiv](https://arxiv.org/abs/2509.25756)
- StructRL (arXiv 2608.15139, 15 Aug 2026) argues that noise injected mid-chain is "diluted" by the remaining denoising steps. It puts structured noise directly in action space and uses "last-step replay" — [arXiv abs](https://arxiv.org/abs/2608.15139)

#### Other 2026 work found (abstract level only)
- ExToken (2607.12931): exploration tokens; "trajectory diversity is fundamentally more important to sample efficiency than the sheer quantity" — [arXiv](https://arxiv.org/abs/2607.12931)
- Offline supervision for VLA RL (2607.19399): about half the RL budget — [arXiv](https://arxiv.org/abs/2607.19399)
- SVA "Look Before You Leap" (2607.03751): MCTS distilled into a Q-evaluator that reranks a frozen VLA's proposals; pass@1 33% vs pass@32 92% — [arXiv](https://arxiv.org/abs/2607.03751)
- ALOE (2602.12691): off-policy evaluation for real-world VLA post-training — [arXiv](https://arxiv.org/abs/2602.12691)
- TEMPO (2608.07314): frozen VLM, two-timescale RL on the projection layer and the action expert — [arXiv](https://arxiv.org/abs/2608.07314)
- GPC (2510.01068, ICLR 2026): training-free convex score composition of diffusion/flow policies — [arXiv](https://arxiv.org/abs/2510.01068)
- EQRL (2606.14375): RL over latent input, denoising budget and chunk length without fine-tuning the VLA — [arXiv](https://arxiv.org/abs/2606.14375)
- ARFM (2509.04063): offline RL for flow VLAs with an adaptively scaled flow loss — [arXiv](https://arxiv.org/abs/2509.04063)
- Self-Improving Embodied Foundation Models (2509.15155, NeurIPS 2025): steps-to-go reward — [arXiv](https://arxiv.org/abs/2509.15155)
- PostBC (2512.16911): standard BC "can fail to ensure coverage" needed for RL fine-tuning — [arXiv](https://arxiv.org/abs/2512.16911)

#### ID / venue ledger
All IDs below were opened on arxiv.org/abs and title and dates confirmed.
- Venues confirmed from the arXiv comments field:
  - iRe-VLA: ICRA 2025
  - RL4VLA: NeurIPS 2025
  - RLinf-VLA: RSS 2026
  - Simple Recipe: RLC 2026
  - Self-Improving EFMs: NeurIPS 2025
  - GPC: ICLR 2026
- Venues confirmed from a proceedings or conference page:
  - ConRFT: RSS 2025 ([rss21/p019](https://www.roboticsproceedings.org/rss21/p019.pdf))
  - RLDG: RSS 2025 ([rss21/p028](https://www.roboticsproceedings.org/rss21/p028.pdf))
  - WMPO: ICLR 2026 ([ML Anthology](https://mlanthology.org/iclr/2026/zhu2026iclr-wmpo/))
  - PLD: ICLR 2026 ([iclr.cc](https://iclr.cc/virtual/2026/poster/10008318))
  - SRPO: CVPR 2026 ([CVF open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_SRPO_Self-Referential_Policy_Optimization_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf))
- Venue from a repo README only:
  - SimpleVLA-RL: ICLR 2026 ([GitHub](https://github.com/PRIME-RL/SimpleVLA-RL))
- Venue from a secondary doc only:
  - DSRL: CoRL 2025 ([RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html))
- Venue **unverified**: RIPT-VLA (an [OpenReview forum](https://openreview.net/forum?id=ytGamErI9m) exists, but its decision was not read), VLA-RL, piRL, TGRPO, VLAC, VLA-RFT, RL-100, PA-RL, CFGRL, pi*0.6 (arXiv only), and all 2026 arXiv-only papers listed above.

### Inferences
- **Likelihood access.** For our pi0.5 prior, a PPO/GRPO/RLOO log-probability needs a stochastic sampler: Flow-SDE, which keeps the marginals, or Flow-Noise, which adds a learnable noise head. Our deterministic 10-step Euler sampler gives no likelihood.
  - piRL's pi0.5 settings (3-5 denoise steps, noise 0.3-0.5) differ from our 10 Euler steps. The composition's guided sampler would have to be made stochastic in the same way, and the success rate at train time may fall under noise. piRL's ablation shows the SFT train-time success dropping from 62.3 to 46.6 as noise rises from 0.2 to 0.8.
- **Methods that need no likelihood.** RECAP, CFGRL, DSRL, SARL, PA-RL, PLD, RLDG and SVA avoid the likelihood problem. They fit a guided flow sampler more naturally.
- **Starting points are not comparable.** The one-demo LIBERO results (SimpleVLA-RL 48.9→96.9; SRPO 48.9→99.2; piRL pi0.5 77.1→98.3 from 40 trajectories) all start from checkpoints SFT'd on the *same* 40 standard LIBERO tasks they are then RL'd on. That is not held-out LIBERO-90 with a frozen prior.
  - SRPO's 48.9 start equals SimpleVLA-RL's one-trajectory OpenVLA-OFT average, which suggests the same checkpoint (inference, not stated in what I read).

### Gaps
- piRL does not say its 40 pi0.5 trajectories are exactly 1 per task, and it gives no GPU-hours.
- If one "train epoch" is 64 envs × 8 rollout epochs, piRL's LIBERO budget is up to about 256K episodes per suite. The paper does not state the total, so this is my arithmetic.
- SimpleVLA-RL gives no training-step count or GPU-hours. I did not extract its ablation of dynamic sampling, clip-higher and temperature.
- I did not extract quantitative LIBERO numbers for pi-StepNFT, StructRL, ExToken, WISE, TEMPO or ARFM (abstracts only).
- The PI "RL Token (RLT)" project page ([pi.website/research/rlt](https://www.pi.website/research/rlt)) returned HTTP 403. I found no arXiv ID; its RLinf description is in Q5.

## Q2. Which methods start from a single demonstration per task, and what success do they reach after how many rollouts?

### Takeaway
Four lines of work reach 90-99% from about one demo per task on LIBERO: RIPT-VLA (QueST, cross-scenario), SimpleVLA-RL and SRPO (OpenVLA-OFT), and piRL (pi0.5, 40 trajectories for 40 tasks). All of them spend about 10^3-10^5 rollouts per task.

The only reports at a budget of tens to about a hundred episodes on held-out LIBERO tasks with a frozen pi-family prior are:
- SARL: pi0.5 on LIBERO-10, near 0 → up to 80% in 60-100 episodes.
- DSRL: pi0 on one unseen LIBERO-90 task, ≈20% → ≈100% in ≈10K samples.

### Cited Findings
- **RIPT-VLA, cross-scenario (1 demo in the new scenario).**
  - Setup: QueST is pretrained on demos of the same goal in Scenario A, SFT'd on 1 demo in Scenario B, then RIPT'd on rollout contexts (initial states) from Scenario B. It is evaluated on 50 test contexts over 5 scenario pairs and 3 seeds.
  - SFT averages "around 5%" and falls "as low as 2%". The best gain was 3.5% → 97.2%.
  - The headline: an "unworkable SFT model (4%)" reaches 97% "within 15 iterations". The single-task setting uses K=16, B=160 rollouts per iteration on 1 GPU — [arXiv HTML §5.4, §5.1](https://arxiv.org/html/2505.17016)
- **RIPT-VLA, cross-goal.** QueST is pretrained on 50 demos of Task A, then gets 3-10 demos of Task B (a different goal in the same scene). With 3 demos, SFT scores 0.7% and RIPT 59.7%; one pair went from near 0 to 84.7%. With 10 demos, SFT scores 29.4% and RIPT 79.7% — [arXiv HTML §5.5](https://arxiv.org/html/2505.17016)
- **RIPT-VLA, multitask LIBERO-LONG with 1 demo per task.** RIPT gives "a 20.8 absolute gain". Adding more rollout contexts (initial states, no extra demos) "significantly improves performance" — [arXiv HTML §5.3, §5.6](https://arxiv.org/html/2505.17016)
- **SimpleVLA-RL, one trajectory per task.** OpenVLA-OFT averages 48.9 → 96.9, and LIBERO-Long goes 17.3 → 91.7. RL runs on 500 scenarios per suite with group size 8 — [arXiv HTML](https://arxiv.org/html/2509.09674v1)
- **SRPO.** 48.9 → 99.2 on LIBERO in 200 RL steps — [arXiv abs](https://arxiv.org/abs/2511.15605)
- **piRL, pi0.5.** A 40-trajectory unified SFT goes 77.1 → 97.9 (Flow-SDE) or 98.3 (Flow-Noise). pi0 uses 58 trajectories (208 for Long) and goes 57.6 → 96.1 or 97.6 — [arXiv HTML v3](https://arxiv.org/html/2510.25889v3)
- **Simple Recipe.** 10 demos (1 per task) start OpenVLA-OFT at 56.9 (Spatial) and 55.6 (Object). After sequential GRPO+LoRA the average is 81.2 and 93.2, using 10,240 episodes per task — [arXiv HTML v3, Tables 1 and 5](https://arxiv.org/html/2603.11653v3)
- **DSRL.** Zero demos on the target: `pi0_libero` goes from ≈20% to ≈100% on an unseen LIBERO-90 task after ≈10,000 online samples. On real robots, pi0 needs about 65-80 episodes — [arXiv HTML §5.5, App. C.5](https://arxiv.org/html/2506.15799v2)
- **SARL.** Zero demos on the target: pi0.5 fine-tuned on LIBERO-90 is tested on LIBERO-10 tasks and goes from near 0% "up to 80% after only 60-100 online episodes" on the tasks it solves (5 of 10 LIBERO-10 tasks) — [arXiv HTML](https://arxiv.org/html/2606.31958)
- **Real-robot, few-episode reports** (not one-demo LIBERO):
  - PA-RL: OpenVLA 40→70% in 40 min; diffusion policies improve within 10-70 episodes — [arXiv HTML](https://arxiv.org/html/2412.06685)
  - VLAC: about 30→90% in 200 episodes — [arXiv abs](https://arxiv.org/abs/2509.15937)
  - ConRFT: 96.3% in 45-90 min — [arXiv abs](https://arxiv.org/abs/2502.05450)

### Inferences
- **RIPT's rollout count.** RIPT's one-demo single-task result implies about 15 × 160 ≈ 2,400 rollouts, not counting the contexts rejected by dynamic sampling. This is my arithmetic from B = 160 and "15 iterations". It is roughly 50-100× our per-task budget.
- **RIPT's regime is ours.** Its cross-scenario setting (pretrained skill, one demo in a new layout, sparse success, a set of rollout init states) is the closest published analogue to our "one demo on a held-out LIBERO-90 task" regime. Its per-task gain (about 5% → 97%) is far larger than our 0.61 → 0.66. But its SFT starting points were near 0%, where our composition starts at about 61%.
- **The low-budget regime is steering.** At a budget of tens of rollouts, the only positive published evidence on LIBERO with a frozen pi-family prior comes from steering methods: SARL (prompts) and DSRL (noise). SARL says action-space steering helps only when the base policy is already near success. At about 61% zero-shot, we are in the regime where it does help.

### Gaps
- No paper found runs RL on pi0.5 from one demo per *held-out LIBERO-90* task with ≤100 rollouts per task and reports success on unseen init states.
- RIPT's per-iteration success curves for the one-demo runs are figures only; I could not read exact per-iteration numbers.
- DSRL's "online samples" unit is ambiguous (environment steps or chunk-level decisions).

## Q3. Which techniques specifically address the plateau of success-only imitation?

### Takeaway
The evidence converges on one point: learn from failures as well as successes, and concentrate updates on init states with mixed outcomes. The tools, with the evidence behind each:
- **Per-init-state baselines** (RLOO leave-one-out, GRPO group normalization) and **dynamic sampling** that drops all-success and all-failure groups: +3.3 average points in RIPT.
- A **success-rate curriculum** peaked near 50% (VLA-RL).
- **Trust-region clipping**: it also preserves old skills (ConSFT).
- Enough **exploration noise or temperature**: noise 0.5-0.8 for Flow-SDE, T = 1.6 for token VLAs.
- **Advantage or outcome conditioning with guidance weight w > 1**, which extrapolates beyond filtered BC and AWR (CFGRL, RECAP).
- **Direct evidence of the plateau**: iterated filtered BC stuck at 50% vs 90% for PA-RL.

### Cited Findings
- **Iterated filtered BC plateaus.** On the real-robot Cup-to-Rack task, iterated filtered BC (online) stayed at 50% while Cal-QL + PA-RL went 55% → 90% — [PA-RL arXiv HTML, Table 3](https://arxiv.org/html/2412.06685)
- **Dynamic sampling (RIPT-VLA).** On QueST, the average was 85.0 for the base, 90.2 for RIPT without dynamic sampling, and 93.5 with it (+3.3). Per suite: Goal 90.6→92.7, Spatial 91.3→95.6, Long 78.3→87.5, LIBERO-90 92.2→94.3. The paper says it "accelerates convergence" and gives "a consistent gradient signal across batches" — [arXiv HTML, Table 3](https://arxiv.org/html/2505.17016)
- **Leave-one-out baseline.** RIPT uses RLOO advantages within groups of K rollouts sharing one initial state, with no value net — [arXiv HTML §4.2](https://arxiv.org/html/2505.17016)
  - Init-state noise: grouping stays robust up to 2.5 cm (1×) of noise and degrades beyond 2×.
  - More distinct rollout contexts (init states) improve generalization "without additional human annotations" — [arXiv HTML §5.6](https://arxiv.org/html/2505.17016)
- **Exploration settings (SimpleVLA-RL).** It keeps only groups with 0 < successes < G, uses clip-higher (ε_high 0.28), removes KL "to … enable broader exploration", and samples rollouts at T = 1.6. It reports "pushcut": the policy discovers pushing where the demos grasp — [arXiv HTML](https://arxiv.org/html/2509.09674v1)
- **Curriculum (VLA-RL).** P(task_j) ∝ exp((0.5 − s_j)/τ) "prioritizes tasks with ~50% success rate as the frontier"; the method also uses critic warmup — [arXiv HTML](https://arxiv.org/html/2505.18719)
- **Noise level for flow VLAs** (piRL ablation: Flow-SDE, pi0, LIBERO-Spatial, eval after 100 RL steps).
  - Noise 0.2: 73.1. Noise 0.5: 94.5. Noise 0.8: 98.1.
  - Denoise steps 1/2/4/8: 88.5 / 97.0 / 94.5 / 86.7.
  - Chunk 5/10/20: 94.5 / 95.5 / 89.2 — [arXiv HTML v3, Table 2](https://arxiv.org/html/2510.25889v3)
- **Trust region.** On pi0, fine-tuning on one LIBERO-Spatial target task, the average prior-task success (Object and Goal) moved from 0.49 as follows: SFT 0.09; PPO without clipping 0.03; PPO with ε = 0.2 0.39. Target success was 0.90, 0.87 and 0.95 respectively — [ConSFT arXiv HTML, Table 1](https://arxiv.org/html/2605.08879)
- **Guidance beyond imitation (CFGRL).** Flow BC corresponds to w = 0 and goal-conditioned BC to w = 1, and "w > 1 lead[s] to improved performance". AWR performance saturates around 1/β = 10, while guidance "continues to improve". The paper cites "Imitating past successes can be very suboptimal" (Eysenbach et al., NeurIPS 2022) — [arXiv HTML](https://arxiv.org/html/2505.23458)
- **Advantage conditioning (RECAP / pi*0.6).** Binarized advantage (top 70% above the 30th-percentile threshold) is used as a text condition, with optional CFG β > 1. AWR was slower and had lower throughput; PPO "fall[s] far short" — [arXiv HTML](https://arxiv.org/html/2511.14759)
- **Using failures (SRPO).** Failed trajectories receive progress rewards measured against in-batch successes in world-model latent space. The paper's reason: "relying on binary success indicators wastes valuable information in failed trajectories". This quote comes from a search-result summary of the arXiv page, not a direct read — [arXiv abs](https://arxiv.org/abs/2511.15605); [arXiv HTML](https://arxiv.org/html/2511.15605v1)
- **Negative-aware losses.** pi-StepNFT uses a push-pull contrastive ranking loss on positives and negatives without a critic — [arXiv abs](https://arxiv.org/abs/2603.02083). Hi-ORS instead filters negatives but adds a reward-weighted objective and human corrections — [arXiv abs](https://arxiv.org/abs/2510.26406)
- **Alternating RL and SL (iRe-VLA).** RL on the action head alternates with SL on successes. The paper says RL on the full VLA was unstable, with drops in 4 of 5 tasks, and that freezing the VLM in both stages lowered the final performance — [arXiv HTML](https://arxiv.org/html/2501.16664)
- **Exploration structure.** ExToken claims diversity matters more than rollout count — [arXiv abs](https://arxiv.org/abs/2607.12931). StructRL shows mid-chain noise gets diluted by later denoising steps — [arXiv abs](https://arxiv.org/abs/2608.15139). SARL finds prompt-space exploration succeeds where action-noise steering fails on long-horizon tasks — [arXiv HTML](https://arxiv.org/html/2606.31958)
- **Initialization coverage (PostBC).** Standard BC "can fail to ensure coverage over the demonstrator's actions, a minimal condition necessary for effective RL finetuning" — [arXiv abs](https://arxiv.org/abs/2512.16911)

### Inferences
- **Why our second round plateaus.** It trains only on successes, so it cannot tell a "lucky" chunk from a "reliable" one on init states that succeed about 61% of the time. The failed rollouts on those same init states are the missing signal.
- **The cheapest change to try first.** Split the existing practice data per init state into success and failure. Use an RLOO-style per-init-state baseline, or simply a success/failure label, as the advantage. Train the fold outcome-conditioned in the RECAP/CFGRL way and sweep the guidance weight at test time. Rollouts on all-success and all-failure init states carry no RLOO/GRPO signal and would be dropped, as in RIPT's dynamic sampling.
- **Where to spend new rollouts.** Spend them on init states near 50% success (VLA-RL curriculum) and on more distinct init states rather than repeats (RIPT's context-size result).

### Gaps
- None of the plateau evidence (PA-RL's filtered BC, CFGRL vs AWR) is on pi0.5 or LIBERO-90 specifically.
- No paper directly compares success-only self-imitation against RLOO or advantage conditioning at a budget of tens of rollouts per task.

## Q4. Do any report forgetting of other tasks, or use per-task adapters?

### Takeaway
Yes. The consistent findings:
- On-policy RL through LoRA forgets little. Simple Recipe reports negative backward transfer (NBT) of 0.3 on LIBERO-Spatial.
- Full fine-tuning or SFT forgets heavily: NBT 40.9 without LoRA, and 78.7 when SFT replaces RL.
- A single-task SFT on pi0.5 cuts prior LIBERO skills from 0.77 to 0.23.
- PPO clipping preserves much of the prior (0.49 → 0.39 on pi0, vs 0.09 for SFT).
- Frozen-base steering (DSRL, SARL) cannot forget, by construction.

Per-task artifacts are common but large: RIPT's per-suite LoRA adaptor plus header is about 1 GB, and DSRL's noise policy has about 500K parameters.

### Cited Findings
- **Simple Recipe** (Table 1). Seq. FT with LoRA scores AVG / NBT / ZS (zero-shot on held-out tasks):
  - libero-spatial: 81.2 / 0.3 / 57.1
  - libero-object: 93.2 / 1.0 / 25.4
  - libero-long-horizon: 89.8 / −2.4 / 86.6
  - The multitask oracle scores 85.8, 95.7 and 90.5 average. Its zero-shot scores are 51.2 (Spatial), 27.6 (Object) and 85.2 (Long), so Seq. FT beats it on held-out tasks for Spatial and Long but not Object.
  - Baselines on Spatial: EWC 66.1, Expert Replay 80.2, DER 73.4, Dynamic Weight Expansion 79.6 (NBT 0.0), SLCA 69.9, RETAIN 66.0 — [arXiv HTML v3](https://arxiv.org/html/2603.11653v3)
- **Simple Recipe ablations** on Spatial (Table 3 / Fig. 6 data).
  - Without LoRA: AVG 7.3, NBT 40.9.
  - SFT instead of RL (432 demos): AVG 29.9, NBT 78.7.
  - Smaller policy (12M CNN): NBT 11.4.
  - LoRA rank 512: NBT 0.6. Full fine-tuning: NBT 40.9, with effective rank std 465.0 vs 5.7 for LoRA r32.
  - Pi-0 on RoboCasa Seq. FT: NBT −0.1.
  - Mechanisms proposed: on-policy RL "only reweight[s] probability mass where πθ already has support"; large-model null space; LoRA's low cross-layer rank variance.
  - Sequences of 30 tasks remain robust — [arXiv HTML v3](https://arxiv.org/html/2603.11653v3)
- **ConSFT** (Zhang et al.; arXiv 2605.08879, v2 19 May 2026). Fine-tuning on one LIBERO-Spatial target task, prior capability (average of Object and Goal):
  - pi0.5: base 0.77, SFT 0.23 (target 1.00), ConSFT 0.43.
  - pi0: base 0.49, SFT 0.09, ConSFT 0.34, LoRA 0.24, ER 0.18, LwF 0.05.
  - PPO baselines were trained on 5× A100 with sparse reward, K = 4 denoise steps, chunk 5 and noise 0.5 — [arXiv HTML, Tables 1-4](https://arxiv.org/html/2605.08879)
- **SimpleVLA-RL** (9 seen / 1 unseen task per suite). SFT "exhibits catastrophic forgetting" on unseen tasks (to 0% on a LIBERO-Goal unseen task). RL keeps or improves them, e.g., LIBERO-Spatial unseen task 1 from 43.3% to 71.8% — [arXiv HTML](https://arxiv.org/html/2509.09674v1) (summarizer extraction)
- **Pretrained VLAs resist forgetting** (Liu, Kim, Liu, Liu, Zhu; arXiv 2603.03818). "Simple Experience Replay (ER) works surprisingly well on VLAs, sometimes achieving zero forgetting even with a small replay data size" — [arXiv abs](https://arxiv.org/abs/2603.03818)
- **Real-world continual learning** (arXiv 2605.26820). Naive sequential fine-tuning "leads to severe catastrophic forgetting", and well-configured ER mitigates it — [arXiv abs](https://arxiv.org/abs/2605.26820)
- **Other continual-RL methods.**
  - LifeLong-RFT (arXiv 2602.10503): +22% average success over SFT on LIBERO continual learning, with 20% of the data — [arXiv abs](https://arxiv.org/abs/2602.10503)
  - CRL-VLA (arXiv 2602.03445): dual critic with a frozen goal-conditioned critic — [arXiv abs](https://arxiv.org/abs/2602.03445)
  - VLA-OPD (arXiv 2603.26666): on-policy distillation with reverse KL "effectively mitigating catastrophic forgetting" — [arXiv abs](https://arxiv.org/abs/2603.26666)
- **Per-task or per-suite artifacts.**
  - RIPT-VLA releases one LoRA adaptor per OpenVLA-OFT LIBERO suite; "RIPT LoRA Adaptor + Header are around 1GB", and the Laplace scale head is about 300 MB — [GitHub README](https://github.com/Ariostgx/ript-vla)
  - DSRL-for-pi0 in RLinf trains about 500K parameters with pi0 frozen — [RLinf DSRL docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)
  - SARL keeps pi0.5 frozen and learns a Q over at most 32 cached prompts for LIBERO — [arXiv HTML](https://arxiv.org/html/2606.31958)
- **iRe-VLA.** Full-model RL "can lead to performance degradation", while freezing the VLM "can avoid collapses" — [arXiv HTML](https://arxiv.org/html/2501.16664)

### Inferences
- **Our per-task fold is the most storage-efficient per-task artifact in this literature.** The 258 KB gated fold is 3-4 orders of magnitude smaller than RIPT's per-suite adaptor, and about 4× smaller than DSRL's roughly 500K parameters (about 1 MB at bf16, my arithmetic).
- **Isolation.** Because each fold is per task and gated, forgetting across tasks is prevented structurally, as with Dynamic Weight Expansion (NBT 0.0 in Simple Recipe). The literature says RL through low-rank adapters with clipping should not hurt the prior even without gating.
- **Risk.** Any RL variant that updates shared pi0.5 weights, as piRL does with the full action expert, would reintroduce the forgetting ConSFT measured on pi0.5 (0.77 → 0.23 under SFT).

### Gaps
- No paper reports forgetting for per-task gated adapters on pi0.5.
- No paper reports forgetting measured on LIBERO-90 specifically under RL.
- The Simple Recipe code README lists only OpenVLA and OpenVLA-OFT, although the paper also ran Pi-0 on RoboCasa.

## Q5. Code availability, compute, and pi0.5 PyTorch support

### Takeaway
**RLinf is the only open framework with RL for pi0/pi0.5 in PyTorch.** Its pi0/pi0.5 port is described as "numerical behavior aligned with the JAX reference". It offers:
- PPO/GRPO with Flow-SDE/Flow-Noise
- LoRA for pi0/pi0.5
- DSRL (pi0)
- RECAP-CFG (pi0.5)
- STEAM, RLT, SAC-Flow
- community extensions such as pi-StepNFT

SimpleVLA-RL, RIPT-VLA and VLA-RL support OpenVLA/OFT/QueST only. Reported compute ranges from one RTX 3070 (DSRL steering) and 4× A5000 (RIPT with OFT) to 8× H100 (piRL) and 32× H100 (WMPO). Only VLA-RL states GPU-hours (48).

### Cited Findings
- **RLinf README ([GitHub](https://github.com/RLinf/RLinf)).**
  - 2025/10: RL fine-tuning for π0 and π0.5.
  - 2025/11: "Add lora support to π₀ and π₀.₅".
  - 2026/01: SAC-Flow.
  - 2026/02: DSRL for Pi0; sim-real co-training for π0/π0.5.
  - 2026/04: RECAP.
  - 2026/06: STEAM.
  - 2026/07: "RLinf reimplements π₀ and π₀.₅ in PyTorch with numerical behavior aligned with the JAX reference implementations"; RLT; OPD.
  - 2026/09: ApxInf-accelerated π₀.₅ LIBERO evaluation.
  - Algorithm list includes PPO, GRPO, Reinforce++, SAC, CrossQ, RLPD, IQL, DSRL, RECAP (CFG), STEAM, RLT, OPD, SAC-Flow, DAgger. Hardware includes NVIDIA, AMD ROCm and Ascend.
- **RLinf pi0/pi0.5 docs.**
  - pi0.5 is selected by `openpi.config_name` (e.g. `pi05_libero`).
  - Flow method is chosen with `noise_method: flow_sde | flow_noise` and `noise_level`.
  - For pi0.5 PPO the doc says "should set `value_after_vlm: True`".
  - LoRA is set with `is_lora: True, lora_rank: 8`, described as for "the VLM part", without gradient checkpointing.
  - A "minimum test case" runs on 4 H100s: 32 envs, 2 rollout epochs, global batch 256, lr 1e-6, with offload — [RLinf pi0 docs (repo source)](https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/pi0.rst)
- **RLinf RECAP.** It requires "pi0.5 base (PyTorch)" converted from openpi. It is offline, with no new rollouts — [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- **RLinf RLT.**
  - Stage 1 SFTs pi0.5 together with an "RL token" transformer.
  - Stage 2 freezes it and trains a compact off-policy actor-critic on z_rl, proprioception and the reference chunk.
  - Examples cover Franka peg insertion and ManiSkill. The official page is "Precise Manipulation with Efficient Online RL" (pi.website/research/rlt) — [RLinf RLT docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)
- **SimpleVLA-RL.** Implemented for OpenVLA and OpenVLA-OFT. "Support advanced diffusion based RL: pi0 and pi0.5 with flow matching RL" is an **unchecked TODO**. Built on veRL; tested on 8× A800 80 GB — [GitHub README](https://github.com/PRIME-RL/SimpleVLA-RL)
- **RIPT-VLA.** Code covers QueST and OpenVLA-OFT, with checkpoints on HF (tanshh97/RIPT_VLA). It recommends 3 GPUs for LIBERO suites and 6 for LIBERO-90 with QueST — [GitHub README](https://github.com/Ariostgx/ript-vla)
- **Simple Recipe.** Code is built on RLinf and supports OpenVLA and OpenVLA-OFT with PPO and GRPO. It uses the SimpleVLA-RL one-trajectory SFT checkpoints (HF `Haozhan72/Openvla-oft-SFT-libero-spatial-traj1`) — [GitHub README](https://github.com/UT-Austin-RobIn/continual-vla-rl)
- **Compute by method.**
  - VLA-RL: 48 GPU hours; one GPU for vLLM inference plus the rest for learning — [arXiv HTML](https://arxiv.org/html/2505.18719); code [GitHub](https://github.com/GuanxingLu/vlarl)
  - DSRL: pi0 runs on a remote server; DSRL trains on an RTX 3070 8 GB. The RLinf docs cite reference code `nakamotoo/dsrl_pi0`; project page [diffusion-steering.github.io](https://diffusion-steering.github.io) — [arXiv HTML](https://arxiv.org/html/2506.15799v2); [RLinf DSRL docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)
  - WMPO: 8 H100 for SFT; 32 H100 for world model plus policy optimization — [arXiv HTML](https://arxiv.org/html/2511.09515); code [GitHub](https://github.com/WM-PO/WMPO)
  - VLA-RFT: 4× A800 — [arXiv HTML](https://arxiv.org/html/2510.00406)
  - piRL: 8× H100 — [arXiv HTML v3](https://arxiv.org/html/2510.25889v3)
  - TGRPO: single A100, no code — [arXiv HTML](https://arxiv.org/html/2506.08440)
  - ConSFT PPO baselines: 5× A100 (1 train, 4 eval) — [arXiv HTML](https://arxiv.org/html/2605.08879)
- **Other code links:**
  - CFGRL: [github.com/kvfrans/cfgrl](https://github.com/kvfrans/cfgrl) (URL from [arXiv HTML](https://arxiv.org/html/2505.23458))
  - ConRFT: [GitHub](https://github.com/cccedric/conrft)
  - pi-StepNFT: [GitHub](https://github.com/wangst0181/pi-StepNFT)
  - PLD: project page [wenlixiao.com/self-improve-VLA-PLD](https://wenlixiao.com/self-improve-VLA-PLD)

### Inferences
- **Reusable code for our PyTorch pi0.5 pipeline:**
  - RLinf's Flow-SDE log-prob code, for any likelihood-based variant.
  - Its RECAP CFG-training path, for advantage-conditioned folds.
  - Its DSRL module, for noise steering. It is written for pi0 but uses 32-dim noise, the same latent dimension as the pi-series noted by SARL.
- **What would need adapting.** RLinf's defaults target multi-H100 runs with 64 parallel envs. Our one-task, tens-of-rollouts regime would need small-batch configs. The 4-H100 "minimum test case" is the closest documented setup.

### Gaps
- No GPU-hour totals for RIPT-VLA, SimpleVLA-RL, piRL, RLinf-VLA, Simple Recipe or SRPO.
- It is unverified whether RLinf's PyTorch pi0.5 matches the openpi PyTorch port our pipeline uses (module names, the LoRA placement "VLM part" vs action expert).
- DSRL in RLinf is documented for pi0 only, not pi0.5.

## Q6. Which methods could run on our pipeline with tens of rollouts per task, and which component would we train?

### Takeaway
At tens of rollouts per task, published policy-gradient RL (RIPT/LOOP, GRPO, piRL-PPO) is 1-4 orders of magnitude over budget. Three published recipes fit, and none needs a flow likelihood; each trains either the fold or a few scalars.
1. **Outcome/advantage-conditioned fold with a guidance weight** (RECAP / CFGRL).
2. **Low-dimensional steering of the frozen prior** — the composition's guide weights or noise (DSRL-style), or prompts (SARL-style).
3. **Sample-rank-distill** (PA-RL-style) into the fold.

RIPT-style LOOP with dynamic sampling on the fold becomes possible if the practice budget grows to a few hundred rollouts per task.

### Cited Findings
- Reported budgets per task or suite (for scale):
  - RIPT one-demo: K=16 × B=160 per iteration, "within 15 iterations" — [arXiv HTML](https://arxiv.org/html/2505.17016)
  - Simple Recipe: 10,240 episodes per task — [arXiv HTML v3](https://arxiv.org/html/2603.11653v3)
  - piRL: 500 epochs × 64 envs × 8 rollout epochs per suite — [arXiv HTML v3](https://arxiv.org/html/2510.25889v3)
  - DSRL: ≈10K samples on one LIBERO-90 task — [arXiv HTML](https://arxiv.org/html/2506.15799v2)
  - SARL: 60-100 episodes — [arXiv HTML](https://arxiv.org/html/2606.31958)
  - PA-RL: 10-70 real episodes — [arXiv HTML](https://arxiv.org/html/2412.06685)
  - WMPO: 128 real trajectories plus a world model — [arXiv HTML](https://arxiv.org/html/2511.09515)
- RECAP and CFGRL train by a plain conditional flow/diffusion loss on an outcome/advantage label, and set improvement strength at inference through the guidance weight. RLinf implements this for pi0.5 PyTorch offline — [RECAP arXiv HTML](https://arxiv.org/html/2511.14759); [CFGRL arXiv HTML](https://arxiv.org/html/2505.23458); [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- DSRL keeps the flow policy frozen and learns only which initial noise to feed it. SARL keeps pi0.5 frozen and learns which prompts to feed it. SARL finds action-space steering works when the base is near success — [DSRL arXiv HTML](https://arxiv.org/html/2506.15799v2); [SARL arXiv HTML](https://arxiv.org/html/2606.31958)
- PA-RL samples candidates from the base, picks them with a Q-function, and distills by supervised learning (via LoRA r32 for OpenVLA). It beat iterated filtered BC (50%) with 55% → 90% — [arXiv HTML](https://arxiv.org/html/2412.06685)
- GPC shows that convex composition of diffusion/flow policy scores, with test-time search over the weights, can beat either parent policy without training — [arXiv abs](https://arxiv.org/abs/2510.01068)
- SVA reranks a frozen VLA's proposals with a distilled Q-evaluator, going from pass@1 33% toward pass@32 92% — [arXiv abs](https://arxiv.org/abs/2607.03751)

### Inferences
Everything in this subsection is my inference, fitted to our pipeline; none of it has been tested.

**1. Advantage-conditioned fold plus a guidance weight** (fits the current budget; trains the fold). Take the practice rollouts already collected per task: about 20-30 init states with successes and failures.
- Label each chunk's outcome, or an RLOO-style advantage, as the per-init-state success minus the leave-one-out mean. The init states with mixed outcomes carry the signal.
- Train the gated LoRA fold with a binary "advantage positive" condition, and keep it able to represent the unconditioned composition. RECAP's `positive_only_conditional` mode with 10% unconditional dropout is the reference recipe.
- At inference, sweep the guidance weight w on held-out init states.
- What gets stored: the fold (≈258 KB) plus one scalar w per task.
- Why it should help: it adds the failure signal that success-only self-imitation discards, with no likelihood and no new rollouts.
- Test: does it move 0.66 on unseen init states?

**2. Guide-weight or noise steering** (trains a few scalars or a tiny noise policy).
- Treat the composition's guide weights (and possibly the per-chunk initial noise) as the action space of a bandit or CEM search. RLOO-style group baselines and dynamic sampling (drop init states where all tries agree) sharpen the search.
- With about 20-30 init states and 2-4 tries each, this fits a budget of tens of rollouts. The storage cost is a handful of floats.
- Full DSRL (a SAC noise policy of ~500K params, about 1 MB) is heavier than the fold, and its reported budget (≈10K samples) is far above ours. A reduced version (a state-independent or low-dimensional noise/guide policy) is what fits.
- Per SARL, this helps where the base is already near success. At about 61% zero-shot success, most of our tasks should be in that regime.

**3. PA-RL-style sample-rank-distill** (trains the fold).
- Sample N candidate chunks per step from the composition, pick among them with a small Q or success classifier fit on practice data, and distill the picks into the fold.
- This generalizes success-only self-imitation: it ranks candidates instead of keeping whole successful episodes, which is where PA-RL beat filtered BC.
- Cost: a per-task Q network (storage) and the risk of Q overfitting on 20-30 init states.

**4. RIPT-style LOOP on the fold** (needs several hundred rollouts per task; trains the fold).
- Make the guided sampler stochastic with Flow-SDE (noise about 0.3-0.5, from piRL's pi0.5 settings) to get per-step Gaussian log-probs.
- Run RLOO with K = 8 per init state, PPO clip ε 0.1-0.2, and dynamic sampling.
- One iteration over 20-30 init states is already 160-240 rollouts, and RIPT's one-demo success needed about 15 iterations. This is out of budget unless practice grows by about 10-100×. Its trust region and LoRA-only update should keep forgetting low (Simple Recipe; ConSFT).

**5. Not recommended at this budget.**
- piRL-style PPO on the full 300M action expert: it breaks both storage and forgetting constraints.
- SimpleVLA-RL / Simple Recipe GRPO: about 10^4 episodes per task.
- World-model RL (WMPO, VLA-RFT, WISE): it trades rollouts for training a world model, at 4-32 GPUs.
- SARL-style prompt RL is plausible at our budget, but it needs a VLM prompt proposer and a reset primitive. That is outside the current composition and may conflict with the "stored cheaply" constraint only mildly: it stores a Q over cached prompts.

### Gaps
- None of the recipes above has been published on pi0.5 with a per-task gated LoRA on held-out LIBERO-90 at tens of rollouts. The effect sizes here are untested.
- No source reports how CFG guidance on a small LoRA fold (as opposed to full-model CFG training as in pi*0.6) behaves.
- I found no study of how many rollouts per init state (K) an RLOO baseline needs when a task's baseline success is about 60%, at budgets below 100.
