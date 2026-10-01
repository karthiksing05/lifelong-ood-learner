# RL fine-tuning of diffusion and flow-matching policies (generative-policy RL), as of 2026-09-27

Scope note. I verified every arXiv ID below by fetching its arXiv abstract page (title, date and authors match), on 2026-09-27. I checked venues on OpenReview (forum links given) or took them from the arXiv "Comments"/"Journal-ref" field. Where neither source confirmed a venue, the entry says "venue unverified". I read full texts (arXiv HTML) for DPPO, ReinFlow, FPO, FPO++, Flow-GRPO, pi_RL, DSRL, pi0-FPO (ICRA'26), RIPT-VLA, Q-chunking, OTQL, Q-VGM, Z-1, UniSteer, RL Token and SimpleVLA-RL. For the rest I read only the abstracts. Repos were checked through the GitHub API (existence, description, last push).

Notation. Our pipeline follows the openpi convention: x_t = t*eps + (1-t)*a, Euler sampling from t=1 (noise) to t=0 (action). Flow-GRPO and pi_RL write their SDE in the same convention (t=1 is noise). Q-VGM uses the opposite convention (tau=0 is noise).

---

## 1. Mechanisms: how each method gets a likelihood or gradient through the denoising chain, how it makes sampling stochastic, and what it trains

### Takeaway
The field has split into five mechanism families:
- **(A) Denoising-as-MDP with Gaussian transitions.** DDPO, DPOK, DPPO, ReinFlow, Flow-GRPO, pi_RL Flow-Noise/Flow-SDE, Z-1 and ProphRL all belong here. They need a stochastic sampler (DDPM, a learned noise net, or an ODE-to-SDE conversion) and give exact per-step log-probs.
- **(B) Likelihood-free surrogates built from the flow-matching loss.** FPO, FPO++, pi0-FPO, AWM, DiffusionNFT, ORW-CFM, ARFM and OTQL. These are sampler-agnostic and keep the pretraining loss.
- **(C) Critic-gradient or critic-weighted methods.** Diffusion-QL, QSM, IDQL, FQL, FlowRL, SAC Flow, Q-Flow, Q-VGM, EXPO, PA-RL and SDAC.
- **(D) Noise-space or latent steering of a frozen generator.** DSRL, UniSteer, DLSRL, HAF-Steer, LPS and SDN.
- **(E) Residual or reference-conditioned small actors.** RL Token, PLD, ResFiT and Policy Decorator.

For a frozen VLA with a small adapter and sparse success reward, (B), (D) and (E) are the families that avoid rewriting the sampler.

### Cited Findings

**Family A: denoising as an MDP with explicit Gaussian likelihoods**
- **DDPO** (Black et al., arXiv 2305.13301; ICLR 2024). The paper poses denoising as a multi-step decision problem, which enables a class of policy-gradient algorithms that beat reward-weighted likelihood approaches on text-to-image objectives (compressibility, aesthetics, VLM-judged alignment). — [arXiv](https://arxiv.org/abs/2305.13301); venue: [OpenReview ICLR 2024](https://openreview.net/forum?id=YCWjhGrJFD). PyTorch code with LoRA support: [kvablack/ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch); original: [jannerm/ddpo](https://github.com/jannerm/ddpo).
- **DPOK** (Fan et al., arXiv 2305.16381; NeurIPS 2023). Online policy gradient on a text-to-image diffusion model with KL regularization to the pretrained model; generally better than supervised fine-tuning on the reward. — [arXiv](https://arxiv.org/abs/2305.16381)
- **DPPO** (Ren et al., arXiv 2409.00588; ICLR 2025 poster).
  - Builds a two-layer "Diffusion Policy MDP": the outer layer is the environment and the inner layer is the denoising chain, optimized with PPO. — [arXiv](https://arxiv.org/abs/2409.00588); [OpenReview ICLR 2025](https://openreview.net/forum?id=mEpqHvbD2h)
  - Efficiency devices: fine-tune only the last K' denoising steps, or fine-tune DDIM with eta=1 during training (exploration noise, which avoids a Dirac likelihood) and eta=0 at evaluation. — [arXiv](https://arxiv.org/abs/2409.00588)
  - Ablations: the critic must depend on the environment state but not on the denoised action; there is a "sweet spot" for clipping the denoising-noise std; fine-tuning fewer than K steps is fine. — [arXiv](https://arxiv.org/abs/2409.00588)
  - Zero-shot sim-to-real on Furniture-Bench One-leg: DPPO reaches 80% (16/20) on hardware, while a Gaussian policy that reaches 88% in sim gets 0% on hardware. — [arXiv](https://arxiv.org/abs/2409.00588)
- **ReinFlow** (Zhang et al., arXiv 2505.22094; NeurIPS 2025 per arXiv journal-ref).
  - Mechanism: a learnable noise-injection network sigma_theta' turns each Euler step into a Gaussian transition (mean = Euler update, std = sigma_theta'). The log-prob of the whole denoising chain is then exact and closed-form, even at 1 to 4 steps.
  - The noise net is trained jointly with the velocity net, and the noise level decays by itself as success approaches 100%.
  - Results: manipulation success improves by +40.34% on average; locomotion reward grows 135.36%; wall time is 62.82% lower than DPPO.
  - The authors say they "prioritize wall-time efficiency ... over sample cost."
  - [arXiv](https://arxiv.org/abs/2505.22094); code: [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow)
- **Flow-GRPO** (Liu et al., arXiv 2505.05470; NeurIPS 2025 poster).
  - ODE-to-SDE conversion that preserves the marginals.
  - Update rule: x_{t+dt} = x_t + [v_theta(x_t,t) + (sigma_t^2/(2t))(x_t + (1-t) v_theta(x_t,t))] dt + sigma_t sqrt(dt) eps, with sigma_t = a*sqrt(t/(1-t)).
  - Each step is an isotropic Gaussian, so the KL to the reference policy has a closed form.
  - "Denoising Reduction": train with about 10 steps and keep the full schedule at inference.
  - The KL term prevents reward hacking.
  - [arXiv](https://arxiv.org/abs/2505.05470); [OpenReview NeurIPS 2025](https://openreview.net/forum?id=oCBKGw5HNf); code [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)
- **Image-domain GRPO successors (mechanism only).**
  - DanceGRPO applies GRPO to both diffusion and rectified flows. — [arXiv 2505.07818](https://arxiv.org/abs/2505.07818)
  - MixGRPO uses SDE sampling and optimization only inside a sliding window of timesteps and ODE elsewhere, with about 50% lower training time than DanceGRPO. — [arXiv 2507.21802](https://arxiv.org/abs/2507.21802)
  - Flow-Factory is a unified framework for RL in flow-matching models. — [arXiv 2602.12529](https://arxiv.org/abs/2602.12529)
  - Venues unverified for all three.
- **pi_RL** (Chen et al., arXiv 2510.25889; venue unverified).
  - **Flow-Noise:** a ReinFlow-style learnable noise net, a one-layer MDP, and the exact joint log-likelihood of the denoising sequence. The noise net is discarded after training, leaving a deterministic policy.
  - **Flow-SDE:** the Flow-GRPO ODE-to-SDE conversion, plus a DPPO-style two-layer MDP, plus hybrid ODE-SDE rollouts. One randomly chosen denoising step per env step is stochastic; the rest are ODE.
  - Optimizer: PPO with GAE.
  - The critic is attached either to the action expert or to the VLM output. For pi0.5 the critic sits on the VLM, because pi0.5 folds the state into the prompt.
  - [arXiv](https://arxiv.org/abs/2510.25889)
- **NCDPO** (arXiv 2505.10482; venue unverified). Pre-samples and fixes the whole noise sequence, treats the diffusion policy as a noise-conditioned deterministic map, and backpropagates through all denoising timesteps. From scratch it reaches sample efficiency comparable to MLP+PPO. — [arXiv](https://arxiv.org/abs/2505.10482)
- **GenPO** (arXiv 2505.18763; NeurIPS 2025 per arXiv comment). Exact diffusion inversion through a "doubled dummy action" gives invertible mappings and exact log-likelihoods, which support entropy/KL estimation for PPO in IsaacLab. — [arXiv](https://arxiv.org/abs/2505.18763)
- **DiPOD** (arXiv 2606.13795; venue unverified). Diagnoses a "double drift": optimizing an ELBO surrogate lets the ELBO separate from the true log-likelihood. The fix adds an on-policy ELBO (self-distillation) regularizer to each diffusion policy-gradient step. — [arXiv](https://arxiv.org/abs/2606.13795)

**Family B: likelihood-free surrogates from the flow-matching (CFM) loss**
- **FPO, Flow Matching Policy Gradients** (McAllister, Ge, Yi, Kanazawa et al., arXiv 2507.21053; ICLR 2026 poster).
  - Replaces the PPO likelihood ratio with r_FPO = exp(ELBO_theta(a|o) − ELBO_old(a|o)), estimated from differences in the conditional flow-matching loss over Monte Carlo draws of (tau, eps).
  - This decomposes into the true likelihood ratio times a KL-gap correction.
  - The paper stresses that it is not bound to a specific sampler and does not lengthen the credit-assignment horizon, unlike methods that treat denoising as an MDP.
  - [arXiv](https://arxiv.org/abs/2507.21053); [OpenReview ICLR 2026](https://openreview.net/forum?id=eoEmoKoQpJ); code [akanazawa/fpo](https://github.com/akanazawa/fpo); blog [flowreinforce.github.io](https://flowreinforce.github.io)
- **FPO++, Flow Policy Gradients for Robot Control** (Yi, Choi et al., arXiv 2602.02481; venue unverified).
  - Uses a separate ratio per (tau, eps) sample instead of one ratio per action.
  - "ASPO" trust region: PPO-clip for positive-advantage actions, and the SPO quadratic penalty for negative advantages.
  - "Zero-sampling" at test time: integrate from eps=0.
  - Manipulation base policies (before fine-tuning) jump from about 10% to over 70% success with zero-sampling versus stochastic sampling.
  - Works for locomotion, humanoid sim-to-real (Booster T1, Unitree G1), and fine-tuning of image-based manipulation policies. It is slightly below a well-tuned Gaussian PPO on motion-tracking returns.
  - [arXiv](https://arxiv.org/abs/2602.02481); code [amazon-far/fpo-control](https://github.com/amazon-far/fpo-control); page [fpocontrol.github.io](https://fpocontrol.github.io/)
- **pi0-FPO** (Lyu et al., arXiv 2510.09976; ICRA 2026 per arXiv comment).
  - A latent (noise-space) actor proposes a latent chunk, and the frozen pi0 decodes it.
  - The actor is trained with a CFM-loss-difference ratio inside a PPO clip, with multi-step latent Euler exploration and a Q-ensemble critic.
  - On LIBERO-90 "pick up the butter and put it in the tray", the full method reaches 78.5%. Ablations: replacing the CFM ratio with SAC in latent space gives 32.4%; removing clipping gives 45.1%; single-step exploration gives 61.7%; a single critic gives 71.2%.
  - [arXiv](https://arxiv.org/abs/2510.09976)
- **AWM, Advantage Weighted Matching** (arXiv 2509.25050; ICML 2026 per arXiv comment).
  - Shows that DDPO is implicitly score/flow matching with noisy targets.
  - AWM keeps the pretraining flow-matching loss and weights each sample by its advantage.
  - Reports up to 34x speedup over Flow-GRPO on SD3.5-M/FLUX. The v1 HTML and the HF page say 24x; the camera-ready abstract says 34x.
  - [arXiv](https://arxiv.org/abs/2509.25050); code [scxue/advantage_weighted_matching](https://github.com/scxue/advantage_weighted_matching) (repo not opened)
- **DiffusionNFT** (Zheng et al., arXiv 2509.16117; ICLR 2026 Oral per arXiv comment).
  - Online RL on the forward process via flow matching: it contrasts positive and negative generations to define an implicit improvement direction.
  - Works with arbitrary black-box solvers, needs no likelihood, and needs only clean samples, not sampling trajectories.
  - Up to 25x more efficient than FlowGRPO.
  - [arXiv](https://arxiv.org/abs/2509.16117); code [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)
- **ORW-CFM-W2** (arXiv 2502.06061; ICLR 2025 poster). Online reward-weighted CFM with Wasserstein-2 regularization. — [arXiv](https://arxiv.org/abs/2502.06061); [OpenReview](https://openreview.net/forum?id=2IoFFexvuw)
- **ARFM** (arXiv 2509.04063; venue unverified). Offline-RL post-training for VLA flow models: an adaptively scaled advantage weight inside the flow loss trades RL signal against gradient variance. — [arXiv](https://arxiv.org/abs/2509.04063)
- **OTQL** (Sochopoulos et al., arXiv 2607.06262; venue unverified). Advantage-weighted conditional optimal-transport flow matching: a critic reshapes the OT coupling toward high-value samples, which both steers the flow and straightens it for fewer inference steps. — [arXiv](https://arxiv.org/abs/2607.06262)
- **FlowPRO / RPRO** (arXiv 2606.05468; venue unverified). Preference optimization for flow-matching VLA heads with a proximal regularizer that fixes the reward hacking of plain Flow-DPO. Pairs come from intervention-and-rollback on a real robot. — [arXiv](https://arxiv.org/abs/2606.05468)
- **CFGRL, "Diffusion Guidance Is a Controllable Policy Improvement Operator"** (Frans et al., arXiv 2505.23458).
  - Derives a direct relation between classifier-free guidance and policy improvement. Increasing the guidance weight reliably increases performance on offline RL tasks, and the method can work without a value function.
  - OpenReview lists it as "Submitted to ICLR 2026"; I found no acceptance.
  - [arXiv](https://arxiv.org/abs/2505.23458); [OpenReview](https://openreview.net/forum?id=PwaUsAk4i6)
- **pi*0.6 / RECAP** (Physical Intelligence, arXiv 2511.14759; venue unverified).
  - RL via advantage-conditioned policies, using demonstrations, on-policy data and teleoperated interventions.
  - Offline-RL pretraining followed by on-robot specialization.
  - "More than doubles task throughput and roughly halves the task failure rate" on the hardest tasks.
  - [arXiv](https://arxiv.org/abs/2511.14759)

**Family C: critic-based (Q-gradient, Q-weighted, one-step distillation)**
- **Diffusion-QL** (arXiv 2208.06193; ICLR 2023). Diffusion BC loss plus a Q-maximization term through the diffusion policy. — [arXiv](https://arxiv.org/abs/2208.06193); code [Zhendong-Wang/Diffusion-Policies-for-Offline-RL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)
- **IDQL** (arXiv 2304.10573; venue unverified). An IQL critic plus importance resampling of samples from a diffusion behavior policy, using critic-derived weights. — [arXiv](https://arxiv.org/abs/2304.10573); code [philippe-eecs/IDQL](https://github.com/philippe-eecs/IDQL)
- **QSM, Q-score matching** (Psenka et al., arXiv 2312.11752; ICML 2024). Matches the policy's score to the action-gradient of Q. It differentiates only through the denoiser, not the whole chain, and is off-policy. — [arXiv](https://arxiv.org/abs/2312.11752); project [michaelpsenka.io/qsm](https://michaelpsenka.io/qsm)
- **Consistency-policy RL.**
  - Ding and Jin (arXiv 2309.16984; ICLR 2024) use a consistency model as the actor in actor-critic for offline, offline-to-online and online RL. — [arXiv](https://arxiv.org/abs/2309.16984)
  - CPQL (arXiv 2310.06343; venue unverified) is a one-step consistency policy with Q-learning. — [arXiv](https://arxiv.org/abs/2310.06343)
  - ConRFT (arXiv 2502.05450; venue unverified) is a consistency-based offline+online fine-tuning of a VLA with human interventions: 96.3% average success on 8 real tasks within 45–90 minutes of online fine-tuning. — [arXiv](https://arxiv.org/abs/2502.05450)
- **FQL, Flow Q-Learning** (Park, Li, Levine, arXiv 2502.02538; ICML 2025). Trains a one-step policy with RL plus distillation from a BC flow policy, which avoids recursive backprop through the flow. Evaluated on 73 OGBench/D4RL tasks. — [arXiv](https://arxiv.org/abs/2502.02538); code [seohongpark/fql](https://github.com/seohongpark/fql)
- **FlowRL** (Lv et al., arXiv 2506.12811; NeurIPS 2025 poster). Maximizes Q through a deterministic-ODE flow policy under a Wasserstein-2 bound to a "behavior-optimal" policy derived implicitly from the replay buffer. Evaluated on DMControl and HumanoidBench. — [arXiv](https://arxiv.org/abs/2506.12811); [OpenReview](https://openreview.net/forum?id=CANUXhPoyn)
- **SAC Flow** (Zhang et al., arXiv 2509.25756; ICLR 2026 poster). Shows the flow rollout is algebraically a residual RNN, which explains its vanishing and exploding gradients. Reparameterizes the velocity net as a gated (Flow-G) or decoded (Flow-T) velocity, and uses a noise-augmented rollout for end-to-end SAC. — [arXiv](https://arxiv.org/abs/2509.25756); [OpenReview](https://openreview.net/forum?id=zZvWj4JrYj)
- **Q-Flow** (arXiv 2605.13435; ICML 2026 per arXiv comment). Propagates the terminal value to intermediate latent states along the deterministic flow, so it gets intermediate value gradients without unrolling the solver. Reports +10.6 pts on OGBench offline. — [arXiv](https://arxiv.org/abs/2605.13435)
- **Q-VGM** (Wang et al., arXiv 2606.08015; venue unverified).
  - Targets the pi0.5 action expert. From an optimal-control (HJB) view, the optimal velocity correction is (1/rho) * grad_x V.
  - At late denoising steps it takes a one-step look-forward x̂1_ref = x_tau + (1−tau) v_ref, improves x̂1 by normalized Q-gradient ascent, and regresses Δv_theta onto (x̂1_Q − x̂1_ref)/(1−tau). Gradients pass only through the local velocity, with no BPTT.
  - Uses an action-sensitive chunk critic on a frozen-VLM "RL token", trained with IQL offline and TD online.
  - [arXiv](https://arxiv.org/abs/2606.08015)
- **EXPO** (arXiv 2507.07986; ICLR 2026 poster). A base expressive policy trained by imitation, plus a light Gaussian "edit" policy; the higher-Q of (base, edited) actions is used for both acting and TD backups. Reports 2–3x sample-efficiency gains. — [arXiv](https://arxiv.org/abs/2507.07986); [OpenReview](https://openreview.net/forum?id=aFjSjkB6CV)
- **PA-RL** (arXiv 2412.06685; WRL@ICLR 2025 workshop oral). Samples multiple actions from the base policy, re-ranks them by Q (global) and takes gradient steps on them (local), then distills the "optimized" actions back into the policy with the ordinary supervised loss. Works for diffusion policies and OpenVLA. — [arXiv](https://arxiv.org/abs/2412.06685); [OpenReview](https://openreview.net/forum?id=VGOFxdc6AD)
- **SDAC / DPMD, reweighted score matching** (arXiv 2502.00361; ICML 2025 poster). — [arXiv](https://arxiv.org/abs/2502.00361); [OpenReview](https://openreview.net/forum?id=6Anv3KB9lz)
- **QVPO** (arXiv 2405.16173, NeurIPS 2024) and **DACER** (arXiv 2405.15177, NeurIPS 2024) are further online diffusion actor-critics (verified IDs; mechanism not studied here). — [QVPO](https://arxiv.org/abs/2405.16173); [DACER](https://arxiv.org/abs/2405.15177)
- **RLDT** (arXiv 2606.08602; venue unverified). Builds a transport field from a max-entropy objective with SVGD and aligns the flow to it, using expected-target estimates at intermediate steps to avoid BPTT. — [arXiv](https://arxiv.org/abs/2606.08602)
- **FASTER** (Dong, Swerdlow, Sadigh, Finn; arXiv 2604.19730; venue unverified). Treats best-of-N candidate filtering during denoising as an MDP and learns a denoising-space value that prunes candidates early. Applied to a pretrained VLA, it keeps performance while cutting compute. — [arXiv](https://arxiv.org/abs/2604.19730); code [alexanderswerdlow/faster](https://github.com/alexanderswerdlow/faster)
- **Adjoint Matching** (arXiv 2409.08861; ICLR 2025 spotlight). Reward fine-tuning of flow and diffusion models via memoryless stochastic optimal control. It needs a differentiable reward, so it does not directly fit a binary success signal. — [arXiv](https://arxiv.org/abs/2409.08861); [OpenReview](https://openreview.net/forum?id=xQBRrtQM8u)

**Family D: noise-space or latent steering of a frozen generator**
- **DSRL** (Wagenmaker, Nakamoto et al., arXiv 2506.15799; CoRL 2025 oral).
  - Runs RL in the latent-noise space: the actor picks the initial noise w, and the frozen diffusion or flow policy maps (s, w) to the action.
  - DSRL-SAC is plain SAC in w-space. DSRL-NA ("noise aliasing") trains an action-space Q with TD and distills it into a noise-space Q, which can use offline data.
  - Noise actions are box-constrained to [-b, b]^d so they stay in-distribution.
  - Needs only black-box sampling access to the base policy.
  - [arXiv](https://arxiv.org/abs/2506.15799); [OpenReview](https://openreview.net/forum?id=jU7AbGq3se); code [ajwagen/dsrl](https://github.com/ajwagen/dsrl), [nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0)
- **UniSteer** (arXiv 2605.10821; venue unverified). A DSRL-style noise actor plus approximate action-to-noise inversion of the frozen flow decoder (fixed-point inversion). Human corrective action chunks become noise-space supervision targets for the same actor that RL trains. — [arXiv](https://arxiv.org/abs/2605.10821)
- **DLSRL** (arXiv 2609.11270; venue unverified). The actor outputs an initial-noise latent plus a representation latent. The latter is mapped to adapter features and injected residually into the hidden states of the frozen generator's action tokens. — [arXiv](https://arxiv.org/abs/2609.11270); code [xianchaoxiu/DLSRL](https://github.com/xianchaoxiu/DLSRL)
- **HAF-Steer** (arXiv 2608.16837; venue unverified). Maps demonstrated actions back to their flow noise (invertibility), compresses them with DCT into low-dimensional "spectral actions", and runs offline-to-online RL there on a frozen flow VLA. — [arXiv](https://arxiv.org/abs/2608.16837)
- **LPS, Latent Policy Steering** (arXiv 2603.05296; RSS 2026 per arXiv comment). Backpropagates action-space Q-gradients through a differentiable one-step MeanFlow policy into a latent-space actor. — [arXiv](https://arxiv.org/abs/2603.05296)
- **EQRL** (arXiv 2606.14375; venue unverified). A lightweight adaptor chooses the latent input, the denoising budget and the chunk length per query for a frozen VLA. — [arXiv](https://arxiv.org/abs/2606.14375)
- **SDN** (arXiv 2606.14084; venue unverified). Training-free test-time selection of diffusion noise for pi0 and GR00T: +8% in sim and +10% in real. — [arXiv](https://arxiv.org/abs/2606.14084)
- **V-GPS** (arXiv 2410.13816; CoRL 2024). Re-ranks a generalist policy's sampled actions with an offline-RL value function; no weight access needed. — [arXiv](https://arxiv.org/abs/2410.13816)

**Family E: residual or reference-conditioned small actors on a frozen VLA**
- **RL Token (RLT)** (Xu, Springenberg, Equi, Amin, Esmail, Levine, Ke; arXiv 2604.23073; venue unverified).
  - The VLA is adapted to expose a compact "RL token". A small off-policy actor-critic is then trained on it.
  - The actor is conditioned on the VLA's sampled reference action chunk and regularized toward it. The authors contrast this with predicting residuals or latent noise.
  - Works over RL chunks of length C < H, where H is the VLA horizon.
  - Results: success rises "from 20% to 65%" on a screw-insertion task within minutes to hours of real practice.
  - [arXiv](https://arxiv.org/abs/2604.23073)
- **PLD** (arXiv 2511.00091; ICLR 2026 poster). Residual RL probes the failure regions, hybrid rollouts collect recovery data, and the result is distilled back into the VLA by SFT. Reports 99% on LIBERO. — [arXiv](https://arxiv.org/abs/2511.00091); [OpenReview](https://openreview.net/forum?id=eUGoqrZ6Ea)
- **ResFiT** (arXiv 2509.19301) learns per-step residuals with off-policy RL from sparse binary reward. — [arXiv](https://arxiv.org/abs/2509.19301)
- **Policy Decorator** (arXiv 2412.13630) is a residual policy; RLT characterizes it as needing "on the order of millions of steps". — [arXiv](https://arxiv.org/abs/2412.13630); [RLT](https://arxiv.org/abs/2604.23073)
- **RL-100** (arXiv 2510.14830). Applies a clipped PPO surrogate inside the denoising process in both offline and online stages, then consistency-distills to one step. Reports 1000/1000 real successes over 8 tasks. The GitHub repo says Science Robotics 2026; I did not verify that. — [arXiv](https://arxiv.org/abs/2510.14830); [Lei-Kun/RL-100](https://github.com/Lei-Kun/RL-100)

### Inferences
- Families B and D are the only ones that leave our guided Euler sampler untouched:
  - B trains from (observation, executed chunk, advantage) tuples with the ordinary flow-matching loss.
  - D only needs the sampler to accept an externally chosen initial noise.
- Family A needs a stochastic transition density at every denoising step, so the PoE guide would have to enter the mean of each Gaussian transition.
- The FPO++ zero-sampling result (about 10% to over 70% for manipulation base policies) suggests a zero-cost ablation: evaluate our pipeline with eps=0. This is untested for pi0.5.

### Gaps
- I did not open the full texts of DDPO, DPOK, IDQL, CPQL, GenPO, SDAC, QVPO, DACER or FlowRL, so their exact estimator variants are not detailed here.
- GFP ([arXiv 2512.03973](https://arxiv.org/abs/2512.03973)) and GoldenStart ([arXiv 2603.14245](https://arxiv.org/abs/2603.14245)) were verified by title only; I did not read their mechanisms.

---

## 2. Which methods have been applied to large pretrained VLAs or to pi0/pi0.5 flow action experts, with what budgets and success numbers

### Takeaway
Methods confirmed on pi0 or pi0.5:
- DSRL: pi0, frozen, including a held-out LIBERO-90 task.
- pi_RL (Flow-Noise = ReinFlow, and Flow-SDE): pi0 and pi0.5 on LIBERO, ManiSkill, MetaWorld and CALVIN.
- pi0-FPO: frozen pi0 plus a latent actor.
- Q-VGM: pi0.5 action expert.
- Z-1: GRPO on pi0.5 in RoboCasa.
- RECAP: pi0.6.
- SDN: test-time noise selection on pi0.

Other flow VLAs: OTQL on SmolVLA. On-policy methods reach about 94–98% on LIBERO, but with thousands to tens of thousands of episodes. Off-policy and steering methods report gains within tens to hundreds of episodes.

### Cited Findings
- **pi_RL on pi0 and pi0.5 (LIBERO).**
  - Starting points: few-shot SFT pi0 (58 trajectories for Spatial/Object/Goal, 208 for Long) and few-shot SFT pi0.5 (40 trajectories across all 40 tasks, i.e. one per task).
  - RL raises pi0 from a 57.6% average to 97.6% (Flow-Noise) and 96.1% (Flow-SDE), above the 94.2% full-data SFT baseline.
  - RL raises pi0.5 from 77.1% to 98.3%, above full-data SFT at 96.9%. LIBERO-Long goes from 43.9% to 94.0%.
  - Setup: VLM frozen and only the ~300M action expert trained. Evaluated over all 500 initial states per suite, on 8×H100.
  - [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
- **pi_RL budgets (Table 11, LIBERO).**
  - 64 parallel environments, 8 rollout epochs, global batch 2048, 500 "train epochs", PPO clip 0.2, gamma 0.99, GAE 0.95, actor lr 5e-6, critic lr 1e-4.
  - pi0.5 settings: action prediction horizon 10; replan horizon 5 (10 for Long); 3–5 denoise steps; Flow-SDE noise level 0.3–0.5.
  - pi0 settings: prediction horizon 50; replan 5 (10 for Long); 4 denoise steps.
  - The hyper-parameter ablation reports results "after 100 training steps".
  - [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
- **pi_RL other results.**
  - Appendix H applies PPO with Flow-SDE to a few-shot GR00T N1.5 on LIBERO: 52.5% to 89.9% (Table 10), with default configurations.
  - Real2Sim2Real: 20 motion-planner demos for SFT, then 100 RL iterations in ManiSkill, deployed zero-shot on a real Franka: SFT 0% versus RL 40%.
  - Adding VLM LoRA to RL training gave no clear benefit on LIBERO.
  - [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
- **DSRL on pi0.**
  - Base checkpoint: the public `pi0_libero` checkpoint, fine-tuned on 1,693 demos from LIBERO Spatial/Object/Goal/10.
  - Target task: LIBERO-90 "pick up the cream cheese and put it in the tray", which is not in that data.
  - DSRL-SAC goes from about 20% to about 100% "after only about 10,000 online samples" (3 seeds). Reward is −1 per step until success.
  - Baselines ResiP (residual RL) and V-GPS "are not able to significantly improve" pi0.
  - [arXiv 2506.15799](https://arxiv.org/abs/2506.15799)
- **DSRL on real robots.**
  - pi0-DROID on a real Franka: 80 online episodes (about 11,000 steps) on a toaster task and 65 episodes (about 10,000 steps) on spoon-to-plate, with significant improvement. The authors describe this as, to their knowledge, the first real-world RL fine-tuning of pi0.
  - "In some cases" real-world success goes from 20% to 90% with fewer than 50 online episodes (the paper's headline claim for real-world refinement).
  - [arXiv 2506.15799](https://arxiv.org/abs/2506.15799)
- **pi0-FPO (ICRA 2026).** LIBERO average 87.2% and LIBERO-Long 65.3%. ALOHA Transfer Cube goes from about 40% to over 65%; the figure marks 0 / 0.8M / 1.6M training steps. — [arXiv 2510.09976](https://arxiv.org/abs/2510.09976)
  - Inference: 87.2% is below the pi_RL numbers and below the full-data SFT of pi0 reported by pi_RL (94.2%), so the comparison is weak.
- **Q-VGM on pi0.5.**
  - Offline, from 150 policy rollouts per suite: few-shot-SFT four-suite average rises from 84.6% to 90.7%.
  - After online training: 98.4%.
  - Reaches 95% rollout success with 5.3x fewer episodes than on-policy RL, on average.
  - Three real bimanual tasks, offline: 66.7% to 98.3%.
  - [arXiv 2606.08015](https://arxiv.org/abs/2606.08015)
- **Z-1 on pi0.5 (RoboCasa, 24 tasks).**
  - Task-wise GRPO with Flow-SDE likelihoods from pi_RL, no KL term.
  - Shared-prefix groups by cloning the simulator state at the branch point, tree-structured branching, success-aware reward decay, and joint VLM+action-expert training only where action-expert-only GRPO stalls.
  - 80.6% average, +13.2 pts over SFT. SFT used 1,199 demos with official openpi JAX code.
  - [arXiv 2606.31846](https://arxiv.org/abs/2606.31846)
- **OTQL on SmolVLA.** The action head is first SFT'd on 10 demos because the public checkpoint failed zero-shot. Real "tape in penholder" goes from 14/30 to 24/30. Overall the VLA rises from 38% to 76% with 70% fewer NFEs, within a 50–60 episode budget. — [arXiv 2607.06262](https://arxiv.org/abs/2607.06262)
- **ReinFlow at VLA scale.** The official repo says ReinFlow has been "scaled up ... to fine-tune VLA models such as pi0 and pi0.5", with LIBERO code in RLinf, and supports GR00T N1.5. The technical report for that is pi_RL. — [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow)
- **ProphRL.** Flow-GRPO plus "intrinsic stepwise reweighting" (FlowScale), trained inside a learned action-to-video world model (Prophet). Reports 5–17% gains on public benchmarks and 24–30% on real robots "across diverse VLA backbones"; I did not confirm which backbones. — [arXiv 2511.20633](https://arxiv.org/abs/2511.20633)
- **RECAP / pi*0.6.** Real-world RL of a PI VLA through advantage conditioning (laundry, boxes, espresso). — [arXiv 2511.14759](https://arxiv.org/abs/2511.14759)
- **HABC.** Advantage-weighted BC from sparse episode outcomes, with viability and efficiency critic heads and intervention-aware credit. Real bimanual SFT baselines of 36/44/12% rise to 92/88/38%. The VLA backbone is not confirmed in the abstract. — [arXiv 2606.17043](https://arxiv.org/abs/2606.17043)
- **Autoregressive/OFT comparison points (not flow).**
  - SimpleVLA-RL: with one demonstration per task, RL on OpenVLA-OFT lifts LIBERO-Long from 17.1% to 91.7%. — [arXiv 2509.09674](https://arxiv.org/abs/2509.09674)
  - RIPT-VLA: with one demo, a QueST SFT model goes from 4% to 97% within 15 iterations; LIBERO-90 reaches 94.3% with QueST. — [arXiv 2505.17016](https://arxiv.org/abs/2505.17016)
  - RLinf-VLA: 98.11% on 130 LIBERO tasks. — [arXiv 2510.06710](https://arxiv.org/abs/2510.06710)

### Inferences
- DSRL's pi0 experiment is the closest published match to our setting:
  - a public LIBERO-finetuned pi checkpoint,
  - one held-out LIBERO-90 task,
  - sparse reward,
  - frozen base weights.
- pi_RL's pi0.5 one-trajectory-per-task SFT plus RL is the closest match on data regime. Its RL budget, however, is orders of magnitude above ours (see section 3).

### Gaps
- pi_RL's exact total episodes to convergence per suite is not stated in one number. The per-step batch (64 envs × 8 rollout epochs) and training-epoch counts are given, but I could not pin down the step at which each curve saturates.
- I could not confirm which VLA backbones ProphRL, PLD, HABC, UniSteer, DLSRL, FASTER or EQRL used, because only their abstracts were read.
- I found no published RL result on the openpi `pi05_libero` checkpoint applied to LIBERO-90 held-out tasks specifically.

---

## 3. Sample efficiency: episodes needed to improve a policy already at about 50–60%, and whether anything works with tens of rollouts per task

### Takeaway
On-policy PPO/GRPO methods for flow VLAs (pi_RL, Z-1, RIPT-style) use hundreds to thousands of rollouts per task. Only off-policy, critic-based or steering methods report gains at our budget:
- Q-VGM: 15 rollouts per task offline, +6.1 pts.
- OTQL: 50–60 episodes, 38% to 76%.
- DSRL: fewer than 50 to about 80 episodes on real robots, and about 10k steps on a LIBERO-90 task.
- UniSteer: 66 min, with human corrections.
- RLT and ConRFT: minutes to 90 minutes of real practice.

No paper demonstrates on-policy policy-gradient improvement with only tens of rollouts per task on a flow VLA.

### Cited Findings
- **Q-VGM.** 150 policy rollouts per LIBERO suite (10 tasks, so about 15 per task) lift pi0.5 few-shot SFT from 84.6% to 90.7% offline. It needs 5.3x fewer episodes than on-policy RL to reach 95%. — [arXiv 2606.08015](https://arxiv.org/abs/2606.08015)
- **OTQL.** "Can finetune and accelerate flows with an interaction budget of 50-60 episodes". Single-task policies go from 36% to 86%; a pre-trained VLA goes from 38% to 76%. — [arXiv 2607.06262](https://arxiv.org/abs/2607.06262)
- **DSRL.**
  - Real-world refinement: "in some cases" 20% to 90% with fewer than 50 episodes.
  - pi0-DROID on a real robot: 65–80 episodes.
  - pi0-LIBERO on a LIBERO-90 task: about 20% to about 100% in about 10k online samples.
  - [arXiv 2506.15799](https://arxiv.org/abs/2506.15799)
- **UniSteer.** Averaged over 4 real tasks, success rises from 20% to 90% in 66 minutes, beating DSRL and DAgger baselines. — [arXiv 2605.10821](https://arxiv.org/abs/2605.10821)
- **RLT.** "Within minutes to a few hours of practice", for example a screw-insertion task from 20% to 65%. — [arXiv 2604.23073](https://arxiv.org/abs/2604.23073)
- **ConRFT.** 96.3% average on 8 real tasks "within 45-90 minutes of online fine-tuning", with human interventions. — [arXiv 2502.05450](https://arxiv.org/abs/2502.05450)
- **RIPT-VLA** (on-policy, LOOP = RLOO + PPO, sparse binary reward).
  - Single-task setting: K=16 rollouts per context, B=160 rollouts per collection step, N=20 optimization iterations per rollout batch, lr 1e-6. The one-demo QueST result (4% to 97%) came "within 15 iterations". The paper's "iteration" unit may be the collect-then-optimize step; I did not resolve this.
  - Dynamic sampling drops groups with zero advantage (all success or all fail), which gives +3.3 pts.
  - More rollout contexts (initial states) "significantly improves performance".
  - OpenVLA-OFT was made stochastic by training a Laplace scale head with NLL for 500 steps.
  - [arXiv 2505.17016](https://arxiv.org/abs/2505.17016)
  - Inference: 15 × 160 ≈ 2,400 rollouts per task.
- **pi_RL.** 64 envs × 8 rollout epochs per training step; the ablations are read "after 100 training steps". — [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
  - Inference: tens of thousands of episodes per suite, i.e. thousands per task.
- **Stochastic exploration costs success (pi_RL, LIBERO-Spatial, pi0 SFT).**
  - Stochastic ("train") success falls as the Flow-SDE noise rises: 62.3% at 0.2, 56.0% at 0.5, 46.6% at 0.8. Deterministic eval stays at 65.2%.
  - After RL, deterministic eval is 73.1% / 94.5% / 98.1% for the same three noise levels. Low noise (0.2) gave instability with a high clip fraction.
  - With 1 denoise step, stochastic SFT success is only 9.4%.
  - [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
- **DPPO versus off-policy.** Demo-augmented off-policy methods (RLPD, Cal-QL, IBRL) "significantly improve sample efficiency vs. DPPO" on HalfCheetah. On Robomimic manipulation, however, off-policy Q-learning was often unstable while DPPO was stable. — [arXiv 2409.00588](https://arxiv.org/abs/2409.00588)
- **ReinFlow** explicitly optimizes wall time rather than sample count. — [arXiv 2505.22094](https://arxiv.org/abs/2505.22094)
- **FPO++.** DPPO baselines "often failed when base policies had low stochastic sampling success rates" (Can at about 10%). FPO++ was more robust to the base policy's starting success. — [arXiv 2602.02481](https://arxiv.org/abs/2602.02481)
- **HABC.** From sparse episode outcomes on real bimanual tasks: 36/44/12% to 92/88/38%. The episode count is not stated in the abstract. — [arXiv 2606.17043](https://arxiv.org/abs/2606.17043)

### Inferences
- Our budget is tens of rollouts per task over 20–30 init states, which gives mixed success/fail groups at about 61% base success.
- At that budget, the evidence favors:
  - (i) critic- or advantage-weighted supervised updates that reuse every rollout, successes and failures alike (Q-VGM, OTQL, AWM/DiffusionNFT-style, HABC);
  - (ii) noise-space steering with a small actor (DSRL/UniSteer).
- It does not favor PPO/GRPO through an SDE-converted sampler.
- Our current success-only self-imitation throws failures away. Every method above that works at small budgets uses negatives (a critic, advantages, or explicit positive/negative contrast). This may explain the plateau, but it is a hypothesis, not tested.

### Gaps
- No paper reports a controlled comparison at "tens of rollouts per task" between on-policy flow RL and critic-based updates on the same VLA.
- Q-VGM's 5.3x figure is relative to an unspecified on-policy baseline in the abstract; I did not extract the baseline's identity.

---

## 4. Credit assignment for action chunks: chunk as action, partial execution with replanning every k steps, and chunk length versus RL stability

### Takeaway
Almost all methods treat the executed chunk as one macro-action, with reward summed over the executed steps and a log-prob or CFM loss summed over the chunk's dimensions. Several report that longer executed chunks give a better imitation start but a lower RL ceiling or less stable RL. The consensus sweet spot is to execute about 5–10 steps.

### Cited Findings
- **pi_RL.** "We treat the entire sequence as a single macro-step and define its corresponding reward as the sum of the per-step rewards" (the RLinf-VLA "chunk-level formulation"). — [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
  - Chunk-size ablation on LIBERO-Spatial with pi0 (5/10/20): SFT eval 65.2/70.5/72.6 and RL eval 94.5/95.5/89.2.
  - A larger chunk "obscures precise reward credit assignment", lowers explained variance, and "limits the ceiling for subsequent RL".
  - The authors use chunk 10 for LIBERO-Long and 5 elsewhere.
  - Their pi0.5 configuration (predict 10, replan every 5) is identical to our querying scheme.
- **DPPO.**
  - Pre-training with a larger prediction horizon T_p and fine-tuning with a smaller executed T_a helps with a UNet head (e.g. T_p=16, T_a=8). With an MLP head, T_a < T_p "leads to training instability".
  - DPPO stays stable with executed chunks up to T_a=16, where Gaussian and GMM policies can fail to improve at all.
  - [arXiv 2409.00588](https://arxiv.org/abs/2409.00588)
- **ReinFlow.** The chunk is flattened into one vector; actions within a chunk are conditionally independent given the denoising state; the chunk log-prob is the sum of per-action log-probs. — [arXiv 2505.22094](https://arxiv.org/abs/2505.22094)
- **FPO++.** "We compute chunk-level ratios by summing CFM losses across all chunk timesteps" (horizon 16). It uses chunk 4 when matching the DPPO/ReinFlow setup. — [arXiv 2602.02481](https://arxiv.org/abs/2602.02481)
- **Q-chunking** (Li, Zhou, Levine, arXiv 2507.07969; NeurIPS 2025).
  - The critic Q(s, a_{t:t+h}) and the policy both work on h-step chunks executed open loop. This gives unbiased n-step backups and temporally coherent exploration.
  - The KL constraint is imposed implicitly by best-of-N sampling from a flow BC policy.
  - On cube-triple, h=10 performs best asymptotically. h=25 learns faster early and is the only length that solves task5, but it ends lower on the others. h=50 fails.
  - [arXiv](https://arxiv.org/abs/2507.07969); code [ColinQiyangLi/qc](https://github.com/ColinQiyangLi/qc)
- **RLT.** Uses an RL chunk C smaller than the VLA horizon H "to give the policy the ability to be more reactive", with a chunk-level C-step value. — [arXiv 2604.23073](https://arxiv.org/abs/2604.23073)
- **SmoothRL** (arXiv 2608.29768; venue unverified). Under asynchronous chunked execution, it splits each chunk into committed, execution and discarded regions, and propagates value gradients only through the executed region. — [arXiv](https://arxiv.org/abs/2608.29768)
- **EQRL.** Treats variable chunk length as a query-level macro-action with chunk-dependent discounting. — [arXiv 2606.14375](https://arxiv.org/abs/2606.14375)
- **DSRL on pi0.** pi0's 50×32 chunk gives a 1,600-dim noise space. DSRL trains a single-step 32-dim noise actor and critic and repeats the sampled noise across the chunk axis, which made SAC "more efficient". — [arXiv 2506.15799](https://arxiv.org/abs/2506.15799)
- **Z-1.** Optimizes GRPO only over a set of "trainable action chunks" per rollout, masking shared-prefix chunks. Shared prefixes come from cloning the sim state so that group comparisons isolate the task-critical phase. — [arXiv 2606.31846](https://arxiv.org/abs/2606.31846)
- **Q-VGM** uses an action-sensitive chunk critic with a gamma^H backup. — [arXiv 2606.08015](https://arxiv.org/abs/2606.08015)

### Inferences
- For us (query every 5 env steps, 10-step chunk), the literature supports treating each query as one macro-action with reward summed over the 5 executed steps.
- Whether the likelihood or CFM surrogate should cover all 10 predicted steps or only the 5 executed ones is unresolved:
  - DPPO and pi_RL use the whole predicted chunk.
  - SmoothRL restricts gradients to executed actions.
  - Restricting to executed steps would avoid rewarding un-executed actions.
- Our fold is gated to contact windows. Z-1's sim-state-cloned shared prefixes are a natural fit, because LIBERO states can be saved and restored: branch K rollouts at the start of a contact window, so advantages attribute success to what the fold does there. This is an inference; nobody has tested it for gated adapters.

### Gaps
- None of the pi-model papers ablates "replan every k" separately from chunk horizon for RL stability. pi_RL varies the executed chunk only.

---

## 5. Code availability and PyTorch-openpi compatibility

### Takeaway
RLinf is the main PyTorch stack for RL on pi0/pi0.5. It provides:
- Flow-Noise (ReinFlow) and Flow-SDE (PPO/GRPO),
- DSRL for pi0,
- LoRA for pi0/pi0.5,
- a July 2026 PyTorch reimplementation of pi0/pi0.5 aligned numerically with JAX.

Official openpi PyTorch does not support LoRA training. DSRL-pi0 is JAX. FPO, FPO++, DPPO and QC are standalone PyTorch or JAX research code for small policies and would have to be ported.

### Cited Findings
- **openpi.** PyTorch implementations of pi0 and pi0.5 were released in Sept 2025 and validated on LIBERO for inference and fine-tuning. Unsupported in PyTorch: pi0-FAST, mixed precision, FSDP, LoRA training and EMA. — [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- **RLinf** (last pushed 2026-09-25): — [RLinf/RLinf](https://github.com/RLinf/RLinf)
  - "[2026/07] RLinf reimplements pi0 and pi0.5 in PyTorch with numerical behavior aligned with the JAX reference implementations".
  - "[2025/11] Add lora support to pi0 and pi0.5".
  - "[2026/02] RLinf supports DSRL ... for Pi0".
  - LIBERO and ManiSkill RL for pi0/pi0.5 via openpi.
  - Papers: [arXiv 2509.15965](https://arxiv.org/abs/2509.15965); RLinf-VLA [arXiv 2510.06710](https://arxiv.org/abs/2510.06710) (RSS 2026 per arXiv comment).
- **pi_RL** is built on RLinf. — [arXiv 2510.25889](https://arxiv.org/abs/2510.25889)
- **ReinFlow**: [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow) (PyTorch; pushed 2026-04). The README advises:
  - Noise level: "When the SFT success rate is low, tune down noise to [0.04, 0.10] or [0.05, 0.12]".
  - Critic warmup: "The stronger your SFT checkpoint is, the more you need a critic warmup".
  - ReinFlow "may not be an optimal method to train RL agents from scratch".
- **DSRL.** Generic code at [ajwagen/dsrl](https://github.com/ajwagen/dsrl). The pi0 version, [nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0), is a "JAX-based implementation" that installs openpi and LIBERO as submodules and is built on jaxrl2.
- **FPO**: [akanazawa/fpo](https://github.com/akanazawa/fpo). **FPO++**: [amazon-far/fpo-control](https://github.com/amazon-far/fpo-control); its manipulation experiments cover FPO++, vanilla FPO and DPPO variants on 5 tasks.
- **DPPO**: [irom-princeton/dppo](https://github.com/irom-princeton/dppo) (last pushed 2025-02).
- **Flow-GRPO**: [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo). **DiffusionNFT**: [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT). **DDPO (PyTorch, LoRA)**: [kvablack/ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch).
- **RIPT-VLA**: [Ariostgx/ript-vla](https://github.com/Ariostgx/ript-vla). It covers QueST and OpenVLA-OFT on LIBERO/LIBERO-90, with RIPT LoRA adaptors for OFT. Venue: FMEA@CVPR 2025 workshop poster ([OpenReview](https://openreview.net/forum?id=ytGamErI9m)); "Submitted to ICLR 2026" ([OpenReview](https://openreview.net/forum?id=oXYZHg7HiZ)) with no acceptance found.
- **Q-chunking**: [ColinQiyangLi/qc](https://github.com/ColinQiyangLi/qc). **FQL**: [seohongpark/fql](https://github.com/seohongpark/fql). **IDQL**: [philippe-eecs/IDQL](https://github.com/philippe-eecs/IDQL). **CleanDiffuser** (diffusion decision-making library): [CleanDiffuserTeam/CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser).
- **DLSRL**: [xianchaoxiu/DLSRL](https://github.com/xianchaoxiu/DLSRL) (0 stars; pushed 2026-09-10). **FASTER**: [alexanderswerdlow/faster](https://github.com/alexanderswerdlow/faster) (repo not opened). **RL-100**: [Lei-Kun/RL-100](https://github.com/Lei-Kun/RL-100).
- **openpi-RLT.** A search result describes an openpi/pi0.5-based real-robot reproduction of RL Token at [Destiny000621/openpi-RLT](https://github.com/Destiny000621/openpi-RLT). I did not open or verify this repo.
- **Not found or not checked:** QSM code is referenced via [michaelpsenka.io/qsm](https://michaelpsenka.io/qsm) (the guessed GitHub path failed); PLD, OTQL, Q-VGM, Z-1, UniSteer and pi0-FPO code; SAC Flow and FlowRL repos.

### Inferences
- Our pipeline is a PyTorch openpi port with a custom gated LoRA fold. Porting a *loss* (FPO/FPO++ ratio, AWM weighting, DiffusionNFT contrast) is the smallest change, because each is a few lines around the existing flow-matching loss.
- Adopting RLinf would mean moving onto its pi0.5 reimplementation and its env workers, which is a larger migration.

### Gaps
- I did not check whether RLinf's pi0.5 PyTorch reimplementation loads the `pi05_libero` checkpoint directly.
- I did not check whether RLinf's LoRA covers only the action expert.

---

## 6. How each method could be applied to our pipeline (frozen pi0.5 + borrowed-chunk PoE guide + gated rank-2 fold; sparse success; tens of rollouts per task over 20–30 init states)

### Takeaway
Ranked by fit to our budget and architecture:
1. **Negative-aware, advantage-weighted flow matching on the fold/LoRA.** AWM, DiffusionNFT, FPO/FPO++ ratio, or HABC weights, with a group baseline per init state (RLOO as in RIPT-VLA). The Euler sampler and guide stay deterministic.
2. **DSRL- or UniSteer-style noise-space actor** over the initial noise, warm-started by inverting demo and composition chunks to noise.
3. **A small chunk critic** trained on all practice rollouts, used for best-of-N re-ranking (QC/V-GPS/FASTER) or for Q-VGM late-step velocity targets on the fold.
4. **Flow-SDE or Flow-Noise PPO/GRPO** (pi_RL/ReinFlow/Z-1) only if the budget can grow to thousands of rollouts per task.

### Cited Findings (the facts each proposal rests on)
- **CFM-ratio methods do not care how the rollout action was sampled.** FPO needs only the executed action and the CFM loss under the current and old parameters, which avoids sampler-bound MDP formulations. FPO++ sums CFM losses over chunk timesteps. — [FPO](https://arxiv.org/abs/2507.21053); [FPO++](https://arxiv.org/abs/2602.02481)
- **Advantage-weighted and contrastive variants reuse the pretraining loss and work with any solver.** AWM weights the flow-matching loss by advantage. DiffusionNFT contrasts positive and negative samples and needs only clean samples. — [AWM](https://arxiv.org/abs/2509.25050); [DiffusionNFT](https://arxiv.org/abs/2509.16117)
- **Group baselines from repeated rollouts of the same initial context work with binary reward.**
  - RLOO baseline b_k = mean of the other K−1 rewards; groups with zero advantage are filtered out.
  - More initial-state contexts improve generalization.
  - [RIPT-VLA](https://arxiv.org/abs/2505.17016)
- **Noise-space RL needs only the ability to choose the initial noise.**
  - DSRL box-constrains the noise to stay in-distribution and repeats one per-step noise across the chunk.
  - UniSteer inverts a frozen flow decoder to turn action chunks into noise targets.
  - HAF-Steer inverts demonstrations to noise and compresses them with DCT.
  - [DSRL](https://arxiv.org/abs/2506.15799); [UniSteer](https://arxiv.org/abs/2605.10821); [HAF](https://arxiv.org/abs/2608.16837)
- **Critic-guided velocity targets touch only late denoising steps and only the local velocity**, without BPTT. — [Q-VGM](https://arxiv.org/abs/2606.08015)
- **Best-of-N with a chunk critic is a KL-bounded improvement over the BC flow**, without training the actor. — [Q-chunking](https://arxiv.org/abs/2507.07969); [V-GPS](https://arxiv.org/abs/2410.13816); [FASTER](https://arxiv.org/abs/2604.19730)
- **Guidance weight as a policy-improvement knob.** CFGRL finds that raising the guidance weight on an optimality-conditioned diffusion policy raises performance. — [CFGRL](https://arxiv.org/abs/2505.23458)
- **Flow-SDE and Flow-Noise requirements** (pi_RL on pi0.5): a stochastic transition density at each denoising step, a critic (on the VLM output for pi0.5), 3–5 denoise steps, noise 0.3–0.5, and large parallel rollout budgets. Exploration noise lowers training-time success. — [pi_RL](https://arxiv.org/abs/2510.25889)
- **Flow-GRPO SDE step** (the same t=1 noise convention as ours): x_{t+dt} = x_t + [v + (sigma_t^2/(2t))(x_t + (1−t)v)] dt + sigma_t sqrt(dt) eps, with sigma_t = a sqrt(t/(1−t)). — [Flow-GRPO](https://arxiv.org/abs/2505.05470)
- **Reference-conditioned small actor.** The actor takes the VLA's reference chunk as input and is regularized toward it. — [RLT](https://arxiv.org/abs/2604.23073)

### Inferences (concrete proposals; none tested)

1. **Advantage-weighted / negative-aware fold training. Best fit; minimal sampler change.**
   - Keep the practice loop as is.
   - For each init state, run K rollouts (e.g. K=4–8) with the full guide+fold pipeline.
   - Compute an RLOO advantage from binary success.
   - For every executed query (observation, executed 10×7 chunk), add a loss term:
     - AWM: A · L_CFM(θ; chunk), with negative A allowed but clipped; or
     - DiffusionNFT: separate positive and negative flow-matching terms; or
     - FPO/FPO++: exp(L_old − L_θ) inside PPO-clip, with ASPO on the negative side.
   - Train only the fold (or the rank-16 LoRA).
   - Caveat: rollouts come from the guided sampler, not from the fold-only model, so they are off-policy for the fold. The ratio or weight then acts as a trust region around the old fold rather than an exact importance weight. The FPO ratio's KL-gap term makes this a biased surrogate anyway.
   - This directly addresses the "success-only self-imitation plateaus" finding, because failures get negative weight.
   - Sampler change: none.
2. **DSRL-style noise actor on top of the frozen pipeline.**
   - Expose eps (the t=1 noise) as the action of a small SAC actor that conditions on a VLM feature (DSRL used the last-token 2,048-d feature for pi0-DROID) plus proprioception.
   - Use a per-step 32-d noise repeated over the 10-step chunk, box-constrained.
   - Warm-start the actor's replay buffer or targets by inverting the demo and the borrowed composition chunks to noise (UniSteer/HAF-style fixed-point inversion through our guided sampler).
   - The composition's zero-training 61% then becomes the actor's prior.
   - Sampler change: accept an external eps.
   - DSRL's 20% to about 100% in about 10k samples on a held-out LIBERO-90 task with `pi0_libero` is the most on-target precedent.
3. **Chunk critic from all practice rollouts, used three ways.**
   - Train Q(s, chunk) on the RL-token or VLM prefix features, using IQL on the logged rollouts; failures included.
   - Use (a) for best-of-N re-ranking over sampled chunks at inference (no weight change);
   - (b) as Q-VGM late-step velocity targets for the fold (Q-VGM got +6.1 pts from about 15 rollouts per task on pi0.5);
   - (c) to gate the fold ("switch on in contact windows") by estimated value gain rather than by a fixed window.
   - Sampler change: run N parallel samples per query, for (a) only.
4. **Guide-weight RL.**
   - Treat the PoE guide mixing weight (and the fold gate) as low-dimensional actions or parameters.
   - Optimize them per task with RLOO/REINFORCE or CFGRL-style weight sweeps.
   - CFGRL's monotone guidance-weight improvement suggests that the guide weight is itself a policy-improvement knob.
   - Cheapest possible RL: a handful of scalars, which suits tens of rollouts.
5. **Flow-SDE/Flow-Noise PPO or GRPO** (pi_RL, ReinFlow, Z-1, Flow-GRPO).
   - Replace the Euler step with the Flow-GRPO SDE step on one randomly chosen denoising step per query (pi_RL's hybrid ODE-SDE), or add a ReinFlow noise head.
   - Include the guide's contribution in the transition mean so the Gaussian log-prob is well defined.
   - Train fold/LoRA parameters with PPO or GRPO, using a VLM-side value head for pi0.5.
   - Expect a drop in practice success during exploration (pi_RL: 65.2% deterministic versus 46.6–62.3% stochastic).
   - Expect budgets of thousands of rollouts per task.
   - Numerical caution: sigma_t = a·sqrt(t/(1−t)) diverges at t=1, so the first denoising step from pure noise needs special handling (e.g. keep it ODE).
   - Sampler change: significant.
6. **Evaluation-time zero-sampling check.** Evaluate the current pipeline with eps=0 (FPO++ zero-sampling) to see whether mode-seeking alone lifts success. This costs no training.

### Gaps
- No published method combines a product-of-experts guide or retrieval-composed chunks with RL fine-tuning of a flow VLA. How the guide interacts with ratio or likelihood estimates is our own analysis, not literature.
- Whether a rank-2 fold (258 KB) has enough capacity for RL gains is unknown. Every published pi0/pi0.5 RL result trains the full ~300M action expert (pi_RL, Q-VGM) or an external actor (DSRL, pi0-FPO, RLT).
- No paper tests RL on gated or contact-window-only adapters.
