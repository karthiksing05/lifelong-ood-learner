# Failure-aware and advantage-based learning from robot practice (catalogue as of 2026-09-27)

Scope note for the report writer. Our setting is a frozen pi0.5 (openpi `pi05_libero`, flow matching, 10-step chunks, Euler sampling) learning held-out LIBERO-90 tasks from one demo each. Practice gives sparse binary success on ~20-30 init states per task, tens of rollouts per task. The round-0 composition solves ~61% of practice instances. A rank-16 LoRA trained on demo + successes (filtered BC) reaches ~0.66 on unseen inits, then plateaus, and ~35% of instances keep failing. We record every rollout (images, states, executed actions) and an alignment pointer into the demo that stalls where things go wrong.

Verification method: I checked every arXiv ID in this file against the arXiv export API (title, first authors, submission date, comments field) on 2026-09-27. I checked the GitHub URLs with an HTTP request (all returned 200). Venues come from the arXiv comments field or the source named inline. "Venue not verified" means I found no primary confirmation. Most per-paper numbers came from the paper's arXiv HTML through a summarising fetch tool. Check them against the PDF before quoting them in a paper. Many 2026 items are unreviewed preprints, and I flag each one.

## Q1. Advantage/return-conditioned and weighted imitation (RECAP, AWR/AWAC/RWR for flow policies, QVPO, weighted flow matching, DT-style return conditioning, CFG-RL)

### Takeaway
Two families exist. Weighting families (AWR, RWR, energy- or advantage-weighted flow matching) scale the flow-matching loss per sample by exp(alpha * A). Conditioning families (RECAP, CFGRL, Decision Transformer/Decision Diffuser) feed an optimality or advantage token and, at inference, condition on "good" and optionally extrapolate with classifier-free guidance. The conditioning family keeps failures in the training set as labelled negatives. The strongest evidence (RECAP; CFGRL) says conditioning beats weighting. RECAP's value function is a distributional steps-to-completion model. Our alignment pointer already supplies a close analogue of that model for free.

### Cited Findings
**RECAP / pi*0.6** (Physical Intelligence, arXiv 2511.14759, Nov 2025; no venue listed)
- Value function: a distributional value p(V | o_t, l) over B = 201 bins, on a 670M VLM backbone initialised from Gemma 3, trained with cross-entropy on discretised returns — [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- Reward: r_t = 0 at the terminal step on success, -C_fail at the terminal step on failure, and -1 otherwise. The value is therefore the negative number of steps to success, normalised per task to (-1, 0) by maximum episode length — [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- Improvement indicator: I_t = 1(A(o_t, a_t, l) > eps_l), with eps_l set to the 30th percentile of predicted values for task l — [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- Conditioning: the indicator enters as a text input, "Advantage: positive" or "Advantage: negative". Training drops the indicator at random, so inference can use I = True (beta = 1) or CFG with beta > 1 — [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- Human corrections get forced I = True. Whole episodes, successes and failures alike, go into the dataset. Failures are therefore trained on with a negative label, not filtered out — [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- Data volume: about 300 trajectories per iteration (laundry) and 600 autonomous plus 360 intervention trials per iteration (box assembly), over 2 iterations — [arXiv HTML](https://arxiv.org/html/2511.14759v1)
- On the same data, AWR reached reasonable success but produced slower policies. PPO needed a small trust region (eta = 0.01) and "does not achieve good performance". Both "fall far short" of advantage conditioning — [arXiv HTML](https://arxiv.org/html/2511.14759v1); [PI blog](https://www.pi.website/blog/pistar06)

**CFGRL, "Diffusion Guidance Is a Controllable Policy Improvement Operator"** (Frans, Park, Abbeel, Levine; arXiv 2505.23458, May 2025). An [OpenReview forum](https://openreview.net/forum?id=PwaUsAk4i6) exists, but I could not confirm the venue.
- Training labels an optimality variable o = 1 if A(s, a) >= 0, else o = 0, with dropout to unconditional. Inference uses score = unconditional + w * (conditional(o = 1) - unconditional). A larger w improves the policy without retraining — [arXiv HTML](https://arxiv.org/html/2505.23458)
- The paper contrasts CFGRL with AWR: AWR's temperature is "chosen beforehand and folded into the optimization objective". CFGRL's gains keep rising with w, while AWR saturates — [arXiv HTML](https://arxiv.org/html/2505.23458)
- In goal-conditioned BC, guidance on the goal needs no value function and beats plain GCBC, e.g. pointmaze-giant 30 +/- 10 vs 2 +/- 2 — [arXiv HTML](https://arxiv.org/html/2505.23458). Code: [github.com/kvfrans/cfgrl](https://github.com/kvfrans/cfgrl)

**DIPOLE, "Dichotomous Diffusion Policy Optimization"** (arXiv 2601.00898, v3 Jul 2026; unreviewed)
- Trains a positive policy weighted by sigmoid(beta * G) and a negative policy weighted by 1 - sigmoid(beta * G). Inference uses eps = (1 + w) eps+ - w eps-, a CFG-like extrapolation away from the failure policy. The bounded sigmoid replaces the unstable exp weighting — [arXiv HTML](https://arxiv.org/html/2601.00898v3)
- Beats CFGRL, FQL and IFQL on OGBench and ExORL. On a 1B driving VLA with LoRA (NAVSIM), PDM rises from 88.3 to 94.8. No LIBERO results — [arXiv HTML](https://arxiv.org/html/2601.00898v3)

**Weighted flow matching for VLAs**
- ARFM (arXiv 2509.04063; published in AAAI proceedings per the [AAAI OJS listing](https://ojs.aaai.org/index.php/AAAI/article/view/38944/42906)): weights w_i proportional to exp(alpha * standardised advantage) inside the pi0 flow-matching loss, with alpha chosen adaptively by bisection. Advantages come from a 13-term dense reward. LIBERO average: pi0 88.1, RWR 90.8, ReinboT 91.2, ARFM 92.1 — [arXiv HTML](https://arxiv.org/html/2509.04063)
- Energy-Weighted Flow Matching / QIPO (arXiv 2503.04975): the arXiv comments say ICLR 2025, not ICML — [arXiv](https://arxiv.org/abs/2503.04975)
- QVPO: Q-weighted variational policy optimisation for diffusion policies, NeurIPS 2024 — [arXiv 2405.16173](https://arxiv.org/abs/2405.16173)
- Online reward-weighted fine-tuning of flow matching with Wasserstein regularisation — [arXiv 2502.06061](https://arxiv.org/abs/2502.06061)
- Advantage Weighted Matching (diffusion image models): ICML 2026 per the arXiv comments — [arXiv 2509.25050](https://arxiv.org/abs/2509.25050)
- Flow Q-Learning: ICML 2025 — [arXiv 2502.02538](https://arxiv.org/abs/2502.02538)
- FPO, "Flow Matching Policy Gradients" — [arXiv 2507.21053](https://arxiv.org/abs/2507.21053)
- FPO for VLAs (arXiv 2510.09976; ICRA 2026 per the arXiv comments): keeps the pi0 decoder frozen and trains a latent actor with a critic ensemble. It uses a likelihood-free ratio from per-sample CFM-loss changes. LIBERO average 87.2 — [arXiv HTML](https://arxiv.org/html/2510.09976)
- Classic baselines: AWR [arXiv 1910.00177](https://arxiv.org/abs/1910.00177); AWAC [arXiv 2006.09359](https://arxiv.org/abs/2006.09359)

**Return conditioning**
- Decision Transformer [arXiv 2106.01345](https://arxiv.org/abs/2106.01345) and Decision Diffuser [arXiv 2211.15657](https://arxiv.org/abs/2211.15657) are the original return-conditioned sequence and diffusion policies. I did not re-verify their venues here.
- ReinboT (ICML 2025, [ICML poster page](https://icml.cc/virtual/2025/poster/45523)): an end-to-end VLA that predicts dense returns to guide actions on CALVIN mixed-quality data — [arXiv 2505.07395](https://arxiv.org/abs/2505.07395)

**Value guidance at test time, with the policy frozen**
- V-GPS (CoRL 2024, per the arXiv comments): Cal-QL or IQL Q-function trained on Bridge + Fractal, with reward -1 per step and 0 on the last 3 steps of successes. It samples K in {10, 50} actions from a frozen generalist and re-ranks them by Q. SIMPLER gains: Octo-small +26.7%, OpenVLA +12.5%, RT1-X +16.7%. Inference costs 1.28-1.59x. No policy weights are touched — [arXiv HTML](https://arxiv.org/html/2410.13816); code [github.com/nakamotoo/V-GPS](https://github.com/nakamotoo/V-GPS)
- Q-Planning (arXiv 2608.21204, Aug 2026; unreviewed): pairs a frozen BC policy with a small Q-function. The Q-function first trains on the BC demos, then absorbs both successful and failed rollouts. It picks actions by a Q-weighted average over BC draws, and only Q is fine-tuned. Filtered SFT stalls at 55% (stack-cups) and 30% (insert-wallet); Q-Planning reaches 90% and 80% in five iterations — [arXiv](https://arxiv.org/abs/2608.21204); [project](https://varungiridhar.github.io/qplanning/)
- "What Frozen VLAs Already Know About Success" (arXiv 2605.28527, May 2026; unreviewed): linear probes on frozen pi0.5 features predict success-like value with R^2 = 0.74 (demo split; the best pi0.5 feature is the vision encoder) and about 0.55 on the task split. Same-task, same-timestep pairwise ordering accuracy is about 92%. Probe-guided selection among K = 16 candidates raises push-plate success from 26.7% to 44.3% — [arXiv HTML](https://arxiv.org/html/2605.28527)

### Inferences
- RECAP's value, negative steps-to-success normalised per task, is almost exactly (demo length - alignment pointer) / demo length. The alignment pointer is therefore a zero-cost value estimate. Per-chunk advantage can be pointer progress in that chunk minus expected progress, binarised at a per-task percentile (RECAP uses the 30th). This removes RECAP's most data-hungry part, the 670M value model, which tens of rollouts could not train.
- For a frozen pi0.5, RECAP's "Advantage: positive/negative" text token has a parameter-light equivalent: two small per-task adapters (LoRA or fold), one fit on positive-labelled chunks and one on negative-labelled chunks. Sample with CFG on the velocity: v = v_pos + w (v_pos - v_neg), or the DIPOLE form (1 + w) v_pos - w v_neg. With Euler sampling this is one extra forward pass of the action expert per step. The guide weight w becomes a per-task knob tuned on practice success, with no retraining (CFGRL's key property).
- Value-guided selection (V-GPS, Q-Planning, the frozen-VLA probe) needs no policy update. That makes it the cheapest test of whether failure data carries usable signal: train a tiny success/progress probe on pi0.5 features from our recorded successes and failures, then pick among K sampled chunks.

### Gaps
- RECAP reports no LIBERO numbers and no ablation against success-only filtered BC on the same data, only against AWR and PPO. The size of its gain over filtered BC in a low-data regime is unknown.
- I found no paper applying CFGRL-style optimality guidance to a frozen pi0/pi0.5 with per-task adapters. DIPOLE's VLA result is on driving, and RECAP fine-tunes the full model.
- ARFM's advantage comes from a 13-term hand-designed dense reward, not binary success. How it behaves with sparse binary reward is untested.

## Q2. Negative-aware training (learning from failed demos, negative/contrastive flow matching, NFT, unlikelihood, DPO/GRAPE, preference RL for diffusion policies)

### Takeaway
Negatives help when three conditions hold: they share the positives' parameterisation (implicit negative policy, as in NFT/DiffusionNFT), they are localised to the failing chunks (RedFlow), or they serve only as a guidance direction at inference (AFIL, DIPOLE). Naive trajectory-level negatives (step-wise DPO) do worse than trajectory-level preference optimisation (GRAPE). LLM evidence shows negative-only reinforcement preserves diversity, while positive-only reinforcement collapses it. That is one plausible mechanism for a filtered-BC plateau.

### Cited Findings
**NFT and DiffusionNFT**
- NFT (arXiv 2505.18116): builds an implicit negative policy that shares parameters with the positive model, so supervised learning can use negatives. It is equivalent to GRPO under strict on-policy training and beats rejection-sampling fine-tuning — [arXiv](https://arxiv.org/abs/2505.18116)
- DiffusionNFT (arXiv 2509.16117; ICLR 2026 Oral per the arXiv comments): defines v+ = (1 - beta) v_old + beta v_theta and v- = (1 + beta) v_old - beta v_theta. The loss is r ||v+ - v||^2 + (1 - r) ||v- - v||^2, with reward mapped to r in [0, 1]. It needs no likelihoods and no CFG, and works with any sampler. GenEval goes from 0.24 to 0.98 in 1k steps, vs FlowGRPO's 0.95 in over 5k. It has no robotics experiments — [arXiv HTML](https://arxiv.org/html/2509.16117); code [github.com/NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)

**Failure-aware flow VLAs (2026)**
- AFIL, "Failing Forward" (arXiv 2605.08434, May 2026; unreviewed): built on pi0.5. A Dual Action Generator (success head and failure head over a shared VLM backbone) is trained on the model's own failed rollouts. Inference applies negative guidance with adaptive strength alpha * (1 - cos(eps_succ, eps_fail)). LIBERO average rises from 96.9 to 98.4, and LIBERO-10 from 92.4 to 95.8 — [arXiv HTML](https://arxiv.org/html/2605.08434v1)
- RedFlow (arXiv 2607.27782, Jul 2026; unreviewed): turns trajectory outcomes into a signed chunk-level score from local progress change plus outcome bias. It clusters chunks in progress-state space (HDBSCAN) and uses the quality-weighted centroid of positive chunks in the same cluster as the correction target for each negative chunk. The loss has attraction, hinge suppression and correction terms, and only v_theta is trained. From pi0 initialised from the piRL few-shot checkpoint, LIBERO goes from 56.2 to 68.2, which is +5.9 over AWR and +8.5 over DPO. It uses 1,536 rollouts per suite and matches PPO/GRPO/DDPO that need 13K-24K — [arXiv HTML](https://arxiv.org/html/2607.27782)
- Failure-Boundary Learning (arXiv 2609.06114, Sep 2026; unreviewed): "Directional Boundary Shaping" reinforces success-producing denoising directions and suppresses failure-producing ones, "without action likelihoods or auxiliary critics", on pi0 and pi0.5. It reports real-robot results only — [arXiv HTML](https://arxiv.org/html/2609.06114)
- ExpReS-VLA (arXiv 2511.06202; ICRA 2026 per the arXiv comments): OpenVLA with rank-32 LoRA. Retrieved successes are positives and a failure buffer supplies negatives, at a 3:2 ratio. The loss is thresholded hybrid contrastive (triplet margin 0.5, or InfoNCE with tau = 0.1). LIBERO (spatial + long) rises from 77.9 to 88.7 with 50 rollouts per task — [arXiv HTML](https://arxiv.org/html/2511.06202)

**Preference optimisation from success/failure pairs**
- GRAPE (arXiv 2411.19309; ICLR 2025 workshop per [ML Anthology](https://mlanthology.org/iclrw/2025/zhang2025iclrw-grape/)): trajectory-wise preference optimisation. It sums per-step log-ratios over the whole trajectory and ranks trajectories by success, VLM-generated stage/keypoint costs, and self log-likelihood. It uses N_t = 5 rollouts per task, top-m vs bottom-m pairs, and 3 iterations of LoRA on OpenVLA. Results: Simpler 29.54 to 43.11, LIBERO +7.36% over SFT, and +33.14% over step-wise DPO — [arXiv HTML](https://arxiv.org/html/2411.19309); code [github.com/aiming-lab/GRAPE](https://github.com/aiming-lab/GRAPE)
- APO, Action Preference Optimization (arXiv 2506.07127; NeurIPS 2025 per the arXiv comments): builds preferences from human interventions vs failed actions, with adaptive reweighting for irreversibility and probability mismatch — [arXiv](https://arxiv.org/abs/2506.07127)
- FDPP, preference fine-tuning of diffusion policies — [arXiv 2501.08259](https://arxiv.org/abs/2501.08259)
- Diffusion-DPO — [arXiv 2311.12908](https://arxiv.org/abs/2311.12908)
- CPL (ICLR 2024) — [arXiv 2310.13639](https://arxiv.org/abs/2310.13639)

**Why positives alone plateau: LLM evidence**
- NSR vs PSR (arXiv 2506.01347; NeurIPS 2025 per the arXiv comments): negative-only reinforcement improves Pass@k across the whole spectrum and preserves diversity. Positive-only reinforcement improves Pass@1 but degrades high-k through lost diversity. NSR "redistribut[es] probability mass toward other plausible candidates, guided by the model's prior" — [arXiv](https://arxiv.org/abs/2506.01347)

**Offline imitation with undesirable demos (low-dim benchmarks)**
- "Learning What to Do and What Not To Do" optimises a difference of KL divergences between expert and undesirable visitation distributions, which is convex when expert data dominates — [arXiv 2505.21182](https://arxiv.org/abs/2505.21182)
- UNIQ — [arXiv 2410.08307](https://arxiv.org/abs/2410.08307)
- DWBC (ICML 2022) — [arXiv 2207.10050](https://arxiv.org/abs/2207.10050)
- T-REX (ICML 2019) — [arXiv 1904.06387](https://arxiv.org/abs/1904.06387)
- Unlikelihood training (text) — [arXiv 1908.04319](https://arxiv.org/abs/1908.04319)

### Inferences
- DiffusionNFT's loss fits our situation well. It needs only the frozen reference velocity v_old (pi0.5 plus the current adapter), the recorded (observation, executed chunk) pairs, and r in {0, 1} or a pointer-progress r. It needs no likelihoods, which pi0.5's Euler sampling lacks. It trains the same LoRA or fold we already train: a filtered-BC loss plus a mirrored term on failures.
- Unlikelihood or hinge suppression on whole failed trajectories is risky: most chunks in a failed LIBERO rollout (approach, pre-grasp) are identical to chunks in successful ones. RedFlow and PCM both localise the negative signal before applying it (see Q5).
- NSR's diversity argument predicts that our filtered-BC LoRA narrows pi0.5's chunk distribution around the round-0 composition. This inference can be checked on our recordings: compare action-chunk entropy or variance across seeds between round 0 and round 1 at the stall positions.

### Gaps
- No paper applies NFT/DiffusionNFT-style implicit negatives to a robot flow policy. AFIL and Failure-Boundary Learning are the closest, and both are unreviewed.
- The GRAPE, RedFlow and AFIL baselines differ (OpenVLA, pi0 few-shot, pi0.5 full-data). I found no head-to-head of negative-aware vs filtered BC on pi0.5 at our one-demo, tens-of-rollouts scale.

## Q3. Dynamic sampling, curricula, SIL, and breaking iterated filtered-BC plateaus

### Takeaway
Group methods (RIPT-VLA, SimpleVLA-RL, DAPO) drop instances whose rollouts are all-success or all-failure because those instances give zero advantage. So the hardest instances contribute nothing, and outcome-only RL "yields only marginal improvements" when initial success is near zero. Published plateau-breakers add signal that does not need a success in the same group:
- a value or Q-function over failures (Q-Planning)
- progress-based rewards (SRPO, steps-to-go)
- resets to demo or failure states (RFCL, U-GROW)
- hindsight relabeling (Q4)
- recovery data (RaC, PLD)

### Cited Findings
**Group RL with dynamic sampling**
- RIPT-VLA (arXiv 2505.17016, May 2025; venue not verified): each context gets K rollouts (K = 8 for OpenVLA-OFT, 16 for QueST). The advantage is leave-one-out, A_k = R_k - mean of the other R_j, under a PPO objective, and a group is skipped if all advantages are 0 (all success or all failure). LIBERO-90 with QueST goes from 88.6 to 94.3 (as extracted from the paper's Table 2). In the one-demo setting, per the abstract, success goes from 4% to 97% within 15 iterations. OFT uses B = 192 rollouts per epoch on 4 A5000s — [arXiv HTML](https://arxiv.org/html/2505.17016); [abs](https://arxiv.org/abs/2505.17016); code [github.com/Ariostgx/ript-vla](https://github.com/Ariostgx/ript-vla)
- SimpleVLA-RL (arXiv 2509.09674, Sep 2025; venue not verified): OpenVLA-OFT with one trajectory per task goes from 48.9 to 96.9 on average (LIBERO-Long 17.3 to 91.7). It uses dynamic sampling (drop all-success/all-failure groups), clip-higher [0.8, 1.28], rollout temperature 1.6, no KL, and 8 rollouts per init state. It also reports "pushcut": new strategies absent from the demos. RL produced no improvement from zero-success models, and "when initial success rates are very low, online RL with outcome rewards yields only marginal improvements" — [arXiv HTML](https://arxiv.org/html/2509.09674); code [github.com/PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- DAPO introduced dynamic sampling for LLM RL — [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)

**Curricula**
- VLA-RL: adaptive task curriculum with P(task_i) proportional to exp((0.5 - s_i) / tau), which prefers tasks near 50% success — [arXiv 2505.18719](https://arxiv.org/pdf/2505.18719)
- RFCL (ICLR 2024): a per-demonstration reverse curriculum resets the simulator to states along the demo, starting near the end. A forward curriculum then widens the initial-state distribution — [arXiv 2405.03379](https://arxiv.org/abs/2405.03379); code [github.com/StoneT2000/rfcl](https://github.com/StoneT2000/rfcl)
- U-GROW (arXiv 2609.22879, Sep 2026; unreviewed): truncates failed trajectories at the "first high-uncertainty state" (diffusion-step disagreement). It starts short branched rollouts from high-uncertainty recorded states, mixed with uniform starts, on pi0.5. LIBERO average 89.1 vs 87.7 for uniform starts — [arXiv HTML](https://arxiv.org/html/2609.22879)

**Plateau evidence and breakers**
- Q-Planning: filtered SFT on successes stalls (55% and 30%), while a Q-function trained on successes and failures, steering the frozen BC policy, reaches 90% and 80% — [arXiv](https://arxiv.org/abs/2608.21204)
- RaC: imitation plateaus even with thousands of demos. Human rewind-and-correct data beats prior state of the art with 10x less data — [arXiv 2509.07953](https://arxiv.org/abs/2509.07953); [project](https://rac-scaling-robot.github.io/)
- ReGuide (arXiv 2606.28939, Jun 2026; unreviewed): guides a diffusion policy at test time with a phase-conditioned dynamics-model cost toward phase-specific latent targets. Guidance acts only in "drifted-but-recoverable" regimes, and guided successes are then recycled into fine-tuning. Matched-data ablations show guided rollouts beat unguided ones at equal data. Tool Hang improves 7.7x (3% to 24%) on robomimic — [arXiv HTML](https://arxiv.org/html/2606.28939)
- Self-Improving Embodied FMs (NeurIPS 2025): the reward is the change in predicted steps-to-go, with on-policy REINFORCE. 10% extra robot time in self-improvement moves success from 45% to 75%, while 8x more imitation data gives 45% to 60% — [arXiv HTML](https://arxiv.org/html/2509.15155)
- SIL, the original self-imitation learning — [arXiv 1806.05635](https://arxiv.org/abs/1806.05635)
- "When should we prefer offline RL over BC" (ICLR 2022) — [arXiv 2204.05618](https://arxiv.org/abs/2204.05618)

### Inferences
- Our plateau matches the published pattern: success-only retraining (Q-Planning's filtered SFT, NSR's PSR) and outcome-only RL on near-zero-success instances (SimpleVLA-RL). Adding more rounds of successes will not reach the ~35% persistent failures, because those instances generate no successes to imitate.
- For the persistent failures, the most direct lever is a reset-based reverse curriculum. LIBERO is MuJoCo and our recordings include states, so we can restore the simulator to (a) the demo state at the pointer position just before the stall, or (b) the recorded failure state at the stall. From there we practise only the remaining segment. That gives nonzero success rates on sub-problems where full-episode success is 0 (RFCL, U-GROW).
- Instance prioritisation: sample practice inits near 50% empirical success (VLA-RL's formula), and treat all-fail instances as reset-curriculum candidates instead of wasting full rollouts on them.

### Gaps
- No paper reports iterated filtered BC vs negative-aware training on pi0.5 with one demo at tens of rollouts per task. RIPT-VLA, SimpleVLA-RL and piRL use hundreds to thousands of rollouts (see Q6).
- I did not verify RIPT-VLA's LIBERO-90 numbers against the PDF table; they are as extracted.

## Q4. Hindsight relabeling for goal/instruction-conditioned manipulation: does it apply to failures of a language-conditioned VLA?

### Takeaway
Yes, and a 2026 preprint does it on pi0.5 in LIBERO. LfH has a VLM write a hindsight instruction for a group of failed rollouts, scores each rollout against it, and trains GRPO on relabelled and original rollouts jointly. It reports 5x sample efficiency. The caveat is noisy positives when the hindsight instruction is wrong or generic.

### Cited Findings
- LfH, "Learning More from Less: Reinforcement Learning from Hindsight" (arXiv 2607.09042, Jul 2026; unreviewed). Base: pi0.5 (RLinf-Pi05-LIBERO-SFT), also GR00T and OpenVLA-OFT. It keeps about 80% of trajectory groups after relabeling vs about 20-40% for standard GRPO, which matches GRPO's final success in about 1/5 of the steps. On a real FR3 robot at 160 rollouts it gets 56% vs 22% for GRPO. It beats a dense progress-reward baseline. It uses importance correction for the instruction switch. Failure mode: relabels can be "semantically unrelated", e.g. "tip over the white mug", and "LfH depends highly on the quality of the failures it receives" — [arXiv HTML](https://arxiv.org/html/2607.09042v1); implemented in [RLinf](https://github.com/RLinf/RLinf)
- SOAR (CoRL 2024): relabels goals to states actually reached, so failures train a goal-conditioned policy. It scores 0.65 vs 0.28 for language-conditioned filtered BC on the same autonomous data. The VLM success detector had precision 0.63 and recall 1.0, which is why filtering hurt — [arXiv HTML](https://arxiv.org/html/2407.20635); code [github.com/rail-berkeley/soar](https://github.com/rail-berkeley/soar)
- Earlier work:
  - HER — [arXiv 1707.01495](https://arxiv.org/abs/1707.01495)
  - GCSL — [arXiv 1912.06088](https://arxiv.org/abs/1912.06088)
  - HIGhER (ADPRL'20) — [arXiv 1910.09451](https://arxiv.org/abs/1910.09451)
  - DIAL, VLM instruction augmentation (RSS 2023) — [arXiv 2211.11736](https://arxiv.org/abs/2211.11736)
  - SPRINT, instruction relabeling (ICRA 2024) — [arXiv 2306.11886](https://arxiv.org/abs/2306.11886)
  - HIR for LLMs — [arXiv 2302.05206](https://arxiv.org/abs/2302.05206)
  - CAST, counterfactual labels for VLAs — [arXiv 2508.13446](https://arxiv.org/abs/2508.13446)

### Inferences
- LIBERO-90 scenes host several tasks each (the same kitchen or living-room layout with different instructions). A failed rollout that ends with a known predicate satisfied, readable from the recorded simulator state via LIBERO's BDDL goal checks (no VLM needed), is a verified success for a sibling instruction. It can be relabelled with an exact reward, which avoids LfH's noisy-VLM failure mode.
- Whether relabelled data helps the target task depends on what we train. For a per-task fold or LoRA, sibling-task data mostly improves the prior's shared sub-skills (reach, grasp, place), not the target composition. Hindsight at the sub-goal level ("the rollout completed demo phases 1-3") matches our pointer better: the pre-stall prefix is a verified success of the prefix sub-task and can be imitated as such.

### Gaps
- LfH is unreviewed and targets LIBERO-PRO perturbations, not held-out LIBERO-90 tasks. I found no result on relabeling to sibling BDDL predicates from simulator state.

## Q5. Credit assignment: where did a failed trajectory go wrong, and how to train only there (failure localization, phase/keyframe credit, divergence from successes, recovery data, DAgger-like self-correction without a human)

### Takeaway
2025-26 methods localise failures in five ways:
- progress deltas: SRPO, RedFlow, steps-to-go
- per-phase success/failure action divergence: PCM
- first high-uncertainty state: U-GROW
- a rewind point: RaC, B2FF
- segment roles (prefix, drift, correction): RePO-VLA

They then train only there, or build self-generated correction targets from successful chunks at the same progress (RedFlow), or from branched rollouts started at the failure state (PLD, U-GROW). Our stalling alignment pointer is a direct, free localiser for all of these.

### Cited Findings
**Progress-based credit**
- SRPO (arXiv 2511.15605, Nov 2025; venue not verified): uses the batch's own successful trajectories as references and assigns failed attempts a progress-wise reward via latent world-model representations. It goes from 48.9 to 99.2 on LIBERO in 200 RL steps — [arXiv](https://arxiv.org/abs/2511.15605)
- RedFlow: the chunk-level signed score is progress change plus outcome bias. The correction target is the centroid of positive chunks in the same progress-state cluster — [arXiv HTML](https://arxiv.org/html/2607.27782)

**Divergence localisation**
- PCM (arXiv 2605.16154, May 2026; unreviewed): 5 gripper-defined phases. It measures per-phase success-vs-failure mean-action divergence and spends GRPO gradient only on high-divergence phases (<20% of chunks). It reaches 98% LIBERO success 2.38x faster in wall-clock, with 60% less activation memory, on OpenVLA-OFT — [arXiv HTML](https://arxiv.org/html/2605.16154)
- U-GROW: truncates failure tails at the first high-uncertainty state and restarts rollouts there — [arXiv HTML](https://arxiv.org/html/2609.22879)

**Segment roles**
- RePO-VLA (arXiv 2605.09410, May 2026; unreviewed): on pi0.5 with a value token, success and recovery suffixes get v = 1, failure prefixes v = 0, and pure failures v(t) = V * (1 - t/T)^3, "retain[ing] useful early motion while rapidly down-weighting states near irreversible failure". Recovery episodes come from planners (sim) or teleop (real). FRBench-Sim improves 15.0 to 37.0 (clean) — [arXiv HTML](https://arxiv.org/html/2605.09410v1)

**Recovery data without a human**
- PLD (arXiv 2511.00091): trains a residual RL actor on a frozen VLA to probe failure regions. Hybrid rollouts run the base policy, then hand over, so they "capture recovery behaviors". The data is distilled back by SFT, reaching about 99% on LIBERO — [arXiv](https://arxiv.org/abs/2511.00091)
- MiDAS (arXiv 2608.11363, Aug 2026; unreviewed): pi0.5 with K = 1 demo. BC uses LoRA on the backbone plus the full action head. Then a frozen BC policy gets a residual actor-critic (3-layer MLP) trained with sparse-reward online RL. The warm buffer mixes demos with successful and failed rollouts. LIBERO-Long reaches 91.2 +/- 4.2 vs BC 33.5, DSRL 33.5 and DICE-RL 89.3 — [arXiv HTML](https://arxiv.org/html/2608.11363); [project](https://minimal-data-adaptation.github.io/)
- DSRL (CoRL 2025, per the official repo): trains an RL policy over the initial noise of a frozen diffusion or flow policy. The base weights stay frozen, and the noise-aliased variant can use offline data. pi0 on LIBERO goes from about 20% to about 100% with about 10k online samples. Real robot: toaster 5/20 to 18/20 in 80 episodes — [arXiv HTML](https://arxiv.org/html/2506.15799); code [github.com/nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0), [github.com/ajwagen/dsrl](https://github.com/ajwagen/dsrl)
- B2FF (arXiv 2606.09258, Jun 2026; unreviewed): the frozen VLA pre-imagines about 12 milestone subgoal images. When a detector flags deviation, a selector trained offline on counterfactual failures chooses the most recoverable milestone. Failure-injected LIBERO improves 56.3 to 74.0 (UD-VLA frozen) — [arXiv HTML](https://arxiv.org/html/2606.09258v1)

**Human-intervention precedents**
- Sirius (RSS 2023) — [arXiv 2211.08416](https://arxiv.org/abs/2211.08416)
- RLIF (ICLR 2024) — [arXiv 2311.12996](https://arxiv.org/abs/2311.12996)
- RaC — [arXiv 2509.07953](https://arxiv.org/abs/2509.07953)
- Hi-ORS: rejection sampling plus reward-weighted SFT plus human corrections, about 1.5 h on a real robot — [arXiv 2510.26406](https://arxiv.org/abs/2510.26406)

**Synthetic corrective data**
- Diffusion Meets DAgger (RSS 2024) — [arXiv 2402.17768](https://arxiv.org/abs/2402.17768)
- CCIL — [arXiv 2310.12972](https://arxiv.org/abs/2310.12972)
- IntervenGen — [arXiv 2405.01472](https://arxiv.org/abs/2405.01472)

**Failure detection and explanation**
- AHA — [arXiv 2410.00371](https://arxiv.org/abs/2410.00371)
- REFLECT (CoRL 2023) — [arXiv 2306.15724](https://arxiv.org/abs/2306.15724)
- RACER — [arXiv 2409.14674](https://arxiv.org/abs/2409.14674)
- RoboFAC — [arXiv 2505.12224](https://arxiv.org/abs/2505.12224)
- VLA-FAIL — [arXiv 2606.21386](https://arxiv.org/abs/2606.21386)
- Hide-and-Seek failure signals — [arXiv 2605.30834](https://arxiv.org/abs/2605.30834)

**Progress/value estimators**
- GVL, in-context VLM value — [arXiv 2411.04549](https://arxiv.org/abs/2411.04549)
- VLAC — [arXiv 2509.15937](https://arxiv.org/abs/2509.15937)

### Inferences
- Our pointer gives, for each failed rollout, the demo index p* where progress stalls, plus the time t* at which it stalled. That is RedFlow's "progress change" signal without a learned reward model. Four ways to use it:
  - **Localised negatives:** apply negative-aware loss (Q2) only to chunks executed in a window of roughly 1-3 chunks before t*, and a positive or neutral label to the pre-stall prefix (RePO-VLA's segment roles).
  - **Self-generated corrections:** successful rollouts from other init states that passed demo index p* supply correction targets: their executed chunks at the matching pointer position, averaged or nearest in state (RedFlow's cluster centroid, keyed on our pointer instead of HDBSCAN).
  - **DAgger without a human:** restore the MuJoCo state at t* (or a few chunks earlier), branch K rollouts with fresh flow noise or with CFG weights, and keep the branches that get the pointer moving again. These give recovery data exactly where the policy fails (PLD's hybrid rollouts, U-GROW's branched rollouts, RaC's rewind, all without a human).
  - **Phase-level budget:** run PCM's success-vs-failure divergence per demo segment (our pointer segments replace PCM's gripper phases) to decide which segments' chunks get any gradient.
- For a strictly frozen pi0.5, DSRL-style noise steering is the other route. A tiny per-task noise policy (the "fold" could be exactly this) learns which initial noise leads the Euler flow to chunks that advance the pointer. DSRL's reported budget (about 10k samples) is above ours, however.

### Gaps
- The DSRL "~10k online samples" figure for LIBERO: I could not confirm whether it counts env steps or chunk decisions, which matters for comparing against our tens of rollouts.
- No paper uses a demo-alignment pointer as the progress signal. The closest are RFCL's demo-state resets, ReGuide's phase targets, and steps-to-go. Whether pointer stalls coincide with the causal error, as opposed to a delayed symptom such as a grasp slip visible only later, is untested. RePO-VLA's "terminal drift" category and U-GROW's "first high-uncertainty state" suggest the causal chunk can precede the stall.

## Q6. Sample efficiency with tens of rollouts; LIBERO numbers; code; verification table

### Takeaway
Methods reported with tens of rollouts per task:
- GRAPE: 5 per task per iteration
- ExpReS-VLA: 50 per task
- LfH: 160 total on a real robot
- RECAP: hundreds per iteration on real tasks

pi0.5 one-demo online-RL results (piRL, MiDAS, SimpleVLA-RL with OFT) reach 91-98% on standard LIBERO suites, but with far larger simulator budgets and on tasks the prior partly knows. They are upper bounds for our setting, not predictions.

### Cited Findings
**LIBERO numbers relevant to one demo per task**
- piRL (arXiv 2510.25889, Oct 2025, v3 Jan 2026; venue not verified): pi0.5 SFT on "only 40 trajectories" (4 suites of 10 tasks, i.e. about 1 per task). SFT averages 77.1 (Long 43.9); Flow-Noise RL reaches 98.3 (Long 94.0) and Flow-SDE 97.9. For pi0, few-shot SFT goes from 57.6 to 97.6. The VLM is frozen and only the 300M action expert is fine-tuned; the pi0.5 critic attaches to the VLM output; training ran on 8 H100s — [arXiv HTML](https://arxiv.org/html/2510.25889); code [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
- SimpleVLA-RL: one-trajectory SFT with OFT goes from 48.9 to 96.9 — [arXiv HTML](https://arxiv.org/html/2509.09674)
- MiDAS: pi0.5 with K = 1 reaches 91.2 on LIBERO-Long — [arXiv HTML](https://arxiv.org/html/2608.11363)
- RIPT-VLA: one demo goes from 4% to 97% in 15 iterations; QueST on LIBERO-90 goes from 88.6 to 94.3 — [arXiv](https://arxiv.org/abs/2505.17016)
- SRPO: 48.9 to 99.2 in 200 steps — [arXiv](https://arxiv.org/abs/2511.15605)
- Failure-aware and offline results on LIBERO: RedFlow 56.2 to 68.2 with 1,536 rollouts per suite; AFIL (pi0.5) 96.9 to 98.4; ARFM (pi0) 88.1 to 92.1; ExpReS 77.9 to 88.7 with 50 rollouts per task; GRAPE +7.36% — sources in Q1 and Q2.

**Other RL-for-VLA references**
- VLA-RL — [arXiv 2505.18719](https://arxiv.org/abs/2505.18719)
- RL4VLA (NeurIPS 2025) — [arXiv 2505.19789](https://arxiv.org/abs/2505.19789)
- ReinFlow (NeurIPS 2025) — [arXiv 2505.22094](https://arxiv.org/abs/2505.22094)
- DPPO — [arXiv 2409.00588](https://arxiv.org/abs/2409.00588)
- Flow-GRPO — [arXiv 2505.05470](https://arxiv.org/abs/2505.05470); code [github.com/yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)
- TGRPO — [arXiv 2506.08440](https://arxiv.org/abs/2506.08440)
- ConRFT — [arXiv 2502.05450](https://arxiv.org/abs/2502.05450)
- CO-RFT — [arXiv 2508.02219](https://arxiv.org/abs/2508.02219)
- RL-100 — [arXiv 2510.14830](https://arxiv.org/abs/2510.14830)
- PA-RL — [arXiv 2412.06685](https://arxiv.org/abs/2412.06685)
- Q-chunking (NeurIPS 2025) — [arXiv 2507.07969](https://arxiv.org/abs/2507.07969)
- Q-Transformer — [arXiv 2309.10150](https://arxiv.org/abs/2309.10150)
- IDQL — [arXiv 2304.10573](https://arxiv.org/abs/2304.10573)
- VLA-RFT — [arXiv 2510.00406](https://arxiv.org/abs/2510.00406)
- WMPO — [arXiv 2511.09515](https://arxiv.org/abs/2511.09515)
- GR-RL — [arXiv 2512.01801](https://arxiv.org/abs/2512.01801)
- Hume — [arXiv 2505.21432](https://arxiv.org/abs/2505.21432)
- RoboMonkey — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811)
- Q-VGM — [arXiv 2606.08015](https://arxiv.org/abs/2606.08015)
- Reflection-based task adaptation — [arXiv 2510.12710](https://arxiv.org/abs/2510.12710)
- Imagining Recovery — [arXiv 2608.14822](https://arxiv.org/abs/2608.14822)
- Contrastive Flow Matching (image diversity, not failures) — [arXiv 2506.05350](https://arxiv.org/abs/2506.05350)

**Verification table: arXiv ID, first-listed date, venue as verified**

| Method | arXiv ID | First-listed date | Venue as verified |
|---|---|---|---|
| RECAP / pi*0.6 | 2511.14759 | 2025-11-18 | none listed |
| RIPT-VLA | 2505.17016 | 2025-05-22 | not verified |
| SimpleVLA-RL | 2509.09674 | 2025-09-11 | not verified |
| piRL | 2510.25889 | 2025-10-29 | not verified |
| SRPO | 2511.15605 | 2025-11-19 | not verified |
| CFGRL | 2505.23458 | 2025-05-29 | OpenReview forum exists; not verified |
| DiffusionNFT | 2509.16117 | 2025-09-19 | ICLR 2026 Oral |
| NFT | 2505.18116 | 2025-05-23 | not verified |
| GRAPE | 2411.19309 | 2024-11-28 | ICLR 2025 workshop |
| DSRL | 2506.15799 | 2025-06-18 | CoRL 2025 (per official repo) |
| V-GPS | 2410.13816 | 2024-10-17 | CoRL 2024 |
| SOAR | 2407.20635 | 2024-07-30 | CoRL 2024 |
| Self-Improving EFMs | 2509.15155 | 2025-09-18 | NeurIPS 2025 |
| QVPO | 2405.16173 | 2024-05-25 | NeurIPS 2024 |
| QIPO / Energy-Weighted FM | 2503.04975 | 2025-03-06 | ICLR 2025 |
| FQL | 2502.02538 | 2025-02-04 | ICML 2025 |
| ARFM | 2509.04063 | 2025-09-04 | AAAI proceedings (OJS listing) |
| ReinboT | 2505.07395 | 2025-05-12 | ICML 2025 |
| RFCL | 2405.03379 | 2024-05-06 | ICLR 2024 |
| ExpReS-VLA | 2511.06202 | 2025-11-09 | ICRA 2026 |
| FPO for VLAs | 2510.09976 | 2025-10-11 | ICRA 2026 |
| APO | 2506.07127 | 2025-06-08 | NeurIPS 2025 |
| NSR | 2506.01347 | 2025-06-02 | NeurIPS 2025 |
| AFIL | 2605.08434 | 2026-05-08 | preprint |
| RePO-VLA | 2605.09410 | 2026-05-10 | preprint |
| PCM | 2605.16154 | 2026-05-15 | preprint |
| Frozen-VLA probe | 2605.28527 | 2026-05-27 | preprint |
| B2FF | 2606.09258 | 2026-06-08 | preprint |
| ReGuide | 2606.28939 | 2026-06-27 | preprint |
| LfH | 2607.09042 | 2026-07-10 | preprint |
| RedFlow | 2607.27782 | 2026-07-30 | preprint |
| MiDAS | 2608.11363 | 2026-08-11 | preprint |
| Q-Planning | 2608.21204 | 2026-08-21 | preprint |
| Failure-Boundary Learning | 2609.06114 | 2026-09-05 | preprint |
| U-GROW | 2609.22879 | 2026-09-19 | preprint |
| DIPOLE | 2601.00898 | — | preprint (v3 Jul 2026) |

Source for the table: arXiv API records (e.g. [2511.14759](https://arxiv.org/abs/2511.14759), [2609.22879](https://arxiv.org/abs/2609.22879)).

### Inferences
- The one-demo pi0.5 RL results (piRL 77.1 to 98.3) show that the pi0.5 action expert alone, with the VLM frozen, has enough capacity to fix most failures once enough rollouts exist. Our bottleneck is the signal per rollout, not adapter capacity. This favours methods that extract more signal per rollout: localised negatives, pointer progress, resets, relabeling. Plain group RL does not do this.

### Gaps
- None of the LIBERO numbers above is on held-out LIBERO-90 tasks from a frozen `pi05_libero` prior, so no directly comparable baseline exists.
- piRL does not report rollout counts or GPU hours for its one-demo pi0.5 run. Its sample cost is unknown relative to our budget.

## Q7. Concrete application to our recordings (what to train, what pairing, where to localize)

### Takeaway
The best-supported ideas are ordered by cost and by how directly they address the 35% persistent failures. All reuse existing recordings and the pointer. The first three need no new rollouts beyond evaluation.

### Cited Findings
- Frozen-policy value selection works on pi0.5 features: probe R^2 about 0.74, and K = 16 selection gives 26.7% to 44.3% — [arXiv HTML](https://arxiv.org/html/2605.28527). Q-Planning breaks the filtered-SFT plateau with a Q-function over successes and failures — [arXiv](https://arxiv.org/abs/2608.21204)
- CFG over positive and negative models gives a tunable improvement knob — [CFGRL](https://arxiv.org/html/2505.23458); [DIPOLE](https://arxiv.org/html/2601.00898v3); [AFIL on pi0.5](https://arxiv.org/html/2605.08434v1). RECAP uses a per-task 30th-percentile advantage threshold — [RECAP](https://arxiv.org/html/2511.14759v1)
- Implicit-negative flow loss needs no likelihoods — [DiffusionNFT](https://arxiv.org/html/2509.16117)
- Chunk-level redirection toward successful chunks at matched progress — [RedFlow](https://arxiv.org/html/2607.27782)
- Resets and branching from recorded or demo states — [RFCL](https://arxiv.org/abs/2405.03379); [U-GROW](https://arxiv.org/html/2609.22879); [PLD](https://arxiv.org/abs/2511.00091)
- Hindsight relabeling on pi0.5 — [LfH](https://arxiv.org/html/2607.09042v1)
- Noise-space steering of frozen pi0 — [DSRL](https://arxiv.org/html/2506.15799)

### Inferences
These are proposals, not results. Each names what to train, the data pairing, and where to localise.

1. **Pointer-advantage labels (no new data).** For every recorded chunk, compute delta_p = pointer advance during the chunk. Label a chunk positive if delta_p is above the per-task 30th percentile and the rollout did not stall within the next ~2 chunks. Label it negative if it falls within a window before t* in a failed rollout. Everything else is neutral or unlabelled. This is RECAP's indicator with the pointer as the value; it makes the success/failure pairing chunk-level and lets failed rollouts contribute their positive prefixes.
2. **Two-adapter CFG ("guide weights").** Train LoRA or fold A+ on positive chunks plus the demo, and A- on negative chunks. At each Euler step use v = v+ + w (v+ - v-). Sweep w per task on practice inits, then report the held-out init success. Watch for a known risk: with few negatives, v- is poorly fit, and a large w amplifies noise. AFIL's cosine-adaptive strength, alpha * (1 - cos(v+, v-)), is a safer default.
3. **DiffusionNFT-style single-adapter loss.** Keep one LoRA and use r ||v+ - v||^2 + (1 - r) ||v- - v||^2 with v_old = the current adapter, r in {0, 1} from item 1's labels, and optionally r = normalised delta_p. It is the closest drop-in replacement for the current filtered-BC loss.
4. **Matched-position corrections, RedFlow-style.** For each negative chunk at pointer p*, find successful rollouts' chunks executed at pointer p* whose recorded state is nearest in object pose. Use their (quality-weighted) mean as a regression target from the failing observation. This pairs success and failure by demo phase, not by init state.
5. **Autonomous DAgger at the stall (small new-rollout budget).** Restore the recorded MuJoCo state about 1-2 chunks before t*. Branch K short rollouts with fresh noise, with the round-0 composition, or with CFG w > 0. Label branches by whether the pointer passes p* + delta, and add the recovering segments as positives. Put the budget on the persistent-failure instances, where full-episode group methods get zero signal.
6. **Reverse curriculum from the demo.** For all-fail instances, reset to demo states just before the stall phase with the instance's object layout if restorable; otherwise use the demo's own states. Practise the tail segment, then move the start earlier (RFCL).
7. **Hindsight via BDDL predicates.** Evaluate every failed rollout's final simulator state against the goal predicates of sibling LIBERO-90 tasks in the same scene, and against the target task's sub-goal predicates. Relabel verified sub-successes as positives for the prefix sub-task. These labels are exact, unlike LfH's VLM labels.
8. **Test-time selection as a cheap diagnostic.** Train a linear probe on frozen pi0.5 features (vision encoder first, per 2605.28527) to predict "pointer advances within the next chunk". Pick among K sampled chunks. If this lifts held-out success above 0.66 without any training, failure data carries exploitable signal, and items 2-5 are worth building.

### Gaps
- All eight items are untested in our setting.
- Items 5-7 depend on restoring LIBERO/MuJoCo states from our recordings. Whether our recorded "states" include the full MuJoCo qpos/qvel for all objects, as LIBERO's `set_init_state` needs, must be checked in our code.
- I found no evidence on how many negative chunks a rank-16 LoRA needs before v- is reliable for CFG. The literature's negative-model results (AFIL, DIPOLE) use hundreds to thousands of rollouts.
