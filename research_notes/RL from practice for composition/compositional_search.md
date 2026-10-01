# Optimizing structured / compositional action programs from sparse execution feedback (state as of September 2026)

How I checked sources: on 2026-09-27 I fetched the arxiv.org/abs page of every arXiv ID below and matched the title, first author, submission date and the comment/journal-ref field (119 IDs; all resolved; 2 unrelated guesses were discarded). "Venue (arXiv)" means the arXiv comment or journal-ref states the venue. "Venue unverified" means the arXiv page does not state it, so any venue I give comes from memory. Numbers marked "(full text)" were read from the arxiv.org/html version of the paper. Numbers marked "approx., from figure" were read off a plot. Facts about OUR pipeline come from the task brief, not from sources.

Our setting, for reference: frozen pi0.5 (`pi05_libero`) with 10-step x 7-dim chunks, one demo per held-out LIBERO-90 task. The program has these parts: per demo frame and per dimension, one of K=8 leaf chunks (policy chunk means at training frames, 10,249 leaves) picked by max coverage, or a segment-wise variant. It runs through a state-aligned product-of-experts (PoE) guide (weight w, teacher variance) with a stall fallback, plus a gated LoRA "fold" in contact windows (-30/+10 frames around gripper events). Practice gives about 20-30 init states per task, tens of rollouts per task and binary success. About 61% of instances succeed zero-shot and about 35% keep failing. Every rollout and the stall location are logged. Imitating the successes plateaus at about 0.66 on unseen init states.

---

## Key Question 1: How are skill / motion-primitive libraries built for manipulation, and how are they adapted from feedback? (LOTUS, SkillDiffuser, PRISE, BUDS, skill chaining, DMP/ProMP policy search, Movement Primitive Diffusion, latent action spaces, hierarchical RL over primitives)

### Takeaway
There are two families. **Library-from-demonstrations** methods (BUDS, LOTUS, PRISE, SkillDiffuser, FIST) segment demos into skills and train a meta-controller by imitation. They adapt by adding or updating skills, and they use no reward-driven search. **Reward-driven search over a low-dimensional primitive parameterization** (PoWER, PI2, PI2-CMA, CMA-ES) is the only family with documented budgets in the tens of rollouts: PoWER learned ball-in-a-cup on a real WAM in about 75 iterations, starting from an imitation initialization. Hierarchical RL over primitives (MAPLE, RAPS, SPiRL, SkiMo) speeds up exploration, but its budgets are RL-scale. For a frozen flow/diffusion policy, the reward-driven analogue of "search in a latent action space" is DSRL (RL over the sampler's initial noise). DSRL improved pi0 on LIBERO and on real robots within tens of episodes.

### Cited Findings

**Catalogue: skill libraries and primitives**

| Method | Mechanism | Adaptation signal / budget | Reported numbers | Code / page | ID, venue |
|---|---|---|---|---|---|
| BUDS | Agglomerative clustering of unsegmented demos into a hierarchical task structure. Recurring patterns become skills (goal-conditioned policies) composed by a meta-controller | Demos only (~30 min of demos) | Skills from multi-task demos raise average task success by 8% over per-task skills | — | [2109.13841](https://arxiv.org/abs/2109.13841), RA-L 2022 (arXiv) |
| LOTUS | Continual skill discovery with an open-vocabulary vision model. Updates existing skills and adds new ones, and a meta-controller composes them in lifelong learning | Few demos per new task, no reward | ">11%" success over state-of-the-art baselines | [project](https://ut-austin-rpl.github.io/Lotus/) | [2311.02058](https://arxiv.org/abs/2311.02058), ICRA 2024 (arXiv) |
| PRISE | Vector-quantizes continuous actions, then runs **BPE** over code sequences to induce variable-length primitives | Demos. Pretrained on LIBERO-90 (50 demos/task), 5-shot on the first 8 LIBERO-LONG tasks | 5-shot LIBERO-LONG: **52.7 ± 0.9** vs 43.5 ± 0.6 without BPE, 40.4 BC from scratch, 42.0 pretrained ACT (full text) | [github.com/FrankZheng2022/PRISE](https://github.com/FrankZheng2022/PRISE) | [2402.10450](https://arxiv.org/abs/2402.10450), ICML 2024 (arXiv) |
| SkillDiffuser | Learns discrete, interpretable skill embeddings that condition a diffusion trajectory planner | Demos | SOTA on Meta-World and LOReL (abstract) | — | [2312.11598](https://arxiv.org/abs/2312.11598), CVPR 2024 (arXiv) |
| FIST | Skills from offline data, an inverse skill-dynamics model and a distance function. Semi-parametric few-shot imitation | Few downstream demos | Outperforms baselines on maze and kitchen (abstract) | — | [2107.08981](https://arxiv.org/abs/2107.08981), venue unverified |
| Adversarial Skill Chaining | Chains skills and regularizes terminal states so that the next skill's initiation set is reached (mechanism from title) | — (not extracted) | — | — | [2111.07999](https://arxiv.org/abs/2111.07999), CoRL 2021 (arXiv) |
| Sequential Dexterity | Chains dexterous sub-policies for long-horizon tasks | — (not extracted) | — | — | [2309.00987](https://arxiv.org/abs/2309.00987), CoRL 2023 (arXiv) |
| Generative Skill Chaining | Skill-centric diffusion models whose distributions are composed at inference, with geometric constraints enforced, for long-horizon plans | Demos, inference-time composition | — | [page](https://generative-skill-chaining.github.io/) | [2401.03360](https://arxiv.org/abs/2401.03360), CoRL 2023 (arXiv) |
| PoWER (DMP policy search) | EM-style reward-weighted averaging of exploration on motor-primitive parameters, initialized by imitation | Episodic return, real robot | "The robot regularly succeeds at bringing the ball into the cup after approximately **75 iterations**." On a benchmark, "PoWER outperforms the other algorithms from 50 rollouts on" | — | [NeurIPS 2008 paper](https://proceedings.neurips.cc/paper_files/paper/2008/file/7647966b7343c29048673252e490f736-Paper.pdf) (not on arXiv) |
| PI2 / PI2-CMA | PI2 belongs to the "probability-weighted averaging" family together with CEM and CMA-ES. PI2-CMA adapts the exploration magnitude automatically | Episodic cost | Conceptual and empirical comparison, no LIBERO | — | [1206.4621](https://arxiv.org/abs/1206.4621), ICML 2012 (arXiv) |
| CMA-ES | Covariance-matrix-adapting evolution strategy (reference tutorial) | Black-box | — | — | [1604.00772](https://arxiv.org/abs/1604.00772) (tutorial) |
| Movement Primitive Diffusion | Diffusion imitation learning whose output is Probabilistic Dynamic Movement Primitive (ProDMP) parameters, giving smooth, gentle motion | Demos (data-efficient) | Beats diffusion-IL baselines on success, motion quality and data efficiency (surgical/deformable tasks) | [page](https://scheiklp.github.io/movement-primitive-diffusion/) | [2312.10008](https://arxiv.org/abs/2312.10008), RA-L 9 (2024) (arXiv jref) |
| Movement primitives survey | 105-page encyclopedic review of DMP/ProMP-style frameworks and their open problems | — | — | — | [2601.02379](https://arxiv.org/abs/2601.02379) (Dec 2025) |

**Hierarchical RL over primitives / skill priors**
- MAPLE: a hierarchical policy picks among a pre-defined library of behavior primitives (grasp, push, ...) and sets their parameters, which augments standard RL. It "outperforms baseline approaches by a significant margin" and transfers to new task variants and hardware — [MAPLE 2110.03655](https://arxiv.org/abs/2110.03655) (ICRA 2022, Outstanding Learning Paper, per arXiv).
- RAPS: a manually specified library of parameterized primitives whose arguments are learned by RL under a **sparse terminal reward**. The changed action interface improves learning efficiency with any underlying RL algorithm — [RAPS 2110.15360](https://arxiv.org/abs/2110.15360) (NeurIPS 2021, arXiv).
- SPiRL learns a state-conditioned prior over skills from offline data and uses it to guide max-entropy RL. The authors show "learned skill priors are essential for effective skill transfer" — [SPiRL 2010.11944](https://arxiv.org/abs/2010.11944) (CoRL 2020, arXiv). SkiMo adds a skill-level world model — [2207.07560](https://arxiv.org/abs/2207.07560) (CoRL 2022, arXiv).
- Q-chunking runs TD-based RL directly in a chunked action space, which lets offline temporally-consistent behavior drive online exploration and uses unbiased n-step backups. It reports gains on long-horizon **sparse-reward** manipulation — [Q-chunking 2507.07969](https://arxiv.org/abs/2507.07969) (NeurIPS 2025, arXiv).

**Latent action spaces and RL in them**
- LAPA trains a VQ-VAE over discrete latent actions between frames, pretrains a latent VLA and fine-tunes it to robot actions with small data — [LAPA 2410.11758](https://arxiv.org/abs/2410.11758) (ICLR 2025, arXiv). LAPO recovers latent-action policies from video, which "can be rapidly fine-tuned into expert-level policies, either offline ... or online with rewards" — [LAPO 2312.10812](https://arxiv.org/abs/2312.10812) (ICLR 2024 spotlight, arXiv). Genie learns latent actions for a generative interactive environment — [2402.15391](https://arxiv.org/abs/2402.15391) (venue unverified). UniVLA learns task-centric latent actions in DINO feature space and beats OpenVLA "with less than 1/20 of pretraining compute and 1/10 of downstream data" — [UniVLA 2505.06111](https://arxiv.org/abs/2505.06111) (RSS 2025, arXiv).
- **DSRL** (diffusion steering via RL) runs RL over the latent *noise* space of a frozen diffusion/flow policy. It needs only black-box access and never modifies the base weights. DSRL-NA uses noise aliasing: a noise-space critic is distilled from an action-space critic — [DSRL 2506.15799](https://arxiv.org/abs/2506.15799). Full-text numbers ([html](https://arxiv.org/html/2506.15799)):
  - pi0 on a LIBERO-90 pick-and-place task: about 20% to about 100% after about 10,000 online samples (approx., from figure).
  - Real "turn on toaster": 5/20 to 18/20 in 80 episodes (~11k steps).
  - Real "put spoon on plate": 15/20 to 19/20 in 65 episodes (~10k steps).
  - Single-task Franka pick-and-place (base policy, not pi0): 2/10 to 9/10 in about 40 episodes (3,500 steps).
  - Code: [github.com/ajwagen/dsrl](https://github.com/ajwagen/dsrl). pi0/JAX variants that list LIBERO and Aloha support include [github.com/nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0) (repo titles say "CoRL 2025"; the venue is not on the arXiv page).
- StructRL finds that noise injected inside the denoising chain is "diluted" by the remaining denoising steps. It moves structured (temporally smooth, per-action-group-scaled) exploration noise into the action space and uses a deterministic ODE decoder and last-step replay — [StructRL 2608.15139](https://arxiv.org/abs/2608.15139) (Aug 2026, venue not stated).

**Residual / refinement over a frozen chunked policy**
- ResiP: a closed-loop residual RL policy on top of a *frozen* chunked BC policy. The authors argue that chunked BC behaves "like trajectory planners" rather than closed-loop controllers — [ResiP 2407.16677](https://arxiv.org/abs/2407.16677) (venue unverified). Policy Decorator is a model-agnostic residual with controlled exploration (ManiSkill, Adroit) — [2412.13630](https://arxiv.org/abs/2412.13630) (venue unverified). PLD probes failure regions with residual RL, collects hybrid rollouts and distills them into the VLA with SFT, reaching "near-saturated 99% task success on LIBERO" — [PLD 2511.00091](https://arxiv.org/abs/2511.00091) (venue unverified).

### Inferences
- Our leaf library is a **non-parametric primitive library**: chunks are borrowed from training frames rather than generated from DMP parameters. Its closest structural relatives are PRISE (discrete action codes composed into sequences) and retrieval methods (STRAP, Key Question 4), not DMPs. The transferable DMP lesson is the *parameterization*: PoWER-type search works in tens of rollouts because it optimizes a **few parameters initialized from the demo**. Our program instead has per-frame x per-dim discrete choices (roughly frames x 7 x 8 options), which is far too many decision variables for tens of binary rollouts. Collapsing it to per-phase choices is the precondition for any search (Key Question 5).
- PRISE's +9.2 pts from BPE over quantized action codes (5-shot LIBERO-LONG, same benchmark family as ours) suggests running BPE (or Stitch, Key Question 2) over **leaf-ID sequences of successful programs** to create macro-leaves. That shrinks the choice set at each decision.
- DSRL is the most directly transplantable reward-driven method. It needs a frozen flow policy (we have one), keeps the weights untouched (it could sit beside the fold), has official pi0/LIBERO code, and reports budgets (40-80 episodes per task in the real world) close to our tens of rollouts per task. Our PoE guide changes the sampler, so the noise-to-action map is guide-dependent. Noise steering would then optimize "which in-guide sample to take" and not replace the program. This is an inference that needs a test.
- StructRL's dilution finding matters if we add per-step noise inside the guided flow sampler to explore: noise injected at intermediate flow times may be washed out. Exploration should instead act on the program (leaf/w choices), on the initial noise (DSRL) or in action space.

### Gaps
- I found no paper that runs PoWER, PI2 or CMA-ES over a VLA's chunk composition, or any DMP-style policy search on LIBERO.
- LOTUS's exact benchmark numbers (I recall LIBERO-GOAL) were not extracted. Its abstract states only ">11%".
- I did not find rollout counts for MAPLE/RAPS/SPiRL comparable to ours; those are RL-scale (not quantified here).
- I did not verify Konidaris & Barto's original skill chaining (NeurIPS 2009), so it is omitted.

---

## Key Question 2: What do program synthesis and library learning with execution feedback (DreamCoder, LILO, Stitch, Voyager, Code-as-Policies, Eureka, RoboCodeX, Inner Monologue, LLM repair) offer for a program over motion chunks: credit assignment to program parts, repair of the failing part, growing the library from successes?

### Takeaway
Three mechanisms transfer cleanly:
1. **Verify-then-store library growth.** Voyager, LRLL and SkillComposer keep only programs that passed execution.
2. **Compression of successful programs into reusable abstractions.** Examples are DreamCoder, Stitch, babble, LILO and SkillComposer's macro skills, and in robotics PRISE's BPE.
3. **Failure localization followed by repair of the failing sub-part only.** REFLECT, AHA, Code-as-Monitor, Self-Debugging and DROC do this.

For searching over discrete program structure with reward only, the strongest evidence (Carvalho et al., ICLR 2024) is that **local search directly in program space beats CEM in a learned program-embedding space**. LLM-in-the-loop generation (Code-as-Policies, RoboCodeX, Eureka) is less relevant because our "program" is a numeric chunk selection, not code.

### Cited Findings

**Library learning (compression of successful programs)**
- DreamCoder uses a wake-sleep loop. It "alternately extends the language with new symbolic abstractions and trains the neural network on imagined and replayed problems" — [DreamCoder 2006.08381](https://arxiv.org/abs/2006.08381) (venue unverified on the arXiv page; PLDI 2021 from memory).
- Stitch (corpus-guided top-down synthesis of abstractions) is "3-4 orders of magnitude faster and uses 2 orders of magnitude less memory" than DreamCoder's compressor, with comparable or better compressivity. It is "robust to terminating the search procedure early" — [Stitch 2211.16605](https://arxiv.org/abs/2211.16605) (POPL 2023, arXiv jref). babble does the same with e-graphs and anti-unification — [2212.04596](https://arxiv.org/abs/2212.04596) (POPL 2023, arXiv).
- LILO combines LLM-guided synthesis, Stitch compression and AutoDoc naming, and "solves more complex tasks and learns richer libraries" than DreamCoder — [LILO 2310.19791](https://arxiv.org/abs/2310.19791) (ICLR 2024, arXiv).
- SkillComposer (robotics, Aug 2026) uses a generate-test loop in which an LLM iteratively revises robot programs before execution. "Successful programs are stored and processed by an online library-learning algorithm that compresses recurring function sequences into reusable macro skills" — [SkillComposer 2608.14944](https://arxiv.org/abs/2608.14944) (submitted to Humanoids 2026, per arXiv).

**Verified skill libraries grown from execution**
- Voyager has three parts: an automatic curriculum, "an ever-growing skill library of executable code", and "iterative prompting ... [with] environment feedback, execution errors, and self-verification". Results: 3.3x more unique items, 2.3x longer distances, milestones up to 15.3x faster (Minecraft) — [Voyager 2305.16291](https://arxiv.org/abs/2305.16291) (venue unverified). [Code](https://voyager.minedojo.org/).
- LRLL has four parts: a soft memory, self-guided exploration, a skill abstractor that distills recent experiences into new library skills, and lifelong growth "while bypassing gradient-based optimization, thus relieving the learner from catastrophic forgetting" — [LRLL 2406.18746](https://arxiv.org/abs/2406.18746) (ICRA 2024, arXiv).
- BOSS grows a skill library by "skill bootstrapping": an LLM proposes which primitive skills to chain, and the agent practices them without task reward — [BOSS 2310.10021](https://arxiv.org/abs/2310.10021) (CoRL 2023 oral, arXiv).
- Uni-Skill requests new skill implementations "when existing ones are insufficient", retrieving from a VerbNet-style video-derived repository — [Uni-Skill 2603.02623](https://arxiv.org/abs/2603.02623) (ICRA 2026, arXiv).

**Failure localization and repair of the failing part**
- REFLECT builds a hierarchical summary of multisensory robot experience. An LLM uses it to explain the failure, and the explanation guides a planner to correct it (RoboFail dataset) — [REFLECT 2306.15724](https://arxiv.org/abs/2306.15724) (CoRL 2023, arXiv). AHA is a VLM for detecting and reasoning over manipulation failures — [2410.00371](https://arxiv.org/abs/2410.00371) (venue unverified). Code-as-Monitor uses constraint-aware visual programs for reactive and proactive failure detection — [2412.04455](https://arxiv.org/abs/2412.04455) (CVPR 2025, arXiv).
- DROC distills online language corrections, which can target both "high-level task plans and low-level skill primitives", into a retrievable knowledge base. It uses "half of the total number of corrections needed in the first round and requires little to no corrections after two iterations" — [DROC 2311.10678](https://arxiv.org/abs/2311.10678) (venue unverified).
- Inner Monologue feeds closed-loop language feedback (success detection, scene description) back into LLM planning — [2207.05608](https://arxiv.org/abs/2207.05608) (venue unverified on arXiv). Self-Debugging makes LLMs debug their own code using execution results — [2304.05128](https://arxiv.org/abs/2304.05128) (venue unverified). Reflexion stores verbal feedback in episodic memory in place of weight updates — [2303.11366](https://arxiv.org/abs/2303.11366) (venue unverified).

**LLM program generation and evolutionary program search**
- Code as Policies has an LLM write policy code over perception and control APIs — [2209.07753](https://arxiv.org/abs/2209.07753) (venue unverified on arXiv). RoboCodeX builds tree-structured multimodal code generation from object-centric units, with iterative self-updating SFT — [2402.16117](https://arxiv.org/abs/2402.16117) (venue unverified). Demo2Code summarizes demos and then synthesizes code — [2305.16744](https://arxiv.org/abs/2305.16744) (venue unverified). ProgPrompt — [2209.11302](https://arxiv.org/abs/2209.11302).
- Eureka performs "evolutionary optimization over reward code". It beats human rewards on 83% of 29 environments, with a 52% average normalized improvement — [Eureka 2310.12931](https://arxiv.org/abs/2310.12931) (ICLR 2024, arXiv). DrEureka extends this to sim-to-real — [2406.01967](https://arxiv.org/abs/2406.01967) (RSS 2024, arXiv). AlphaEvolve is an LLM-driven evolutionary coding agent — [2506.13131](https://arxiv.org/abs/2506.13131).

**Reward-only search over program structure (the closest analogue to our discrete program)**
- LEAPS learns a program embedding space without supervision, then **searches it (CEM) for a program that maximizes return**, using the reward signal alone — [LEAPS 2108.13643](https://arxiv.org/abs/2108.13643) (NeurIPS 2021, arXiv). HPRL composes programs hierarchically — [2301.12950](https://arxiv.org/abs/2301.12950) (ICML 2023, arXiv).
- Carvalho et al. find that "algorithms searching in the programmatic space significantly outperform those in LEAPS and HPRL", and that search "[is] more likely to stop at local maxima when searching in the latent space" — [2410.12166](https://arxiv.org/abs/2410.12166) (ICLR 2024, arXiv). PIRL is the earlier example of searching for programmatic policies — [1804.02477](https://arxiv.org/abs/1804.02477) (ICML 2018, arXiv).

### Inferences
Mapping onto a program over motion chunks:
- **Credit assignment.** The stall point of our alignment pointer plays the same role as REFLECT's failure localization or a failing unit test in Self-Debugging: it names the program part to repair. The program-repair pattern is to **freeze the prefix that executed correctly and search only the segment at or just before the stall**. That cuts the search space from all phases to one phase. The analogy is mine; no paper does this for motion chunks.
- **Repair operator.** Carvalho et al. imply that the neighborhood should be defined in the *native* program space: swap one segment's leaf for the next-ranked leaf, change one segment's w, shift one fold-window boundary. A learned embedding of programs would be worse. Hill climbing or top-two Thompson sampling over these local edits (Key Question 3) matches that evidence.
- **Library growth.** Voyager and LRLL store only *verified* programs. For us, each successful rollout's (init-state embedding, per-phase leaf choices, w, window) tuple is a verified program. New leaves could be added from successful practice chunks, which grows the library in place of LoRA-distilling successes. We already know the LoRA-distilled route plateaus at ~0.66 on unseen init states.
- **Compression.** Stitch or BPE over successful leaf-ID sequences, across tasks, would find recurring multi-segment "macro-leaves" (for example approach-then-grasp). PRISE shows this helps on LIBERO-LONG. Stitch is fast and robust to early stopping, so it fits a nightly consolidation step.
- LLM-in-the-loop repair (Eureka, Voyager-style prompting) would matter only if the program had a symbolic layer (for example "which phase to re-borrow and from which task"). With numeric leaf choices, bandits or local search are cheaper and need no LLM.

### Gaps
- I found no work applying DreamCoder, Stitch or LILO compression to *continuous motion-chunk* programs. PRISE (BPE over VQ codes) is the only robotics analogue found.
- There are no rollout-budget numbers for LLM program-repair loops in manipulation comparable to our tens of rollouts. Their "budget" is usually LLM calls plus a few executions per repair, and the reports do not quantify it in a way that transfers.
- The venues of Voyager, Code as Policies, Inner Monologue, RoboCodeX, Reflexion and Self-Debugging are not stated on their arXiv pages and were left unverified.

---

## Key Question 3: With very few rollouts, which black-box optimizers (CEM, CMA-ES, bandits / Thompson sampling, Bayesian optimization, evolution strategies, Diffusion-ES, MCTS over chunks) work, and how many rollouts buy how many points of success?

### Takeaway
Published budgets split into three bands:
- **Tens of rollouts per task:** PoWER on imitation-initialized primitives (~75); DSRL noise-steering (40-80 real episodes, +35 to +65 pts on single tasks); Q-VGM offline (150 logged rollouts per 10-task suite → +6.1 pts); Intelligent Trial and Error with a precomputed behavior map (<2 min of trials).
- **Hundreds per configuration:** GPC's weight grid (11 weights x 200 rollouts → +2 to +7.5 pts).
- **Thousands, with GPU clusters:** full RL on flow VLAs. SimpleVLA-RL went from 48.9 to 96.9 on LIBERO from 1-demo SFT. piRL took pi0.5 from 77.1 to 98.3.

Test-time **search/selection** over the frozen policy's own samples shows large headroom:
- SVA: pass@1 33% → pass@32 92%.
- V-VLAPS: MCTS with a simulator took LIBERO-90 from 57% to 90% with Octo.
- VGAS: best-of-N with an offline critic, 5-shot LIBERO +9.2 pts.

For discrete choices, top-two Thompson sampling or best-arm identification is the right tool. Binomial arithmetic says per-task budgets of 20-30 rollouts can only resolve large effects.

### Cited Findings

**Classical black-box optimizers**
- iCEM adds temporally-correlated actions and memory to CEM and needs "2.7-22x less samples" for "1.2-10x" performance — [iCEM 2008.06389](https://arxiv.org/abs/2008.06389) (venue unverified). CMA-ES tutorial — [1604.00772](https://arxiv.org/abs/1604.00772). PI2-CMA — [1206.4621](https://arxiv.org/abs/1206.4621) (ICML 2012). Evolution strategies as a scalable RL alternative — [1703.03864](https://arxiv.org/abs/1703.03864).
- Intelligent Trial and Error (a precomputed map of high-performing behaviors guides a trial-and-error search) lets robots "adapt to damage in less than two minutes", across 5 hexapod damages and 14 arm joint failures — [Cully et al. 1407.3501](https://arxiv.org/abs/1407.3501) (Nature 2015 from memory; not stated on the arXiv page). MAP-Elites — [1504.04909](https://arxiv.org/abs/1504.04909).
- BO for robot controllers degrades in high dimensions. Domain-knowledge feature transforms that cut dimensionality "accelerate learning on hardware" on the ATRIAS biped — [Rai et al. 1709.06047](https://arxiv.org/abs/1709.06047) (submitted to ICRA 2018 per arXiv).
- BO over discrete and mixed spaces: BOCS [1806.08838](https://arxiv.org/abs/1806.08838) (ICML 2018); COMBO, a graph-Cartesian-product kernel with a Horseshoe prior for variable selection, [1902.00448](https://arxiv.org/abs/1902.00448) (NeurIPS 2019); CoCaBO [1906.08878](https://arxiv.org/abs/1906.08878) (venue unverified); CASMOPOLITAN, "local optimisation with a tailored kernel" for high-dimensional categorical spaces, [2102.07188](https://arxiv.org/abs/2102.07188) (ICML 2021).
- Thompson sampling tutorial — [Russo et al. 1707.02038](https://arxiv.org/abs/1707.02038) (FnT ML 2018, arXiv jref). Top-two Thompson / top-two probability sampling for **best-arm identification** reaches the optimal exponential convergence rate of the posterior on the best design — [Russo 1602.08448](https://arxiv.org/abs/1602.08448).
- Active Offline Policy Selection "use[s] OPE estimates to warm start the online evaluation". It then picks which policy to evaluate next "based on a Bayesian optimization method with a kernel that represents policy similarity", under a restricted interaction budget, including on real robotics — [AOPS 2106.10251](https://arxiv.org/abs/2106.10251) (NeurIPS 2021, arXiv).

**Composition weights searched by rollouts**
- GPC (General Policy Composition) takes convex combinations of distributional scores from several diffusion/flow policies, including pi0 + Flow Policy, and searches the weight on a grid "from 0.0 to 1.0 in steps of 0.1". Each weight is scored by rollout success rate. Robomimic used 200 rollouts per configuration, about 2.5 h for a full search. Gains were +2.22 to +7.55 on Robomimic/PushT, up to +7 on RoboTwin, pi0+FP at 88.94 (+2.52), and 14/20 in the real world vs 7/20 and 12/20 for the parents (full text) — [GPC 2510.01068](https://arxiv.org/abs/2510.01068) (ICLR 2026, arXiv). PoCo composes diffusion policies across modalities and domains, and with analytic costs, at inference — [2402.02511](https://arxiv.org/abs/2402.02511) (RSS 2024, arXiv).

**Evolutionary / tree search over trajectories and chunks**
- Diffusion-ES samples trajectories from a diffusion model, scores them with a black-box reward and mutates the elites by truncated noising/denoising. It reached SOTA on nuPlan and can optimize non-differentiable, LLM-written rewards — [Diffusion-ES 2402.06559](https://arxiv.org/abs/2402.06559) (venue unverified).
- VLAPS runs MCTS biased by VLA action priors, using a model of the environment, and improves success "by as much as 67 percentage points" — [VLAPS 2508.12211](https://arxiv.org/abs/2508.12211).
- V-VLAPS adds a value head (an MLP of ~2.4M parameters trained on 7.8k-18.8k offline rollout examples per suite) with an Octo backbone. Planning uses a simulator matching the evaluation environment, 300 action samples per node and 600 s or 1800 s per episode (full text) — [V-VLAPS 2601.00969](https://arxiv.org/abs/2601.00969). Success rates:

  | Suite | Octo alone | V-VLAPS at 600 s | V-VLAPS at 1800 s |
  |---|---|---|---|
  | libero_90 | 57% | 89% | 90% |
  | libero_10 | 38% | 75% | 85% |
  | 5-suite average | 60.2% | 87.4% | 91.6% |

- SVA (Search, Value, Act) reports that "frozen VLAs already contain competent behaviors in their output distribution, with overall success rates rising from 33% at pass@1 to 92% at pass@32". It distills MCTS into a Q-model (256 simulations on SimplerEnv/RoboTwin) and picks among 16 candidates at deployment without a simulator. SimplerEnv went from 38.5 to 50.7 and RoboTwin 2.0 from 36.0 to 43.5 (full text) — [SVA 2607.03751](https://arxiv.org/abs/2607.03751). Monte Carlo Tree Diffusion — [2502.07202](https://arxiv.org/abs/2502.07202) (ICML 2025 spotlight, arXiv).

**Sample-and-select with a verifier or value (no weight updates)**
- V-GPS re-ranks a generalist policy's sampled actions with a value function from offline RL. It works for 5 policies across 12 tasks without touching their weights — [2410.13816](https://arxiv.org/abs/2410.13816) (CoRL 2024, arXiv).
- RoboMonkey combines sampling, Gaussian perturbation, majority voting and a VLM verifier for +25% absolute OOD and +9% ID — [2506.17811](https://arxiv.org/abs/2506.17811) (venue unverified).
- Hume adds a value-query head, resamples candidates and selects by value — [2505.21432](https://arxiv.org/abs/2505.21432).
- BID (Bidirectional Decoding) selects among samples by backward coherence and forward contrast — [2408.17355](https://arxiv.org/abs/2408.17355).
- VGAS applies best-of-N with a geometry-aware chunk critic trained offline from 5-shot data. On LIBERO 5-shot with SmolVLA-0.5B it reached 49.0% vs 39.8% for BC (full text) — [VGAS 2602.07399](https://arxiv.org/abs/2602.07399), [code](https://github.com/Jyugo-15/VGAS).
- ITPS steers diffusion sampling with human-specified alignment objectives. Its stochastic sampling gives the best alignment vs distribution-shift trade-off — [2411.16627](https://arxiv.org/abs/2411.16627) (ICRA 2025, arXiv).

**RL on flow/diffusion VLAs (budget reference points)**
- SimpleVLA-RL (OpenVLA-OFT, **1-demo-per-task SFT**) with binary outcome reward, groups of 8 samples, 8x A800 and RL on 500 scenarios per suite ([full text](https://arxiv.org/html/2509.09674)) — [SimpleVLA-RL 2509.09674](https://arxiv.org/abs/2509.09674), [code](https://github.com/PRIME-RL/SimpleVLA-RL):

  | LIBERO suite | 1-demo SFT | After RL |
  |---|---|---|
  | Spatial | 54.9 | 98.7 |
  | Object | 59.6 | 98.8 |
  | Goal | 17.3 | 91.7 |
  | Long | 63.6 | 98.2 |
  | Average | 48.9 | 96.9 |

  Exploration tricks: dynamic sampling (drop all-success or all-fail groups), clip-higher (0.8 to 1.28) and temperature 1.0 to 1.6. RL discovered a "pushcut" (pushing in place of grasp-move-place).
- piRL (Flow-Noise and Flow-SDE, PPO, binary reward, 8x H100) ([full text](https://arxiv.org/html/2510.25889v2)) — [piRL 2510.25889](https://arxiv.org/abs/2510.25889), [code RLinf](https://github.com/RLinf/RLinf). pi0.5 was SFT'd on 40 trajectories per suite:

  | pi0.5, LIBERO suite | Few-shot SFT | After RL (Flow-Noise) |
  |---|---|---|
  | Spatial | 84.6 | 99.6 |
  | Object | 95.4 | 100 |
  | Goal | 84.6 | 99.6 |
  | Long | 43.9 | 94.0 |
  | Average | 77.1 | 98.3 |

  pi0 went from 57.6 to 97.6.
- Q-VGM (offline-to-online critic-guided velocity matching for flow VLAs): "improves the few-shot-SFT four-suite average from 84.6% to 90.7% offline using **150 policy rollouts per suite**, and reaches 98.4% after online training". It claims 5.3x better sample efficiency than on-policy RL — [Q-VGM 2606.08015](https://arxiv.org/abs/2606.08015) (abstract of the revised version). A search snippet of the same paper instead said pi0.5 "79.0% to 92.5%"; this is probably a different version or setting and was **not reconciled**.
- More data points:
  - FPO (flow policy optimization for pi0 on LIBERO/ALOHA) — [2510.09976](https://arxiv.org/abs/2510.09976) (ICRA 2026, arXiv).
  - VLA-RL: OpenVLA-7B +4.5% over the best fine-tuned baseline on 40 LIBERO tasks — [2505.18719](https://arxiv.org/abs/2505.18719).
  - PA-RL (re-rank base-policy samples by Q, then distill): OpenVLA went from 40% to 70% in the real world in 40 minutes — [2412.06685](https://arxiv.org/abs/2412.06685).
  - DPPO — [2409.00588](https://arxiv.org/abs/2409.00588). ReinFlow — [2505.22094](https://arxiv.org/abs/2505.22094) (NeurIPS 2025, arXiv jref).
  - Self-Improving Embodied Foundation Models (steps-to-go reward and success detector, autonomous practice) — [2509.15155](https://arxiv.org/abs/2509.15155) (NeurIPS 2025, arXiv).
  - SAILOR (learning to search with a world model and reward model beats diffusion-policy BC even with 5-10x more demos) — [2506.05294](https://arxiv.org/abs/2506.05294).

### Inferences
- **Binary-reward arithmetic.** These are my calculations: two-proportion test, α=0.05 two-sided, power 0.8.

  | Comparison | Rollouts needed per arm |
  |---|---|
  | 0.6 vs 0.7 | ~356 |
  | 0.66 vs 0.76 | ~323 |
  | 0.6 vs 0.8 | ~82 |
  | 0.6 vs 0.9 | ~32 |
  | 0.1 vs 0.4 | ~32 |

  At n=20 the standard error of a success rate near 0.6 is ±0.11. So **per-task** tuning with 20-30 rollouts can only find *large* effects, such as rescuing an instance class from ~0.1 to ~0.4 or more. Small shared improvements (w, teacher variance, window offsets) have to be estimated **pooled across tasks**. GPC's 200 rollouts per configuration fit that picture: its +2 to +7.5 pt gains would be invisible at our per-task budget.
- **Discrete choices** (leaf rank per phase, prefix-frozen repair candidates) are a best-arm-identification problem, so top-two Thompson sampling (Russo) is the principled allocator. Beta-Bernoulli posteriors per arm can be warm-started from logged rollouts, the way AOPS warm-starts from OPE. Combinatorial BO (COMBO, CASMOPOLITAN) only pays off when the categorical space stays large after reparameterization; with one to three phases x 8 leaves, plain TS is enough.
- **Continuous shared knobs** (w, teacher variance, fold window start/end, stall threshold) have roughly 4-6 dimensions. That suits CMA-ES or GP-BO where each evaluation is a batch of rollouts pooled across held-out tasks. The PoWER, PI2 and CEM family is reward-weighted averaging and has the same data profile, since it was designed for episodic returns.
- **Headroom check before building search.** SVA (pass@1 33% → pass@32 92%) and V-VLAPS (LIBERO-90 57 → 90 with simulator MCTS) show that frozen-policy sample diversity can hide large headroom. For our ~35% persistent failures, the first measurement should be pass@k: k fresh-seed reruns per failing instance, with the fold, guide and program fixed. If per-rollout success is q, pass@k covers the chance of at least one success:

  | q | k = 10 | k = 20 | k = 32 |
  |---|---|---|---|
  | 0.05 | 0.40 | 0.64 | 0.81 |
  | 0.10 | 0.65 | 0.88 | 0.97 |

  **Project gotcha from our own notes, not the literature:** seed-0 reruns are exact replays, so pass@k must use fresh seeds. If pass@k ≈ 0, no selection or search over the current program's samples will fix those instances, and the geometry has to change (Key Question 4).
- Simulator MCTS (VLAPS/V-VLAPS) needs matching simulator state resets, which is privileged. It is useful as an **oracle diagnostic**, answering "which phase, and which leaf or noise choice, would have succeeded". It is not a deployable method in our framing. SVA's recipe (search offline, distill into a Q-model, select among candidates at deployment) is the deployable version.
- **Caution on the big RL numbers.** SimpleVLA-RL's full-text description says RL was "conducted on 500 corresponding simulation scenarios" for each suite, which appear to be the same init states used for evaluation. If so, its 96.9% measures seen-init performance. It does not transfer to our unseen-init metric, where imitation of successes plateaus at 0.66. piRL's train/eval init split was not stated in the parts I read.

### Gaps
- I found no source reporting sample counts for Thompson sampling or BO over discrete chunk or primitive choices in manipulation with binary reward. The best-arm arithmetic above is mine, not from a paper.
- piRL's total environment episodes and its train/test init-state protocol were not visible (they are in appendix tables 7-8 that I did not see).
- The DSRL LIBERO number (~20% → ~100% in ~10k samples) was read from a figure and is approximate. Which pi0 checkpoint and how many LIBERO tasks were used was not extracted.
- Q-VGM's two conflicting headline numbers (84.6 → 90.7 offline vs 79.0 → 92.5) were not reconciled.
- The venues of Diffusion-ES, iCEM, RoboMonkey, ResiP, Policy Decorator and PA-RL are not on their arXiv pages (unverified).

---

## Key Question 4: How can a single demonstration be adapted to a new instance (object pose changes) at test time — trajectory retargeting, keypoint/object-centric alignment, DemoGen, MimicGen/SkillMimicGen — and could this make our composition adapt to the instance instead of following the demo path?

### Takeaway
One pattern recurs across Coarse-to-Fine, DINOBot, DITTO, MimicGen, SkillGen, DemoGen and VLBiMan: **split the demo into free-space motion and object-relative contact/skill segments; transform the contact segments rigidly with the new object pose (preserving the end-effector-to-object relative pose); and re-plan or interpolate the free-space motion between them**. From one demo, DemoGen took real-world success from 11.0% to 74.6%. DemoDiffusion used a retargeted trajectory as a *prior* that a pretrained diffusion policy refines, reaching 83.8% vs 13.8% for the base policy and 52.5% for retargeting alone. That setup has the same shape as our PoE guide: a demo path fused with the policy's sampler. It indicates that our guide should follow an **object-relative, instance-transformed** target in contact windows, not the raw demo path.

### Cited Findings

**Align then replay (one demo, no training on the new task)**
- Coarse-to-Fine IL models the task as a coarse approach followed by a fine interaction. It estimates the bottleneck end-effector pose relative to the object, moves there on a linear path, then "the original demonstration's end-effector velocities are simply replayed". It was tested on 8 real tasks from a single demo — [2105.06411](https://arxiv.org/abs/2105.06411) (ICRA 2021, arXiv).
- DINOBot retrieves "the most visually similar object experienced during human demonstrations" with DINO image-level features. It then aligns the end-effector using pixel-level features and replays — [2402.13181](https://arxiv.org/abs/2402.13181) (ICRA 2024, arXiv). [Page](https://www.robot-learning.uk/dinobot).
- DITTO segments the manipulated objects and their relative motion, then at test time re-detects the objects and "warp[s] the demonstration trajectory to the current scene" — [2403.15203](https://arxiv.org/abs/2403.15203) (IROS 2024, arXiv).
- IMOP learns task-invariant regions and matches them between demo and test scene to compute end-effector poses. It gains +11.5% average on 22 novel tasks from a single demo (RLBench) — [2405.13178](https://arxiv.org/abs/2405.13178) (RSS 2024, arXiv).
- Other alignment-style methods:
  - KAT uses keypoint-token in-context imitation with an off-the-shelf LLM — [2403.19578](https://arxiv.org/abs/2403.19578) (RSS 2024, arXiv).
  - Instant Policy does graph-diffusion in-context imitation from 1-2 demos — [2411.12633](https://arxiv.org/abs/2411.12633) (venue unverified).
  - Neural Descriptor Fields (SE(3)-equivariant) — [2112.05124](https://arxiv.org/abs/2112.05124) (venue unverified).
  - Robot See Robot Do — [2409.18121](https://arxiv.org/abs/2409.18121) (CoRL 2024, arXiv).
  - ReKep uses relational keypoint constraints — [2409.01652](https://arxiv.org/abs/2409.01652) (venue unverified).
- VLBiMan derives reusable skills from one human example, "preserving invariant primitives as anchors while dynamically adapting adjustable components via vision-language grounding", and composes long-horizon tasks by "atomic skill splicing" — [2509.21723](https://arxiv.org/abs/2509.21723) (ICLR 2026, arXiv).

**Segment transformation for data generation (generate, then filter by success)**
- MimicGen parses demos into object-centric segments and maps each segment with T_new = T_obj,new · (T_obj,src)^-1 · T_ee,src, which preserves "the relative poses between the target pose frame and the object frame". It linearly interpolates from the current pose to the transformed segment's start. Source segments are chosen "at random or by using the relevant object poses", and the choice matters "for certain tasks" (full text) — [MimicGen 2310.17596](https://arxiv.org/abs/2310.17596) (CoRL 2023, arXiv). [Page](https://mimicgen.github.io).
  - Data-generation success varies widely: Stack 82.3%, Coffee 52.1%, Gear Assembly 8.2%, Three-Piece Assembly D0 35.6%.
  - The authors say generation success rate and trained-agent performance are not necessarily correlated.
  - Policy success with 10 source demos vs 1,000 generated demos:

    | Task | 10 source demos | 1,000 generated demos |
    |---|---|---|
    | Stack | 26.0% | 100% |
    | Threading | 19.3% | 98.0% |
    | Coffee | 74.0% | 100% |
    | Three-Piece Assembly | 1.3% | 82.0% |

  - The abstract reports 50K demos generated from ~200.
- SkillMimicGen (SkillGen) segments demos into skills, adapts each skill to the new context and stitches them with free-space motion planning. It adds a Hybrid Skill Policy that learns skill initiation, control and termination. It generated 24K demos from 60 across 18 task variants, with agents "on average 24% more successful" — [2410.18907](https://arxiv.org/abs/2410.18907) (CoRL 2024, arXiv jref). DexMimicGen extends this to bimanual dexterous manipulation — [2410.24185](https://arxiv.org/abs/2410.24185) (ICRA 2025, arXiv).
- DemoGen works from **one demo per task**. Skill segments are transformed with their object via A_ee · (T_obj,0)^-1 · T_obj,0', and motion segments are re-planned with motion planning between transformed skill segments (full text) — [DemoGen 2502.16932](https://arxiv.org/abs/2502.16932) (venue unverified). [Page](https://demo-generation.github.io).
  - Real world, 8 tasks: 74.6% vs 11.0% for the source demo alone.
  - Simulation: 91/88 (max/mean) vs 16/13 from 1 demo. 25 human demos reach 94/91.
  - 100-432 demos were generated per task.

**Retargeted trajectory as a prior for a pretrained policy (closest to our PoE guide)**
- DemoDiffusion converts human hand motion into "a rough open-loop robot motion trajectory via kinematic retargeting". A pretrained generalist diffusion policy then modifies it so that it "follows the human motion and remains within the distribution of plausible robot actions". On 8 real tasks: **83.8%** vs 13.8% for the pretrained policy and 52.5% for kinematic retargeting alone — [DemoDiffusion 2506.20668](https://arxiv.org/abs/2506.20668) (ICRA 2026, arXiv). [Page](https://demodiffusion.github.io/).

**Segmentation and retrieval of sub-trajectories**
- AWE decomposes a demo into the minimal set of waypoints whose linear interpolation approximates the trajectory within a threshold. It raises success "by up to 25% in simulation and by 4-28% on real-world bimanual" tasks and reduces the decision horizon "by up to a factor of 10" — [AWE 2307.14326](https://arxiv.org/abs/2307.14326) (venue unverified). [Page](https://lucys0.github.io/awe/).
- STRAP segments trajectories where "‖ẋ‖<ε" (minimum chunk length 20) and retrieves sub-trajectories with subsequence DTW over DINOv2 features. With 5 LIBERO-10 target demos and the 4,500 LIBERO-90 trajectories as the prior, it reached **45.6%** vs 37.9% for BC and 41.4% for full-trajectory retrieval (full text) — [STRAP 2412.15182](https://arxiv.org/abs/2412.15182) (venue unverified). [Page](https://weirdlabuw.github.io/strap/). Related work: Behavior Retrieval [2304.08742](https://arxiv.org/abs/2304.08742); COLLAGE, which fuses multiple retrieval cues weighted by how well each subset predicts target actions, [2508.01131](https://arxiv.org/abs/2508.01131) (CoRL 2025, arXiv); VINN [2112.01511](https://arxiv.org/abs/2112.01511).
- Generative trajectory stitching through diffusion composition — [2503.05153](https://arxiv.org/abs/2503.05153).

**Few-shot or one-shot adaptation of VLAs**
- ControlVLA adds object-centric conditions to a pretrained VLA through ControlNet-style zero-initialized projections. It reached 76.7% on 6 real tasks with 10-20 demos — [2506.16211](https://arxiv.org/abs/2506.16211) (venue unverified).
- DART adapts VLAs to camera or embodiment shift from a single demonstration via weight-vector arithmetic with subspace alignment — [2607.00666](https://arxiv.org/abs/2607.00666) (ECCV 2026, arXiv), [code](https://github.com/snumprlab/dart).

### Inferences
- The **~35% persistent failures on new init states** plausibly come from the design itself: the composition follows the *demo path*, so whenever the object pose differs, the contact phases are aimed at the wrong place. Every one-demo method above that generalizes spatially (Coarse-to-Fine, DITTO, DemoGen, MimicGen) makes the contact segment **object-relative**. The concrete change for us:
  - In each contact window (already defined as -30/+10 frames around gripper events), express the borrowed chunk trajectory in the frame of the object being manipulated.
  - Apply MimicGen's transform T_new = T_obj,new · (T_obj,demo)^-1 · T_ee,demo to obtain the guide target for this instance.
  - Let the free-space phases be policy-driven, with a low guide weight or a DemoGen-style interpolation to the start of the transformed segment.

  DemoDiffusion's result (a retargeted prior plus a pretrained policy at 83.8%, beating both parts) directly supports keeping the pi0.5 flow sampler in the loop, with the guide re-targeted.
- LIBERO actions are end-effector *delta* commands (my understanding of the robosuite OSC controller; not re-verified here). Transforming a segment therefore means integrating the deltas to absolute poses, transforming them and differencing back. Object poses are readable from LIBERO sim state. That is **privileged** information, and a paper would have to disclose it or replace it with a pose or keypoint estimator (DINOBot, IMOP and DITTO use vision). This caveat is mine.
- MimicGen's finding that **source segment selection matters for some tasks** maps onto our leaf choice. Choosing the leaf whose *source object pose* (the object pose at the leaf's training frame) is nearest to the current instance is a zero-rollout heuristic worth testing against max coverage. This is my extrapolation.
- AWE waypoints or STRAP-style low-velocity split points would give **data-driven phase boundaries** for the segment-wise program in place of fixed 10-step segments. Fewer, semantically aligned segments shrink the search space in Key Question 5.
- MimicGen's "generate then keep successes" is exactly what our practice-then-imitate-successes loop does. Its note that generation success and final policy performance are not necessarily correlated is consistent with our plateau at 0.66: filtered imitation of the current program's successes cannot learn behaviors the program never produces on hard instances.

### Gaps
- None of the segment-transform methods (MimicGen, SkillGen, DemoGen) has been combined with a pretrained VLA's flow sampler as a *guide* on LIBERO, as far as I found. DemoDiffusion (real world, human video) is the nearest.
- I did not find LIBERO numbers for the one-demo alignment methods (DINOBot, Coarse-to-Fine, DITTO, KAT).
- DemoGen's and AWE's venues are not stated on arXiv (unverified; the DemoGen fetch explicitly said v1 had no venue).

---

## Key Question 5: What are concrete ways to optimize OUR composition choices (which leaves, per-dimension choices, guide weights, fold windows, recomposition of the failing phase) with sparse binary reward and tens of rollouts per task?

### Takeaway
The recommendation is sequential, and mostly **search and selection with frozen weights**. It rests on four pieces of evidence:
- Reward-driven methods succeed at our budget only with few decision variables (PoWER; DSRL; binomial arithmetic).
- Failure localization plus local repair beats global search in program space (REFLECT; Carvalho et al.).
- Frozen-policy selection has large headroom (SVA; V-VLAPS; VGAS).
- One-demo spatial generalization comes from object-relative contact segments (MimicGen; DemoGen; DemoDiffusion).

The steps are:
1. Measure pass@k headroom on failing instances.
2. Collapse per-frame x per-dim choices into a handful of per-phase variables.
3. Warm-start arm posteriors and a chunk critic from the logged rollouts.
4. Run top-two Thompson sampling only over the stalled phase with the prefix frozen.
5. Tune shared continuous knobs with CMA-ES or BO pooled across tasks.
6. Add DSRL-style noise steering as a per-task adapter.
7. Make contact-window guide targets object-relative.
8. Grow the library from verified programs (retrieval plus BPE/Stitch compression) in place of LoRA-distilling successes.

### Cited Findings
(Evidence behind each step; details and sources are in Key Questions 1-4.)
- Few-parameter, demo-initialized reward search converges in tens of trials: PoWER, ~75 iterations — [NeurIPS 2008](https://proceedings.neurips.cc/paper_files/paper/2008/file/7647966b7343c29048673252e490f736-Paper.pdf).
- Noise-space RL on a frozen pi0 improves single tasks in 40-80 episodes — [DSRL](https://arxiv.org/html/2506.15799).
- Offline critic from logged rollouts: +6.1 pts from 150 rollouts per suite — [Q-VGM](https://arxiv.org/abs/2606.08015). +9.2 pts from 5-shot data with best-of-N — [VGAS](https://arxiv.org/html/2602.07399). Value re-ranking without weight access — [V-GPS](https://arxiv.org/abs/2410.13816).
- Frozen policies hide success in their sample distribution: pass@1 33% → pass@32 92% — [SVA](https://arxiv.org/abs/2607.03751).
- Local search in native program space beats latent-space CEM — [Carvalho et al.](https://arxiv.org/abs/2410.12166).
- Best-arm identification with top-two Thompson sampling is rate-optimal — [Russo](https://arxiv.org/abs/1602.08448). OPE warm start plus BO for policy selection under a small interaction budget — [AOPS](https://arxiv.org/abs/2106.10251).
- Composition weights can be tuned by rollout-scored grid search, but it cost 200 rollouts per configuration for +2 to +7.5 pts — [GPC](https://arxiv.org/html/2510.01068).
- Object-relative segment transforms give one-demo spatial generalization — [MimicGen](https://arxiv.org/html/2310.17596), [DemoGen](https://arxiv.org/html/2502.16932). A retargeted prior refined by a pretrained policy beats both — [DemoDiffusion](https://arxiv.org/abs/2506.20668).
- BPE-compressed action primitives help 5-shot LIBERO-LONG — [PRISE](https://arxiv.org/html/2402.10450). Verified-program libraries and macro-skill compression — [Voyager](https://arxiv.org/abs/2305.16291), [SkillComposer](https://arxiv.org/abs/2608.14944), [Stitch](https://arxiv.org/abs/2211.16605).
- Noise injected inside the denoising chain gets diluted, so exploration should act on actions or initial noise — [StructRL](https://arxiv.org/abs/2608.15139).
- Full-weight RL on flow VLAs reaches 94-100% on LIBERO suites but with GPU-cluster budgets and possibly seen init states — [SimpleVLA-RL](https://arxiv.org/html/2509.09674), [piRL](https://arxiv.org/html/2510.25889v2).

### Inferences (concrete plan, ordered by cost)

**Step 0: headroom diagnostic (about 10-20 rollouts per failing instance, no new code).**
- For a sample of persistently failing (task, init) pairs, rerun with **fresh seeds** and the current program fixed. Record pass@k and the stall-point distribution.
- Optionally use LIBERO sim resets as an *oracle*: at the stall phase, try each of the 8 leaf ranks x 3 w values from the same restored state. This labels which phase and which choice would have rescued the instance, in the spirit of V-VLAPS and SVA. It is privileged and diagnostic only.
- Decision rule:
  - pass@k clearly above 0 → selection or search over existing choices can help (Steps 2-5).
  - pass@k ≈ 0 and the oracle also fails → the program's *geometry* is wrong, and Step 6 is required.

**Step 1: reparameterize the program into few decision variables.**
- Define phases by gripper events (already used for fold windows), AWE waypoints or recorded stall points.
- Per phase, keep:
  - one leaf **rank** r ∈ {1..8} tied across dimensions, or at most tied within 3 groups (translation, rotation, gripper);
  - one guide weight multiplier from {0.5, 1, 2};
  - the fold on/off flag and window offsets.
- With 3-5 phases this is tens of variables, not thousands. By Carvalho et al., neighborhoods should be native edits: change one phase's leaf rank, change one w, shift one window edge.

**Step 2: zero-rollout warm start from logs.**
- Every logged rollout is (instance features, program, stall index, success).
- Fit per-(phase, choice) Beta posteriors, or a logistic model with init-state features, as the OPE-style prior (as AOPS does).
- Train a small chunk-value head on pi0.5 features (or DINOv2 plus proprioception) from the same logs, V-GPS/VGAS/SVA-style. At the stall phase, sample N candidate chunks (different leaf ranks or noise seeds) and execute the argmax. This changes no weights and applies to unseen init states immediately.

**Step 3: prefix-frozen repair with top-two Thompson sampling.**
- On failing instances, freeze every phase before the stall and run top-two TS over the stalled phase's arms (for example 8 leaf ranks x 3 w values = 24 arms, or fewer after pruning by the Step 2 prior).
- Pool across a task's init states when the stall phase is the same; otherwise treat each init cluster separately.
- Stop when the posterior best arm has P(best) > 0.8 or the budget ends.
- Arithmetic: a large rescue (0.1 → 0.4) can be separated with ~32 rollouts per arm, so 24 arms needs pruning to about 3-6 plausible arms first.

**Step 4: shared continuous knobs pooled across tasks.**
- Tune w, teacher variance, fold-window start and end, and the stall-fallback threshold with CMA-ES (default population about 4 + 3 ln n ≈ 9 for n=6, from the Hansen tutorial) or with GP-BO using task as context.
- Each evaluation samples (task, init) pairs at random.
- Effects of 5-10 pts need hundreds of rollouts per configuration (GPC used 200), which is only affordable pooled across all held-out tasks, never per task.

**Step 5: DSRL-style noise steering as a per-task adapter.**
- A small actor outputs the initial flow noise per chunk, trained with SAC (DSRL-NA-style critic distillation) on binary reward, with pi0.5, the guide and the fold frozen.
- DSRL's 40-80 episodes per task is the closest published budget to ours.
- Compare it head-to-head with the LoRA-on-successes route (0.652) on unseen inits. Unlike filtered imitation, it learns from failures.

**Step 6: object-relative contact segments (targets the "follows the demo path" failures).**
- In contact windows, transform the borrowed trajectory into the manipulated object's frame and retarget it to the current object pose (MimicGen/DemoGen transform). Feed that as the PoE guide target, the way DemoDiffusion feeds a retargeted prior.
- In free-space phases, drop w or interpolate to the transformed segment's start.
- Test leaf selection by nearest source-object pose against max coverage.
- Caveat: object poses from sim state are privileged. Disclose this or estimate them (DINOBot/IMOP-style alignment).

**Step 7: library growth and consolidation.**
- Store each verified success as (init embedding, per-phase leaf ranks, w, window) — the Voyager/LRLL pattern.
- At a new init, retrieve the nearest successful program as the starting point (VINN/DINOBot-style retrieval) in place of imitating all successes into a LoRA.
- Periodically run BPE (PRISE) or Stitch over successful leaf-ID sequences across tasks to create macro-leaves and prune never-chosen leaves.
- New leaves = successful practice chunks at the phases where they rescued a failure.

**What not to do at our budget.**
- Full-weight PPO/GRPO on pi0.5 (SimpleVLA-RL, piRL, FPO): thousands of rollouts, multi-GPU, and LIBERO numbers possibly on seen inits.
- Per-task grid search over w (GPC-scale budgets).
- LLM-in-the-loop code repair: our program is numeric.
- Deploying simulator MCTS: privileged resets.

**Expected gains, anchored to the literature and not guaranteed.**
- Value-guided best-of-N: +6 to +12 pts (Q-VGM, VGAS, SVA).
- DSRL-style adapters: large single-task gains within tens of episodes.
- Object-relative segment transforms: the largest documented one-demo effect (DemoGen 11.0 → 74.6 real; MimicGen 1.3-26 → 82-100 with generated data). These are not same-benchmark comparisons.

### Gaps
- No published method optimizes a PoE-guided, borrowed-chunk program for a frozen VLA. Every step above is transplanted from an adjacent setting, and the effect sizes are unknown for our pipeline.
- Whether the ~35% persistent failures have pass@k > 0 under fresh seeds is unknown until Step 0 runs. The whole plan branches on it.
- I found no data on how the PoE guide changes the noise-to-action map that DSRL relies on (Step 5), or on whether guide-induced stalls interact with value re-ranking.
- LIBERO's action convention (delta end-effector commands) and the openpi `pi05_libero` action normalization were not re-verified. They determine how Step 6's transforms must be implemented.
