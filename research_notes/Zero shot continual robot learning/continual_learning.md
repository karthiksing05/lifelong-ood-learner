# Continual / lifelong learning for robot manipulation policies (pretrained VLAs and diffusion policies, 2022-2026)

Scope note: the setting compared against throughout is a frozen pi0.5 plus one stored unit per task (a 9-byte composition code, an 18 KB key/value prefix token, or an 18.7 MB LoRA). Frozen shared weights make backward forgetting zero by construction, so the questions that matter are forward transfer, per-task bytes, and selecting the right unit at test time. Metric conventions differ between papers. LIBERO's NBT is positive when the model forgets, and lower is better. TAIL reports BWT, where negative means forgetting. Numbers from different papers use different architectures, pretraining and protocols, so they cannot be compared directly across papers.

## Q0. Which classic CL baselines were run on LIBERO, and what did they show?

### Takeaway
The original LIBERO benchmark (NeurIPS 2023) tested from-scratch policies of about 15-26M parameters. Sequential fine-tuning (SeqL) had the best forward transfer of any method but forgot badly (NBT 0.6-0.8). ER and PackNet cut forgetting, and both lowered FWT. EWC did poorly. Supervised pretraining on LIBERO-90 hurt downstream lifelong learning. In 2026, large pretrained VLAs reversed that last result (see Q1b).

### Cited Findings
- Metrics: FWT is the average success while learning each new task (averaged over the evaluation checkpoints during that task's training). NBT is the drop in success on earlier tasks after later ones are learned; positive means forgetting. AUC combines the two. Algorithms tested were SeqL, ER, EWC, PackNet, and multitask learning (MTL) as an upper bound. Architectures were ResNet-RNN, ResNet-T and ViT-T. — [LIBERO, Liu et al., NeurIPS 2023 D&B, arXiv 2306.03310](https://arxiv.org/html/2306.03310)
- ResNet-T results (FWT / NBT / AUC):
  - LIBERO-Long: SeqL 0.54/0.63/0.15; ER 0.48/0.32/0.32; PackNet 0.22/0.08/0.25.
  - LIBERO-Spatial: SeqL 0.72/0.81/0.20; ER 0.65/0.27/0.56; PackNet 0.55/0.07/0.63.
  - LIBERO-Object: SeqL 0.78/0.76/0.26; ER 0.67/0.43/0.44; PackNet 0.60/0.17/0.60.
  - LIBERO-Goal: SeqL 0.77/0.82/0.22; ER 0.64/0.34/0.49; PackNet 0.63/0.06/0.75.
  - [LIBERO](https://arxiv.org/html/2306.03310)
- "SeqL shows the best FWT over all task suites... all lifelong learning algorithms we consider actually hurt forward transfer." — [LIBERO](https://arxiv.org/html/2306.03310)
- "Basic supervised pretraining can have a negative impact on the learner's downstream performance in LLDM": agents trained from scratch did better than agents pretrained on LIBERO-90. — [LIBERO](https://arxiv.org/html/2306.03310)
- No single visual encoder was best at every kind of transfer: ViT did better on object variety, ResNet-T on spatial tasks. PackNet had the lowest NBT but gave up FWT, especially on LIBERO-Long. — [LIBERO](https://arxiv.org/html/2306.03310)
- A later re-run with a different setup (FWT/NBT/AUC, %):
  - EWC: Object 56/69/16; Goal 32/48/6; Spatial 23/33/6.
  - ER: Object 56/24/49; Goal 53/36/47; Spatial 65/27/56.
  - Sequential: Object 62/63/30; Goal 55/70/23; Spatial 72/81/20.
  - The ER baseline used a 1000-trajectory buffer.
  - [M2Distill, Roy et al., ICRA 2025, arXiv 2410.00064](https://arxiv.org/html/2410.00064)
- With pretrained VLAs (2026), EWC barely reduces forgetting. On LIBERO-Object, pi0 NBT was 0.696 with Sequential, 0.608 with EWC and -0.007 with ER. GR00T NBT was 0.752 Sequential, 0.766 EWC and 0.004 ER. — [Liu, Kim, Liu, Liu, Zhu, arXiv 2603.03818](https://arxiv.org/html/2603.03818)

### Inferences
- In LIBERO the two goals pull against each other: methods that cut forgetting (ER, PackNet, EWC) lower FWT. This is the baseline any "frozen backbone plus per-task unit" claim gets measured against.
- EWC failing on both 15M from-scratch policies and 3B VLAs suggests regularization-based CL is not a serious competitor in this area.

### Gaps
- I did not pull the LIBERO paper's EWC rows or its ResNet-RNN and ViT-T rows. The fetched summary gave only the ResNet-T SeqL/ER/PackNet rows.

## Q1. Which approaches show positive forward transfer and near-zero forgetting on LIBERO?

### Takeaway
Methods that freeze or isolate parameters (TAIL-LoRA, CORAL, CLARE, DMPEL, IsCiL, SDP, PPL) reach NBT of about 0 to 2%. Of these, only TAIL-LoRA and PPL report FWT clearly above full fine-tuning. CLARE matches full sequential fine-tuning's FWT (75.0 vs 76.1 on LIBERO-Long) with NBT 1.9. Skill-library methods (LOTUS, IsCiL) sometimes show negative NBT, meaning old tasks improve, but at lower FWT. For large pretrained VLAs, plain ER with a small buffer gives near-zero forgetting and keeps forward learning.

### Cited Findings
**Parameter-efficient, task-isolated adapters**
- **TAIL (ICLR 2024).**
  - Setup: base model of CLIP-base spatial and instruction encoders plus a GPT-2 temporal decoder, 172.24M parameters in total. Pretrained on 40 LIBERO-90 Kitchen tasks, then adapted in sequence to Spatial, Goal, Object, Living Room and Study Room (8 tasks each), with LIBERO-10 evaluated separately.
  - LIBERO-10 (FWT / BWT): FFT 0.48 / -0.55; ER 0.45 / -0.49; EWC 0.30 / -0.43; TAIL-LoRA 0.70 / 0.00; TAIL-Prefix 0.51 / 0.00; TAIL-Bottleneck 0.46 / 0.00; TAIL-RoboAdapter 0.42 / 0.00.
  - Ranking by integration style: parallel (LoRA) > prefix token > sequential (bottleneck adapters).
  - FFT "contaminates the rich CLIP features." The authors read LoRA's success as "a well-pretrained model has a surprisingly low intrinsic dimension for imitation learning tasks."
  - [TAIL, Liu et al., arXiv 2310.05905](https://arxiv.org/html/2310.05905)
- **CLARE (arXiv 2601.09512, 2026).**
  - Setup: DiT-decoder policy with frozen DINOv2 and CLIP encoders, pretrained on LIBERO-90. Adapters go into the observation-conditioning and AdaLN modules. A new adapter is added only where layer-wise feature similarity says one is needed. An autoencoder-discriminator router picks adapters without task IDs.
  - LIBERO-Long (AUC / FWT / NBT):

    | Method | AUC | FWT | NBT |
    |---|---|---|---|
    | CLARE | 75.1 | 75.0 | 1.9 |
    | ER | 60.5 | 76.6 | 22.7 |
    | SeqFFT | 22.4 | 76.1 | 74.7 |
    | SeqLoRA | 21.4 | 73.1 | 71.6 |
    | PackNet | 4.8 | 37.2 | 41.3 |
    | LOTUS | 52.9 | 58.1 | -7.2 |
    | DMPEL | 58 | 55 | 7 |

  - Real FR3 robot, 5 tasks: AUC 63.3%, NBT -2.9%, against ER 51.1% and SeqFFT 23.8%.
  - Scales to 40 sequential LIBERO tasks, where ER's NBT is 23%.
  - [CLARE, Römer, Zhang, Li, Schoellig](https://arxiv.org/html/2601.09512)
- **DMPEL (arXiv 2506.05985, 2025).** Builds a progressive library of LoRA experts on a policy pretrained on LIBERO-90 with frozen CLIP ViT-B/16. A new expert is added per task and frozen afterwards, with Gram-Schmidt orthogonalization. An MLP router mixes the top-3 experts. "Coefficient replay" stores router inputs and outputs in place of demonstrations. The authors report better success rates than prior lifelong methods on all four LIBERO suites. — [DMPEL, Lei et al.](https://arxiv.org/html/2506.05985v1), [code](https://github.com/HarryLui98/DMPEL). CLARE's re-run put DMPEL at LIBERO-Long AUC 58, FWT 55, NBT 7. — [CLARE](https://arxiv.org/html/2601.09512)
- **CORAL (arXiv 2603.09298, 2026).**
  - One frozen VLA backbone with one LoRA expert per task. The language instruction selects the expert directly, with no learned gate. Experts are fully isolated and need no replay. Switching experts takes under 100 ms.
  - Results: 99.3% average success on all 40 LIBERO tasks, 97.9% on WidowX, 84.9% on Google Robot. On a real Galaxea R1 it matches independent full fine-tuning.
  - [CORAL](https://arxiv.org/html/2603.09298v1)

**Skill-library and retrieval methods**
- **LOTUS (ICRA 2024).** Discovers skills without supervision using open-vocabulary vision models, keeps a growing skill library, and composes skills with a meta-controller. Results (FWT / NBT / AUC):
  - LIBERO-Object: 74 / 11 / 65, against ER 56 / 24 / 49.
  - LIBERO-Goal: 61 / 30 / 56.
  - LIBERO-50: 39 / 43 / 45, against ER 35 / 49 / 36.
  - About 11% higher success than prior methods.
  - [LOTUS, Wan, Zhu, Shah, Zhu](https://arxiv.org/html/2311.02058)
- **IsCiL (NeurIPS 2024).** Uses prototype-based skill retrieval (20 K-means bases per skill) over a state embedding, with a rank-4 LoRA adapter per skill on a pretrained diffusion skill decoder.
  - Evolving Kitchen-Complete, 20 stages: FWT 79.3, BWT 11.0 (positive backward transfer), AUC 89.8. TAIL-τ scored 86.2 / 0.0 / 86.2; the multitask oracle reached AUC 92.3.
  - When demonstrations are incomplete (missing sub-goals), the missing skills are retrieved from other tasks.
  - Not evaluated on LIBERO.
  - [IsCiL, arXiv 2410.22658](https://arxiv.org/html/2410.22658), [NeurIPS poster](https://neurips.cc/virtual/2024/poster/95159)
- **M2Distill (ICRA 2025).** Multi-modal distillation that keeps the vision, language and action latents and the GMM policy consistent between steps. It also uses replay. Results (FWT / NBT / AUC):
  - Object: 75 / 8 / 69.
  - Goal: 71 / 20 / 57.
  - Spatial: 74 / 11 / 61.
  - [M2Distill](https://arxiv.org/html/2410.00064)
- **PPL, "Think Small, Act Big" (CVPR 2025).** A diffusion-transformer policy with motion-aware primitive prompts. New "lifelong prompts" are appended next to the frozen pretrained prompts.
  - MimicGen: FWT 0.83 and BWT +0.03, against TAIL-LoRA FWT 0.78, ER 0.61 / -0.46 and Sequential 0.65 / -0.56.
  - Real Franka: 0.68 vs ER 0.58.
  - [PPL, arXiv 2504.00420](https://arxiv.org/html/2504.00420v1), [CVPR](https://mlanthology.org/cvpr/2025/yao2025cvpr-think/)
- **SCE, Skill-Compositional Experts (arXiv 2606.15685, 2026).** Builds on pi0. It segments demonstrations into skills, grounds them with a VLM, and adds one execution expert (a bottleneck adapter) per new skill plus a transition expert. The protocol has 10 stages with 5 replay demonstrations per old task.
  - LIBERO-Goal: AUC 82.9 and final success 83.9, against Task-MoE 72.3 / 66.0.
  - Real robot: final success 80.0, NBT 0.6.
  - [SCE](https://arxiv.org/html/2606.15685)
- **Stellar VLA (arXiv 2511.18085).** Knowledge-guided expert routing over a self-evolving skill knowledge space. It adds no network parameters and is "competitive" with only 1% data replay. — [Stellar VLA](https://arxiv.org/abs/2511.18085)

**Expert growth in diffusion policies**
- **Sparse Diffusion Policy (CoRL 2024).** A transformer diffusion policy with MoE layers. Each new task freezes the old experts and routers, then adds 8 new experts per layer and a task router. Active parameters stay at 9.2M, while a LoRA baseline grows 9.0M → 12.0M → 14.9M.
  - Robomimic sequence Can → Lift → Square: FFT drops to 0.00 on old tasks. SDP keeps Can at 0.94 and reaches Square 0.75; LoRA reaches Square 0.73.
  - This is a 3-task robomimic study, not LIBERO.
  - [SDP, arXiv 2407.01531](https://arxiv.org/html/2407.01531)

**Replay on large pretrained VLAs**
- With 20% replay (1000 samples per task), on Spatial / Object (success, NBT):
  - pi0: 0.879, 0.019 / 0.897, -0.011.
  - GR00T N1.5: 0.940, 0.007 / 0.975, 0.019.
  - BC-Transformer from scratch: 0.659, 0.299 / 0.595, 0.132.
  - [arXiv 2603.03818](https://arxiv.org/html/2603.03818)
- **Memory Anchors (arXiv 2608.26545, Aug 2026).** A small subset of replay samples, found where new-task observations collapse onto old-task ones but need different actions, does most of the work. Dropping 10% of these anchors raises forgetting 4.5x. Enriching the buffer with them cuts forgetting on high-conflict tasks by 63%. — [Memory Anchors, Du et al.](https://arxiv.org/abs/2608.26545)
- **Real-world, 10-task stream (arXiv 2605.26820).** Single-arm and bimanual tasks. Naive sequential fine-tuning forgets severely. A well-configured ER gives stable continual learning across all 10 tasks and beats joint multitask training at equal compute. — [Zhu et al.](https://arxiv.org/abs/2605.26820)

**RL-based continual post-training of VLAs**
- **LifeLong-RFT (arXiv 2602.10503).** Reinforcement fine-tuning with no online environment feedback and no reward model. It reports +22% average success over SFT on LIBERO continual learning while using 20% of the data. — [LifeLong-RFT](https://arxiv.org/abs/2602.10503)
- **CRL-VLA (arXiv 2602.03445).** A frozen critic plus a trainable estimator, with goal-conditioned value constraints. It claims to beat baselines on both anti-forgetting and forward adaptation on LIBERO; the abstract gives no numbers. — [CRL-VLA](https://arxiv.org/abs/2602.03445)

### Inferences
- "Near-zero forgetting" is already standard for parameter-isolated methods (TAIL, CORAL, CLARE, PPL, SDP), and frozen pi0.5 plus per-task units belongs to the same class. The differences among these methods are FWT, bytes per task, and routing without a task ID.
- Positive forward transfer in TAIL-style isolation comes from the frozen pretrained representation, not from earlier tasks: TAIL adapters are independent. Methods that do share across tasks (DMPEL's expert reuse, IsCiL's retrieval, LOTUS's skills, SCE's skill experts) show that sharing in AUC or NBT. None of them shows clearly lower per-task cost as tasks accumulate.
- CORAL's 99.3% on LIBERO-40 with language-indexed LoRAs shows that, on LIBERO, per-task LoRA on a strong VLA is close to the success ceiling. A smaller per-task unit (18 KB prefix or 9 B code) therefore has to match success while cutting bytes; there is little success headroom to win on LIBERO-40.

### Gaps
- No paper here reports "later tasks learned faster or with fewer demonstrations as the library grows" as a curve of demos or bytes per task against task index for a pretrained VLA.
- An earlier fetch returned a DMPEL table for LIBERO-Goal (FWT 63.8, NBT 0.3, AUC 57.9, with baselines). A second fetch could not find that table in the HTML; the results appear to be in a figure. Treat those numbers as unverified. The verified DMPEL numbers are CLARE's re-run on LIBERO-Long.
- CALVIN: I found no continual-learning protocol results on CALVIN. AtomicVLA reports better average task length on CALVIN, but that is not a continual-learning metric ([AtomicVLA, arXiv 2603.07648](https://arxiv.org/abs/2603.07648)).
- Exact numbers for CRL-VLA, LifeLong-RFT and Stellar were not available from the abstracts.

## Q1b. What is known about forgetting in large pretrained policies (VLAs)?

### Takeaway
Large pretrained VLAs still forget heavily under naive sequential fine-tuning: pi0 NBT was about 0.70 on LIBERO-Object. A small replay buffer stops it, and the knowledge that looks lost comes back with under 10% of the original training steps. The VL backbone is the main source of forgetting. A separate problem is that the VLM's language and VQA abilities degrade during VLA fine-tuning. Knowledge insulation, LoRA-only training, conservative SFT and weight interpolation each address that problem.

### Cited Findings
- **Liu, Kim, Liu, Liu, Zhu, "Pretrained VLAs are surprisingly resistant to forgetting" (arXiv 2603.03818, Mar 2026).**
  - Setup: pi0 (3B) and GR00T N1.5 (3B) against from-scratch BC-Transformer, BC-Diffusion and BC-ViT (15-26M), on the 10-task LIBERO-Spatial, Object, Goal and 10 suites in a fixed order.
  - At a 2% buffer (100 samples per task), the VLAs' NBT is about 0.1-0.2 against 0.4-0.5 for the from-scratch models, 2-4x less forgetting. The gap widens as the buffer shrinks.
  - Recovery: retraining an old task on the backbone obtained after the next task recovers peak success in 0.066 / 0.067 / 0.105 of the original steps for pi0 (Spatial / Object / 10), against 1.36 / 1.80 / 1.87 for BC-Transformer.
  - Component swaps show "the VL backbone is the dominant source of forgetting." Forgetting grows with task diversity: LIBERO-10 drops more than Object.
  - Pretraining variants (pi0 from VL+action / VL only / scratch): Spatial success 0.889 / 0.920 / 0.528. Pretraining keeps high forward transfer and low forgetting. Flow-matching vs L2 loss changes little.
  - [arXiv 2603.03818](https://arxiv.org/html/2603.03818)
- This reverses LIBERO's 2023 finding that supervised pretraining on LIBERO-90 hurt lifelong learning in small models. — [LIBERO](https://arxiv.org/html/2306.03310), [arXiv 2603.03818](https://arxiv.org/html/2603.03818)
- **Knowledge Insulation (pi0.5 + KI, Driess et al., May 2025).** Adding untrained continuous action experts (diffusion or flow) to a pretrained VLM "significantly harms both training speed and knowledge transfer." The fix insulates the VLM backbone with gradient isolation and co-training. — [arXiv 2505.23705](https://arxiv.org/abs/2505.23705)
- **VLM2VLA, "Actions as Language" (arXiv 2509.22195).** Writing actions as natural language allows LoRA-only VLA training, which preserves VQA and multilingual abilities. Evaluated with VQA studies and over 800 real-robot experiments. — [arXiv 2509.22195](https://arxiv.org/abs/2509.22195), [project](https://vlm2vla.github.io/)
- **ConSFT (arXiv 2605.08879, May 2026).** Unconstrained fine-tuning of flow-matching VLAs (pi0, pi0.5, GR00T-N1.6) "drives dense parameter overwrites." Confidence-modulated conservative SFT retains over 20 points more capability than vanilla SFT and matches ER without prior data. — [ConSFT](https://arxiv.org/abs/2605.08879)
- **RETAIN (arXiv 2512.08333).** Linearly interpolates the pretrained generalist weights with the fine-tuned specialist weights. On real DROID setups it gives about 40% higher out-of-distribution success than standard fine-tuning and keeps generalist abilities on non-target tasks. — [RETAIN](https://arxiv.org/html/2512.08333v1), [project](https://retain.yajatyadav.com/)
- **Within-task forgetting under full fine-tuning.** TAIL found FFT overfits and degrades the rich CLIP features. On revisits, FFT lost -0.26 on Spatial, -0.22 on Goal and -0.15 on Object. — [TAIL](https://arxiv.org/html/2310.05905)

### Inferences
- "Resistant to forgetting" means resistant when given replay. Without replay, pi0's NBT (0.70) is as bad as a from-scratch model's. A frozen pi0.5 avoids this entirely: no replay buffer, no stored demonstrations, and no NBT to measure.
- The less-than-10% recovery result suggests most forgetting in VLAs is misalignment of a still-present representation, not erased knowledge. That fits with small per-task units (prefix tokens, short codes) being able to steer a frozen backbone, but none of the cited papers tests this directly.
- Freezing pi0.5 also avoids the VLM-ability degradation that KI, VLM2VLA, ConSFT and RETAIN address.

### Gaps
- No work found measures forgetting specifically in pi0.5 under LIBERO lifelong sequential fine-tuning (pi0 and GR00T N1.5 were tested). ConSFT covers pi0.5, but its abstract does not give lifelong-protocol numbers.

## Q2. Typical bytes per task stored for adapter, prompt and replay approaches?

### Takeaway
Adapter methods store about 1-3% of model parameters per task, which comes to about 2-26 MB per task on 0.17-0.8B models. Replay methods store demonstrations: 100-1000 samples per task, or about 28 GB for TAIL's ER Kitchen buffer. Prompt-pool and coefficient-replay methods are smaller, but none of the papers gives a byte count per task. An 18 KB prefix is 2-3 orders of magnitude below published adapter costs, and a 9-byte code is about 6 orders below.

### Cited Findings
- **TAIL.** LoRA trains 2.02M parameters (1.17% of 172.24M), counting 0.84M of fusion and head parameters. Prefix tuning trains 1.66M (0.93%), bottleneck 2.55M (1.48%), RoboAdapter 2.53M (1.47%). Storage per LoRA adapter is reported as about 7.8 MB, against 28 GB for the Kitchen demonstration data used by ER. The same summary also says one CLIP encoder "can accommodate as many as 84 task-specific adapters," each about 2 MB, so the reported per-adapter size differs depending on what is counted. — [TAIL](https://arxiv.org/html/2310.05905)
- **CORAL.** A rank-16 LoRA expert takes about 26 MB on a 0.8B-parameter base, about 100x smaller than a full checkpoint. 40 experts take about 1 GB, against about 3 GB for one full model. — [CORAL](https://arxiv.org/html/2603.09298v1)
- **CLARE.** Adds about 2% parameters per task, with about 2% VRAM growth per task and under 3 ms added latency. In simulation the adapters total 3.2M parameters and the discriminators 1.4M. On the real robot the adapters total 0.38M and the discriminators 6.08M. Raising the expansion threshold γ from 0 to 20 cuts the number of adapters from 60 to 16 with little performance loss. — [CLARE](https://arxiv.org/html/2601.09512)
- **DMPEL.** 1.2M trainable parameters in total (under 0.7% of the policy). Coefficient replay at a 5% ratio costs about 6.7% of the storage of full demonstration replay. — [DMPEL](https://arxiv.org/html/2506.05985v1)
- **IsCiL.** A rank-4 LoRA per skill adds 0.37-1.48% parameters depending on rank. Skill prototypes are about 0.3% of model size. Retrieval takes 3.6 ms and adaptation 3.0 ms. Unlearning a task means deleting its adapter and prototype, at a cost of 1.8-5.2% degradation. — [IsCiL](https://arxiv.org/html/2410.22658)
- **SDP.** Adds 8 experts per MoE layer per task. Active parameters stay at 9.2M; per-task LoRA grows about 3M parameters per task (9.0 → 12.0 → 14.9M). — [SDP](https://arxiv.org/html/2407.01531)
- **PPL.** Each new skill adds M×D prompt components per layer, with no absolute size given. — [PPL](https://arxiv.org/html/2504.00420v1)
- **Stellar VLA.** No added parameters, 1% data replay. — [Stellar VLA](https://arxiv.org/abs/2511.18085)
- **Replay on VLAs.** Buffer sizes of 0.2%, 2% and 20% of each task's data, where 2% = 100 samples per task and 20% = 1000 samples per task. — [arXiv 2603.03818](https://arxiv.org/html/2603.03818). The ER baseline in M2Distill uses 1000 trajectories. — [M2Distill](https://arxiv.org/html/2410.00064)
- **Merging instead of storing.** PolicyWeave (GR00T N1.5) restricts each expert to hidden-layer LoRA with a shared action interface, then merges experts. On RoboCasa365 (18 tasks), LIBERO-10 and a real Piper arm, keeping the shared interface raised average merged success from 17.0% to 52.8% across four merging methods. PolicyWeave reached 64.7% with SFT experts and 74.1% after per-task RL, against 60.7% for joint RL. — [PolicyWeave, arXiv 2609.33125](https://arxiv.org/abs/2609.33125)

### Inferences
- Per-task cost scale, from published methods and the user's own units:

  | Per-task unit | Approximate size | Source |
  |---|---|---|
  | Published LoRA / adapter | ~2-26 MB | TAIL, CORAL |
  | User's pi0.5 LoRA | 18.7 MB | user's setting |
  | CLARE adapter (3.2M params) | ~12.8 MB in fp32 (my estimate) | CLARE |
  | User's KV prefix | 18 KB | user's setting |
  | User's composition code | 9 B | user's setting |
  | Replay of 100-1000 image-observation samples | likely tens to hundreds of MB per task (my estimate) | sample counts from 2603.03818 |

- The 18 KB prefix is about 1000x smaller than the LoRAs. The 9 B code is about 10^6x smaller. The replay estimate is my own: the source papers give sample counts, not bytes.
- No published robot CL method found stores a per-task unit in the KB or byte range. The closest are prompt pools (PPL, L2M) and DMPEL's coefficient replay, and none of them reports bytes per task.
- Merging (RETAIN, PolicyWeave) is the consolidation route: per-task storage could fall to zero after merging. Merged success drops well below per-expert success (PolicyWeave 64.7% vs CORAL-style isolated experts near 99% on LIBERO). These are different benchmarks, so the comparison is only rough.

### Gaps
- No paper found reports bytes per task for prompt-pool robot methods (PPL, L2M) or for KV and prefix-token units on VLAs.
- I found no work that distills many per-task VLA adapters into one model under a continual protocol on LIBERO and reports forgetting. RETAIN and PolicyWeave are single-step or one-shot merging, not sequential.
- The TAIL adapter size is inconsistent within the summary (about 7.8 MB vs about 2 MB). Read the source to settle it.

## Q3. Does any work address new tasks that recombine old factors (compositional continual learning)?

### Takeaway
Only partly. Pre-VLA compositional lifelong RL (Mendez and Eaton's modular composition on CompoSuite) shows zero-shot generalization to new combinations of known modules. In the VLA era, skill-library methods reuse skills and recombine them for long horizons: LEGION, LOTUS, IsCiL, SCE, PPL, AtomicVLA, SkillMemo. None of them runs a protocol in which held-out tasks are exactly new combinations of learned factors and the per-task stored cost is measured.

### Cited Findings
- **Mendez and Eaton.**
  - Modular Lifelong RL via Neural Composition (ICLR 2022): compositional representations "dramatically" outperform non-compositional ones in lifelong RL and enable "zero-shot generalization to new tasks that are combinations of known modules."
  - CompoSuite (CoLLAs 2022): robot × object × objective × obstacle combinations give hundreds of manipulation RL tasks.
  - [Modular lifelong RL, arXiv 2207.00429](https://arxiv.org/pdf/2207.00429), [CompoSuite, PMLR v199](https://proceedings.mlr.press/v199/mendez22a.html), [project](https://www.cis.upenn.edu/~eeaton/projects/2022CompositionalContinualLearning/)
- **LEGION (Nature Machine Intelligence 7:256-269, 2025).** Lifelong off-policy RL with a Dirichlet-process mixture knowledge space plus LLM language embeddings, evaluated on Meta-World MT10 and a KUKA robot. The agent "can tackle challenging real-world long-horizon tasks by combining and reapplying its acquired knowledge," with no forgetting metrics given in the README. — [Nature MI](https://www.nature.com/articles/s42256-025-00983-2), [GitHub](https://github.com/Ghiara/LEGION), [TUM](https://portal.fis.tum.de/en/publications/preserving-and-combining-knowledge-in-robotic-lifelong-reinforcem/)
- **IsCiL.** Explicitly tests knowledge sharing in the Semi-Complete scenario: when a task's demonstrations lack some sub-goals, its missing skills are retrieved from other tasks. — [IsCiL](https://arxiv.org/html/2410.22658)
- **L2M.** A modulation pool retrieved by key matching (L2P-style) over a frozen pretrained Decision Transformer. State of the art on Continual-World while keeping pretraining-task performance. IsCiL reports that L2M struggles to retrieve the correct skill. — [L2M, arXiv 2306.14884](https://arxiv.org/abs/2306.14884), [IsCiL](https://arxiv.org/html/2410.22658)
- **SCE.** Decomposes tasks into a skill base and adds experts only for new skills. The authors do not evaluate whether new tasks are strict recombinations of old skills. — [SCE](https://arxiv.org/html/2606.15685)
- **SIL-C (NeurIPS 2025 spotlight).** Maps the subtask space to the skill space so that skill improvements carry over to existing policies without retraining. — [SIL-C, arXiv 2509.20612](https://arxiv.org/abs/2509.20612)
- **Atomic-skill experts.**
  - AtomicVLA: skill-guided MoE in which new atomic skills get dedicated experts. +10% over pi0 on LIBERO-Long; +18.3% / +21% in real long-horizon / continual scenarios. — [AtomicVLA](https://arxiv.org/abs/2603.07648)
  - "Emergent Compositional Skills in MoE VLAs" (ICML 2026 workshop): experts are heavily reused across tasks and match distinct low-level behaviors, without supervision. — [arXiv 2607.20771](https://arxiv.org/abs/2607.20771)
  - SkillMemo: MoE skill segmentation plus a key/value episodic skill memory. It generalizes compositionally to unseen configurations and outperforms pi0.5, but it is not a continual protocol. — [SkillMemo, arXiv 2608.05970](https://arxiv.org/abs/2608.05970)
- **Reuse inside routed libraries.** DMPEL reports "significant knowledge sharing" across tasks, with action-head experts widely reused and later task-specific experts inactive. — [DMPEL](https://arxiv.org/html/2506.05985v1). CLARE's router switches adapters mid-trajectory when the language command changes. — [CLARE](https://arxiv.org/html/2601.09512)

### Inferences
- The gap the user's setting can fill: a protocol where each new LIBERO task is a recombination of factors already seen (objects, goals, layouts, verbs), the frozen VLA plus a byte-scale code expresses it, and the per-task cost shrinks toward the code size once the factors are known. No paper found reports this. The closest are CompoSuite-style zero-shot composition (RL, not VLA) and IsCiL's retrieval of missing sub-goals.
- LIBERO-Spatial, Object and Goal are themselves factorized variation sets (same objects in new layouts, new objects, new goals). That makes them a natural testbed for a "recombination costs fewer bytes than a new factor" claim, but the standard lifelong protocol does not label tasks by factor novelty.
- Prior skill-library methods (LOTUS, IsCiL, SCE) store per-skill adapters or controllers that are kilobytes to megabytes in size. A 9-byte composition code is a different kind of unit: an index into known factors, not a learned module. The natural comparison is FWT at equal bytes.

### Gaps
- I found no VLA-era benchmark that splits tasks into "new factor" and "recombination of old factors" and reports FWT and NBT separately for each.
- LEGION's quantitative forgetting and recombination results were not reachable: the Nature page redirected to a login.
- I did not run a dedicated search for L2P/DualPrompt prompt pools adapted to VLAs beyond PPL and L2M. One may exist that I missed.
