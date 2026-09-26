#!/bin/bash
# A wave of held-out LIBERO-90 tasks through the whole protocol, from ONE demo each:
#   prep (experiments/prep.sbatch: raw program, dimension-wise codes, the 258 KB fold if the demo grasps, LoRA rank 16)
#   practice: zero-shot on the development states 0-9 (m3_compose)         -- decides whether the prior is kept
#   final (m3_final, states 30-49, 20 episodes each): zeroshot, raw_replay, dimcode_s5, dimcode_s1, dimfold_s1_last2, lora
#   TASKS=1+6+34 bash experiments/launch_wave.sh
#   TASKS=35+36 SKIP_PREP=1 PRACTICE=0 ARMS="leaf_s5 leaffold_s1_last2 shared_s1" bash experiments/launch_wave.sh   (extra arms)
#   leaf_* = borrowing from the policy's own chunk means (the marginal's components, r9 "leaf"); shared_* = the t = 0.4 level alone
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
TASKS=${TASKS:?}; NICE=${NICE:-0}; ARMS=${ARMS:-"zeroshot raw_replay dimcode_s5 dimcode_s1 dimfold_s1_last2 lora"}
R=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot
PY=~/flash/miniconda3/envs/chunks/bin/python
C="w0=1.0&sched=const&var=0.001&key=pose&stall=3"
for tid in ${TASKS//+/ }; do
  FOLD=$($PY -c "
import sys, importlib.util; sys.path.insert(0, '.')
from lifelong_ood.policy import data as D
s = importlib.util.spec_from_file_location('r3p', 'experiments/r3_prepare.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
texts, _ = m.libero90_texts(); d = D.Demos90(); ep = d.load(d.episodes_of_text(texts[$tid])[0])
print(1 if len(D.gripper_events(ep['actions'])) else 0)" 2>/dev/null)
  if [ "${SKIP_PREP:-0}" = 1 ]; then DEP=""; else
    P=$(sbatch --parsable --nice=$NICE --export=ALL,TID=$tid,FOLD=$FOLD experiments/prep.sbatch); DEP="--dependency=afterok:$P"
    echo "task $tid prep -> job $P (fold $FOLD)"
  fi
  if [ "${PRACTICE:-1}" = 1 ]; then
    j=$(sbatch --parsable --nice=$NICE --export=ALL,MODE=ours,SUITE=libero_90,TASKS=$tid,TRIALS=10,FIRST=0,EXPERIMENT=m3_compose,TAG=zeroshot,PROGRAM_BYTES=0,OUT=$R/m3_compose_${tid}_zeroshot.json experiments/roll.sbatch)
    echo "task $tid practice -> job $j"
  fi
  for arm in $ARMS; do
    D=$DEP; PB=0
    case $arm in
      zeroshot)         MODE=ours; D="" ;;
      raw_replay)       P0=$R/r3_prog_${tid}_raw_lib40.npz; MODE="program:$P0?$C"; PB="prog:$P0" ;;
      dimcode_s5)       P0=$R/r7_dim_${tid}_s5.npz; MODE="program:$P0?$C&window=4"; PB="prog:$P0" ;;
      dimcode_s1)       P0=$R/r7_dim_${tid}_s1.npz; MODE="program:$P0?$C&window=20"; PB="prog:$P0" ;;
      dimfold_s1_last2) [ "$FOLD" = 1 ] || continue
                        P0=$R/r7_dim_${tid}_s1.npz; F0=$R/r4_${tid}_approach_last2_program.npz
                        MODE="hybrid:$P0?expert=$R/r4_${tid}_approach_last2_expert.pt&foldprog=$F0&$C&window=20"; PB="prog:$P0@$F0" ;;
      leaf_s5)          P0=$R/r9_leaf_${tid}_s5.npz; MODE="program:$P0?$C&window=4"; PB="prog:$P0" ;;
      leaf_s1)          P0=$R/r9_leaf_${tid}_s1.npz; MODE="program:$P0?$C&window=20"; PB="prog:$P0" ;;
      leaffold_s1_last2) [ "$FOLD" = 1 ] || continue
                        P0=$R/r9_leaf_${tid}_s1.npz; F0=$R/r4_${tid}_approach_last2_program.npz
                        MODE="hybrid:$P0?expert=$R/r4_${tid}_approach_last2_expert.pt&foldprog=$F0&$C&window=20"; PB="prog:$P0@$F0" ;;
      shared_s1)        P0=$R/r9_shared_${tid}_s1.npz; MODE="program:$P0?$C&window=20"; PB="prog:$P0" ;;
      seg_s5)           P0=$R/r10_seg_${tid}_s5.npz; MODE="program:$P0?$C&window=4"; PB="prog:$P0" ;;
      lora)             MODE="lora:$R/r3_lora_${tid}.pt" ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    j=$(sbatch --parsable --nice=$NICE $D --export=ALL,MODE="$MODE",SUITE=libero_90,TASKS=$tid,TRIALS=20,FIRST=30,EXPERIMENT=m3_final,TAG=$arm,PROGRAM_BYTES="$PB",OUT=$R/m3_final_${tid}_${arm}.json experiments/roll.sbatch)
    echo "task $tid $arm -> job $j"
  done
done
