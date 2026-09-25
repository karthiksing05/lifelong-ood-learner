#!/bin/bash
# Wave 1 after the 40-task library (results/robot/r2b_libero_40_tree.npz) lands:
#   programs + novelty + growth on CPU, then M3 arms, few-shot LoRA, and M4 folds on the GPU queue.
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
PY=~/flash/miniconda3/envs/chunks/bin/python
T=results/robot/r2b_libero_40_tree.npz; L=results/robot/r2b_libero_40.npz
R=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot
M3=${M3:-36+67+19}; M4=${M4:-39+35+42}; TRIALS=${TRIALS:-10}
$PY experiments/r3_prepare.py --tree $T --library-npz $L --tasks ${M3//+/,},${M4//+/,} --levels 2,3,4 --tag _lib40 2>&1 | grep -v Warning
$PY experiments/r5_growth.py --tree $T --library $L --order 42+40+41+18+35+33+39+20 --shuffles 2 --tag _lib40 2>&1 | grep -v Warning
TASKS=$M3+$M4 TRIALS=$TRIALS ARMS="code_replay" PROGTAG=_lib40 bash experiments/launch_m3.sh   # zeroshot + raw_replay were launched from the _demo programs
# few-shot LoRA baselines were launched separately (35, 39, 42, 36)
TASKS=$M4 TREE=$T LIB=$L TRIALS=$TRIALS RANK=4 TAG=_lib40 bash experiments/launch_r4.sh
