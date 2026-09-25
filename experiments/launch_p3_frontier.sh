#!/bin/bash
# Launch the Phase-3 fold frontier: one GPU job per (arm, seed), each well under the 4 h limit.
#
#     bash experiments/launch_p3_frontier.sh                 # default arms, seeds 0 1 2, tag _F1
#     ARMS="lora-all:film-512" SEEDS="0" TAG=_try bash experiments/launch_p3_frontier.sh
#
# The reference arm (lora-all-c1@gate=label) must reproduce C1's 0.716 / 0.715; every other arm runs
# under the ANALYTIC gate.  Jobs go to overcap on A40s (hal's TITAN Xp made a 1-minute setup take 20).
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
ARMS="${ARMS:-lora-all-c1@gate=label:lora-all-c1:lora-all:lora-b32:lora-b16:film-all:film-512:film-128:token-emb:nonparam}"
SEEDS="${SEEDS:-0 1 2}"
TAG="${TAG:-_F1}"
IFS=':' read -r -a arms <<< "$ARMS"
for s in $SEEDS; do
  for a in "${arms[@]}"; do
    name=$(echo "$a" | tr '@=' '__')
    sbatch -p overcap -A overcap --qos=short --gres=gpu:a40:1 --job-name="p3_${name}_s${s}" \
      --export=ALL,SCRIPT="experiments/p3_fold_frontier.py --seed $s --arms $a --tag $TAG" slurm/jobs/c_gpu.sbatch
  done
done
