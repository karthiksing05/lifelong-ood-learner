#!/bin/bash
#SBATCH --job-name=loln41s
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-2
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# All three seeds re-run at --n-score 8, including seed 0: the earlier seed-0 result scored only
# 4 concepts, so mixing it in would pool two different measurements under one mean.
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/n41_addressing.py \
    --n-concepts 24 --n-score 8 --seed $SLURM_ARRAY_TASK_ID
rc=$?
echo "exit=$rc | $(date)"
exit $rc
