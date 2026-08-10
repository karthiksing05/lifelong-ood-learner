#!/bin/bash
#SBATCH --job-name=loln37b
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# Budget set BY N35: nothing renders on this backbone below ~80 epochs (seen_joint 0.000 at 40,
# 0.383 at 80), so the 32-epoch run leaves seen_joint degenerate at 0.000 for every arm and only
# the loss comparison is readable.  4 blocks x 30 = 120 epochs puts every arm above the threshold.
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/n37_progressive_seal.py \
    --blocks 4 --epochs-per-block 30 --seed 0 --tag _e120
echo "exit=$? | $(date)"
