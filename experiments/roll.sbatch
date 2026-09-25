#!/bin/bash
#SBATCH --job-name=lolroll
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/lifelong-ood-learner
mkdir -p results/robot
echo "[roll] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
bash experiments/roll.sh
rc=$?
echo "exit=$rc | $(date)"
exit $rc
