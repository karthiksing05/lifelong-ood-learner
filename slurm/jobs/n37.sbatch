#!/bin/bash
#SBATCH --job-name=loln37
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
srun $PY experiments/n37_progressive_seal.py --quick || exit 1
srun $PY experiments/n37_progressive_seal.py --blocks 4 --epochs-per-block 8 --seed 0
echo "exit=$? | $(date)"
