#!/bin/bash
#SBATCH --job-name=lolw0
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# Quantifies infidelity (a) from docs/SETTINGS.md: CelebA guided at w0=1.2 (the paper's
# Color-MNIST value) instead of its Table 8 CelebA value of 0.5.
srun $HOME/flash/miniconda3/envs/chunks/bin/python tests/diag_celeba_w0.py
rc=$?
echo "exit=$rc | $(date)"
exit $rc
