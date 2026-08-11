#!/bin/bash
#SBATCH --job-name=lolproto
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# The paper's CelebA is a fine-tune of ddpm-celebahq-256.  If we cannot get clean faces out of that
# model unmodified, nothing downstream can replicate their figures.
srun $HOME/flash/miniconda3/envs/chunks/bin/python tests/diag_prototype_quality.py
rc=$?
echo "exit=$rc | $(date)"
exit $rc
