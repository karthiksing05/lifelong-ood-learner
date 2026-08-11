#!/bin/bash
#SBATCH --job-name=loln20
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The CelebA headline, rerun after docs/SETTINGS.md found poe-guided was being sampled at
# w0=1.2 -- the paper's COLOR-MNIST guidance -- instead of Table 8's CelebA w0=0.5.  On the
# pretrained backbone that error alone moved poe-guided roughness from 2.50-4.82x real down to
# 1.45-1.67x, so the published arm was being handicapped by a factor it never chose.
# Combination holdout, then the stronger pair holdout (a different backbone).
srun $PY experiments/n20_celeba_composition.py || exit 1
srun $PY experiments/n20_celeba_composition.py \
    --ckpt results/cache/ddpm_celeba_factored_nopairSmiling-Big_Nose.pt --tag _pair
rc=$?
echo "exit=$rc | $(date)"
exit $rc
