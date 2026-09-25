#!/bin/bash
#SBATCH --job-name=loln44
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# Rerun after docs/SETTINGS.md found CelebA was guided at w0=1.2 (the paper's Color-MNIST
# value) instead of Table 8's CelebA w0=0.5.  The first pass of this job, at 1.2, produced
# poe-guided samples 2.5-4.8x ROUGHER than real faces on the paper's OWN pretrained backbone,
# while the ascent prototypes feeding them sat at 0.78-1.40 -- i.e. the break is in guided
# SAMPLING, not in the backbone or in discovery.  This measures the same thing at 0.5.
# both discovery settings, so the effect of OUR deviations is measured rather than assumed
srun $PY experiments/n44_celeba_paper.py --n-cases 3 --paper-discovery --tag _paperdisc_w05
srun $PY experiments/n44_celeba_paper.py --n-cases 3 --tag _oursdisc_w05
rc=$?
echo "exit=$rc | $(date)"
exit $rc
