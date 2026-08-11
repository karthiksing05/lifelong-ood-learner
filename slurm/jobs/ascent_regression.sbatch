#!/bin/bash
#SBATCH --job-name=lolareg
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The plain-gradient ascent default was chosen on the pretrained CelebA backbone.  It changes
# discovery EVERYWHERE, including ColorMNIST, where the pipeline was already working -- so it has
# to be shown not to cost anything there before it can stand as a global default.
# The bar: the canonical case study last measured ours joint 0.667 at recall ~0.93 under Adam.
srun $PY experiments/make_case_studies.py --n-cases 4
rc=$?
echo "exit=$rc | $(date)"
exit $rc
