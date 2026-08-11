#!/bin/bash
#SBATCH --job-name=lolcases
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
# The canonical README figure, at the SELECTED dial (0.9) and the paper's discovery budget.
# The sweep at results/case_colormnist*.json shows 0.9 dominates: joint 0.708 at recall 0.881,
# against 0.667/0.495 at 0.8 and 0.542/0.907 at 1.0.
srun $PY experiments/make_case_studies.py --n-cases 4
rc=$?
echo "exit=$rc | $(date)"
exit $rc
