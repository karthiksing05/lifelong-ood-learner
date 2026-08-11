#!/bin/bash
#SBATCH --job-name=loln45
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
# What the permuted input layout costs: two matched backbones, only `layout` differs.
srun $PY experiments/n45_layout.py --quick || exit 1
# 30 epochs at n_per_slot 60 left BOTH arms at the conditioning floor (joint 0.010 vs
# 0.005), so only the held-out loss could discriminate -- it moved 0.05286 -> 0.03727, a
# 29.5% improvement.  Rerun at the real backbone's recipe (n_per_slot 200, 60 epochs) so
# the DOWNSTREAM metrics are above floor and the layout's cost is visible where it matters.
srun $PY experiments/n45_layout.py --epochs 60 --n-per-slot 200 --tag _full
rc=$?
echo "exit=$rc | $(date)"
exit $rc
