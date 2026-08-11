#!/bin/bash
#SBATCH --job-name=loln38
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
srun $PY experiments/n38_hierarchy_growth.py --quick || exit 1
srun $PY experiments/n38_hierarchy_growth.py --n-concepts 40 --n-orders 5 --seed 0
echo "exit=$? | $(date)"
