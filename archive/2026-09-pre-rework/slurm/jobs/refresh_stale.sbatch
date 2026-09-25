#!/bin/bash
#SBATCH --job-name=lolstale
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
# tests/test_figures_fresh.py found these older than the code that draws them.  Two of them --
# the f14 hierarchy panels -- are RESULTS 24's evidence for the ll-vs-cu diagnosis, so the paper
# leans on images that predate edits to their generator.
srun $PY experiments/n38_hierarchy_growth.py --rule ll --tag ""      || exit 1
srun $PY experiments/n38_hierarchy_growth.py --rule cu --tag "_cu"   || exit 1
srun $PY experiments/n45_layout.py --epochs 60 --n-per-slot 200      || exit 1
rc=$?
echo "exit=$rc | $(date)"
exit $rc
