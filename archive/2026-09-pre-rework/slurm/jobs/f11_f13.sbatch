#!/bin/bash
#SBATCH --job-name=lolf11
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
# F11 with the sample strip: the curves are an abstraction OF these images, so showing both lets a
# reader check the abstraction instead of trusting it.
srun $PY experiments/make_n30_figure.py --with-cases || exit 1
# F13 redrawn under the tree rule that is now the DEFAULT (cu).  n36 has no --rule flag, so it
# picks the default up automatically -- this run is what makes the paper's hierarchy a cu drawing.
srun $PY experiments/n36_hierarchy_probe.py
rc=$?
echo "exit=$rc | $(date)"
exit $rc
