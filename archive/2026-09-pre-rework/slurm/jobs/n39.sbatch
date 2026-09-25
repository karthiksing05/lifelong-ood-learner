#!/bin/bash
#SBATCH --job-name=loln39
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
srun $PY experiments/n39_placement.py --quick || exit 1
srun $PY experiments/n39_placement.py --n-concepts 40 --seed 0
rc=$?
# Propagate the real exit code.  `echo` as the last command always succeeds, so SLURM
# reported COMPLETED for a run that bus-errored at epoch 34/90 and any monitor reading
# sacct saw a success.
echo "exit=$rc | $(date)"
exit $rc
