#!/bin/bash
#SBATCH --job-name=lolcond
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
# PHASE B backbone: identical data, UNet, recipe and epoch count to the shipped ColorMNIST
# backbone -- ONLY the conditioning scheme differs.  That is the control §13 never had, because it
# compared entangled ColorMNIST against factored CelebA and so confounded conditioning with dataset.
srun $PY experiments/n1_train.py --cond factored --epochs 200 --n-per-slot 200 --seed 0 \
    --out results/cache/ddpm_colormnist_factored.pt
rc=$?
# Propagate the real exit code.  `echo` as the last command always succeeds, so SLURM
# reported COMPLETED for a run that bus-errored at epoch 34/90 and any monitor reading
# sacct saw a success.
echo "exit=$rc | $(date)"
exit $rc
