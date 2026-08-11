#!/bin/bash
#SBATCH --job-name=lolchwc
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-1
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# diag_band_artifact confirmed the banding artifact IS the layout permutation: the row-difference
# spikes exactly at rows 21 and 43 -- where the permutation's three "channels" meet -- scoring 2.314
# against real data's 0.906, and an hwc-trained backbone scores 1.004, i.e. no seam at all.
#
# So: a real CelebA backbone in the correct layout, long enough for the EMA to converge (60 epochs
# at ~40 steps/epoch is still short, so n_per_slot is raised to give more steps per epoch).
# Arm 0 is the fix; arm 1 is the matched control, so the comparison stays paired at full length.
L=$([ $SLURM_ARRAY_TASK_ID -eq 0 ] && echo hwc || echo permuted)
srun $PY experiments/n1_train.py --dataset celeba --img 64 --cond factored \
    --epochs 60 --n-per-slot 300 --layout $L --seed 0 \
    --out results/cache/ddpm_celeba_$L.pt
rc=$?
echo "exit=$rc | $(date)"
exit $rc
