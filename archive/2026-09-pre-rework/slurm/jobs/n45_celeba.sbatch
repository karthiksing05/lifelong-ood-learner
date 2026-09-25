#!/bin/bash
#SBATCH --job-name=loln45c
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-1
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# ColorMNIST measured the layout permutation at -28.7% held-out loss and +0.031 joint accuracy.
# But ColorMNIST is the EASIEST possible case for a scrambled CNN -- a near-binary stroke on flat
# colour, almost all low frequency, which a permuted network can fit by brute force.  CelebA is the
# opposite: dense local texture is most of the signal.  If the effect is dataset-dependent in the
# direction the mechanism predicts, it should be substantially LARGER here, and that is the number
# that decides whether the CelebA backbones are retrained.
# Short matched runs (same epochs, same data, same seed) -- only `layout` differs.
L=$([ $SLURM_ARRAY_TASK_ID -eq 0 ] && echo permuted || echo hwc)
srun $PY experiments/n1_train.py --dataset celeba --img 64 --epochs 12 --cond factored \
    --layout $L --seed 0 --out results/cache/ddpm_celeba_layout_$L.pt
rc=$?
echo "exit=$rc | $(date)"
exit $rc
