#!/bin/bash
#SBATCH --job-name=lolceleb128
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=30:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --requeue
cd /coc/flash10/ksingara3/lifelong-ood-learner
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
export PYTHONIOENCODING=utf-8
# --save-every 5 is now live (n19 writes <out>.partial atomically every 5 epochs).  The previous
# pair of these jobs was PREEMPTED at epochs 64/90 and 72/90 after ~18 h each and left NOTHING on
# disk, because they were launched before that fix existed.  --requeue lets SLURM restart them.
if [ "$SPLIT" = "pair" ]; then
  srun $PY experiments/n19_celeba_train.py --img 128 --epochs 90 --batch 128 --base 64 --seed 0 \
      --save-every 5 --holdout-pair Smiling Big_Nose \
      --out results/cache/ddpm_celeba128_nopairSmiling-Big_Nose.pt
else
  srun $PY experiments/n19_celeba_train.py --img 128 --epochs 90 --batch 128 --base 64 --seed 0 \
      --save-every 5 --out results/cache/ddpm_celeba128.pt
fi
echo "exit=$? | $(date)"
