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
# RETRAINED IN THE CORRECT LAYOUT.  The previous pair ran ~20 h each in the permuted layout,
# which is the one RESULTS 30a shows produces a measurable seam at rows H/3 and 2H/3 (score
# 2.509 against real data's 0.906).  n19 now defaults to --layout hwc and records it in the
# checkpoint; the old .partial files are parked as *.permuted_obsolete so --resume cannot
# silently continue a run in the wrong layout.  128px is also the paper's own resolution.

PY=$HOME/flash/miniconda3/envs/chunks/bin/python
export PYTHONIOENCODING=utf-8
# --save-every 5 is now live (n19 writes <out>.partial atomically every 5 epochs).  The previous
# pair of these jobs was PREEMPTED at epochs 64/90 and 72/90 after ~18 h each and left NOTHING on
# disk, because they were launched before that fix existed.  --requeue lets SLURM restart them.
# SPLIT selects which backbone this job trains and MUST be exported at submit time:
#     sbatch --export=ALL,SPLIT=main slurm/jobs/celeba128.sbatch
#     sbatch --export=ALL,SPLIT=pair slurm/jobs/celeba128.sbatch
# Submitting without it silently sends BOTH jobs down the else-branch, where they race on
# the same output file -- which is what happened when SLURM requeued a preempted run.
if [ -z "${SPLIT:-}" ]; then
  echo "SPLIT is unset -- refusing to run. Pass --export=ALL,SPLIT=main|pair" >&2
  exit 2
fi
if [ "$SPLIT" = "pair" ]; then
  srun $PY experiments/n19_celeba_train.py --img 128 --epochs 90 --batch 128 --base 64 --seed 0 \
      --save-every 5 --resume --holdout-pair Smiling Big_Nose \
      --out results/cache/ddpm_celeba128_nopairSmiling-Big_Nose.pt
else
  srun $PY experiments/n19_celeba_train.py --img 128 --epochs 90 --batch 128 --base 64 --seed 0 \
      --save-every 5 --resume --out results/cache/ddpm_celeba128.pt
fi
rc=$?
# Propagate the real exit code.  `echo` as the last command always succeeds, so SLURM
# reported COMPLETED for a run that bus-errored at epoch 34/90 and any monitor reading
# sacct saw a success.
echo "exit=$rc | $(date)"
exit $rc
