#!/bin/bash
#SBATCH --job-name=lolroll
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/lifelong-ood-learner
mkdir -p results/robot
# a job admitted without its GPU (seen 2026-09-25: AllocTRES without gres) requeues itself, at most 3 times
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
echo "[roll] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# run a private copy: bash reads scripts incrementally, so editing roll.sh under a running job corrupts it
mkdir -p results/robot/.drivers
cp experiments/roll.sh results/robot/.drivers/roll_${SLURM_JOB_ID}.sh
bash results/robot/.drivers/roll_${SLURM_JOB_ID}.sh
rc=$?
echo "exit=$rc | $(date)"
exit $rc
