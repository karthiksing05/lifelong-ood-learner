#!/bin/bash
# Everything one held-out LIBERO-90 task needs before its rollouts, from ONE demo (IPEC episode 0 of its text):
#   r3_prepare (raw demo program, tag _lib40) -> r7_dimcode (dimension-wise borrowing, strides 1 and 5)
#   -> r4_fold (the 258 KB last2 fold on the approach / contact windows; only if the demo grasps) -> r3_fewshot (LoRA rank 16)
#   sbatch --export=ALL,TID=37,FOLD=1 experiments/prep.sbatch
#SBATCH --job-name=lolprep
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
set -eo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
T=results/robot/r2b_libero_40_tree.npz; L=results/robot/r2b_libero_40.npz
echo "[prep] task $TID fold=$FOLD $(date) on $(hostname)"
$PY experiments/r3_prepare.py --tree $T --library-npz $L --tasks $TID --levels 2,3,4 --tag _lib40 2>&1 | grep -v Warning
$PY experiments/r7_dimcode.py --tasks $TID 2>&1 | grep -v Warning
if [ "${FOLD:-0}" = 1 ]; then
  $PY experiments/r4_fold.py --task $TID --tree $T --library $L --rank 2 --steps 1000 --tag _approach_last2 --arrival contact --contact 10 --pre 30 --scope last2 2>&1 | grep -v Warning
fi
$PY experiments/r3_fewshot.py --task $TID 2>&1 | grep -v Warning
echo "exit=0 | $(date)"
