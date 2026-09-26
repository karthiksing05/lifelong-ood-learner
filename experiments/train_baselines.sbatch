#!/bin/bash
# The trained one-demo baselines of one task: the equal-bytes adapter (rank 2, last two layers, whole demo: 258 KB), a
# rank-64 LoRA, and full fine-tuning of the action expert.   sbatch --export=ALL,TID=39 experiments/train_baselines.sbatch
#SBATCH --job-name=lolbase
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
set -eo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
echo "[base] task $TID $(date) on $(hostname)"
[ -f results/robot/r3_lora_${TID}_eq.pt ] || $PY experiments/r3_fewshot.py --task $TID --rank 2 --scope last2 --tag _eq 2>&1 | grep -v Warning
[ -f results/robot/r3_lora_${TID}_r64.pt ] || $PY experiments/r3_fewshot.py --task $TID --rank 64 --tag _r64 2>&1 | grep -v Warning
[ -f results/robot/r3_full_${TID}.pt ] || $PY experiments/r3_fewshot.py --task $TID --full 2>&1 | grep -v Warning
echo "exit=0 | $(date)"
