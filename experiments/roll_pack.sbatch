#!/bin/bash
# Several rollouts on ONE GPU: pi0.5 inference leaves the A40 mostly idle while the simulator steps (7-39% utilisation,
# 8.4 of 48 GB), so N instances of roll.sh share the allocation.  SPECS = a file, one instance per line, tab-separated
# KEY=VALUE pairs (MODE, SUITE, TASKS, TRIALS, FIRST, EXPERIMENT, TAG, PROGRAM_BYTES, OUT).
#   sbatch --export=ALL,SPECS=results/robot/.packs/p_0001.tsv experiments/roll_pack.sbatch
#SBATCH --job-name=lolpack
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/lifelong-ood-learner
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
echo "[pack] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); specs $SPECS"
mkdir -p results/robot/.drivers
DRV=results/robot/.drivers/roll_${SLURM_JOB_ID}.sh
cp experiments/roll.sh $DRV
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
base=$((9000 + (SLURM_JOB_ID % 440) * 2))
i=0; pids=()
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  IFS=$'\t' read -r -a kv <<< "$line"
  env "${kv[@]}" PORT=$((base + i)) LOG=$PWD/results/robot/server_${SLURM_JOB_ID}_$i.log bash $DRV > slurm/${SLURM_JOB_ID}_$i.out 2>&1 &
  pids+=($!); echo "[pack] instance $i: ${kv[*]}" | cut -c1-300
  i=$((i + 1)); sleep 20        # stagger the model loads
done < "$SPECS"
rc=0
for p in "${pids[@]}"; do wait $p || rc=$?; done
echo "exit=$rc | $(date)"
exit $rc
