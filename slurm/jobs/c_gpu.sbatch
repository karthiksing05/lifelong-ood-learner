#!/bin/bash
# Single-GPU runner for the c-series (Cobweb-form update) experiments.
#
#     sbatch --export=ALL,SCRIPT="experiments/c1_cobweb_update.py --seed 0" slurm/jobs/c_gpu.sbatch
#
# Differs from gpu.sbatch in two ways, both from cluster incidents:
#   * the dedicated tail-lab partition, not overcap -- overcap has handed jobs a slot with no usable
#     GPU, and torch then runs on CPU until the wall limit;
#   * a CUDA PREFLIGHT that fails in seconds instead of burning the allocation.
# sbatch splits --export on COMMAS: pass lists with colons (--arms a:b:c).
#SBATCH --job-name=lolc
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[c_gpu] $SCRIPT"
echo "[c_gpu] $(date) on $(hostname)"
$PY -c "import torch,sys; ok=torch.cuda.is_available(); print('[c_gpu] cuda', ok, torch.cuda.get_device_name(0) if ok else '-'); sys.exit(0 if ok else 3)" || { echo "[c_gpu] NO GPU -- aborting"; exit 3; }
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
