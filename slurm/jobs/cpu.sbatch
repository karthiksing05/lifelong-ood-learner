#!/bin/bash
# CPU runner for tests and probes that import torch.  The login node SIGTERMs bare torch processes,
# so anything heavier than a numpy unit test goes through here:
#
#     sbatch --wait --export=ALL,SCRIPT="-m pytest tests -q" slurm/jobs/cpu.sbatch
#     sbatch --export=ALL,SCRIPT="experiments/p0_sampler_parity.py --cpu" slurm/jobs/cpu.sbatch
#
# sbatch splits --export on COMMAS: pass lists with colons.
#SBATCH --job-name=lolcpu
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
NT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT" NUMEXPR_NUM_THREADS="$NT"
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[cpu] $SCRIPT"
echo "[cpu] $(date) on $(hostname)"
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
