#!/bin/bash
# Generic single-GPU runner.  Submit any experiment through SCRIPT:
#
#     sbatch --export=ALL,SCRIPT="experiments/n62_router_calibrate.py --seeds 4" slurm/jobs/gpu.sbatch
#
# CAUTION -- sbatch splits --export on COMMAS.  A value containing a comma is truncated at the
# first one, silently, and the job runs with a different argument than the header prints.  This
# cost a whole sweep once: "--batch-sizes 1,2,4,8,16" arrived as "1" and every run measured N=1
# while the log showed the full list.  Use COLON-separated lists in scripts that take them.
#
# This file lives in slurm/jobs/ because that directory is tracked; an untracked copy under
# experiments/slurm/ was deleted by a restructure while jobs still referenced it.
#SBATCH --job-name=lolgpu
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[gpu.sbatch] $SCRIPT"
echo "[gpu.sbatch] $(date) on $(hostname)"
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
