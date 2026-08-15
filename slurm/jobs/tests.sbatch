#!/bin/bash
#SBATCH --job-name=loltests
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# run_all.py GLOBS tests/test_*.py.  This file used to hardcode the list, which silently drifted:
# test_factored and test_figures_fresh existed on disk and were run by run_all, while this job
# reported "ALL PASS" over 11 of 13.  A hardcoded list cannot fail to notice a test it does not
# know about, so the glob is the single source of truth and this only launches it.
srun $PY tests/run_all.py
fail=$?
echo "exit=$fail | $(date)"
