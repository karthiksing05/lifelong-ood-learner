#!/bin/bash
# FRESH init states in the LIBERO client's venv (experiments/fresh_states.py):
#   sbatch --export=ALL,TASKS=10+15,N=20,BASE=100000 experiments/fresh_states.sbatch
#SBATCH --job-name=lolfresh
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
cd /coc/flash10/ksingara3/robot/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PYTHONPATH:/coc/flash10/ksingara3/robot/openpi/third_party/libero
examples/libero/.venv/bin/python /coc/flash10/ksingara3/lifelong-ood-learner/experiments/fresh_states.py 2>&1 | grep -v "WARNING\|overload"
