#!/bin/bash
#SBATCH --job-name=lolcasesw
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-2
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
DIALS=(0.8 0.9 1.0)
D=${DIALS[$SLURM_ARRAY_TASK_ID]}
# the dial swept AT the paper's discovery budget, so the prototypes are chosen from a pool 20x
# larger and the dial can be judged on prototypes that are actually relevant
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/make_case_studies.py \
    --n-cases 4 --guidance 2.0 --paper-budget --dial $D
rc=$?
echo "exit=$rc | $(date)"
exit $rc
