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
# NOTE: do NOT pass --guidance here.  Its default of 1.5 is what n9's label-free
# calibration selects on this backbone, and the tables use it.  These scripts previously
# passed --guidance 2.0, which over-guides the DISTILLATION POOL that `ours` learns from
# (the other arms sample the teacher directly and never touch it).  That is why `ours`
# collapsed to joint accuracy 0.00 with wrong backgrounds while poe-guided stayed clean --
# the asymmetry in the figure was the fingerprint.  make_case_studies' own help warns the
# colour factor breaks entirely by 3.0.
export PYTHONIOENCODING=utf-8
DIALS=(0.8 0.9 1.0)
D=${DIALS[$SLURM_ARRAY_TASK_ID]}
# the dial swept AT the paper's discovery budget, so the prototypes are chosen from a pool 20x
# larger and the dial can be judged on prototypes that are actually relevant
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/make_case_studies.py \
    --n-cases 4 --paper-budget --dial $D
rc=$?
echo "exit=$rc | $(date)"
exit $rc
