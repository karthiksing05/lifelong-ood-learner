#!/bin/bash
#SBATCH --job-name=loln46a
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The identity check passed at 0.802 against chance 0.006, so the composition machinery is correct
# and the failure is in DISCOVERY.  Under the paper's simplex constraint Eq. 11 provably collapses
# to a convex combination of conditionals -- a weighted AVERAGE -- which cannot express a
# CONJUNCTION like digit AND fg AND bg.  This ablation drops the constraint to test whether that is
# what limits the method here, rather than asserting it.
srun $PY experiments/n46_unsup_concepts.py --K 18 --disc-steps 1500 --no-simplex \
    --identity-check --tag _k18_nosimplex
rc=$?
echo "exit=$rc | $(date)"
exit $rc
