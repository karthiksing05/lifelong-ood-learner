#!/bin/bash
#SBATCH --job-name=loln46i
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# The SEEN control stayed at chance after the init fix, so the question is no longer "does discovery
# work" but "is the composition machinery correct at all".  The identity check answers it without
# involving discovery: library = the backbone's own class embedding, w one-hot, which reduces Eq. 11
# to the ordinary class conditional and MUST reproduce the class.
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/n46_unsup_concepts.py \
    --K 8 --identity-check --disc-steps 200 --n-cases 2 --n-control 6 --tag _ident
rc=$?
echo "exit=$rc | $(date)"
exit $rc
