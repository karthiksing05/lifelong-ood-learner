#!/bin/bash
#SBATCH --job-name=lolcases
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# NOTE: do NOT pass --guidance here.  Its default of 1.5 is what n9's label-free
# calibration selects on this backbone, and the tables use it.  These scripts previously
# passed --guidance 2.0, which over-guides the DISTILLATION POOL that `ours` learns from
# (the other arms sample the teacher directly and never touch it).  That is why `ours`
# collapsed to joint accuracy 0.00 with wrong backgrounds while poe-guided stayed clean --
# the asymmetry in the figure was the fingerprint.  make_case_studies' own help warns the
# colour factor breaks entirely by 3.0.
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The user asked for s=0.8; §20b measured F1 0.410 there against 0.669 at s=0.90, so BOTH are
# rendered and the comparison is visible rather than asserted.
for D in 0.8 0.9; do
  srun $PY experiments/make_case_studies.py --n-cases 4 --dial $D || exit 1
done
rc=$?
echo "exit=$rc | $(date)"
exit $rc
