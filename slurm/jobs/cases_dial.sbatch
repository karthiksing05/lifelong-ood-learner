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
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The user asked for s=0.8; §20b measured F1 0.410 there against 0.669 at s=0.90, so BOTH are
# rendered and the comparison is visible rather than asserted.
for D in 0.8 0.9; do
  srun $PY experiments/make_case_studies.py --n-cases 4 --guidance 2.0 --dial $D || exit 1
done
rc=$?
echo "exit=$rc | $(date)"
exit $rc
