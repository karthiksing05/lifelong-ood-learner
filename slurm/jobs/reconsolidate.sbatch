#!/bin/bash
#SBATCH --job-name=lolrecon
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-5
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The two ColorMNIST headline tables re-run under TODAY'S defaults, which the stored results
# predate on two counts: placement is now `cu` rather than `ll`, and mode ascent is plain gradient
# rather than Adam.  Neither changed for a cosmetic reason -- cu because ll builds a caterpillar
# (RESULTS 24), plain ascent because Adam marches past modes (RESULTS 30c) -- but both change
# numbers the documents quote, so the documents have to be re-derived rather than assumed intact.
#
# Tagged `_recon` so the canonical files are not overwritten until the comparison is read.  The
# ColorMNIST backbone itself is untouched by the layout work: it is permuted, consistently so, and
# retraining it is a separate decision from re-deriving the tables on it.
S=$(( SLURM_ARRAY_TASK_ID % 3 ))
if [ $(( SLURM_ARRAY_TASK_ID / 3 )) -eq 0 ]; then
  srun $PY experiments/n9_showdown.py --seed $S --tag _recon --mcmc-steps 10 --mcmc-lam 0.3
else
  srun $PY experiments/n5_classinc.py --seed $S --tag _recon
fi
rc=$?
echo "exit=$rc | $(date)"
exit $rc
