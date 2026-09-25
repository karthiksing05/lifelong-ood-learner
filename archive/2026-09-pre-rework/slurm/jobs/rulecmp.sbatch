#!/bin/bash
#SBATCH --job-name=lolrule
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-11
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# 12 = 2 experiments x 2 rules x 3 seeds.  The rule is set by env so the experiment code is
# byte-identical between arms -- the only difference is the placement criterion.
EXPS=(n5 n9)
E=${EXPS[$(( (SLURM_ARRAY_TASK_ID / 6) ))]}
R=$([ $(( (SLURM_ARRAY_TASK_ID / 3) % 2 )) -eq 0 ] && echo ll || echo cu)
S=$(( SLURM_ARRAY_TASK_ID % 3 ))
export LIFELONG_TREE_RULE=$R
echo "=== $E rule=$R seed=$S ==="
if [ "$E" = "n5" ]; then
  srun $PY experiments/n5_classinc.py --arms ours --seed $S --tag _rule$R
else
  # n9 has no --arms; it runs the full showdown.  That is fine and in fact better --
  # every baseline is recomputed identically under both rules, so any movement in the
  # non-bank arms would flag a leak rather than a placement effect.
  srun $PY experiments/n9_showdown.py --seed $S --tag _rule$R
fi
rc=$?
# Propagate the real exit code.  `echo` as the last command always succeeds, so SLURM
# reported COMPLETED for a run that bus-errored at epoch 34/90 and any monitor reading
# sacct saw a success.
echo "exit=$rc | $(date)"
exit $rc
