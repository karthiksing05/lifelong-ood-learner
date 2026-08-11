#!/bin/bash
#SBATCH --job-name=loln20c
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-5
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The CelebA tables rerun at the CANONICAL configuration from reproduce.sh -- comp-scale 7.5 and
# mcmc-steps 10 / lam 0.3, which is what feeds the aliased keys the docs quote -- now that
# docs/SETTINGS.md corrected poe-guided's guidance from w0=1.2 (the paper's Color-MNIST value) to
# Table 8's CelebA w0=0.5.
#
# An earlier rerun of mine used the DEFAULT flags instead and wrote straight over the aliased
# result files, which silently dropped the composable-mcmc arm entirely (--mcmc-steps defaults to
# 0, which disables it) and changed comp-scale.  Those outputs are parked as *_w05probe.json.
# The configuration has to match or the w0 change is not the only thing that moved.
P=results/cache/ddpm_celeba_factored_nopairSmiling-Big_Nose.pt
S=$(( SLURM_ARRAY_TASK_ID % 3 ))
if [ $(( SLURM_ARRAY_TASK_ID / 3 )) -eq 0 ]; then
  srun $PY experiments/n20_celeba_composition.py --ckpt $P --tag _pairfull --seed $S \
       --comp-scale 7.5 --mcmc-steps 10 --mcmc-lam 0.3
else
  srun $PY experiments/n20_celeba_composition.py --seed $S --tag _full \
       --comp-scale 7.5 --mcmc-steps 10 --mcmc-lam 0.3
fi
rc=$?
echo "exit=$rc | $(date)"
exit $rc
