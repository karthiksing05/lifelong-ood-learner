#!/bin/bash
#SBATCH --job-name=loln48s
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# De-risk the critical path.  n48 has never run, and n47 is a 13 h investment; finding a loader or
# API bug in n48 AFTER that completes would cost another cycle.  Make a throwaway backbone in
# minutes and run n48 against it end to end.
srun $PY experiments/n47_paper_backbone.py --quick || exit 1
srun $PY experiments/n48_paper_composition.py --quick \
     --ckpt results/cache/ddpm_celeba_paper128_smoke.pt
rc=$?
echo "exit=$rc | $(date)"
exit $rc
