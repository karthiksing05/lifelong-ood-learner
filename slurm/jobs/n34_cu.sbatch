#!/bin/bash
#SBATCH --job-name=loln34cu
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
# THE decisive follow-up: §24b measured no transfer under the CATERPILLAR, where the "family"
# ancestor was shared by 32/40 concepts.  Under rule=cu families are real (purity 0.60 vs 0.27,
# 20 internal nodes vs 4).  Does transfer appear?  Same wrong-family control, so "any ancestor"
# and "THIS ancestor" stay distinguishable.
srun $PY experiments/n34_reuse.py --quick --rule cu || exit 1
srun $PY experiments/n34_reuse.py --k-family 3 --seed 0 --rule cu --tag _cu
echo "exit=$? | $(date)"
