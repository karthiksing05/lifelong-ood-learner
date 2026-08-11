#!/bin/bash
#SBATCH --job-name=loln46
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# Liu et al. ICCV 2023 unsupervised concept discovery -- the closest published baseline to our
# addressing tier, since both store a small coordinate against a shared library.
srun $PY experiments/n46_unsup_concepts.py --quick || exit 1
# K=8 (under-capacity) and K=18 (matched to ColorMNIST's 10+4+4 factor values).  The first pass
# scored 0.000 on every held-out composition at K=8, which is not reportable without the SEEN
# control: a port that cannot reproduce IN-DISTRIBUTION compositions indicts the implementation,
# not the method.  Both runs now carry that control, and --oracle upper-bounds the method by
# letting discovery see the held-out queries.
srun $PY experiments/n46_unsup_concepts.py --K 8  --tag _k8
srun $PY experiments/n46_unsup_concepts.py --K 18 --tag _k18
srun $PY experiments/n46_unsup_concepts.py --K 18 --oracle --tag _k18_oracle
rc=$?
echo "exit=$rc | $(date)"
exit $rc
