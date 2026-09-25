#!/bin/bash
#SBATCH --job-name=loln47
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=20:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The paper's CelebA backbone: FINE-TUNE ddpm-celebahq-256 at 128px with a |C|+1 class table on the
# time embedding, linear beta T=1000, bf16, AdamW 2e-4.  We cannot match 500k steps on four GPUs,
# but fine-tuning a photorealistic model to accept conditioning is a far smaller ask than learning
# the image distribution from scratch, which is what we were doing.
srun $PY experiments/n47_paper_backbone.py --quick || exit 1
# lr: their Table 5 says 2e-4, which is a from-scratch-scale setting at batch 128 across
# four GPUs.  At batch 16 on one GPU it DIVERGED -- loss 0.013 at step 6400, exactly 1.0
# by 8000, and 1.0 is what eps-prediction loss becomes when the network emits zero.  The
# run now uses 5e-5 with warmup and aborts if the loss stays above 0.5, instead of
# spending 14 h producing an unusable checkpoint.
srun $PY experiments/n47_paper_backbone.py --hours 13 --img 128 --batch 16
rc=$?
echo "exit=$rc | $(date)"
exit $rc
