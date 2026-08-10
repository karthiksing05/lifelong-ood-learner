#!/bin/bash
#SBATCH --job-name=loln4k
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%A_%a.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --array=0-14
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
KS=(1 2 4 8 12)
K=${KS[$((SLURM_ARRAY_TASK_ID % 5))]}
S=$((SLURM_ARRAY_TASK_ID / 5))
# The headline batch claim (2.14x faster, +0.052 ACC at K=12) rested on ONE seed and only two
# points in the matched-discovery configuration; K=2/4/8 existed only in the pre-fix runs whose
# sequential arm used different discovery settings.  n4_batch now calls learn_concept_disc with the
# batch path's own disc, so every run here is matched by construction.
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/n4_batch.py \
    --n-concepts 12 --batch-size $K --seed $S --tag _fairK$K
echo "exit=$? K=$K seed=$S | $(date)"
