#!/bin/bash
#SBATCH --job-name=loln35
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
C=results/cache
srun $PY experiments/n35_competence.py --quick \
    --ckpts $C/ddpm_colormnist_ladder_ep2.pt $C/ddpm_colormnist_ladder.pt || exit 1
srun $PY experiments/n35_competence.py --seed 0 --ckpts \
    $C/ddpm_colormnist_ladder_ep2.pt  $C/ddpm_colormnist_ladder_ep5.pt \
    $C/ddpm_colormnist_ladder_ep10.pt $C/ddpm_colormnist_ladder_ep20.pt \
    $C/ddpm_colormnist_ladder_ep40.pt $C/ddpm_colormnist_ladder_ep80.pt \
    $C/ddpm_colormnist_ladder_ep130.pt $C/ddpm_colormnist_ladder.pt
echo "exit=$? | $(date)"
