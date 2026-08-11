#!/bin/bash
#SBATCH --job-name=lolf13
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
# f13 was regenerated under the new cu default before its title said so.  A hierarchy figure that
# does not name its placement rule is exactly how ll and cu drawings got confused in the first place.
srun $HOME/flash/miniconda3/envs/chunks/bin/python experiments/n36_hierarchy_probe.py
rc=$?
echo "exit=$rc | $(date)"
exit $rc
