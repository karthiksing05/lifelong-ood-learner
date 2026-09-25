#!/bin/bash
#SBATCH --job-name=lolpretrk
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The ENTIRE pretrained track, rerun on a field that hands ddpm-celebahq-256 a real image instead
# of a permuted row.  Under the old layout the field returned a 3x3 tiled greyscale mosaic while
# DDPMPipeline on the same weights returns photorealistic faces, so every image-quality number
# from this track was void.
fail=0
for t in test_pretrained test_pretrained_bank; do
  srun $PY tests/$t.py > /tmp/pre_$t.log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -12 /tmp/pre_$t.log; fail=1; }
done
srun $PY experiments/n25_pretrained_case.py || fail=1     # the pretrained case studies
srun $PY experiments/n26_batch_pretrained.py || fail=1    # batch consolidation, pretrained
echo "exit=$fail | $(date)"
exit $fail
