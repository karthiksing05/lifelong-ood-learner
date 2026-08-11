#!/bin/bash
#SBATCH --job-name=lolsuite
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
# The whole suite, under the CURRENT defaults: cu placement, hwc on pretrained backbones,
# gamut-clamped mode ascent.  This is the reconsolidation checkpoint.
fail=0
for t in test_layout test_speckle test_fid test_smoke test_neural test_merge test_ortho_merge \
         test_unbounded test_pretrained test_pretrained_bank test_claims; do
  srun $PY tests/$t.py > /tmp/suite_$t.log 2>&1
  if [ $? -eq 0 ]; then echo "PASS $t"; else echo "FAIL $t"; tail -16 /tmp/suite_$t.log; fail=1; fi
done
echo "===== suite: $([ $fail -eq 0 ] && echo ALL PASS || echo SOME FAILED) ====="
echo "exit=$fail | $(date)"
exit $fail
