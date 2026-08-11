#!/bin/bash
#SBATCH --job-name=loltestcu
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python
# The whole suite under the CANDIDATE rule.  test_unbounded and test_pretrained_bank are the ones
# that matter: preservation is supposed to be a property of SEALING, not of placement, so a
# placement change must not touch bit-identity.  If it does, cu is disqualified regardless of how
# well-shaped its trees are.
# cu is now the DEFAULT (adapters.py), so this runs the suite as shipped.  Kept
# explicit so the job still states which rule it is validating.
export LIFELONG_TREE_RULE=cu
fail=0
for t in test_layout test_speckle test_fid test_smoke test_neural test_merge test_ortho_merge \
         test_unbounded test_pretrained test_pretrained_bank test_claims; do
  srun $PY tests/$t.py > /tmp/cu_$t.log 2>&1
  if [ $? -eq 0 ]; then echo "PASS $t"; else echo "FAIL $t"; tail -14 /tmp/cu_$t.log; fail=1; fi
done
echo "===== rule=cu: $([ $fail -eq 0 ] && echo ALL PASS || echo SOME FAILED) ====="
echo "exit=$fail | $(date)"
