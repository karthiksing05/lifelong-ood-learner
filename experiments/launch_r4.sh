#!/bin/bash
# M4: fold then roll.  Per LIBERO-90 task: r4_fold.py (GPU) -> rollouts after it succeeds:
#   hybrid  (compose where the library explains the demo, the gated expert where it does not)
#   expert  (the folded expert switched on everywhere: what the gate buys)
#   TASKS=39+35 TREE=results/robot/r2b_libero_40_tree.npz LIB=results/robot/r2b_libero_40.npz TRIALS=20 bash experiments/launch_r4.sh
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
TASKS=${TASKS:?}; TREE=${TREE:?}; LIB=${LIB:?}; TRIALS=${TRIALS:-20}; W0=${W0:-1.0}; SCHED=${SCHED:-const}; TVAR=${TVAR:-0.001}
RANK=${RANK:-4}; STEPS=${STEPS:-1000}; TAG=${TAG:-}
R=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot
for tid in ${TASKS//+/ }; do
  f=$(sbatch --parsable --partition=tail-lab --account=tail-lab --export=ALL,ENV=openpi,SCRIPT="experiments/r4_fold.py --task $tid --tree $TREE --library $LIB --rank $RANK --steps $STEPS --tag $TAG" slurm/jobs/robot_gpu.sbatch)
  echo "task $tid fold -> job $f"
  base=$R/r4_${tid}${TAG}
  h=$(sbatch --parsable --dependency=afterok:$f --export=ALL,MODE="hybrid:${base}_program.npz?expert=${base}_expert.pt&w0=$W0&sched=$SCHED&var=$TVAR&key=pose&stall=3",SUITE=libero_90,TASKS=$tid,TRIALS=$TRIALS,EXPERIMENT=m4_fold,TAG=hybrid${TAG},OUT=$R/m4_${tid}_hybrid${TAG}.json experiments/roll.sbatch)
  e=$(sbatch --parsable --dependency=afterok:$f --export=ALL,MODE="lora:${base}_expert.pt",SUITE=libero_90,TASKS=$tid,TRIALS=$TRIALS,EXPERIMENT=m4_fold,TAG=expert_on${TAG},OUT=$R/m4_${tid}_expert_on${TAG}.json experiments/roll.sbatch)
  echo "task $tid hybrid -> job $h; expert always on -> job $e"
done
