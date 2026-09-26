#!/bin/bash
# FINAL evaluation on init states no development run, pilot or earlier held-out run used (30-49), 20 episodes per arm.
#   bash experiments/final_eval.sh            (EXP=m3_final)
# Arms, all six LIBERO-90 tasks: zeroshot, raw_replay (the demo's chunks), code_replay (library codes), lora (rank 16 on
# the demo); hybrid_approach (library codes + rank-4 fold switched on in the approach/contact windows) on the tasks whose
# demo grasps or releases.  The practice-then-compose rule is derived afterwards (m34_report --final).
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
R=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot
EXP=${EXP:-m3_final}; FIRST=${FIRST:-30}; TRIALS=${TRIALS:-20}; DEP=${DEP:-}
TASKS=67+19+39+35+36+42 TRIALS=$TRIALS FIRST=$FIRST EXP=$EXP ARMS="zeroshot raw_replay code_replay" PROGTAG=_lib40 bash experiments/launch_m3.sh
for tid in 67 19 39 35 36 42; do
  d=""; [ -n "$DEP" ] && [ "$tid" = 19 -o "$tid" = 67 ] && d="--dependency=afterok:$DEP"
  j=$(sbatch --parsable $d --export=ALL,MODE="lora:$R/r3_lora_${tid}.pt",SUITE=libero_90,TASKS=$tid,TRIALS=$TRIALS,FIRST=$FIRST,EXPERIMENT=$EXP,TAG=lora,OUT=$R/final_${tid}_lora.json experiments/roll.sbatch)
  echo "task $tid lora -> $j"
done
for tid in 39 36 42; do
  base=$R/r4_${tid}_approach
  PB=$(~/flash/miniconda3/envs/chunks/bin/python -c "import sys,json,glob; sys.path.insert(0,'.'); from lifelong_ood import ledger as L; print(int(L.mean('r4_fold','bytes.total',tag='_approach',keys__task=$tid)[0]))")
  j=$(sbatch --parsable --export=ALL,MODE="hybrid:${base}_program.npz?expert=${base}_expert.pt&w0=1.0&sched=const&var=0.001&key=pose&stall=3",SUITE=libero_90,TASKS=$tid,TRIALS=$TRIALS,FIRST=$FIRST,EXPERIMENT=$EXP,TAG=hybrid_approach,PROGRAM_BYTES=$PB,OUT=$R/final_${tid}_hybrid_approach.json experiments/roll.sbatch)
  echo "task $tid hybrid_approach -> $j ($PB B)"
done
