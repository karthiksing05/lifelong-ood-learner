#!/bin/bash
# M3 arms for LIBERO-90 tasks: zero-shot (our sampler, no hooks), program:code (prototype PoE), program:raw (the
# demo chunks themselves), lora (few-shot flat adapter; needs results/robot/r3_lora_<task>.pt or a dependency).
#   TASKS=19,35 TRIALS=20 W0=1.2 ARMS="zeroshot code raw lora" bash experiments/launch_r3.sh
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
TASKS=${TASKS:?}; TRIALS=${TRIALS:-20}; W0=${W0:-1.2}; ARMS=${ARMS:-"zeroshot code raw"}; TAGX=${TAGX:-}; SCHED=${SCHED:-variance}; TVAR=${TVAR:-}
R=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot
for tid in ${TASKS//,/ }; do
  for arm in $ARMS; do
    case $arm in
      zeroshot) MODE=ours; PB=0 ;;
      code|raw) MODE="program:$R/r3_prog_${tid}_${arm}${PROGTAG:-}.npz:$W0:20:$SCHED:$TVAR"
                PB=$(~/flash/miniconda3/envs/chunks/bin/python -c "import sys; sys.path.insert(0,'.'); from lifelong_ood.policy.compose import DemoProgram; print(DemoProgram.load('$R/r3_prog_${tid}_${arm}${PROGTAG:-}.npz').bytes())") ;;
      lora) MODE="lora:$R/r3_lora_${tid}.pt"; PB=0 ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    SFX=""; case $arm in code|raw) SFX="_w${W0}_${SCHED}${TVAR:+_v$TVAR}";; esac
    OUT=$R/r3_${tid}_${arm}${TAGX}${SFX}.json
    j=$(sbatch --parsable --export=ALL,MODE="$MODE",SUITE=libero_90,TASKS=$tid,TRIALS=$TRIALS,EXPERIMENT=r3_compose,TAG=${arm}${TAGX}${SFX},PROGRAM_BYTES=$PB,OUT=$OUT experiments/roll.sbatch)
    echo "task $tid arm $arm -> job $j ($OUT)"
  done
done
