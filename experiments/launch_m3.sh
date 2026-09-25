#!/bin/bash
# M3 arms (named) for LIBERO-90 tasks, programs from r3_prepare with PROGTAG (e.g. _lib40):
#   zeroshot      our sampler, the task's own instruction, nothing else
#   raw_replay    the demo's chunks as the teacher, constant weight 1 (open-loop program in chunk space), pose key, stall fallback
#   code_replay   the demo's prototype codes as the teacher (the library's PoE), same settings: COMPOSE at zero weights
#   raw_blend     the demo's chunks at weight 0.7 (the policy keeps a say)
#   code_blend    the codes at weight 0.7
#   prompt        the prompt program: the policy under the instruction of the library task that explains the aligned frame
#   lora          few-shot flat adapter (results/robot/r3_lora_<task>.pt)
#   TASKS=36+67 TRIALS=10 ARMS="zeroshot raw_replay code_replay" PROGTAG=_lib40 bash experiments/launch_m3.sh
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
TASKS=${TASKS:?}; TRIALS=${TRIALS:-10}; ARMS=${ARMS:-"zeroshot raw_replay code_replay"}; PROGTAG=${PROGTAG:-_lib40}; TAGX=${TAGX:-}
NICE=${NICE:-0}; COMMON=${COMMON:-"key=pose&stall=3"}; FIRST=${FIRST:-0}; EXP=${EXP:-m3_compose}
R=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot
PY=~/flash/miniconda3/envs/chunks/bin/python
for tid in ${TASKS//+/ }; do
  for arm in $ARMS; do
    PB=0
    case $arm in
      zeroshot)    MODE=ours ;;
      raw_replay)  P=$R/r3_prog_${tid}_raw${PROGTAG}.npz;  MODE="program:$P?w0=1.0&sched=const&var=0.001&$COMMON" ;;
      code_replay) P=$R/r3_prog_${tid}_code${PROGTAG}.npz; MODE="program:$P?w0=1.0&sched=const&var=0.001&$COMMON" ;;
      raw_blend)   P=$R/r3_prog_${tid}_raw${PROGTAG}.npz;  MODE="program:$P?w0=0.7&sched=const&var=0.001&$COMMON" ;;
      code_blend)  P=$R/r3_prog_${tid}_code${PROGTAG}.npz; MODE="program:$P?w0=0.7&sched=const&var=0.001&$COMMON" ;;
      prompt)      P=$R/r3_prog_${tid}_code${PROGTAG}.npz; MODE="program:$P?how=prompt&$COMMON" ;;
      lora)        MODE="lora:$R/r3_lora_${tid}.pt" ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    case $arm in raw_*|code_*|prompt) PB=$($PY -c "import sys; sys.path.insert(0,'.'); from lifelong_ood.policy.compose import DemoProgram; print(DemoProgram.load('$P').bytes())") ;; esac
    OUT=$R/m3_${tid}_${arm}${TAGX}.json
    j=$(sbatch --parsable --nice=$NICE --export=ALL,MODE="$MODE",SUITE=libero_90,TASKS=$tid,TRIALS=$TRIALS,FIRST=$FIRST,EXPERIMENT=$EXP,TAG=${arm}${TAGX},PROGRAM_BYTES=$PB,OUT=$OUT experiments/roll.sbatch)
    echo "task $tid arm $arm -> job $j"
  done
done
