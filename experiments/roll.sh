#!/bin/bash
# One node: a policy server (openpi env) + our LIBERO client (libero env).  Env vars:
#   MODE    openpi (openpi's Policy.infer, eager, warmed) | ours | proto:<tree>:<node>[:w0] | compose:<spec>
#   SUITE   libero_goal | libero_90 | ...     TASKS  all | 0+1+5 (not commas: sbatch --export splits on them)
#   TRIALS  n      FIRST  first init-state index
#   OUT     results/robot/<name>.json
set -uo pipefail
MODE=${MODE:-openpi}; SUITE=${SUITE:-libero_goal}; TASKS=${TASKS:-all}; TRIALS=${TRIALS:-10}; FIRST=${FIRST:-0}
TASKS=${TASKS//+/,}      # sbatch --export splits on commas: pass task lists as 0+1+2
REPO=/coc/flash10/ksingara3/lifelong-ood-learner
OUT=${OUT:-$REPO/results/robot/roll_${SUITE}_${SLURM_JOB_ID:-x}.json}
ROBOT=/coc/flash10/ksingara3/robot/openpi
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
PORT=$((9000 + RANDOM % 900))
LOG=$REPO/results/robot/server_${SLURM_JOB_ID:-x}.log
cd $ROBOT
.venv/bin/python $REPO/experiments/serve_guided.py --port $PORT --mode "$MODE" > $LOG 2>&1 &
SRV=$!
for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then break; fi; if ! kill -0 $SRV 2>/dev/null; then echo "[roll] server died"; tail -30 $LOG; exit 3; fi; done
echo "[roll] server up on $PORT after $((i*5))s (mode $MODE)"
PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/libero_client.py \
    --port $PORT --suite $SUITE --tasks $TASKS --trials $TRIALS --first-trial $FIRST --out $OUT
rc=$?
kill $SRV 2>/dev/null
echo "[roll] client exit=$rc -> $OUT"
if [ -f "$OUT" ]; then
  .venv/bin/python $REPO/experiments/roll_record.py --json $OUT --experiment ${EXPERIMENT:-rollout} --mode "$MODE" --tag "${TAG:-}" --job ${SLURM_JOB_ID:-x} --program-bytes ${PROGRAM_BYTES:-0}
fi
exit $rc
