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
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$REPO/results/robot/server_${SLURM_JOB_ID:-x}.log}
cd $ROBOT
# a port another process on the node holds (co-located packs, other users): pick a free one, and retry on a bind race
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  .venv/bin/python $REPO/experiments/serve_guided.py --port $PORT --mode "$MODE" --seed ${SERVE_SEED:-0} > $LOG 2>&1 &
  SRV=$!
  up=0
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[roll] port $PORT taken (attempt $attempt); retrying"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[roll] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[roll] no free port after 4 attempts"; exit 3; }
echo "[roll] server up on $PORT after $((i*5))s (mode $MODE)"
PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/libero_client.py \
    --port $PORT --suite $SUITE --tasks $TASKS --trials $TRIALS --first-trial $FIRST --out $OUT ${RECORD:+--record} ${OBJECTS:+--objects} ${BRANCHES:+--branches $BRANCHES --videos 0}
rc=$?
kill $SRV 2>/dev/null
echo "[roll] client exit=$rc -> $OUT"
# PROGRAM_BYTES=prog:<program.npz>[@<fold program.npz>]: size the program in the job (it may be built by a dependency)
case "${PROGRAM_BYTES:-0}" in
  prog:*) SPEC=${PROGRAM_BYTES#prog:}
          PROGRAM_BYTES=$(.venv/bin/python -c "
import sys; sys.path.insert(0, '$REPO')
from lifelong_ood.policy.compose import DemoProgram
p, _, f = '$SPEC'.partition('@'); prog = DemoProgram.load(p)
if f: prog.fold = DemoProgram.load(f).fold
print(prog.bytes())" 2>/dev/null || echo 0) ;;
esac
if [ -f "$OUT" ]; then
  .venv/bin/python $REPO/experiments/roll_record.py --json $OUT --experiment ${EXPERIMENT:-rollout} --mode "$MODE" --tag "${TAG:-}" --job ${SLURM_JOB_ID:-x} --program-bytes ${PROGRAM_BYTES:-0}
fi
exit $rc
