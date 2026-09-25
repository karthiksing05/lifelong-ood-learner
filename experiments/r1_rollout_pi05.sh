#!/bin/bash
# M1(a): pi05_libero rollouts in LIBERO through openpi's server (openpi env) + client (libero env), one node.
#
#   sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r1_rollout_pi05.sh libero_goal 10" slurm/jobs/robot_gpu.sbatch
#   (SCRIPT is run by the openpi python only if it ends in .py; robot_gpu.sbatch runs `srun $PY $SCRIPT`, so use
#    SCRIPT="-c 'import subprocess;subprocess.call([\"bash\",\"experiments/r1_rollout_pi05.sh\",\"libero_goal\",\"10\"])'"
#    or submit directly:  sbatch experiments/r1_rollout_pi05.sbatch)
set -uo pipefail
SUITE=${1:-libero_goal}; TRIALS=${2:-10}
ROBOT=/coc/flash10/ksingara3/robot/openpi
CKPT=/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
PORT=$((8000 + RANDOM % 1000))
cd $ROBOT
MODE=${MODE:-openpi}     # openpi: scripts/serve_policy.py (the reference); ours: our sampler through the same server
LOG=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/server_${SLURM_JOB_ID:-x}.log
if [ "$MODE" = openpi ]; then
  .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config pi05_libero --policy.dir $CKPT > $LOG 2>&1 &
else
  .venv/bin/python /coc/flash10/ksingara3/lifelong-ood-learner/experiments/serve_guided.py --port $PORT --mode "$MODE" > $LOG 2>&1 &
fi
SRV=$!
for i in $(seq 1 120); do sleep 5; if grep -q -E "Creating server|Serving|serving on|listening" $LOG 2>/dev/null; then break; fi; done
echo "[r1] server pid $SRV on port $PORT after $((i*5))s"
export PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero
examples/libero/.venv/bin/python examples/libero/main.py --args.host 127.0.0.1 --args.port $PORT --args.task-suite-name $SUITE --args.num-trials-per-task $TRIALS --args.video-out-path /coc/flash10/ksingara3/lifelong-ood-learner/results/robot/videos_${SUITE}_${MODE}_${SLURM_JOB_ID:-x} 2>&1 | tee /coc/flash10/ksingara3/lifelong-ood-learner/results/robot/client_${SLURM_JOB_ID:-x}.log
rc=${PIPESTATUS[0]}
kill $SRV 2>/dev/null
echo "[r1] client exit=$rc"
exit $rc
