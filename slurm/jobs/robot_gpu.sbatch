#!/bin/bash
# GPU runner for the embodied track.  ENV selects the interpreter:
#   ENV=openpi  -> /coc/flash10/ksingara3/robot/openpi/.venv (jax + torch + openpi + lerobot; the policy)
#   ENV=libero  -> /coc/flash10/ksingara3/robot/openpi/examples/libero/.venv (py3.8, robosuite 1.4.1; the sim client)
#
#   sbatch --export=ALL,ENV=libero,SCRIPT="experiments/r0_env_smoke.py" slurm/jobs/robot_gpu.sbatch
#
# Headless MuJoCo rendering through EGL on the allocated GPU; PYOPENGL_PLATFORM=egl for robosuite.
#SBATCH --job-name=lolrobot
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
# a job admitted without its GPU (seen 2026-09-25: AllocTRES without gres) requeues itself, at most 3 times
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
ROBOT=/coc/flash10/ksingara3/robot/openpi
case "${ENV:-openpi}" in
  openpi) PY=$ROBOT/.venv/bin/python ;;
  libero) PY=$ROBOT/examples/libero/.venv/bin/python; export PYTHONPATH=$PYTHONPATH:$ROBOT/third_party/libero ;;
  *) echo "unknown ENV=$ENV"; exit 2 ;;
esac
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[robot] ENV=$ENV PY=$PY"
echo "[robot] $SCRIPT"
echo "[robot] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
