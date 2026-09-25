#!/bin/bash
# Wait for an unresponsive SLURM controller to come back, then submit a job and report.
#
# The scheduler went down mid-investigation with the fidelity sweep still unrun.  Polling it by hand
# is unreliable, so this waits on the controller itself: it probes `squeue` with a hard timeout (a
# dead controller HANGS rather than erroring, so a bare call would block forever), submits once the
# probe succeeds, and then waits on the job's own completion marker.
#
#   experiments/watch_and_submit.sh "<script and args>" [max_wait_minutes]
set -uo pipefail
cd ~/flash/lifelong-ood-learner

SCRIPT_ARGS="${1:?usage: watch_and_submit.sh \"experiments/foo.py --flag\" [max_minutes]}"
MAX_MIN="${2:-720}"
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
LOG=experiments/slurm/watch.log
mkdir -p experiments/slurm
echo "[watch] $(date): waiting for SLURM, then submitting: ${SCRIPT_ARGS}" | tee -a "$LOG"

n=0
while true; do
  if timeout 25 squeue --me -h >/dev/null 2>&1; then
    echo "[watch] $(date): controller responded after ${n} probes" | tee -a "$LOG"
    break
  fi
  n=$((n+1))
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[watch] $(date): gave up after ${MAX_MIN} min (${n} probes)" | tee -a "$LOG"
    exit 2
  fi
  sleep 60
done

JOB=$(timeout 60 sbatch --parsable --export=ALL,SCRIPT="${SCRIPT_ARGS}" experiments/run_gpu.sh 2>&1)
if ! [[ "$JOB" =~ ^[0-9]+$ ]]; then
  echo "[watch] $(date): submit failed: ${JOB}" | tee -a "$LOG"
  exit 3
fi
echo "[watch] $(date): submitted job ${JOB}" | tee -a "$LOG"

OUT="experiments/slurm/gpu-${JOB}.out"
while ! grep -q "exit=" "$OUT" 2>/dev/null; do
  sleep 30
  if [ "$(date +%s)" -gt "$((DEADLINE + 7200))" ]; then
    echo "[watch] $(date): job ${JOB} still running at deadline" | tee -a "$LOG"; exit 4
  fi
done
echo "[watch] $(date): job ${JOB} finished" | tee -a "$LOG"
tail -40 "$OUT" | tee -a "$LOG"
