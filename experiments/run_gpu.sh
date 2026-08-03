#!/bin/bash
#SBATCH --job-name=lolgpu
#SBATCH --account=tail-lab
#SBATCH --partition=tail-lab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --qos=short
#SBATCH --output=experiments/slurm/gpu-%j.out
#SBATCH --error=experiments/slurm/gpu-%j.err

# GPU runner for the NEURAL track (a trained DDPM backbone).  The analytic track needs no GPU and
# uses experiments/run.sh instead.
#
#   sbatch --export=ALL,SCRIPT="experiments/n1_train.py" experiments/run_gpu.sh
#   sbatch --export=ALL,SCRIPT="experiments/n2_composition.py" experiments/run_gpu.sh
#   sbatch --export=ALL,SCRIPT="experiments/n3_lifelong.py" experiments/run_gpu.sh
set -uo pipefail
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate chunks                     # torch + cobweb + matplotlib
PY="${CONDA_PREFIX}/bin/python"

export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT"

cd ~/flash/lifelong-ood-learner
mkdir -p experiments/slurm results/cache

SCRIPT="${SCRIPT:-experiments/n1_train.py}"
echo "node=$(hostname) | script=${SCRIPT} | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[warn] no GPU visible"
"$PY" - <<'EOF' || exit 1
import torch
print(f"[deps] torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
EOF

srun --unbuffered "$PY" -u ${SCRIPT}
echo "exit=$? | $(date)"
