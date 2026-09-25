#!/bin/bash
#SBATCH --job-name=lol
#SBATCH --account=tail-lab
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --qos=short
#SBATCH --output=experiments/slurm/lol-%j.out
#SBATCH --error=experiments/slurm/lol-%j.err

# sbatch runner for lifelong-ood-learner.  CPU-only by design: cobweb-private ifit is serial C++,
# and the entire diffusion side (score field, DDIM, mode ascent, PoE) is analytic numpy.
#
#   sbatch --export=ALL,SCRIPT="tests/test_smoke.py" experiments/run.sh
#   sbatch --export=ALL,SCRIPT="experiments/e1_manifold.py" experiments/run.sh
#   sbatch --export=ALL,SCRIPT="experiments/e2_composition.py --n-slots 40" experiments/run.sh
#   sbatch --export=ALL,SCRIPT="experiments/e3_lifelong.py --n-concepts 20" experiments/run.sh
set -uo pipefail
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate cobweb
PY="${CONDA_PREFIX}/bin/python"

export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
NT="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT" NUMEXPR_NUM_THREADS="$NT"

cd ~/flash/lifelong-ood-learner
mkdir -p experiments/slurm results

SCRIPT="${SCRIPT:-tests/test_smoke.py}"
echo "node=$(hostname) | script=${SCRIPT} | $(date)"
"$PY" - <<'EOF' || exit 1
import numpy
from cobweb.cobweb_continuous import CobwebContinuousTree
print(f"[deps] numpy {numpy.__version__}, cobweb-private importable")
EOF

srun --unbuffered "$PY" -u ${SCRIPT}
echo "exit=$? | $(date)"
