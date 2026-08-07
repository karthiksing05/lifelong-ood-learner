#!/bin/bash
# Submit the CelebA experiment suite once its backbones exist.
#
#   bash experiments/submit_celeba_suite.sh smoke     # --quick versions first, to catch wiring
#   bash experiments/submit_celeba_suite.sh full      # the real runs, seed 0
#   bash experiments/submit_celeba_suite.sh seeds     # seeds 1 and 2 of the headline runs
#
# Everything goes to the overcap account so it does not consume the 4-A40 tail-lab budget that the
# backbone training needs.  Backbone names carry every distinguishing flag (RESULTS 5p), so a run
# against the wrong split fails loudly on verify_hold_ids rather than scoring against trained data.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MODE="${1:-smoke}"
SUB=(sbatch --account=overcap --partition=overcap --qos=short --gres=gpu:a40:1)
CACHE=results/cache

need () { [ -f "$CACHE/$1" ] || { echo "  MISSING $CACHE/$1 -- skipping the runs that need it"; return 1; }; }

go () { "${SUB[@]}" --export=ALL,SCRIPT="$1" experiments/run_gpu.sh | tail -1; }

find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
sync

Q=""
[ "$MODE" = smoke ] && Q=" --quick"

case "$MODE" in
  smoke|full)
    if need ddpm_celeba_factored.pt; then
      go "experiments/n20_celeba_composition.py --seed 0$Q"
      go "experiments/n22_celeba_lifelong.py --seed 0$Q"
    fi
    if need ddpm_celeba_factored_nopairSmiling-Big_Nose.pt; then
      go "experiments/n20_celeba_composition.py --ckpt $CACHE/ddpm_celeba_factored_nopairSmiling-Big_Nose.pt --tag _pair --seed 0$Q"
    fi
    if need ddpm_celeba_factored_noGray_Hair.pt; then
      go "experiments/n21_celeba_vocabulary.py --attr Gray_Hair --seed 0$Q"
    fi
    if need ddpm_colormnist_oracle.pt; then
      go "experiments/n23_native_parity.py --seed 0$Q"
    fi
    ;;
  seeds)
    for s in 1 2; do
      need ddpm_celeba_factored.pt && go "experiments/n20_celeba_composition.py --seed $s"
      need ddpm_celeba_factored_nopairSmiling-Big_Nose.pt && \
        go "experiments/n20_celeba_composition.py --ckpt $CACHE/ddpm_celeba_factored_nopairSmiling-Big_Nose.pt --tag _pair --seed $s"
      need ddpm_celeba_factored_noGray_Hair.pt && go "experiments/n21_celeba_vocabulary.py --attr Gray_Hair --seed $s"
      need ddpm_colormnist_oracle.pt && go "experiments/n23_native_parity.py --seed $s"
      need ddpm_celeba_factored.pt && go "experiments/n22_celeba_lifelong.py --seed $s"
    done
    ;;
  *) echo "usage: $0 {smoke|full|seeds}"; exit 1;;
esac
echo "submitted ($MODE)"
