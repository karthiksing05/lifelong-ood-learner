#!/bin/bash
# Reproduce every number and every figure in this repository.
#
#   bash reproduce.sh tests      # 7 assertions, ~5 min, no backbone needed
#   bash reproduce.sh backbones  # train the 5 backbones everything else loads (~12 GPU-hours)
#   bash reproduce.sh colormnist # the ColorMNIST experiments, 3 seeds
#   bash reproduce.sh celeba     # the CelebA experiments, 3 seeds
#   bash reproduce.sh analytic   # the analytic (no-gradient) track, 3 seeds
#   bash reproduce.sh supporting # the experiments behind the negative results and the cost claims
#   bash reproduce.sh figures    # aggregate + all figures + TABLES.md  (CPU, seconds)
#   bash reproduce.sh all        # everything, in dependency order
#
# Every stage is idempotent and submits to SLURM; `figures` is the only one that runs locally.
# Stages after `backbones` assume results/cache/*.pt exists.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
STAGE="${1:-all}"
SUB=(sbatch --gres=gpu:a40:1)          # add --account=overcap --partition=overcap to use spare capacity
CPU=(sbatch)
PY="${PY:-$HOME/flash/miniconda3/envs/chunks/bin/python}"

gpu () { "${SUB[@]}" --export=ALL,SCRIPT="$1" experiments/run_gpu.sh | tail -1; }
cpu () { "${CPU[@]}" --export=ALL,SCRIPT="$1" experiments/run.sh     | tail -1; }

# Stale bytecode on NFS silently runs old code; clear it before every submission wave.
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
sync

case "$STAGE" in
tests|all)
  echo "== tests =="
  for t in test_layout test_smoke test_neural test_merge test_ortho_merge test_unbounded \
           test_pretrained; do gpu "tests/$t.py"; done
  [ "$STAGE" = tests ] && exit 0 ;;&

backbones|all)
  echo "== backbones (each writes results/cache/ddpm_*.pt) =="
  gpu "experiments/n1_train.py --epochs 200 --n-per-slot 200 --seed 0"                    # ColorMNIST
  gpu "experiments/n1_train.py --epochs 200 --n-per-slot 200 --seed 0 --holdout-bg 3"     # vocabulary
  gpu "experiments/n1_train.py --all-slots --epochs 200 --n-per-slot 200 --seed 0"        # native oracle
  gpu "experiments/n19_celeba_train.py --epochs 60 --batch 256 --seed 0"                  # CelebA combos
  gpu "experiments/n19_celeba_train.py --epochs 60 --batch 256 --seed 0 --holdout-attr Gray_Hair"
  gpu "experiments/n19_celeba_train.py --epochs 60 --batch 256 --seed 0 --holdout-pair Smiling Big_Nose"
  echo "   wait for these to finish before the stages below"
  [ "$STAGE" = backbones ] && exit 0 ;;&

colormnist|all)
  echo "== ColorMNIST =="
  gpu "experiments/prebuild_celeba_probes.py"      # shared probe cache, also used by CelebA
  for s in 0 1 2; do
    gpu "experiments/n9_showdown.py --seed $s"                                   # composition
    gpu "experiments/n11_basis_extension.py --norm-match --tag _nmadapt --seed $s"  # combination cost
    gpu "experiments/n11_basis_extension.py --ckpt results/cache/ddpm_colormnist_nobg3.pt --tag _B --seed $s"
    gpu "experiments/n18_router.py --seed $s"                                    # self-routing
    gpu "experiments/n23_native_parity.py --seed $s"                             # native parity
  done
  for s in 0 1 2 3; do gpu "experiments/n5_classinc.py --seed $s"; done          # forgetting
  [ "$STAGE" = colormnist ] && exit 0 ;;&

celeba|all)
  echo "== CelebA =="
  P=results/cache/ddpm_celeba_factored_nopairSmiling-Big_Nose.pt
  for s in 0 1 2; do
    gpu "experiments/n20_celeba_composition.py --ckpt $P --tag _pair --seed $s"  # strong split
    gpu "experiments/n20_celeba_composition.py --seed $s"                        # weaker split
    gpu "experiments/n21_celeba_vocabulary.py --attr Gray_Hair --seed $s"        # new vocabulary
  done
  gpu "experiments/n22_celeba_lifelong.py --seed 0 --arms ours independent finetune replay ewc"
  [ "$STAGE" = celeba ] && exit 0 ;;&

supporting|all)
  echo "== supporting results (every number cited in README/TABLES that the stages above miss) =="
  gpu "experiments/e0_tune.py"                                  # tuning of record: tau, K, schedule
  gpu "experiments/n8_idealwords.py"                            # the factor basis itself
  gpu "experiments/n14_hierarchy.py"                            # X1: the hierarchy does NOT pay
  for s in 0 1 2; do
    gpu "experiments/n16_alternatives.py --seed $s"             # C6: hypernet / guidance / ortho
    gpu "experiments/n17_ortho_merge.py --seed $s"              # C6: kept vs merged, 2.8x
    gpu "experiments/n3_lifelong.py --n-concepts 10 --seed $s"  # per-concept retention detail
  done
  gpu "experiments/n2_composition.py"                           # composition on the neural backbone
  gpu "experiments/n4_batch.py"                                 # batch vs sequential addition
  "$PY" experiments/n24_growth.py                               # F8: the growth accounting (CPU)
  [ "$STAGE" = supporting ] && exit 0 ;;&

analytic|all)
  echo "== analytic track (no gradient descent anywhere) =="
  cpu "experiments/e1_manifold.py"
  for s in 0 1 2; do
    cpu "experiments/e2_composition.py --n-slots 40 --seed $s"
    cpu "experiments/e3_lifelong.py --n-concepts 12 --seed $s"
  done
  [ "$STAGE" = analytic ] && exit 0 ;;&

figures|all)
  echo "== aggregate, figures, tables (local, no GPU) =="
  "$PY" experiments/aggregate.py
  "$PY" experiments/make_paper_figures.py
  "$PY" experiments/make_paper_tables.py > TABLES.md && echo "  wrote TABLES.md"
  ;;
*)
  echo "usage: $0 {tests|backbones|colormnist|celeba|analytic|supporting|figures|all}"; exit 1 ;;
esac
