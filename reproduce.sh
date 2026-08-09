#!/bin/bash
# Reproduce every number and every figure in this repository.
#
#   bash reproduce.sh tests      # 7 assertions, ~5 min, no backbone needed
#   bash reproduce.sh backbones  # train the 5 backbones everything else loads (~12 GPU-hours)
#   bash reproduce.sh colormnist # the ColorMNIST experiments, 3 seeds
#   bash reproduce.sh celeba     # the CelebA experiments, 3 seeds
#   bash reproduce.sh pretrained # google/ddpm-celebahq-256: preservation, acquisition, batch
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
  for t in test_layout test_speckle test_fid test_smoke test_neural test_merge test_ortho_merge \
           test_unbounded test_pretrained test_pretrained_bank; do gpu "tests/$t.py"; done
  # CPU-only, and it gates the PROSE: every number the docs quote must match the aggregate
  "$PY" tests/test_claims.py
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
    gpu "experiments/n9_showdown.py --seed $s --tag _cal --mcmc-steps 10 --mcmc-lam 0.3"  # composition
    gpu "experiments/n11_basis_extension.py --norm-match --tag _nmadapt --seed $s"  # combination cost
    gpu "experiments/n11_basis_extension.py --ckpt results/cache/ddpm_colormnist_nobg3.pt --tag _B --seed $s"
    gpu "experiments/n18_router.py --seed $s"                                    # self-routing
    gpu "experiments/n23_native_parity.py --seed $s"                             # native parity
  done
  for s in 0 1 2; do gpu "experiments/n5_classinc.py --seed $s --tag _full"; done  # forgetting, 11 arms
  gpu "experiments/make_case_studies.py --n-cases 3 --guidance 2.0"              # F9 case study
  gpu "tests/diag_poe_faithful.py"                                               # baseline fairness check
  gpu "tests/diag_composable_scale.py --n-cases 3 --n-samples 16"                # composable-diff scale
  gpu "tests/diag_ours_quality.py --n-cases 2 --n-samples 8"                     # pool-guidance choice
  gpu "experiments/n27_factor_conditionals.py --n-cases 3 --n-samples 16"        # what a composition inherits
  gpu "experiments/n28_update_rules.py --n-cases 3 --n-samples 16"               # which update rule composes best
  for K in 1 2 4 8 12; do
    gpu "experiments/n4_batch.py --n-concepts 12 --batch-size $K --tag _K$K"     # how far batching scales
  done
  # held-out SEVERITY: thin the combination grid without removing a primitive, then re-run the
  # showdown on each thinned backbone.  MOSAIC (ICML 2026) reports compositional generalization
  # collapsing as coverage thins, so one split cannot be read as the result.
  for E in 20 40; do
    gpu "experiments/n1_train.py --epochs 200 --n-per-slot 200 --seed 0 --extra-holdout $E --out results/cache/ddpm_colormnist_sev$E.pt"
    gpu "experiments/n9_showdown.py --ckpt results/cache/ddpm_colormnist_sev$E.pt --seed 0 --tag _sev$E --mcmc-steps 10 --mcmc-lam 0.3"
  done
  [ "$STAGE" = colormnist ] && exit 0 ;;&

celeba|all)
  echo "== CelebA =="
  P=results/cache/ddpm_celeba_factored_nopairSmiling-Big_Nose.pt
  for s in 0 1 2; do
    gpu "experiments/n20_celeba_composition.py --ckpt $P --tag _pairfull --seed $s --comp-scale 7.5 --mcmc-steps 10 --mcmc-lam 0.3"  # strong split
    gpu "experiments/n20_celeba_composition.py --seed $s --tag _full --comp-scale 7.5 --mcmc-steps 10 --mcmc-lam 0.3"  # weaker split
    gpu "experiments/n21_celeba_vocabulary.py --attr Gray_Hair --seed $s"        # new vocabulary
  done
  for s in 0 1 2; do gpu "experiments/n22_celeba_lifelong.py --seed $s --tag _full"; done   # 11 arms
  [ "$STAGE" = celeba ] && exit 0 ;;&

pretrained|all)
  echo "== pretrained backbone (google/ddpm-celebahq-256, 113.7M params, 256px) =="
  # needs the HF cache; nothing here trains a backbone, the model is downloaded once
  export HF_HOME="${HF_HOME:-$HOME/flash/.cache/huggingface}"
  gpu "tests/test_pretrained_bank.py"                                            # preservation certificate
  for s in 0 1 2; do
    gpu "experiments/n25_pretrained_case.py --attrs Eyeglasses Smiling --n-train 128 --epochs 40 --lr 5e-4 --rank 8 --t-min 600 --concept-guidance 2.0 --seed $s --tag _final"
    gpu "experiments/n26_batch_pretrained.py --attrs Eyeglasses Smiling Wearing_Hat Male --seed $s --tag _v1"
  done
  gpu "tests/diag_pretrained_lora.py --epochs 30"                                # chooses rank/lr/noise range
  [ "$STAGE" = pretrained ] && exit 0 ;;&

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
  # tagged re-runs feed the canonical keys the figures and tables ask for
  "$PY" experiments/aggregate.py n9_showdown_cal            --as n9_showdown
  "$PY" experiments/aggregate.py n5_classinc_full           --as n5_classinc
  "$PY" experiments/aggregate.py n22_celeba_lifelong_full   --as n22_celeba_lifelong
  "$PY" experiments/aggregate.py n20_celeba_composition_full --as n20_celeba_composition
  "$PY" experiments/aggregate.py n20_celeba_composition_pairfull --as n20_celeba_composition_pair
  "$PY" experiments/make_architecture_figure.py     # F0: the architecture, no results needed
  "$PY" experiments/make_n30_figure.py              # F11: the prototype-distance dial
  "$PY" experiments/make_paper_figures.py
  # PYTHONIOENCODING: redirected stdout defaults to latin-1 on this cluster and the tables
  # contain em-dashes, so without it the file is written truncated at the first one.
  PYTHONIOENCODING=utf-8 "$PY" experiments/make_paper_tables.py > TABLES.md \
    && echo "  wrote TABLES.md"
  ;;
*)
  echo "usage: $0 {tests|backbones|colormnist|celeba|analytic|supporting|figures|all}"; exit 1 ;;
esac
