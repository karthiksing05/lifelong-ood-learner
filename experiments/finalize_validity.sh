#!/bin/bash
# After the validity waves land: statistics, inspection reads, library ablation, figures, audits (login node: numpy / mpl
# only; the strips and galleries use the openpi venv for PyAV).
set -euo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
PY=~/flash/miniconda3/envs/chunks/bin/python
OPY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
$PY experiments/m3_stats.py --record 2>&1 | grep -v Warning
for src in band leaf seg; do $PY experiments/r8_inspect.py --source $src --record 2>&1 | grep -v Warning | grep -E "mean word|lenders|ledger"; done
$PY experiments/r9_library_ablation.py --record 2>&1 | grep -v Warning | tail -14
$PY experiments/fig_m3.py 2>&1 | grep -v Warning
$PY experiments/fig_hierarchy.py 2>&1 | grep -v Warning
for t in ${STRIPS:-39 36 6 1}; do $OPY experiments/fig_strips.py --task $t --trial 30 2>&1 | grep -v -i warn | tail -1; done
for t in ${SHEETS:-39 6 36}; do $PY experiments/fig_segments.py --task $t 2>&1 | grep -v Warning; done
$PY experiments/audit_numbers.py 2>&1 | grep -v Warning | tail -1
$PY experiments/audit_coverage.py 2>&1 | grep -v Warning | tail -1
