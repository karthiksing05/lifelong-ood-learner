#!/bin/bash
# Post-run consolidation, in the ONE order that keeps the canonical tables canonical.
#
# The trap this encodes: results/aggregate.json is not rebuilt automatically, and the headline
# tables are ALIASES -- reproduce.sh runs n20/n9 under _full/_pairfull/_cal tags with
# --comp-scale 7.5 --mcmc-steps 10 and aliases them onto the plain names.  Rebuilding without
# re-aliasing leaves the plain names pointing at whatever last wrote them, and test_claims will
# happily pass against a stale aggregate either way.
set -u
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8
PY=$HOME/flash/miniconda3/envs/chunks/bin/python

echo "== 1. aggregate =="
"$PY" experiments/aggregate.py >/dev/null || exit 1
echo "== 2. re-establish the canonical aliases =="
"$PY" experiments/aggregate.py n9_showdown_recon                 --as n9_showdown            >/dev/null
"$PY" experiments/aggregate.py n5_classinc_recon                --as n5_classinc            >/dev/null
"$PY" experiments/aggregate.py n22_celeba_lifelong_full        --as n22_celeba_lifelong    >/dev/null
"$PY" experiments/aggregate.py n20_celeba_composition_full     --as n20_celeba_composition >/dev/null
"$PY" experiments/aggregate.py n20_celeba_composition_pairfull --as n20_celeba_composition_pair >/dev/null
echo "== 3. figures + tables =="
"$PY" experiments/make_paper_figures.py | tail -2
"$PY" experiments/make_paper_tables.py > docs/TABLES.md && echo "  wrote docs/TABLES.md"
echo "== 4. claims (now meaningful, because the aggregate is fresh) =="
"$PY" tests/test_claims.py
