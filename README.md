# lifelong-ood-learner

Lifelong, out-of-distribution learning as **composing seen prototypes**, and **folding genuinely new
ones into the manifold** of a diffusion model, with the tree of prototypes as the index of what is
known and the noise level as its depth.

Rework in progress (started 2026-09-25). The plan, the audit of the previous architecture and the
literature positioning are in [docs/PLAN_2026-09.md](docs/PLAN_2026-09.md); the review that retired the
previous headline claims is [docs/REVIEW_2026-09.md](docs/REVIEW_2026-09.md); the theory the rework
rests on (the ideal denoiser's one-instance update IS Cobweb's incremental update, noise level = tree
depth) is [docs/COBWEB_UPDATE.md](docs/COBWEB_UPDATE.md). The previous repository is preserved under
[archive/2026-09-pre-rework/](archive/2026-09-pre-rework/README.md) with a map from each retired
claim to the review's verdict.

## The three operators

| Cobweb | here | cost |
|---|---|---|
| categorise / best child | DECOMPOSE a query into the K prototypes (per noise level) that explain it | 0 weights; a code of node ids and weights |
| incorporate | ADDRESS: if the composition explains the query in the read band, store the code, update counts | 0 weights |
| create | FOLD a genuinely new prototype: new nodes at the levels where the query is unexplained, realised by a responsibility-gated minimal expert trained with anchoring on the model's own samples of the sibling nodes | the expert (the frontier experiment decides its size) |
| merge / promote | CONSOLIDATE: distil an accrued node into the base weights under the same gate, delete the expert | net zero or negative memory |

## The ladder of results

Every number quoted anywhere in this repository comes from a row in `ledger/` and is re-derived by
`experiments/audit_numbers.py` (`tests/test_audit.py` fails otherwise). Rungs are added as their rows land.

| rung | status |
|---|---|
| propositions 1-7 (numerical checks) | `tests/test_cobweb_correspondence.py` |
| the reference the fold must reproduce: C1's label-gated rank-1 LoRA on ColorMNIST (backfilled at af8f07a) | taught joint 0.716, never-taught joint 0.715, old-class drop 0.000, at 275,576 bytes; on Shapes3D 0.972 / 0.972 |
| the reference for storage: C5's router (backfilled) | tree store new-value joint 0.771, never-taught 0.795 at 277,601 bytes; flat store 0.646, 0.550 at 1,844,635 bytes |
| the reference for addressing: C3's flagship (backfilled; addressing what the backbone already renders) | 0.980 from one image at 72 bytes, 0.913 for a 1 KB inversion, 0.001 nearest trained class |
| sampler parity after the restructure | PASS: `sample(backbone)` equals the fixtures' `ddim_sample` bit for bit on the ColorMNIST UNet and the Shapes3D VE bridge, max abs diff 0.0 (`ledger/p0_sampler_parity`) |
| prototype tree (phase 0), decompose (1), gate (2), fold frontier (3), compose-after-fold (4), consolidate (5) | pending |

## Reproduce

    PY=$HOME/flash/miniconda3/envs/chunks/bin/python
    $PY -m pytest tests/test_ledger.py tests/test_cobweb_correspondence.py -q     # numpy, login node
    for t in tests/test_keyed_lora.py tests/test_layout.py tests/test_field_contract.py \
             tests/test_sample_concept_path.py; do $PY $t; done                    # torch, seconds
    sbatch --wait --export=ALL,SCRIPT="-m pytest tests -q" slurm/jobs/cpu.sbatch  # off the login node
    sbatch --export=ALL,SCRIPT="experiments/legacy/c1_cobweb_update.py --bench colormnist --seed 0" slurm/jobs/c_gpu.sbatch
    $PY experiments/ledger_backfill.py --dry-run
    $PY experiments/audit_numbers.py && $PY experiments/audit_coverage.py && $PY experiments/make_numbers.py

## Layout

    lifelong_ood/backbones/   schedule, layout (one flat<->image pair), the Backbone handle, THE sampler, score ops
    lifelong_ood/neural/      UNet, DDPM training, VE bridge, pretrained shims, KeyedLoRA + anchoring (cobweb_update.py)
    lifelong_ood/data/        ColorMNIST, Shapes3D (rows="correct"), CelebA, the novel-value edits, multi-object scenes
    lifelong_ood/evals.py     probes, coverage, KID, vendi, memorisation;  instrument.py: ceilings and Wilson resolution
    lifelong_ood/ledger.py    the tracked results ledger
    lifelong_ood/tree/        the prototype tree: nodes at noise levels, build, responsibility, update
    lifelong_ood/experts/     the expert interface and the minimal experts (LoRA, band LoRA, FiLM, token, nonparametric)
    lifelong_ood/operators/   decompose / address, fold (gate, gated losses, stop rule), anchor; consolidate (pending)
    lifelong_ood/streams/     the benches and the arrival streams with the new primitives
    lifelong_ood/accounting.py  charge(): every byte a method keeps
    experiments/legacy/       c1 (fold reference arms), c5 (the label-gated router), c_report, diag_read_by_u
    experiments/p*.py         the probes: p0 sampler parity, p1 tree build, p3 fold frontier
    experiments/audit_*.py    the number audit and its coverage; make_numbers.py regenerates paper/numbers.json
    tests/                    the kept tests plus one per new module; everything else is in the archive
    ledger/                   one small JSON per run, tracked
    archive/2026-09-pre-rework/   the previous repository, with its claim map
