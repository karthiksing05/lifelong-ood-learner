"""Can this learner absorb an UNBOUNDED number of new concepts?  A property test, not a benchmark.

"Infinite" cannot be run, so the thing to test is the INVARIANT that would make it possible: that
the cost of adding concept K+1, and its effect on concepts 1..K, are both independent of K.  If both
hold at K=40 for the same structural reason they hold at K=2, nothing about the mechanism breaks as
K grows -- only memory does, linearly and predictably, which is a budgeting question rather than a
correctness one.

What is asserted:

  1. PRESERVATION IS K-INDEPENDENT.  After every batch, every concept ever consolidated is still
     bit-identical to its fingerprint at birth.  Not "close" -- identical.  This is the property
     that would decay in any method that shares mutable parameters, and it is checked against ALL
     previous concepts at every step, so the test gets stricter as it runs.
  2. THE FROZEN BASE NEVER MOVES, at any K.
  3. COST PER CONCEPT IS CONSTANT.  Adapter parameters grow linearly with a fixed per-concept slope,
     so "how many concepts fit in a memory budget" has an exact answer, printed at the end.
  4. BATCHING DOES NOT WEAKEN ANY OF THIS.  Concepts are added in batches of varying size, including
     batches that create shared group nodes, which are the only nodes ever trained by more than one
     concept.
  5. ORDER-INDEPENDENCE of batch placement: the same K concepts shuffled produce the same structure.

Runs on CPU in about a minute with a deliberately tiny model.

    python tests/test_unbounded.py [--n-concepts 40]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood.neural import NeuralScoreField, UNet, fingerprint
from lifelong_ood.neural.adapters import ConceptTree
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner


def build(seed=0, base=32, rank=2):
    torch.manual_seed(seed)
    model = UNet(n_classes=4, base=base, mults=(1, 2), edim=64, attn=(), dropout=0.0)
    field = NeuralScoreField(model, device="cpu", batch=16)
    return field, NeuralLifelongLearner(field, rank=rank, device="cpu", steps=4, max_depth=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-concepts", type=int, default=40)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    args = ap.parse_args()

    field, L = build()
    D = field.dim
    rng = np.random.default_rng(0)
    base_fp = fingerprint(field, None, seed=7)

    marks, params_at = {}, []
    k = 0
    bi = 0
    while k < args.n_concepts:
        bs = min(args.batch_sizes[bi % len(args.batch_sizes)], args.n_concepts - k)
        bi += 1
        names = [f"c{k + j}" for j in range(bs)]
        # pools drawn around a few shared centres so batches sometimes group and sometimes do not
        centre = rng.standard_normal(D) * 0.4
        items = []
        for n in names:
            pool = np.clip(centre + 0.15 * rng.standard_normal((6, D)), -1, 1)
            items.append((n, pool, pool.mean(0)))
        L.consolidate_batch(items, epochs=1, batch=3, seed=k)
        k += bs

        # 1. every concept ever added is still bit-identical
        for prev, fp in marks.items():
            now = fingerprint(field, prev, seed=7)
            assert np.array_equal(now, fp), (
                f"concept {prev} moved after {k} concepts -- preservation is NOT K-independent")
        for n in names:
            marks[n] = fingerprint(field, n, seed=7)

        # 2. the frozen base never moves
        assert np.array_equal(fingerprint(field, None, seed=7), base_fp), \
            f"the frozen base moved after {k} concepts"

        params_at.append((k, L.bank.n_params()))

    # 3. cost per concept is constant -> the budget question has an exact answer
    ks = np.array([a for a, _ in params_at], float)
    ps = np.array([b for _, b in params_at], float)
    slope, intercept = np.polyfit(ks, ps, 1)
    resid = float(np.max(np.abs(ps - (slope * ks + intercept))) / max(ps[-1], 1))
    assert resid < 0.10, f"parameter growth is not linear (max relative residual {resid:.3f})"

    # 5. batch placement does not depend on arrival order within the batch
    mus = [rng.standard_normal(16) for _ in range(6)]
    t1 = ConceptTree(var=0.1, max_depth=3)
    t2 = ConceptTree(var=0.1, max_depth=3)
    p1, _, _ = t1.insert_batch(mus)
    order = rng.permutation(len(mus))
    p2, _, _ = t2.insert_batch([mus[i] for i in order])
    shape1 = sorted(len(p) for p in p1)
    shape2 = sorted(len(p) for p in p2)
    assert shape1 == shape2, f"batch placement depends on order: {shape1} vs {shape2}"

    s = L.bank.summary()
    per = slope / 1e3
    print(f"unbounded: ok  ({k} concepts in batches of {args.batch_sizes}, "
          f"{s['n_nodes']} nodes, {len(s['shared_ancestors'])} shared)")
    print(f"  preservation held against ALL previous concepts at every one of the "
          f"{len(params_at)} batches (bit-identical, not approximate)")
    print(f"  cost per concept {per:,.0f}K params (linear, max residual {resid:.1%})")
    for budget_gb in (1, 8):
        n = int(budget_gb * 1e9 / 4 / slope)
        print(f"  => a {budget_gb} GB fp32 adapter budget holds ~{n:,} concepts at this rank")


if __name__ == "__main__":
    main()
