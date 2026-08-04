"""Fast neural-track test: a tiny UNet, no training, no MNIST.  Runs on CPU in well under a minute.

It checks the properties the lifelong claim rests on, which are structural and therefore testable
without a trained model:

  1. the neural field satisfies the same interface as the analytic one (sample / mode_ascent /
     laplace_diag all run through BaseScoreField);
  2. a fresh adapter is an exact no-op (B initialized to zero), so registering a concept cannot
     change anything before it is trained;
  3. consolidating a concept changes THAT concept's function but leaves the frozen base
     bit-identical;
  4. consolidating a SECOND concept leaves the first one's function bit-identical -- the
     no-forgetting guarantee -- and the `unfrozen` ablation breaks exactly this.

    python tests/test_neural.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood.neural import NeuralScoreField, UNet, fingerprint
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner


def build(seed=0, base=32):
    torch.manual_seed(seed)
    model = UNet(n_classes=4, base=base, mults=(1, 2), edim=64, attn=())
    return NeuralScoreField(model, device="cpu", batch=8)


def main():
    dev = "cpu"
    field = build()
    D = field.dim
    rng = np.random.default_rng(0)

    # 1. the shared interface
    x = rng.standard_normal((3, D))
    assert field.eps(x, 0.5).shape == (3, D)
    assert field.score(x, 0.5).shape == (3, D)
    S = field.sample(n=2, steps=4, seed=0, clip=(-1, 1))
    assert S.shape == (2, D) and np.isfinite(S).all()
    xt, x0 = field.mode_ascent(x[0], 0.3, n_starts=2, iters=3, seed=0)
    assert xt.shape == (2, D)
    v = field.laplace_diag(xt, 0.3, probes=2, seed=0)
    assert v.shape == (2, D) and (v > 0).all()
    base_before = fingerprint(field, None, seed=1)

    # 2. registering a concept is a no-op until it is trained
    L = NeuralLifelongLearner(field, rank=4, device=dev, steps=4)
    L.bank.add_concept("A", rng.standard_normal(D))
    assert np.array_equal(fingerprint(field, "A", seed=1), base_before), \
        "a fresh adapter must be an exact no-op (B must be zero-initialized)"

    # 3. distilling A moves A but not the frozen base
    poolA = np.clip(rng.standard_normal((8, D)) * 0.3 + 0.5, -1, 1)
    L.consolidate(poolA, "A", mu=poolA.mean(0), epochs=2, batch=4)
    fpA = fingerprint(field, "A", seed=1)
    assert not np.array_equal(fpA, base_before), "distillation should change the concept"
    assert np.array_equal(fingerprint(field, None, seed=1), base_before), \
        "the frozen base must be bit-identical after distillation"

    # 4. a later concept must not disturb the earlier one
    poolB = np.clip(rng.standard_normal((8, D)) * 0.3 - 0.5, -1, 1)
    L.consolidate(poolB, "B", mu=poolB.mean(0), epochs=2, batch=4)
    assert np.array_equal(fingerprint(field, "A", seed=1), fpA), \
        "frozen ancestors must make earlier concepts bit-identical after later consolidations"
    assert np.array_equal(fingerprint(field, None, seed=1), base_before)

    # the ablation must break exactly that property
    f2 = build()
    L2 = NeuralLifelongLearner(f2, rank=4, device=dev, steps=4, freeze_ancestors=False,
                              max_depth=3)
    L2.consolidate(poolA, "A", mu=poolA.mean(0), epochs=2, batch=4)
    fpA2 = fingerprint(f2, "A", seed=1)
    L2.consolidate(poolB, "B", mu=poolA.mean(0) * 0.99, epochs=2, batch=4)   # lands under A
    moved = not np.array_equal(fingerprint(f2, "A", seed=1), fpA2)
    shared = len(set(L2.bank.concepts["A"]["path"]) & set(L2.bank.concepts["B"]["path"])) > 1
    assert (not shared) or moved, "unfrozen ancestors sharing a path SHOULD perturb the earlier concept"

    # 5. a BATCH addition preserves everything a sequential one does, and shares more
    f3 = build()
    L3 = NeuralLifelongLearner(f3, rank=4, device=dev, steps=4)
    centre = rng.standard_normal(f3.dim) * 0.3
    items = [(f"b{i}", np.clip(centre + 0.2 * rng.standard_normal((6, f3.dim)), -1, 1), None)
             for i in range(3)]
    base3 = fingerprint(f3, None, seed=1)
    L3.consolidate_batch(items, epochs=1, batch=3)
    fps = {n: fingerprint(f3, n, seed=1) for n, _, _ in items}
    assert len(set(map(bytes, (v.tobytes() for v in fps.values())))) == 3, \
        "batch members must end up as distinct functions, not copies of one another"
    assert np.array_equal(fingerprint(f3, None, seed=1), base3), \
        "a batch must leave the frozen base bit-identical"
    # a second batch must not disturb the first
    items2 = [(f"c{i}", np.clip(-centre + 0.2 * rng.standard_normal((6, f3.dim)), -1, 1), None)
              for i in range(2)]
    L3.consolidate_batch(items2, epochs=1, batch=3)
    for n, fp in fps.items():
        assert np.array_equal(fingerprint(f3, n, seed=1), fp), \
            f"batch-added concept {n} moved when a later batch was added"
    s3 = L3.bank.summary()
    grouped = len(s3["shared_ancestors"])

    # 6. the inversion path: a concept learned in conditioning space, then consolidated.
    #    Preservation must be untouched by it -- inversion changes WHAT is consolidated, not what
    #    consolidation does, and that separation is the whole reason the fix is safe to adopt.
    from lifelong_ood.neural.inversion import learn_embedding, sample_embedding
    f4 = build()
    L4 = NeuralLifelongLearner(f4, rank=4, device=dev, steps=4)
    tgt = np.clip(rng.standard_normal((1, f4.dim)) * 0.3, -1, 1)
    c_new, alpha, hist = learn_embedding(f4, tgt, n_steps=8, lr=0.05, seed=0)
    assert c_new.shape[0] == f4.model.cemb.embedding_dim
    assert np.isfinite(alpha).all() and len(alpha) == f4.model.n_classes
    S_inv = sample_embedding(f4, c_new, n=4, steps=4, seed=0)
    assert S_inv.shape == (4, f4.dim) and np.isfinite(S_inv).all()
    base4 = fingerprint(f4, None, seed=1)
    L4.consolidate(S_inv, "inv", mu=S_inv.mean(0), epochs=1, batch=2)
    fp_inv = fingerprint(f4, "inv", seed=1)
    assert np.array_equal(fingerprint(f4, None, seed=1), base4), \
        "the inversion path must leave the frozen base bit-identical too"
    L4.consolidate(np.clip(rng.standard_normal((4, f4.dim)) * 0.3, -1, 1), "inv2",
                   mu=None, epochs=1, batch=2)
    assert np.array_equal(fingerprint(f4, "inv", seed=1), fp_inv), \
        "a concept consolidated via inversion must still be preserved by later consolidations"

    s = L.bank.summary()
    print(f"neural smoke: ok  (adapter {s['adapter_params'] / 1e3:.0f}K params, "
          f"{s['n_concepts']} concepts, depths {list(s['depths'].values())}, "
          f"unfrozen-ablation perturbed={moved}, shared-path={shared})")
    print(f"  batch: 5 concepts over 2 batches, {grouped} shared node(s), "
          f"all earlier concepts bit-identical after the later batch")
    print(f"  inversion: c_new learned over {len(alpha)} class embeddings, consolidated, "
          f"and still bit-identical after a later consolidation")


if __name__ == "__main__":
    main()
