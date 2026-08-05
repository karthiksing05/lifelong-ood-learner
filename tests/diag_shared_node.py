"""Does the shared node's LoRA delta ever become REAL, or does the leaf absorb everything?

The open contradiction from RESULTS 5m: N3 reports the `unfrozen` arm's concepts as bit-identical
while its shared node sits in the optimizer and demonstrably moves.  Both can only be true if the
shared node's OUTPUT contribution stays ~zero -- `B` is zero-initialized, so `A` can wander without
changing anything the module emits until `B` grows.

This inspects exactly that after two consolidations in the unfrozen configuration:
  ||A||, ||B|| and ||B@A|| for the shared root and for each leaf,
  and whether concept 1's fingerprint actually moves when concept 2 is added.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.neural import NeuralLifelongLearner, fingerprint, load_field

dev = "cuda" if torch.cuda.is_available() else "cpu"
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
L = NeuralLifelongLearner(field, steps=20, seed=0, device=dev, flat=True, freeze_ancestors=False)
rng = np.random.default_rng(0)
D = field.dim
poolA = np.clip(rng.standard_normal((32, D)) * 0.3 + 0.4, -1, 1)
poolB = np.clip(rng.standard_normal((32, D)) * 0.3 - 0.4, -1, 1)

L.consolidate(poolA, "A", mu=poolA.mean(0), epochs=40, batch=16)
fpA = fingerprint(field, "A", seed=1)

def norms(nid):
    a = b = ba = 0.0
    for w in L.bank.wrapped:
        k = str(nid)
        if k in w.A:
            A, B = w.A[k].detach(), w.B[k].detach()
            a += float((A ** 2).sum()); b += float((B ** 2).sum())
            if A.dim() == 4:                       # conv: contribution is B (1x1) applied after A
                ba += float((B ** 2).sum()) * float((A ** 2).sum())
            else:
                ba += float(((B @ A) ** 2).sum())
    return np.sqrt(a), np.sqrt(b), np.sqrt(ba)

print(f"  {'node':>10s}{'||A||':>12s}{'||B||':>12s}{'||B@A||':>12s}")
for nid in sorted({n for c in L.bank.concepts.values() for n in c["path"]}):
    a, b, ba = norms(nid)
    print(f"  {nid:>10d}{a:12.4f}{b:12.4f}{ba:12.6f}")

L.consolidate(poolB, "B", mu=poolB.mean(0), epochs=40, batch=16)
print("\n  after adding a SECOND concept (shared root trainable):")
print(f"  {'node':>10s}{'||A||':>12s}{'||B||':>12s}{'||B@A||':>12s}")
for nid in sorted({n for c in L.bank.concepts.values() for n in c["path"]}):
    a, b, ba = norms(nid)
    print(f"  {nid:>10d}{a:12.4f}{b:12.4f}{ba:12.6f}")

moved = not np.array_equal(fingerprint(field, "A", seed=1), fpA)
delta = float(np.abs(fingerprint(field, "A", seed=1) - fpA).max())
print(f"\n  concept A's function moved: {moved}   max |delta eps| = {delta:.3e}")
print("  => if ||B@A|| for the shared root stays ~0 while the leaves grow, the leaf absorbs")
print("     everything and 'unfrozen' is unfrozen in name only on this configuration.")
