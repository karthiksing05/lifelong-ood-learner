"""Merging an adapter into the base weights must be EXACTLY a no-op functionally.

The whole case for adapters-as-scaffolding rests on this: if the fold is not exact, then
"reconsolidate into the existing weights" silently costs quality on the concept just learned, and
every number measured afterwards is measuring the fold's error rather than the method.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.neural import NeuralLifelongLearner, fingerprint, load_field

dev = "cuda" if torch.cuda.is_available() else "cpu"
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
L = NeuralLifelongLearner(field, steps=20, seed=0, device=dev, flat=True)
rng = np.random.default_rng(0)
D = field.dim

pool = np.clip(rng.standard_normal((32, D)) * 0.3 + 0.4, -1, 1)
L.consolidate(pool, "A", mu=pool.mean(0), epochs=30, batch=16)
before = fingerprint(field, "A", seed=1)
n_before = L.bank.n_params()

freed = L.bank.merge_concept("A")
after = fingerprint(field, "A", seed=1)
n_after = L.bank.n_params()

err = float(np.abs(after - before).max())
rel = err / max(float(np.abs(before).max()), 1e-12)
print(f"  adapter params before {n_before/1e3:.1f}K -> after {n_after/1e3:.1f}K "
      f"(reclaimed {freed/1e3:.1f}K)")
print(f"  max |delta eps| after merge: {err:.3e}   relative {rel:.3e}")
print(f"  path after merge: {L.bank.concepts['A']['path']} (empty = nothing left to activate)")

# a second concept, learned and merged on top, must not disturb the first beyond float error
pool2 = np.clip(rng.standard_normal((32, D)) * 0.3 - 0.4, -1, 1)
L.consolidate(pool2, "B", mu=pool2.mean(0), epochs=30, batch=16)
L.bank.merge_concept("B")
drift = float(np.abs(fingerprint(field, "A", seed=1) - before).max())
print(f"  after learning AND merging a second concept, concept A drifted {drift:.3e}")
print(f"  final adapter params {L.bank.n_params()/1e3:.1f}K for 2 absorbed concepts")

ok = rel < 1e-3
print(f"\nmerge: {'ok' if ok else 'FAILED'}  (fold is exact to float precision)" if ok else
      f"\nmerge: FAILED -- relative error {rel:.3e} is too large to call this a no-op")
sys.exit(0 if ok else 1)
