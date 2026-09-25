"""Can a merged update be prevented from overwriting what the weights already knew?

`test_merge` established both halves of the problem: folding an adapter into the base weights is
exact for the concept just learned (relative error 1.1e-06), but learning and merging a SECOND
concept then drifts the first by 1.086.  Merging is free; merging repeatedly is destructive.

This tests the fix.  Before each consolidation, sample the activations that already-known
conditionings produce and project the update's input-side factor out of that subspace, so the write
lands where nothing already known reads.  If drift collapses while the new concept is still learned,
then weights can be reconsolidated indefinitely with no per-concept parameters at all.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.neural import (NeuralLifelongLearner, activation_subspace, fingerprint,
                                 load_field)

dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = os.path.join(CACHE, "ddpm_colormnist.pt")
ck = torch.load(CK, map_location="cpu", weights_only=False)
C = ck["ema"]["cemb.weight"].numpy()

from lifelong_ood.data.datasets import load_colormnist
from lifelong_ood.neural import to_model
_d = load_colormnist(n_per_slot=20, seed=0)
_seen = np.where(_d["seen"])[0]
REAL = to_model(_d["X"][_seen[:256]])          # real images of KNOWN compositions

for energy in (None, 0.97, 0.995, 0.999, 0.9999):
    field = load_field(CK, device=dev)
    L = NeuralLifelongLearner(field, steps=20, seed=0, device=dev, flat=True)
    rng = np.random.default_rng(0)
    D = field.dim
    P = None
    if energy is not None:
        # the conditionings we already know -- sampled, not stored
        conds = [C[i] for i in range(0, min(len(C), 40), 4)]
        P = activation_subspace(L.bank, field, conds, n=4, energy=energy, seed=0, x0=REAL)
        kept = np.mean([float(np.linalg.matrix_rank(p.cpu().numpy(), tol=1e-4)) / p.shape[0]
                        for p in list(P.values())[:6]])
        tag = f"project @ energy {energy}"
    else:
        tag = "no projection (plain merge)"

    pool = np.clip(rng.standard_normal((32, D)) * 0.3 + 0.4, -1, 1)
    L.consolidate(pool, "A", mu=pool.mean(0), epochs=30, batch=16, project=P)
    first_loss = L.last_curve["final"]
    before = fingerprint(field, "A", seed=1)
    L.bank.merge_concept("A")

    pool2 = np.clip(rng.standard_normal((32, D)) * 0.3 - 0.4, -1, 1)
    L.consolidate(pool2, "B", mu=pool2.mean(0), epochs=30, batch=16, project=P)
    second_loss = L.last_curve["final"]
    L.bank.merge_concept("B")

    drift = float(np.abs(fingerprint(field, "A", seed=1) - before).max())
    extra = f", free dims {kept:.0%}" if energy is not None else ""
    print(f"  {tag:30s} concept-A drift after merging B: {drift:8.4f}   "
          f"(learned A to {first_loss:.4f}, B to {second_loss:.4f}{extra})", flush=True)

print("\n  drift is max |delta eps| on concept A.  Plain merge is the baseline to beat; a projection")
print("  that kills drift but also stops B from being learned has solved nothing, which is why both")
print("  losses are printed next to it.")
