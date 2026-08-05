"""Does this backbone's class-embedding table admit an additive (ideal-words) factorization?

The composer reads a target composition as mu + D_d + F_f + B_b in the model's OWN class-embedding
space, so its quality is bounded by how well that space factorizes.  On the original backbone the
ridge fit reached R^2 = 0.816.  A retrained backbone has a different embedding table, and if it
factorizes worse the composer degrades even though nothing about the method changed -- which is a
robustness property worth knowing rather than a bug.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.neural.idealwords import factorize, slot_factors

for name in ("ddpm_colormnist.pt",):
    p = os.path.join(CACHE, name)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    fac = factorize(C, slot_factors(slots))
    print(f"\n  {name}: {len(slots)} seen classes, embedding dim {C.shape[1]}")
    print(f"    ideal-words R^2 = {fac['r2']:.3f}   (original backbone: 0.816)")
    print(f"    per-factor sizes {fac['sizes']}")
    n = np.linalg.norm(C, axis=1)
    print(f"    ||c|| mean {n.mean():.3f} +- {n.std():.3f}")
    for i, V in enumerate(fac["V"]):
        vn = np.linalg.norm(V, axis=1)
        print(f"    factor {i}: mean direction norm {vn.mean():.3f} "
              f"(range {vn.min():.3f}-{vn.max():.3f})")
