"""Folding an orthogonal delta into the base weights must be exact, like the LoRA fold.

`R(Wx + b) = (RW)x + Rb`, so there is no approximation involved -- but the einsum has to index the
right axis, and a silent error there would show up later as "orthogonal deltas merge badly" when the
real fault was the fold.  This checks the function against its own definition before any experiment
relies on it.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.neural import NeuralLifelongLearner, load_field
from lifelong_ood.neural.alternatives import OrthoDelta, OrthoHook

dev = "cuda" if torch.cuda.is_available() else "cpu"
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
L = NeuralLifelongLearner(field, steps=20, seed=0, device=dev, flat=True)
reg = {"active": None}
for i, w in enumerate(L.bank.wrapped):
    w.register_forward_hook(OrthoHook(reg, i))

od = OrthoDelta(L.bank.wrapped, block=16, device=dev)
with torch.no_grad():                                   # a non-trivial rotation
    for S in od.S:
        S.normal_(0, 0.15)
R0 = [od.rotation(i).clone() for i in range(3)]
orth = max(float((R @ R.transpose(1, 2) - torch.eye(R.shape[1], device=R.device)).abs().max())
           for R in R0)
print(f"  rotations are orthogonal: max |R R^T - I| = {orth:.3e}")

X = np.random.default_rng(0).standard_normal((4, field.dim)) * 0.5
reg["active"] = od
hooked = field.eps(X, 0.5)
reg["active"] = None
base_before = field.eps(X, 0.5)

od.merge_into(L.bank.wrapped)                           # fold, then the hook is off
merged = field.eps(X, 0.5)

err = float(np.abs(merged - hooked).max())
rel = err / max(float(np.abs(hooked).max()), 1e-12)
moved = float(np.abs(merged - base_before).max())
print(f"  rotation ON (hook) vs MERGED into weights: max |delta| {err:.3e}  relative {rel:.3e}")
print(f"  merged vs untouched base: max |delta| {moved:.3e}  (must be > 0, or nothing was folded)")
ok = rel < 1e-4 and moved > 1e-3
print(f"\northo merge: {'ok -- the fold is exact' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
