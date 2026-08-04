"""Does the teacher's prior over trained classes express the COMPOSITION?

Earlier version asked whether the top-weighted classes individually share the target's factors.
That is the wrong question on this benchmark.  The held-out slots are held out as (fg, bg) PAIRS, so
a class matching the target's foreground necessarily has a different background -- no single trained
class can share both, and demanding it measures an impossibility.

The right question is whether alpha puts mass on BOTH constituent groups: some classes supplying the
foreground, others supplying the background.  That is what a composition in class space looks like,
and it is what the inversion needs as a starting point.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.datasets import load_colormnist
from lifelong_ood.neural import NeuralLifelongLearner, load_field, teacher_init_alpha, to_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=100, seed=0)
ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu", weights_only=False)
slots = sorted(ck["remap"])
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev)
qs = ood_queries(data, n_slots=4, seed=0)
teachers = [(q, L.compose(to_model(data["X"][q["query"]]).ravel(), u_grid=(0.1,0.2,0.3,0.4),
                          n_starts=16, iters=120)) for q in qs]

print(f"  {'prior':34s}{'mass on fg-match':>18s}{'mass on bg-match':>18s}{'both':>8s}{'balance':>9s}")
print(f"  {'(chance)':34s}{0.25:18.2f}{0.25:18.2f}{0.06:8.2f}{1.00:9.2f}")
for tag, kw in (("global (original)", dict(regions=False)),
                ("per-factor PRODUCT", dict(regions=True, combine="product")),
                ("per-factor MIXTURE", dict(regions=True, combine="mixture"))):
    for T in (0.005, 0.001):
        mf = mb = mboth = 0.0
        for q, t in teachers:
            a = teacher_init_alpha(field, t, to_model(data["X"][q["query"]]), temperature=T,
                                   seed=0, **kw)
            sl = np.array(slots)
            fgm = ((sl % 16)//4 == q["f"]); bgm = (sl % 4 == q["b"])
            mf += a[fgm].sum(); mb += a[bgm].sum(); mboth += a[fgm & bgm].sum()
        n = len(teachers)
        bal = min(mf, mb) / max(max(mf, mb), 1e-9)
        print(f"  {tag + f' T={T}':34s}{mf/n:18.2f}{mb/n:18.2f}{mboth/n:8.2f}{bal:9.2f}")
print("\n  balance = min(fg,bg)/max(fg,bg): 1.0 means alpha covers both constituents equally.")
print("  'both' is near zero by construction -- the (fg,bg) pair is exactly what is held out.")
