"""The control that should have come first: is the BASE DDPM itself diverse?

Every diversity fix tried so far (guidance schedule, teacher covariance, pooling over queries,
stochastic sampling) left recall at 0.00.  If the backbone's own class-conditional samples -- with no
composition, no teacher, no guidance -- also fail to cover their class, then nothing downstream can
be blamed and nothing downstream can fix it.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner/experiments")
from common import CACHE
from lifelong_ood.data.datasets import load_colormnist
from lifelong_ood.evals import composition_accuracy, fit_probes, precision_recall, sample_geometry
from lifelong_ood.neural import load_field, to_data

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=100, seed=0)
probes = fit_probes(data, seed=0, epochs=30)
ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu", weights_only=False)
remap = ck["remap"]
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)

print(f"{'seen slot':>10s}{'acc':>6s}{'prec':>7s}{'recall':>8s}{'F1':>7s}"
      f"{'content div':>13s}{'real':>7s}{'ratio':>7s}")
rows = []
for s in list(remap)[:6]:
    d, f, b = int(s)//16, (int(s)%16)//4, int(s)%4
    idx = np.where(data["slot"] == int(s))[0]
    R = data["X"][idx[:100]]
    S = to_data(field.sample(n=32, steps=50, seed=0, clip=(-1,1), branch=remap[int(s)])).astype(np.float32)
    pr = precision_recall(S, R)
    g = sample_geometry(S, R[0], R)
    acc = composition_accuracy(probes, S, d, f, b)
    rows.append((pr, g, acc))
    print(f"{int(s):10d}{acc['joint']:6.2f}{pr['precision']:7.2f}{pr['recall']:8.2f}{pr['f1']:7.2f}"
          f"{g['content_diversity']:13.3f}{g['content_ref_spread']:7.3f}{g['content_ratio']:7.2f}")
m = lambda k, i: float(np.mean([r[i][k] for r in rows]))
print(f"\nBASE MODEL, class-conditional, NO composition and NO guidance:")
print(f"  precision {m('precision',0):.2f}  recall {m('recall',0):.2f}  F1 {m('f1',0):.2f}"
      f"  content div/ref {m('content_ratio',1):.2f}")
half = len(R)//2
ref = precision_recall(R[:half], R[half:])
print(f"  real-vs-real ceiling: precision {ref['precision']:.2f} recall {ref['recall']:.2f}")
print("\n  => if recall is ~0 here, the backbone cannot cover its own TRAINING classes,")
print("     and no composition or guidance change downstream can create coverage it lacks.")
