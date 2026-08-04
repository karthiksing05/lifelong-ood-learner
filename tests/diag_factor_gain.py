"""Why is background lost even when the composed embedding is CORRECT?

N8 established the isolation.  Factored inversion reads the composition right (7/8 slots recover
(d,f,b) exactly from one query), and the ideal-words ORACLE composes straight from the true labels
with no optimization at all -- yet both produce digit 0.89 and fg 0.96 against bg 0.36.  So the
failure is not in FINDING the embedding, it is in the model's response to it.

Two measurable suspects:
  1. the background factor is simply the weakest direction (norms: digit 0.785, fg 0.687, bg 0.552),
     so its contribution may be swamped in a sum;
  2. the additive model leaves 18% residual (R^2 = 0.816), which may be exactly where
     combination-specific background information lives.

This sweeps a per-factor GAIN, c = mu + a*D + b*F + g*B, to see whether background is recoverable by
amplifying the direction that is already correct.  If bg accuracy climbs with the gain, the factor is
right and merely under-weighted -- a one-parameter fix.  If it does not, background is not carried by
the additive component at all and the residual is the thing to model.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.datasets import load_colormnist
from lifelong_ood.evals import composition_accuracy, fit_probes, precision_recall, sample_geometry
from lifelong_ood.neural import load_field, sample_embedding, to_data
from lifelong_ood.neural.idealwords import factorize, slot_factors

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=100, seed=0)
probes = fit_probes(data, seed=0, epochs=30)
ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu", weights_only=False)
remap = ck["remap"]; slots = np.array(sorted(remap))
C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots]]
fac = factorize(C, slot_factors(slots))
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
qs = ood_queries(data, n_slots=6, seed=0)

def run(gains, n=24):
    A, PR, G = [], [], []
    for qi, q in enumerate(qs):
        c = fac["mu"] + gains[0]*fac["V"][0][q["d"]] + gains[1]*fac["V"][1][q["f"]] \
            + gains[2]*fac["V"][2][q["b"]]
        S = to_data(sample_embedding(field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                                     n=n, steps=50, seed=qi)).astype(np.float32)
        R = data["X"][q["rest"][:100]]
        A.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"]))
        PR.append(precision_recall(S, R)); G.append(sample_geometry(S, R[0], R))
    m = lambda k, L: float(np.mean([x[k] for x in L]))
    return (m("joint",A), m("digit",A), m("fg",A), m("bg",A),
            m("recall",PR), m("f1",PR), m("content_ratio",G))

print(f"  {'gains (d,f,bg)':>18s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}{'recall':>8s}{'F1':>6s}{'div':>6s}")
for g in [(1,1,1),(1,1,2),(1,1,3),(1,1,5),(1,0.5,3),(0.7,0.7,3),(1,1,8)]:
    r = run(g)
    print(f"  {str(g):>18s}{r[0]:6.2f}{r[1]:7.2f}{r[2]:5.2f}{r[3]:5.2f}{r[4]:8.2f}{r[5]:6.2f}{r[6]:6.2f}")
print("\n  bg climbing with the gain => the direction is right and merely under-weighted.")
