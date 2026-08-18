"""Are the displayed prototypes COLLAPSED -- one generic mode repeated -- or genuinely varied?

Raised from the Shapes3D case study, where forcing `init="random"` replaced "every prototype is
the query" with "every prototype is the same dark purple capsule", in every block, whatever the
target was.  Both are broken and neither is visible in any accuracy number.

At u=0.05 a random start is far OFF-manifold -- x_t should already look like a clean image at
abar~0.99 -- so ascent falls into whichever basin dominates, identically every time.  This
measures the spread AMONG the displayed prototypes against two scale bars: the spread among
REAL images of the class, and among unconditional samples.  Collapse is spread << both.

    python tests/diag_proto_collapse.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.data.factored import decode_slot, load_factored
from lifelong_ood.neural.ve_bridge import load_ve_field
from n58_shapes3d_results import mode_ascent
dev = "cuda" if torch.cuda.is_available() else "cpu"
field, ck = load_ve_field(os.path.join(CACHE, "ddpm_shapes3d_pre2.pt"), device=dev)
sp = ck["split"]
d = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                  value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                  fix_nuisance=sp["fix_nuisance"])
X, slots = np.asarray(d["X"], np.float32), np.asarray(d["slot"])
seen = set(int(k) for k in ck["remap"])
held = sorted(set(int(s) for s in np.unique(slots)) - seen)
def pw(A):
    return float(np.mean([np.linalg.norm(A[i]-A[j]) for i in range(len(A)) for j in range(i+1,len(A))]))
for init in ("random", "hybrid"):   # hybrid rows report only the query-agnostic half
    dq, spread = [], []
    for s in held[:3]:
        idx = np.where(slots == int(s))[0]; xq = X[int(idx[0])]
        M, src = mode_ascent(field, xq.reshape(64,64,3).transpose(2,0,1), 0.05,
                             n_starts=4, init=init, seed=9973*int(s))
        M = np.asarray(M)
        keep = [i for i,q in enumerate(src) if q == "random"] or list(range(len(M)))
        dq.append(float(np.mean(np.linalg.norm(M[keep] - xq[None], axis=1))))
        spread.append(pw(M[keep]) if len(keep) > 1 else float("nan"))
    print(f"init={init:7s} d(proto,query) {np.mean(dq):7.2f} | spread AMONG protos {np.nanmean(spread):7.2f}")
# scale bars
s0 = held[0]; R = X[np.where(slots == int(s0))[0][:6]]
print(f"scale: real-vs-real spread {pw(R):.2f}")
U = np.asarray(field.sample(n=6, steps=50, seed=3))
print(f"       unconditional-sample spread {pw(U):.2f}")
