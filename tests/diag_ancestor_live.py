"""Is a shared GROUP node functionally live, or is B still ~0 so it contributes nothing?

The question that decides whether ANY reuse experiment can show anything.  `B` is zero-initialized,
so a node can have a large `A` and still emit exactly nothing until `B` grows -- `diag_shared_node`
raised that possibility for the unfrozen arm, and if it held for group nodes then "the newcomer
inherits the family's structure" would be vacuous and every arm of N34 would be identical by
construction.

Measured instead of assumed, and the answer is the opposite of the worry: the group node carries
MORE relative magnitude than any individual leaf.  The root is the one that is genuinely inert --
it is allocated but never enters `_open` (it exists before the first insertion), so it never trains
and contributes exactly 0.

    python tests/diag_ancestor_live.py
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner/experiments")
from common import CACHE, ood_queries
from lifelong_ood.datasets import load_colormnist
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=60, seed=0)
qs = ood_queries(data, seed=0)
by = {}
for q in qs: by.setdefault((q["f"], q["b"]), []).append(q)
fam = max(by.values(), key=len)[:3]
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
L = NeuralLifelongLearner(field, steps=20, seed=0, device=dev)
names = [f"c{q['slot']}" for q in fam]
L.learn_concepts_batch([to_model(data["X"][q["query"]]).ravel() for q in fam], names,
                       n_pool=32, epochs=30)
bank = L.bank
paths = {n: bank.concepts[n]["path"] for n in names}
print("paths:", paths)
shared = set(paths[names[0]])
for n in names[1:]: shared &= set(paths[n])
print("shared nodes:", sorted(shared))
for nid in sorted({x for p in paths.values() for x in p}):
    k = str(nid)
    na = nb = nba = nw = 0.0
    for w in bank.wrapped:
        if k not in w.A: continue
        A, B = w.A[k].detach(), w.B[k].detach()
        d = torch.einsum("or,rikl->oikl", B[:, :, 0, 0], A) if w.is_conv else B @ A
        na += float(A.norm())**2; nb += float(B.norm())**2
        nba += float(d.norm())**2; nw += float(w.base.weight.detach().norm())**2
    owners = [n for n in names if nid in paths[n]]
    print(f"  node {nid:3d} owners={len(owners)}  ||A||={na**0.5:9.3f}  ||B||={nb**0.5:9.5f}  "
          f"||BA||={nba**0.5:9.5f}  ||BA||/||W||={(nba**0.5)/(nw**0.5+1e-12):.3e}")
