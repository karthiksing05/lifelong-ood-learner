"""No gradient, or no sharing?  diag_pool_source showed EXACTLY zero movement on the factored path.

Zero -- not small -- means either the shared parameters get no gradient, or the concepts never
actually share a node.  These imply different claims, so this prints the paths each placement
produces and the gradient norm actually reaching the first concept's nodes.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.datasets import load_colormnist
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_model
from lifelong_ood.neural.idealwords import slot_factors

dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = os.path.join(CACHE, "ddpm_colormnist.pt")
data = load_colormnist(n_per_slot=100, seed=0)
qs = ood_queries(data, n_slots=4, seed=0)
ck = torch.load(CK, map_location="cpu", weights_only=False)
remap = ck["remap"]; slots_seen = np.array(sorted(remap))
C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]

for source in ("factored", "real"):
    field = load_field(CK, device=dev)
    L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev,
                              flat=False, freeze_ancestors=False)
    L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 2.0])
    print(f"\n=== {source} ===", flush=True)
    for i, q in enumerate(qs):
        nm = f"c{q['slot']}"
        if source == "factored":
            L.learn_concept_factored(to_model(data["X"][q["query"]]).ravel(), nm,
                                     n_pool=48, epochs=40, inv_steps=400, seed=i)
        else:
            idx = np.concatenate([[q["query"]], q["rest"]])[:48]
            X = to_model(data["X"][idx])
            L.consolidate(X.reshape(len(idx), -1), nm, mu=X.reshape(len(idx), -1).mean(0), epochs=40)
        print(f"  {nm}: path={L.bank.concepts[nm]['path']}  "
              f"n_trainable_tensors={len(L.bank.trainable_batch([nm]))}", flush=True)
    p0 = set(L.bank.concepts[f"c{qs[0]['slot']}"]["path"])
    print(f"  first concept path={sorted(p0)}")
    for nm in L.bank.concepts:
        ov = p0 & set(L.bank.concepts[nm]["path"])
        print(f"    {nm}: shares {sorted(ov)} with the first concept")
