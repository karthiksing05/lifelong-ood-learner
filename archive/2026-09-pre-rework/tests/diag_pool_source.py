"""The corrected interference diagnostic: SAME code path as N3, pool source the only variable.

`diag_interference_regime` compared a "model-conditional pool" against real images and concluded the
on-manifold hypothesis was refuted.  That comparison was not the one N3 actually runs: its pool was
drawn from a class branch and then consolidated under a DIFFERENT (fresh) concept embedding, so the
model did not already fit it -- first loss 0.0445, higher than real data's 0.0183, which is the
opposite of on-manifold.  It measured a mismatch it had introduced itself.

This runs the real thing.  Both arms use `learn_concept_factored`'s own pool (sampled from `c_new`,
the same embedding consolidation conditions on) versus real held-out images, in the unfrozen
configuration where interference is possible, and reports what actually matters:

  * first/final distillation loss                -- is there any gradient signal at all?
  * ||delta (B@A)|| summed over SHARED nodes     -- did the shared path's real contribution move?
  * fingerprint drift of the FIRST concept       -- did an earlier concept's function change?

If the factored pool shows near-floor loss, ~0 shared movement and 0 drift while real images show
all three, then composed-pool experiments cannot test forgetting: consolidation is self-distillation
onto a target the model already generates.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.data.datasets import load_colormnist
from lifelong_ood.neural import NeuralLifelongLearner, fingerprint, load_field, to_model
from lifelong_ood.neural.idealwords import slot_factors

dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = os.path.join(CACHE, "ddpm_colormnist.pt")
data = load_colormnist(n_per_slot=100, seed=0)
qs = ood_queries(data, n_slots=4, seed=0)
ck = torch.load(CK, map_location="cpu", weights_only=False)
remap = ck["remap"]; slots_seen = np.array(sorted(remap))
C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]


def ba(L, nid):
    """The node's real output contribution: ||B@A||, not ||A|| (which moves freely while B~0)."""
    tot = 0.0
    for w in L.bank.wrapped:
        k = str(nid)
        if k in w.A:
            A, B = w.A[k].detach(), w.B[k].detach()
            tot += (float((B ** 2).sum()) * float((A ** 2).sum()) if A.dim() == 4
                    else float(((B @ A) ** 2).sum()))
    return np.sqrt(tot)


for source in ("factored pool (what N3 uses)", "REAL held-out images (what N5 uses)"):
    field = load_field(CK, device=dev)
    L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev,
                              flat=False, freeze_ancestors=False)      # the `unfrozen` arm
    L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 2.0])
    print(f"\n=== {source} ===", flush=True)
    first = None
    for i, q in enumerate(qs):
        nm = f"c{q['slot']}"
        if source.startswith("factored"):
            L.learn_concept_factored(to_model(data["X"][q["query"]]).ravel(), nm,
                                     n_pool=48, epochs=40, inv_steps=400, seed=i)
        else:
            idx = np.concatenate([[q["query"]], q["rest"]])[:48]
            X = to_model(data["X"][idx])
            L.consolidate(X.reshape(len(idx), -1), nm, mu=X.reshape(len(idx), -1).mean(0), epochs=40)
        cur = L.last_curve
        shared = sorted({n for c in L.bank.concepts.values() for n in c["path"]}
                        & set(L.bank.concepts[f"c{qs[0]['slot']}"]["path"]))
        now = {n: ba(L, n) for n in shared}
        if first is None:
            first = dict(mark=fingerprint(field, nm, seed=1), ba=dict(now), name=nm)
            print(f"  learn {nm}: loss {cur['first']:.4f} -> {cur['final']:.4f}   "
                  f"shared ||B@A|| {[f'{v:.2f}' for v in now.values()]}", flush=True)
        else:
            d = float(np.abs(fingerprint(field, first["name"], seed=1) - first["mark"]).max())
            moved = np.sqrt(sum((now[n] - first["ba"].get(n, 0.0)) ** 2 for n in now))
            print(f"  learn {nm}: loss {cur['first']:.4f} -> {cur['final']:.4f}   "
                  f"shared ||B@A|| {[f'{v:.2f}' for v in now.values()]}   "
                  f"delta-shared {moved:.3f}   {first['name']} drift {d:.3e}", flush=True)
