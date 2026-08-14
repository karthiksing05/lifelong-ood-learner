"""When does forgetting actually happen?  Measure the weight change consolidation causes.

Three experiments have now shown the same puzzle: with pools drawn from the MODEL'S OWN conditional,
the `unfrozen` arm does not forget (N3), and unprotected fine-tuning matches us in the showdown even
after a 16-concept stream (N9) -- yet on REAL held-out images forgetting is large and reproducible
(N5: BWT -0.134 +/- 0.023 over 12 tasks).  Both cannot be "how much interference there is"; something
about the two regimes differs.

The hypothesis, from the distillation losses: a pool drawn from the model's own conditional is
already nearly on-manifold, so fitting it barely moves any weight, and shared parameters that barely
move cannot destroy what depends on them.  Real held-out images sit further from what the model
currently generates, so fitting them moves weights a lot, and THAT is when sharing becomes dangerous.

This measures it directly: consolidate the same concept from (a) a model-conditional pool and (b)
real images of the same composition, and report the starting loss and the resulting parameter
displacement.  If the hypothesis holds, (b) moves weights by a large multiple of (a), and the
no-forgetting guarantee is insurance that matters exactly when the incoming data is far from what
the model already produces.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.data.datasets import load_colormnist
from lifelong_ood.evals import fit_probes
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_model
from lifelong_ood.neural.idealwords import slot_factors

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=100, seed=0)
ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu", weights_only=False)
remap = ck["remap"]; slots = np.array(sorted(remap))
C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
qs = ood_queries(data, n_slots=4, seed=0)

def displacement(pool, tag, epochs=40):
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev, flat=True,
                              freeze_ancestors=False)      # shared root IS trainable
    L.attach_factorization(C, slot_factors(slots), gains=[1, 1, 2.0])
    L.consolidate(pool, "a", mu=np.asarray(pool).mean(0), epochs=1, batch=16)   # register
    shared = [p for w in L.bank.wrapped for k in ("0",) if k in w.A for p in (w.A[k], w.B[k])]
    before = [p.detach().clone() for p in shared]
    L.consolidate(pool, "b", mu=np.asarray(pool).mean(0), epochs=epochs, batch=16)
    d = float(np.sqrt(sum(float(((p - b) ** 2).sum()) for p, b in zip(shared, before))))
    n = float(np.sqrt(sum(float((b ** 2).sum()) for b in before))) + 1e-9
    first = L.last_curve["first"]; final = L.last_curve["final"]
    return dict(tag=tag, first_loss=first, final_loss=final, disp=d, rel=d / n)

print(f"  {'source of the pool':34s}{'first loss':>12s}{'final loss':>12s}"
      f"{'||delta shared||':>18s}")
rows = []
for q in qs[:3]:
    xq = to_model(data["X"][q["query"]]).ravel()
    field0 = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    L0 = NeuralLifelongLearner(field0, steps=50, seed=0, device=dev)
    L0.attach_factorization(C, slot_factors(slots), gains=[1, 1, 2.0])
    c_new, _, _ = L0.compose_factored(xq, inv_steps=400, seed=0)
    from lifelong_ood.neural.idealwords import sample_factored_pool
    gen = sample_factored_pool(field0, c_new, n=64, steps=50, seed=0)
    real = to_model(data["X"][q["rest"][:64]])
    for pool, tag in ((gen, "model-conditional pool"), (real, "REAL held-out images")):
        r = displacement(pool, tag)
        rows.append(r)
        print(f"  {tag:34s}{r['first_loss']:12.4f}{r['final_loss']:12.4f}{r['disp']:18.4f}")
m = lambda t, k: float(np.mean([r[k] for r in rows if r["tag"] == t]))
print(f"\n  MEAN model-conditional : first {m('model-conditional pool','first_loss'):.4f}  "
      f"displacement {m('model-conditional pool','disp'):.4f}")
print(f"  MEAN real images       : first {m('REAL held-out images','first_loss'):.4f}  "
      f"displacement {m('REAL held-out images','disp'):.4f}")
r = m('REAL held-out images','disp') / max(m('model-conditional pool','disp'), 1e-9)
print(f"  => real data moves shared weights {r:.1f}x as far")
print("     If that ratio is large, freezing is insurance that matters exactly when the incoming")
print("     data is far from what the model already generates -- which is the N5 regime, and is")
print("     why the composed-pool experiments (N3/N9) show no forgetting for anyone.")
