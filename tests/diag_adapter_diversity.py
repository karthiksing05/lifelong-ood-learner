"""Does distillation RECOVER the diversity that guided sampling destroys?

Established by the preceding diagnostics:
  * the base DDPM, class-conditional with no guidance, covers its classes (recall 0.90, content
    diversity 0.97x real) -- the backbone is not the problem;
  * every teacher-guided pool collapses (recall 0.00) regardless of guidance schedule, teacher
    covariance, stochastic sampling, or pooling across queries.

Guiding toward an analytic Gaussian contracts trajectories toward its single mean, and no knob on
the guidance changes that -- it is what a unimodal teacher IS.

But the paper's own table says +LoRA has HIGHER recall than PoE (88.0 vs 80.7 faithful, 78.8 vs 69.8
general).  That ordering is the clue: after distillation you no longer sample the analytic teacher,
you sample the MODEL conditioned on the new concept -- and the model's conditional machinery is the
thing we just measured at recall 0.90.  Distillation would then be not a convenience for reuse but
the step that restores diversity.

This measures exactly that: pool diversity going IN, adapter-sample diversity coming OUT.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.datasets import load_colormnist
from lifelong_ood.evals import composition_accuracy, fit_probes, precision_recall, sample_geometry, save_grid
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=100, seed=0)
probes = fit_probes(data, seed=0, epochs=30)
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev)
qs = ood_queries(data, n_slots=3, seed=0)

print(f"{'slot':>6s}{'stage':>22s}{'acc':>6s}{'prec':>7s}{'recall':>8s}{'F1':>7s}{'content div/ref':>17s}")
agg = {}
for q in qs:
    R = data["X"][q["rest"][:100]]
    xq = to_model(data["X"][q["query"]]).ravel()
    t = L.compose(xq, u_grid=(0.1,0.2,0.3,0.4), n_starts=16, iters=120)
    pool = L.sample_teacher(t, n=96, seed=0)
    P01 = to_data(pool).astype(np.float32)
    L.consolidate(pool, f"c{q['slot']}", mu=t["mu"], epochs=60)
    S01 = to_data(L.sample_concept(f"c{q['slot']}", n=32, seed=0)).astype(np.float32)
    for tag, X in (("pool (teacher-guided)", P01), ("adapter (distilled)", S01)):
        pr = precision_recall(X, R); g = sample_geometry(X, data["X"][q["query"]], R)
        a = composition_accuracy(probes, X, q["d"], q["f"], q["b"])
        agg.setdefault(tag, []).append((a["joint"], pr["precision"], pr["recall"], pr["f1"], g["content_ratio"]))
        print(f"{q['slot']:6d}{tag:>22s}{a['joint']:6.2f}{pr['precision']:7.2f}{pr['recall']:8.2f}"
              f"{pr['f1']:7.2f}{g['content_ratio']:17.2f}")
    save_grid(np.concatenate([[data["X"][q["query"]]], P01[:7], S01[:8]]),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "figures", f"diag_adapter_{q['slot']}.png"), ncol=8,
              title=f"slot {q['slot']}: query + 7 teacher-guided pool | 8 distilled-adapter samples")
print()
for tag, v in agg.items():
    v = np.array(v)
    print(f"  MEAN {tag:>22s}: acc {v[:,0].mean():.2f} precision {v[:,1].mean():.2f} "
          f"recall {v[:,2].mean():.2f} F1 {v[:,3].mean():.2f} content div/ref {v[:,4].mean():.2f}")
