"""Can the model's class-embedding span express a held-out composition AT ALL?

Twelve routes have now failed to recover background fidelity (guidance schedules, teacher
covariance, pooling, ancestral sampling, distillation, anchoring, region-weighted loss, mu_T as
target, longer optimization, CFG on the embedding, per-region composition of conditionals,
nearest-class).  Background sits at 0.39-0.55 for every inversion variant against 0.76 for
teacher-guided sampling.  Before trying a thirteenth, establish the ceiling.

The ORACLE fits alpha on 64 REAL held-out images of the target slot -- data the method never gets --
so it measures what the span itself can do, with the inversion's data problem removed.

  If the oracle succeeds, the span is expressive enough and the failure is that ONE query
  underdetermines 120 parameters: a data problem, and the teacher should be able to help.
  If the oracle fails, no combination of trained class embeddings produces this composition, and
  conditioning in that span is the wrong vehicle no matter how it is optimized.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.data.datasets import load_colormnist
from lifelong_ood.evals import composition_accuracy, fit_probes, precision_recall, sample_geometry
from lifelong_ood.neural import (learn_embedding, load_field, sample_embedding, to_data, to_model)

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = load_colormnist(n_per_slot=100, seed=0)
probes = fit_probes(data, seed=0, epochs=30)
field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
qs = ood_queries(data, n_slots=6, seed=0)

print(f"  {'slot':>5s}{'target':>28s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
      f"{'recall':>8s}{'F1':>6s}{'div/ref':>9s}")
rows = []
for q in qs:
    R = data["X"][q["rest"][:100]]
    # ORACLE: fit on many REAL images of the held-out slot
    tgt = to_model(R[:64])
    c, a, _ = learn_embedding(field, tgt, n_steps=600, lr=0.05, seed=0)
    S = to_data(sample_embedding(field, c, n=32, steps=50, seed=0)).astype(np.float32)
    acc = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
    pr = precision_recall(S, R); g = sample_geometry(S, R[0], R)
    rows.append((acc, pr, g))
    print(f"  {q['slot']:5d}{f'digit {q["d"]} fg{q["f"]} bg{q["b"]}':>28s}"
          f"{acc['joint']:6.2f}{acc['digit']:7.2f}{acc['fg']:5.2f}{acc['bg']:5.2f}"
          f"{pr['recall']:8.2f}{pr['f1']:6.2f}{g['content_ratio']:9.2f}")
m = lambda k, i: float(np.mean([r[i][k] for r in rows]))
print(f"\n  ORACLE (alpha fitted on 64 REAL images of the held-out slot):")
print(f"    acc {m('joint',0):.2f}  digit {m('digit',0):.2f}  fg {m('fg',0):.2f}  bg {m('bg',0):.2f}"
      f"  recall {m('recall',1):.2f}  F1 {m('f1',1):.2f}  content div/ref {m('content_ratio',2):.2f}")
print(f"  for comparison: inversion from ONE query reaches acc 0.23, bg 0.49, F1 0.32")
print(f"                  base model on a SEEN class reaches acc 0.97, bg 1.00, F1 0.96")
print("\n  => high oracle accuracy means the span CAN express these compositions and the")
print("     bottleneck is inverting from a single query; low means the span cannot, and")
print("     conditioning in it is the wrong vehicle however it is optimized.")
