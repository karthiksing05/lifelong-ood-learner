"""Why does the composition survive an interference stream even with NO protection?

N9 with a real-image interference stream (RESULTS 5m fixed the pool source) still shows
`finetune-stream` matching `ours`: 0.42 accuracy / F1 0.59 against 0.42 / 0.57.  The stream demonstrably
moves the shared adapter path on real data, so the composition is surviving something that does
damage in N5.  The candidate explanation is architectural: a factored concept's semantics live in
`_emb_extra[name]`, the per-concept embedding written by the ideal-words read, which is PRIVATE and
never on any other concept's path.  Only the LoRA path is shared.  If that is right, the shared path
can move a long way while the composition is unharmed.

Measured here, in the unprotected configuration and on the same case:
  * shared ||B@A|| and the case's fingerprint, before and after the stream  -- did the shared path move?
  * the case's composition accuracy, before and after                        -- did the content survive?
  * the same accuracy with the embedding REPLACED by the null embedding      -- how much was the
    embedding carrying?  If accuracy collapses without it, it was carrying the composition.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE, ood_queries
from lifelong_ood.data.datasets import load_colormnist
from lifelong_ood.evals import composition_accuracy, fit_probes
from lifelong_ood.neural import (NeuralLifelongLearner, fingerprint, load_field, to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors

dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = os.path.join(CACHE, "ddpm_colormnist.pt")
data = load_colormnist(n_per_slot=100, seed=0)
probes = fit_probes(data, seed=0, epochs=30)
qs = ood_queries(data, n_slots=7, seed=0)
case, stream = qs[0], qs[1:]
ck = torch.load(CK, map_location="cpu", weights_only=False)
remap = ck["remap"]; slots_seen = np.array(sorted(remap))
C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]

field = load_field(CK, device=dev)
L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev, flat=True, freeze_ancestors=False)
L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 2.0])
nm = f"c{case['slot']}"
L.learn_concept_factored(to_model(data["X"][case["query"]]).ravel(), nm,
                         n_pool=48, epochs=40, inv_steps=400, seed=0)


def ba(nid):
    t = 0.0
    for w in L.bank.wrapped:
        k = str(nid)
        if k in w.A:
            A, B = w.A[k].detach(), w.B[k].detach()
            t += (float((B ** 2).sum()) * float((A ** 2).sum()) if A.dim() == 4
                  else float(((B @ A) ** 2).sum()))
    return np.sqrt(t)


def acc(tag, zero_emb=False):
    """Sampling always goes through the normal concept path; `zero_emb` only blanks the private
    embedding for the duration, so the two rows differ by that one tensor and nothing else."""
    keep = L.bank._emb_extra[nm].detach().clone()
    if zero_emb:
        L.bank._emb_extra[nm].data.zero_()
    try:
        S = to_data(L.sample_concept(nm, n=24, seed=7)).astype(np.float32)
    finally:
        L.bank._emb_extra[nm].data.copy_(keep)
    a = composition_accuracy(probes, S, case["d"], case["f"], case["b"])
    print(f"    {tag:38s} joint {a['joint']:.2f}  digit {a['digit']:.2f}  "
          f"fg {a['fg']:.2f}  bg {a['bg']:.2f}", flush=True)
    return a["joint"]


shared = sorted(set(L.bank.concepts[nm]["path"]))
before_ba = {n: ba(n) for n in shared}
before_fp = fingerprint(field, nm, seed=1)
print(f"\n  case {nm}, path {shared}")
print(f"  BEFORE the stream: shared ||B@A|| {[f'{before_ba[n]:.2f}' for n in shared]}")
a_before = acc("accuracy before the stream")

for si, q in enumerate(stream):
    idx = np.concatenate([[q["query"]], q["rest"]])[:64]
    X = to_model(data["X"][idx]).reshape(-1, field.dim)
    L.consolidate(X, f"s{q['slot']}", mu=X.mean(0), epochs=40)
after_ba = {n: ba(n) for n in shared}
moved = np.sqrt(sum((after_ba[n] - before_ba[n]) ** 2 for n in shared))
drift = float(np.abs(fingerprint(field, nm, seed=1) - before_fp).max())
print(f"\n  AFTER {len(stream)} REAL-image concepts on an unprotected shared path:")
print(f"  shared ||B@A|| {[f'{after_ba[n]:.2f}' for n in shared]}   moved {moved:.3f}   "
      f"fingerprint drift {drift:.3e}")
a_after = acc("accuracy after the stream")
a_null = acc("same, but with the embedding ZEROED", zero_emb=True)
print(f"\n  shared path moved {moved:.2f} and the function drifted {drift:.2e}, "
      f"yet accuracy went {a_before:.2f} -> {a_after:.2f}.")
print(f"  Without the private embedding it is {a_null:.2f} -- that is what the embedding was carrying.")
