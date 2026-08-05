"""Same unfrozen configuration, only the TOPOLOGY differs -- does the fingerprint move?

diag_shared_node showed that with flat=True and ancestors trainable, adding a second concept moves
the shared root's real contribution (||B@A|| 5.66 -> 8.39) and changes the first concept's function
by 0.88.  Yet N3's `unfrozen` arm, which is the same freeze_ancestors=False with flat=False, reports
retention_exact = 1.00 -- bit-identical.  Only `flat` differs, so this runs both and prints which
nodes each concept's path actually contains and whether the fingerprint moves.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import CACHE
from lifelong_ood.neural import NeuralLifelongLearner, fingerprint, load_field

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

for flat in (True, False):
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    L = NeuralLifelongLearner(field, steps=20, seed=0, device=dev,
                              flat=flat, freeze_ancestors=False)
    D = field.dim
    print(f"\n=== flat={flat}, freeze_ancestors=False ===")
    marks = {}
    for i in range(4):
        pool = np.clip(rng.standard_normal((32, D)) * 0.3 + (i - 1.5) * 0.4, -1, 1)
        L.consolidate(pool, f"c{i}", mu=pool.mean(0), epochs=30, batch=16)
        marks[f"c{i}"] = fingerprint(field, f"c{i}", seed=1)
        paths = {n: L.bank.concepts[n]["path"] for n in L.bank.concepts}
        opt = len(L.bank.trainable_batch([f"c{i}"]))
        print(f"  after c{i}: paths={paths}  #trainable tensors={opt}")
    print("  fingerprint drift of each earlier concept at the end:")
    for n, m in marks.items():
        d = float(np.abs(fingerprint(field, n, seed=1) - m).max())
        print(f"    {n}: identical={np.array_equal(fingerprint(field, n, seed=1), m)}  max|delta|={d:.3e}")
