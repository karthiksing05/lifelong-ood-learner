"""Print the Shapes3D split at a given configuration, without training anything."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lifelong_ood.shapes3d import load_shapes3d                        # noqa: E402

for n_per_slot, stride, hold in ((200, 2, 25), (120, 2, 25)):
    d = load_shapes3d(n_per_slot=n_per_slot, seed=0, value_stride=stride, n_holdout=hold)
    F = d["factors"][d["seen"]]
    print(f"n_per_slot={n_per_slot} stride={stride} holdout={hold}: sizes {d['sizes']} -> "
          f"{int(np.prod(d['sizes']))} compositions, {d['X'].shape[0]} images, "
          f"{np.bincount(d['slot']).max()} per slot; seen "
          f"{len(np.unique(d['slot'][d['seen']]))} / held {len(np.unique(d['slot'][~d['seen']]))}; "
          f"values per factor in SEEN {[len(np.unique(F[:, i])) for i in range(F.shape[1])]}",
          flush=True)
