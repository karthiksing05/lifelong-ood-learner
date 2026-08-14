"""Does the backbone's loss RESPOND to its class label?  The one number that says whether a
conditional model is actually conditional -- measured on TRAINING data, where it cannot fail for any
honest reason.

Shapes3D trained twice (300 sparse classes, then 100 dense ones at ColorMNIST's exact density) and
both times produced excellent samples that ignore the label: seen-slot joint 0.000 then 0.070 against
a 0.010 chance rate, embedding pairwise cosine 0.007.  Before a third retrain, find out WHERE it
breaks, and compare against backbones that demonstrably work.

The statistic is deliberately blunt:

    gap = E[ loss(x, WRONG class) ] - E[ loss(x, RIGHT class) ]     on images the model TRAINED on

If a model uses its conditioning, the right class explains its own images better than a random other
class, so gap > 0 and grows with how much the label tells you.  If gap ~ 0 the label is decorative,
whatever the sample quality looks like.  Reported next to:

    null gap     loss under the CFG null token minus loss under the right class -- how much the
                 conditioning channel carries AT ALL
    emb drift    RMS distance of the embedding table from a fresh init at the same scale, i.e.
                 whether gradient ever reached the table
    pairwise cos mean cosine between class embeddings -- near 0 means they never differentiated

Run on every conditional backbone in the cache, so "broken" is defined against models that work
rather than against intuition.

    python tests/diag_conditioning_gap.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.neural.ddpm import load_field, to_model              # noqa: E402
from lifelong_ood.neural.router import conditional_losses              # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")

# (checkpoint, loader) -- each loader returns (X01 rows, slot ids)
CASES = [
    ("ddpm_colormnist.pt", "colormnist"),
    ("ddpm_colormnist_nobg3.pt", "colormnist"),
    ("ddpm_shapes3d.pt", "shapes3d"),
]


def load_data(kind, n_per_slot=8):
    if kind == "colormnist":
        from lifelong_ood.data.datasets import load_colormnist
        d = load_colormnist(n_per_slot=n_per_slot, seed=0)
        return np.asarray(d["X"]), np.asarray(d["slot"])
    from lifelong_ood.data.factored import load_factored
    d = load_factored("shapes3d", n_per_slot=max(n_per_slot, 40), seed=0,
                      value_stride=2, n_holdout=25)
    return np.asarray(d["X"]), np.asarray(d["slot"])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    print(f"{'backbone':32s} {'classes':>8} {'gap':>10} {'null gap':>10} "
          f"{'emb drift':>10} {'pair cos':>9}")
    print("-" * 84)
    for name, kind in CASES:
        p = os.path.join(CACHE, name)
        if not os.path.exists(p):
            print(f"{name:32s} {'-':>8} {'(absent)':>10}")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        remap = ck["remap"]
        W = ck["ema"]["cemb.weight"].numpy()
        C = W[[remap[int(s)] for s in sorted(remap)]]
        field = load_field(p, device=dev)
        X, slots = load_data(kind)
        trained = np.array(sorted(remap))

        gaps, nulls = [], []
        null_row = W[ck.get("null_id", len(W) - 1)]
        for s in rng.permutation(trained)[:12]:
            idx = np.where(slots == int(s))[0][:4]
            if not len(idx):
                continue
            right = C[list(trained).index(int(s))]
            wrong = C[int(rng.integers(len(C)))]
            for i in idx:
                x = to_model(X[i]).ravel()
                L = conditional_losses(field, x, np.stack([right, wrong, null_row]),
                                       u_grid=(0.15, 0.30, 0.45), n_noise=4, seed=0)
                gaps.append(float(L[1] - L[0]))
                nulls.append(float(L[2] - L[0]))
        # how far the table moved from a comparable random init
        init = rng.normal(0, float(np.std(C)), size=C.shape)
        drift = float(np.sqrt(((C - C.mean(0)) ** 2).sum(1)).mean()
                      / (np.sqrt(((init - init.mean(0)) ** 2).sum(1)).mean() + 1e-12))
        U = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        cos = float((U @ U.T)[~np.eye(len(U), dtype=bool)].mean())
        print(f"{name:32s} {len(C):>8} {np.mean(gaps):>10.5f} {np.mean(nulls):>10.5f} "
              f"{drift:>10.3f} {cos:>9.4f}", flush=True)

    print("\n[gap] A conditional model has gap > 0 on its OWN training images: the right class")
    print("[gap] explains them better than a random wrong one. gap ~ 0 means the label is")
    print("[gap] decorative no matter how good the samples look.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
