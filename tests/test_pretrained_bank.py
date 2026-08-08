"""The adapter bank must attach to a PRETRAINED UNet without changing what it already computes.

Three assertions, in the order they can fail:

  1. an injected bank with NO active path reproduces the pretrained model exactly -- if this fails,
     every number measured on the adapted model is measured on a different model;
  2. a freshly registered concept is an exact no-op before training (B is zero-initialized), so a
     concept starts from the pretrained behaviour rather than from a perturbation of it;
  3. training one concept leaves a previously consolidated concept BIT-IDENTICAL, which is the
     preservation claim, restated on a model we did not train.

Run:  python tests/test_pretrained_bank.py            (needs the cached google/ddpm checkpoint)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood.neural.adapters import HierarchicalAdapterBank   # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained         # noqa: E402

MODEL = os.environ.get("LOL_PRETRAINED", "google/ddpm-cifar10-32")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, img = load_pretrained(MODEL, device=dev, batch=4)
    print(f"[test] {MODEL}: {img}x{img}, emb_dim {field.emb_dim}, "
          f"{sum(p.numel() for p in field.unet.parameters()) / 1e6:.1f}M params", flush=True)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, field.dim)) * 0.5
    u = 0.4
    ref = field.eps(X, u)                                   # before any injection

    bank = HierarchicalAdapterBank(field.model, rank=4, device=dev)
    field.bank = bank
    after = field.eps(X, u)
    d0 = float(np.abs(after - ref).max())
    print(f"[test] injected, no active path      : max|delta| = {d0:.3e}")
    assert d0 == 0.0, "injection alone changed the pretrained model"

    bank.add_concept("c0", np.zeros(field.dim))
    with bank.activate("c0"):
        fresh = field.eps(X, u)
    d1 = float(np.abs(fresh - ref).max())
    print(f"[test] fresh concept, untrained      : max|delta| = {d1:.3e}")
    assert d1 == 0.0, "a fresh concept is not a no-op"

    # give c0 a nonzero delta, snapshot it, then consolidate a SECOND concept and re-check c0
    with torch.no_grad():
        for w in bank.wrapped[:40]:
            for k in list(w.B.keys()):
                w.B[k].add_(torch.randn_like(w.B[k]) * 1e-3)
    with bank.activate("c0"):
        mark = field.eps(X, u).copy()

    bank.add_concept("c1", np.ones(field.dim))
    params = bank.trainable("c1")
    opt = torch.optim.Adam(params, lr=1e-3)
    x0 = torch.from_numpy(X).float().to(dev).reshape(-1, 3, img, img)
    for _ in range(3):
        e = torch.randn_like(x0)
        uu = torch.full((len(x0),), 0.5, device=dev)
        xt = 0.7 * x0 + 0.7 * e
        with bank.activate("c1"):
            pred = field.model(xt, uu, emb_override=bank.embedding("c1"))
        loss = torch.nn.functional.mse_loss(pred, e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with bank.activate("c0"):
        again = field.eps(X, u)
    d2 = float(np.abs(again - mark).max())
    print(f"[test] c0 after training c1          : max|delta| = {d2:.3e}")
    assert d2 == 0.0, f"training c1 moved c0 by {d2:.3e} -- preservation is not structural here"

    print("[test] PASS: bank attaches to a pretrained UNet, and preservation still holds")


if __name__ == "__main__":
    main()
