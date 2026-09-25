"""A stored concept must be sampled THROUGH its adapter on every field family.

`sample_concept` has two routes.  On fields with a guidance callback it samples through
`field.eps(x, u, name)`, which activates the concept's adapter path itself.  On the VE bridge, which
has no callback, it hands the concept's embedding straight to `field.sample` -- and the first version
of that branch never activated the path, so the adapter was trained and then ignored: RESULTS 7kk's
`stored` arm came back identical to `address` to 16 digits on all five Shapes3D concepts and was
reported as "storage cost +0.000".

This pins the property on a stub with the VE bridge's sampler signature (no `guide_eps`):

  1. the concept's samples DIFFER from the same embedding sampled with the path inactive, and
  2. they EQUAL the same embedding sampled with the path explicitly active.

CPU only, a few seconds.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner  # noqa: E402
from lifelong_ood.neural.unet import UNet                               # noqa: E402


class _StubVE:
    """The VE bridge's sampler SURFACE: `emb_override`, and no guidance callback."""
    data_space = "data"

    def __init__(self, model):
        self.model, self.device, self.img, self.layout = model, "cpu", 8, "permuted"
        self.dim = 3 * 8 * 8

    @torch.no_grad()
    def sample(self, n=4, steps=3, seed=0, emb_override=None, **unsupported):
        self.model.eval()                     # dropout off: a sampler is deterministic given seed
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 3, 8, 8, generator=g)
        e = torch.as_tensor(np.asarray(emb_override, np.float32)).reshape(1, -1).expand(n, -1)
        for u in np.linspace(1.0, 0.0, steps + 1)[:-1]:
            x = x - 0.1 * self.model(x, torch.full((n,), float(u)), emb_override=e)
        return x.reshape(n, -1).numpy()


def main():
    torch.manual_seed(0)
    model = UNet(n_classes=3, base=32, mults=(1,), edim=32, attn=(), img=8)
    with torch.no_grad():
        for p in model.parameters():          # nonzero everywhere, so the base output is not 0
            p.normal_(0, 0.05)
    field = _StubVE(model)
    L = NeuralLifelongLearner(field, rank=2, device="cpu", flat=True, steps=3)
    L.bank.add_concept("c", np.zeros(field.dim))
    leaf = str(L.bank.concepts["c"]["leaf"])
    with torch.no_grad():
        for w in L.bank.wrapped:
            if leaf in w.B:
                w.B[leaf].normal_(0, 0.5)     # a trained, non-trivial adapter
    emb = L.bank.embedding("c").detach().numpy()

    s_concept = L.sample_concept("c", n=4, steps=3, seed=0)
    s_inactive = field.sample(n=4, steps=3, seed=0, emb_override=emb)
    with L.bank.activate("c"):
        s_active = field.sample(n=4, steps=3, seed=0, emb_override=emb)

    gap = float(np.abs(s_concept - s_inactive).max())
    assert gap > 1e-4, f"sample_concept ignored the adapter path (max |diff| {gap:.2e})"
    assert np.array_equal(s_concept, s_active), "sample_concept is not the path-active sample"
    print(f"PASS  stored concept sampled through its adapter on a VE-surface field "
          f"(adapter moves samples by {gap:.3f}; equals the path-active sample exactly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
