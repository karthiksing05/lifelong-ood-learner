"""Does mode ascent leave the valid image range?  The neon prototypes say yes; this measures it.

`BaseScoreField.mode_ascent` is Adam with a fixed step size and NO constraint on where it goes.
Adam's update is normalized, so each coordinate moves by about `lr` per iteration whatever the
gradient is: at the paper's CelebA settings (lr 0.09, 100 iterations) a pixel can travel **9.0** in
a space where valid images live in [-1, 1].

Nothing pulls it back.  A clean image x0 lies in [-1, 1]; at noise level u a valid x_t is
`sqrt(abar)*x0 + sqrt(1-abar)*eps`, so x_t is concentrated in roughly
`+-(sqrt(abar) + 3*sqrt(1-abar))`.  Ascent that wanders outside that box is not finding a mode of
p_t -- it is off the support of the data the density was built from, and the x0-space mode it
returns (`x / sqrt(abar)`) is out of gamut.  Out-of-gamut is exactly what saturated neon colour is.

Reports, per noise level, what fraction of the returned x0-space modes fall outside [-1, 1] and how
far the worst coordinate goes.  Run on the pretrained CelebA backbone, which is where the artifact
was seen.

    python tests/diag_ascent_range.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from lifelong_ood.score import cosine_abar                                     # noqa: E402


def main():
    from lifelong_ood.data.celeba import load_celeba
    from lifelong_ood.neural import to_model
    from lifelong_ood.neural.pretrained import load_pretrained
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, img = load_pretrained("google/ddpm-celebahq-256", device=dev, batch=4)
    data = load_celeba(n_per_class=4, img=img, seed=0, min_count=25, max_rows=4000)
    xq = to_model(np.asarray(data["X"])[:1]).ravel()

    print(f"[range] pretrained backbone at {img}px; valid images live in [-1, 1]\n")
    print(f"  {'u':>6s} {'mode':>15s} {'frac |x0|>1':>12s} {'max |x0|':>10s} {'mean |x0|':>10s}")
    rows = []
    for u in (0.05, 0.15, 0.3, 0.5):
        # `tweedie, no clip` is the honest one: with clamp=True the returned x0 is clipped, so
        # "fraction out of range = 0" would be true by construction and prove nothing.
        for mode, kw in (("rescale", dict(clamp=False, tweedie=False)),
                         ("tweedie/noclip", dict(clamp=False, tweedie=True)),
                         ("tweedie+clip", dict(clamp=True, tweedie=True))):
            _, x0 = field.mode_ascent(xq, u, n_starts=4, iters=100, lr=0.09, seed=0,
                                      init="query", **kw)
            clamp = mode
            x0 = np.asarray(x0)
            frac = float((np.abs(x0) > 1.0).mean())
            print(f"  {u:6.2f} {str(clamp):>15s} {frac:12.4f} {np.abs(x0).max():10.3f} "
                  f"{np.abs(x0).mean():10.3f}")
            rows.append((u, clamp, frac, float(np.abs(x0).max())))
    print("\n[range] a large out-of-range fraction means ascent is off the data's support, and the\n"
          "        x0-space mode it returns cannot be rendered -- which is what saturated neon is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
