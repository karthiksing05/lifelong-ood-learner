"""Is mode ascent finding a MODE, or just marching to the boundary?

The prototypes are posterized into flat regions of pure saturated colour.  That is not "a bit out of
range": it is a large fraction of coordinates sitting at EXACTLY the boundary, which is what
clipping does to values that were driven past it.  Clamping therefore changed the numbers and not
the pictures -- the values still arrive at the edge, they just stop there.

So the question is why ascent drives 30-40% of coordinates to the edge at all, and the suspect is
the optimizer.  `mode_ascent` uses **Adam**, whose update is normalized: every coordinate moves by
about `lr` per iteration regardless of how large its gradient is.  Near a mode the gradient goes to
zero and a plain gradient step goes to zero with it -- Adam's does not, it keeps taking
same-sized steps and marches.  Over 100 iterations at lr 0.09 that is a walk of up to 9 units in a
space one unit wide.

This measures, for each optimizer and step size:

  * `frac |x0| > 1`  -- how much of the result is off the valid range
  * `eps error`      -- the model's own denoising error at the returned point, which is the thing
                        ascent is supposed to be REDUCING.  If a setting produces a worse eps error
                        than the query it started from, it has not found a mode of anything.

The query's own eps error is printed as the bar to beat.

    python tests/diag_ascent_quality.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from lifelong_ood.score import cosine_abar                                     # noqa: E402


def eps_err(field, x0, u, rng):
    """Mean squared eps-prediction error of the model at a candidate x0, at noise level u."""
    a = float(cosine_abar(u))
    e = rng.standard_normal(np.atleast_2d(x0).shape)
    xt = np.sqrt(a) * np.atleast_2d(x0) + np.sqrt(1 - a) * e
    return float(np.mean((field.eps(xt, u) - e) ** 2))


def main():
    from lifelong_ood.data.celeba import load_celeba
    from lifelong_ood.neural import to_model
    from lifelong_ood.neural.pretrained import load_pretrained
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, img = load_pretrained("google/ddpm-celebahq-256", device=dev, batch=4)
    data = load_celeba(n_per_class=4, img=img, seed=0, min_count=25, max_rows=4000)
    xq = to_model(np.asarray(data["X"])[:1]).ravel()
    rng = np.random.default_rng(0)

    u = 0.15
    print(f"[quality] pretrained backbone, u={u}; the QUERY's own eps error is the bar to beat\n")
    q = eps_err(field, xq, u, np.random.default_rng(1))
    print(f"  {'setting':34s} {'frac |x0|>1':>12s} {'eps error':>11s}   (query {q:.4f})")
    for label, kw in (
            ("Adam  lr 0.09  100it (the paper)", dict(lr=0.09, iters=100, adam=True)),
            ("Adam  lr 0.02  100it",             dict(lr=0.02, iters=100, adam=True)),
            ("Adam  lr 0.09   20it",             dict(lr=0.09, iters=20, adam=True)),
            ("plain lr 0.09  100it",             dict(lr=0.09, iters=100, adam=False)),
            ("plain lr 0.30  100it",             dict(lr=0.30, iters=100, adam=False)),
            ("plain lr 1.00  100it",             dict(lr=1.00, iters=100, adam=False)),
            ("Adam+x0proj lr 0.09 100it",        dict(lr=0.09, iters=100, adam=True, clamp="x0"))):
        kw = dict(kw); cl = kw.pop("clamp", False)
        _, x0 = field.mode_ascent(xq, u, n_starts=4, seed=0, init="query", clamp=cl,
                                  tweedie=False, **kw)
        x0 = np.asarray(x0)
        frac = float((np.abs(x0) > 1.0).mean())
        err = eps_err(field, np.clip(x0, -1, 1), u, np.random.default_rng(1))
        flag = "  <-- worse than the query" if err > q else ""
        print(f"  {label:34s} {frac:12.4f} {err:11.4f}{flag}")
    print("\n[quality] Adam's normalized step does not shrink as the gradient vanishes, so it keeps\n"
          "          moving after it has arrived.  A plain gradient step scales with the gradient and\n"
          "          settles.  Whichever gives a low eps error AND stays in range is the right one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
