"""Conditional mode ascent: the prototype of p(x | c) is not the prototype of p(x).

Both `mode_ascent` implementations climbed the NULL-embedding density until 2026-08-19, because
neither passed a conditioning vector to the network. That is the correct prototype for an
unconditional panel and the wrong one for a composed concept, and the difference is invisible in a
figure -- an unconditional Shapes3D mode is a perfectly plausible-looking room. So the property is
asserted here rather than judged by eye.

Three things, and the first is the one that protects every existing number:

  1. with no `branch`/`emb` the ascent is BIT-IDENTICAL to what it computed before -- it must call
     `score(x, u, None)` exactly, or the prototype claims (n30, n42, d5) silently move;
  2. conditioning moves the mode TOWARD the concept it names;
  3. a field family whose `eps` cannot accept an embedding REFUSES rather than ignoring the keyword.
     The sampler-signature divergences that cost six runs were all silent-ignore faults.

CPU only, no checkpoint: the arithmetic is exercised on a toy field whose modes are known in closed
form, which is the only way (2) has a ground truth to be checked against.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.score import BaseScoreField, cosine_abar          # noqa: E402


class _Toy(BaseScoreField):
    """Isotropic Gaussian around one of three known centres; `eps` honours `emb`."""

    dim = 8

    def __init__(self):
        self.mu = np.random.default_rng(0).standard_normal((3, 8))

    def _s(self, u):
        return np.sqrt(1.0 - float(cosine_abar(u))) + 0.3

    def score(self, X, u, branch=None):
        c = self.mu[0] if branch is None else self.mu[int(branch)]
        a = float(cosine_abar(u))
        return (np.sqrt(a) * c - np.atleast_2d(X)) / self._s(u) ** 2

    def eps(self, X, u, branch=None, emb=None):
        c = self.mu[0] if emb is None else np.asarray(emb, np.float64)
        a = float(cosine_abar(u))
        return -np.sqrt(1.0 - a) * (np.sqrt(a) * c - np.atleast_2d(X)) / self._s(u) ** 2


class _NoEmb(_Toy):
    """A field family whose `eps` predates conditioning -- the analytic `ScoreField` case."""

    def eps(self, X, u, branch=None):
        return np.zeros_like(np.atleast_2d(X))


def main():
    f, ok = _Toy(), True
    xq = f.mu[0] + 0.01

    xt, x0 = f.mode_ascent(xq, u=0.1, n_starts=4, iters=30, seed=0)
    same = np.array_equal(f._ascent_score(xt, 0.1, None, None), f.score(xt, 0.1, None))
    print(f"  unconditional path identical to score(x, u, None): {same}")
    ok &= same

    _, x0c = f.mode_ascent(xq, u=0.1, n_starts=4, iters=30, seed=0, emb=f.mu[2])
    d_un = float(np.linalg.norm(x0.mean(0) - f.mu[2]))
    d_cd = float(np.linalg.norm(x0c.mean(0) - f.mu[2]))
    print(f"  distance to the named concept: unconditional {d_un:.3f} -> conditional {d_cd:.3f}")
    ok &= d_cd < 0.5 * d_un

    try:
        _NoEmb().mode_ascent(xq, u=0.1, n_starts=2, iters=2, seed=0, emb=f.mu[1])
        print("  FAIL: an unsupported `emb` was ignored instead of refused")
        ok = False
    except TypeError as exc:
        print(f"  unsupported `emb` refused: {str(exc)[:60]}...")

    print("[mode_ascent_cond] PASS" if ok else "[mode_ascent_cond] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
