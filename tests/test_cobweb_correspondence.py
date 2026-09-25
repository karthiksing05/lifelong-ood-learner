"""Numerical check of every identity in docs/COBWEB_UPDATE.md.  numpy only, CPU, seconds.

  P2  adding one instance moves the ideal denoiser's posterior mean and covariance by Cobweb's
      incremental update with a soft share r(x_t)            (vs recomputing from n+1 points)
  P2' the same identity for the ideal epsilon-predictor
  C   with r = 1/(c+1) it IS Welford / COBWEB-3's node update  (vs recomputing from c+1 points)
  P3  r -> 1/(n+1) at high noise; r -> 1 / 0 at low noise near the new / an old instance
  P4  |D_{n+1} - D_n| <= r |x_{n+1} - D_n| everywhere
  P6  top-k by g^2/F is the best size-k support of the separable quadratic (brute force)
"""
import itertools
import sys

import numpy as np


def posterior(X, xt, a, s):
    """-> (weights, mean, covariance) of x_0 | x_t under the empirical prior on rows of X."""
    logk = -((xt[None] - a * X) ** 2).sum(1) / (2 * s ** 2)
    w = np.exp(logk - logk.max())
    w /= w.sum()
    m = w @ X
    R = X - m
    return w, m, (w[:, None] * R).T @ R


def main():
    rng = np.random.default_rng(0)
    n, D = 40, 6
    X = rng.normal(size=(n, D))
    xn = rng.normal(size=D) * 1.5
    Xp = np.vstack([X, xn])
    worst = dict(mean=0.0, cov=0.0, eps=0.0, bound=0.0)
    for s in (0.05, 0.3, 0.7, 0.95, 0.999):
        a = np.sqrt(1 - s ** 2)
        for _ in range(50):
            xt = a * X[rng.integers(n)] + s * rng.normal(size=D) * rng.uniform(0.2, 3.0)
            _, m0, S0 = posterior(X, xt, a, s)
            w1, m1, S1 = posterior(Xp, xt, a, s)
            r = w1[-1]
            d = xn - m0
            worst["mean"] = max(worst["mean"], np.abs(m1 - (m0 + r * d)).max())
            worst["cov"] = max(worst["cov"],
                               np.abs(S1 - ((1 - r) * S0 + r * (1 - r) * np.outer(d, d))).max())
            e0, e1 = (xt - a * m0) / s, (xt - a * m1) / s
            worst["eps"] = max(worst["eps"], np.abs(e1 - (e0 - (a / s) * r * d)).max())
            worst["bound"] = max(worst["bound"],
                                 np.linalg.norm(m1 - m0) - r * np.linalg.norm(d))
    assert worst["mean"] < 1e-9, worst
    assert worst["cov"] < 1e-9, worst
    assert worst["eps"] < 1e-8, worst
    assert worst["bound"] < 1e-9, worst

    # C: Cobweb/Welford with c members is P2 at r = 1/(c+1)
    y = rng.normal(size=12)
    for c in range(1, 11):
        mu, var = y[:c].mean(), y[:c].var()
        r = 1.0 / (c + 1)
        mu1, var1 = mu + r * (y[c] - mu), (1 - r) * var + r * (1 - r) * (y[c] - mu) ** 2
        assert abs(mu1 - y[:c + 1].mean()) < 1e-12 and abs(var1 - y[:c + 1].var()) < 1e-12

    # P3: the limits.  r -> 1/(n+1) as a -> 0, with deviation O(a): check it SHRINKS and ends small
    # (a = sqrt((1-s)(1+s)) avoids the cancellation in 1 - s^2 at s near 1)
    devs = []
    for gap in (1e-6, 1e-10, 1e-14):
        s_hi = 1 - gap
        a_hi = np.sqrt(gap * (1 + s_hi))
        r_hi = [posterior(Xp, rng.normal(size=D), a_hi, s_hi)[0][-1] for _ in range(20)]
        devs.append(float(np.abs(np.asarray(r_hi) - 1 / (n + 1)).max()))
    assert devs[0] > devs[1] > devs[2] and devs[2] < 1e-6, devs
    s_lo = 1e-3
    a_lo = np.sqrt(1 - s_lo ** 2)
    r_new = posterior(Xp, a_lo * xn + s_lo * rng.normal(size=D) * 0.1, a_lo, s_lo)[0][-1]
    r_old = posterior(Xp, a_lo * X[3] + s_lo * rng.normal(size=D) * 0.1, a_lo, s_lo)[0][-1]
    assert r_new > 1 - 1e-6 and r_old < 1e-6, (r_new, r_old)

    # P6: top-k by g^2/F is the optimal support (brute force over every size-k subset)
    for _ in range(200):
        m, k, nn = 9, 3, 50
        g, F = rng.normal(size=m), rng.gamma(1.0, 1.0, size=m) + 1e-3
        gain = g ** 2 / (2 * nn * F)
        best = max(itertools.combinations(range(m), k), key=lambda S: gain[list(S)].sum())
        top = np.argsort(-(g ** 2 / F))[:k]
        assert np.isclose(gain[list(best)].sum(), gain[top].sum()), (best, top)

    print(f"PASS  Cobweb correspondence: mean {worst['mean']:.1e}, cov {worst['cov']:.1e}, "
          f"eps {worst['eps']:.1e}; Welford exact; r limits 1/(n+1) and 1/0; locality bound holds; "
          f"g^2/F support optimal on 200 brute-force cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
