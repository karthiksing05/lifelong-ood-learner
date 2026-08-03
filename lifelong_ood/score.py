"""ScoreField: the Cobweb tree AS a diffusion model, with everything analytic.

A diffusion model learns s_theta(x_t, t) ~ grad log p_t(x_t) for the noisy marginals p_t.  Here no
network is needed: the tree's effective leaves define a Gaussian mixture estimate of p_data, and the
Gaussian convolution that defines p_t keeps the mixture Gaussian component-by-component,

    p_t(x_t) = sum_j w_j N(x_t; sqrt(abar_t) m_j,  abar_t V_j + (1 - abar_t) I),

so the score, Tweedie denoiser, and epsilon-prediction are all closed-form.  Every operation the
poe-diffusion paper performs against a trained UNet -- reverse sampling, mode ascent on p_t,
classifier-free guidance against a teacher -- runs here against this analytic field.

The hierarchy enters twice:

  * mode="leaves"  uses the leaf mixture at every t.  The CLAIM (validated by experiment E1) is that
    the modes of these smoothed marginals at coarser t coincide with the tree's own INTERNAL nodes:
    the hierarchy Cobweb built incrementally is the same density-mode hierarchy the paper recovers
    from a DDPM by gradient ascent (their Appendix A), but ours is explicit and free.
  * mode="frontier" swaps the component set at each t for the tree frontier at a depth matched to the
    noise level (deep/fine at t~0, root-ward/coarse at t~1), i.e. the reverse process anneals DOWN
    the tree.  This makes the depth<->noise correspondence an architectural statement rather than an
    emergent one.

Guidance: `sample(guide_eps=...)` implements the paper's Eq. 13 with the variance-aware schedule of
their Eq. 14, with the analytic PoE teacher (see compose.py) as the conditional branch.
"""
import numpy as np

from .tree import frontier_mask, frontier_weights


def cosine_abar(u, s=0.008):
    """Continuous cosine schedule: u in [0, 1] -> abar in (0, 1], abar(0) = 1."""
    u = np.asarray(u, np.float64)
    f = np.cos((u + s) / (1 + s) * np.pi / 2) ** 2
    f0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return np.clip(f / f0, 1e-5, 1.0)


class ScoreField:
    def __init__(self, snap, mode="leaves", level_gamma=1.0, chunk=128):
        self.snap = snap
        self.mode = mode
        self.level_gamma = float(level_gamma)
        self.chunk = int(chunk)
        self.max_depth = int(snap["depth"].max())
        self._masks = {}

    # ── which components are active at noise level u ─────────────────────────
    def components(self, u, branch=None):
        snap = self.snap
        if self.mode == "frontier":
            lvl = int(round(self.max_depth * (1.0 - float(u)) ** self.level_gamma))
            lvl = max(lvl, 1) if self.max_depth >= 1 else 0
            key = (lvl, branch)
            if key not in self._masks:
                m = frontier_mask(snap, lvl)
                if branch is not None:
                    m = m & (snap["branch"] == branch)
                    if not m.any():  # branch shallower than this level: use its leaves
                        m = snap["is_leaf"] & (snap["branch"] == branch)
                self._masks[key] = (m, frontier_weights(snap, m))
            return self._masks[key]
        key = ("leaves", branch)
        if key not in self._masks:
            m = snap["is_leaf"].copy()
            if branch is not None:
                m = m & (snap["branch"] == branch)
            w = np.where(m, snap["count"], 0.0).astype(np.float64)
            self._masks[key] = (m, w / w.sum())
        return self._masks[key]

    # ── the analytic score of p_t ────────────────────────────────────────────
    def _post(self, X, u, branch=None):
        """Responsibilities r [B, K] and per-component (mu_t [K, D], s_t [K, D]) at noise level u."""
        mask, w = self.components(u, branch)
        idx = np.where(mask)[0]
        a = float(cosine_abar(u))
        mu = np.sqrt(a) * self.snap["M"][idx]
        s = a * self.snap["V"][idx] + (1.0 - a)
        X = np.atleast_2d(np.asarray(X, np.float64))
        B, K = len(X), len(idx)
        ll = np.empty((B, K))
        logw = np.log(w[idx])
        cn = -0.5 * np.log(2 * np.pi * s).sum(1)
        for j0 in range(0, K, self.chunk):
            j1 = min(j0 + self.chunk, K)
            d = X[:, None, :] - mu[None, j0:j1, :]
            ll[:, j0:j1] = cn[j0:j1] - 0.5 * (d * d / s[None, j0:j1, :]).sum(2)
        lj = ll + logw
        lz = lj.max(1, keepdims=True)
        r = np.exp(lj - lz)
        Z = r.sum(1, keepdims=True)
        logp = (lz + np.log(Z)).ravel()
        return r / Z, mu, s, logp, a

    def logp(self, X, u=0.0, branch=None):
        """Mixture log-density of the noisy marginal at level u (u=0: model log-likelihood of data)."""
        return self._post(X, u, branch)[3]

    def score(self, X, u, branch=None):
        r, mu, s, _, _ = self._post(X, u, branch)
        X = np.atleast_2d(np.asarray(X, np.float64))
        out = np.zeros_like(X)
        K = mu.shape[0]
        for j0 in range(0, K, self.chunk):
            j1 = min(j0 + self.chunk, K)
            d = (mu[None, j0:j1, :] - X[:, None, :]) / s[None, j0:j1, :]
            out += np.einsum("bk,bkd->bd", r[:, j0:j1], d)
        return out

    def eps(self, X, u, branch=None):
        a = float(cosine_abar(u))
        return -np.sqrt(1.0 - a) * self.score(X, u, branch)

    def tweedie(self, X, u, branch=None):
        a = float(cosine_abar(u))
        return (np.atleast_2d(X) + (1.0 - a) * self.score(X, u, branch)) / np.sqrt(a)

    # ── reverse-diffusion sampling (DDIM), with optional analytic guidance ───
    def sample(self, n=16, steps=50, seed=0, u_start=1.0, x_init=None,
               guide_eps=None, guide_w=None, branch=None, clip=(-0.2, 1.2)):
        """DDIM sampler on the analytic field.

        guide_eps(x, u) -> conditional epsilon (e.g. the PoE teacher of compose.py, their Eq. 12);
        guide_w(u) -> guidance weight (their Eq. 14).  Composed as classifier-free guidance, Eq. 13:
        eps_hat = eps_uncond + w(u) * (eps_cond - eps_uncond).  `branch` restricts the UNCONDITIONAL
        field to one adapter's components (conditional generation from a consolidated concept).
        """
        rng = np.random.default_rng(seed)
        D = self.snap["M"].shape[1]
        a0 = float(cosine_abar(u_start))
        if x_init is None:
            x = rng.standard_normal((n, D))
        else:
            x = np.sqrt(a0) * np.atleast_2d(x_init) + np.sqrt(1 - a0) * rng.standard_normal((n, D))
        us = np.linspace(u_start, 0.0, steps + 1)
        for i in range(steps):
            u, un = float(us[i]), float(us[i + 1])
            a, an = float(cosine_abar(u)), float(cosine_abar(un))
            e = self.eps(x, u, branch)
            if guide_eps is not None:
                w = float(guide_w(u)) if guide_w is not None else 1.0
                e = e + w * (guide_eps(x, u) - e)
            x0 = (x - np.sqrt(1 - a) * e) / np.sqrt(a)
            x0 = np.clip(x0, *clip)
            x = np.sqrt(an) * x0 + np.sqrt(1 - an) * e
        return np.clip(x, 0.0, 1.0)

    # ── mode ascent on p_t (the paper's Eq. 4, against the analytic score) ───
    def mode_ascent(self, xq, u, n_starts=16, iters=100, lr=0.05, seed=0, init="query"):
        """Adam ascent on log p_t from noised copies of the query -> (modes in x_t space, x0 = /sqrt(abar))."""
        rng = np.random.default_rng(seed)
        a = float(cosine_abar(u))
        D = self.snap["M"].shape[1]
        if init == "query" and xq is not None:
            Q = np.atleast_2d(np.asarray(xq, np.float64))
            Q = Q[np.arange(n_starts) % len(Q)]
            x = np.sqrt(a) * Q + np.sqrt(1 - a) * rng.standard_normal((n_starts, D))
        else:
            x = rng.standard_normal((n_starts, D))
        m = np.zeros_like(x); v = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(1, iters + 1):
            g = self.score(x, u)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** i); vh = v / (1 - b2 ** i)
            x = x + lr * mh / (np.sqrt(vh) + eps)
        return x, x / np.sqrt(a)

    def laplace_diag(self, xt, u, probes=4, delta=5e-3, seed=0, var_floor=1e-4, var_cap=1.0):
        """Diagonal Laplace covariance of p_t at x_t via Hutchinson finite differences on the score,
        pulled back to x0-space:  Sigma_t = -1/diag(H),  Sigma_0 = (Sigma_t - (1-abar))/abar."""
        rng = np.random.default_rng(seed)
        xt = np.atleast_2d(xt)
        a = float(cosine_abar(u))
        h = np.zeros_like(xt)
        for _ in range(probes):
            vsg = rng.choice([-1.0, 1.0], size=xt.shape)
            gp = self.score(xt + delta * vsg, u)
            gm = self.score(xt - delta * vsg, u)
            h += vsg * (gp - gm) / (2 * delta)
        h /= probes
        st = np.where(h < -1e-9, -1.0 / h, var_cap)
        s0 = (st - (1.0 - a)) / a
        return np.clip(s0, var_floor, var_cap)
