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


class BaseScoreField:
    """Everything a diffusion manifold must provide, expressed in terms of ONE primitive: `eps`.

    Reverse sampling, mode ascent and the Laplace covariance are all written against
    `self.eps(X, u, branch) -> epsilon prediction at noise level u`, so a backbone only has to supply
    that.  Two do: `ScoreField` (a Cobweb tree read as an analytic Gaussian mixture) and
    `neural.NeuralScoreField` (a trained UNet).  Because `compose.py` and `LifelongLearner` also
    speak only this interface, the entire composition-and-consolidation stack runs unchanged on
    either -- which is what lets the experiments attribute results to the METHOD rather than to the
    backbone that happens to implement it.

    Subclasses must define `dim` and exactly one of `eps` / `score` (the two are related by
    eps = -sqrt(1-abar) * score, and each default is written in terms of the other).  `logp` is
    optional: the analytic field has it in closed form, a UNet does not.
    """
    dim = None

    def eps(self, X, u, branch=None):
        a = float(cosine_abar(u))
        return -np.sqrt(1.0 - a) * self.score(X, u, branch)

    def score(self, X, u, branch=None):
        a = float(cosine_abar(u))
        return -self.eps(X, u, branch) / np.sqrt(1.0 - a)

    def tweedie(self, X, u, branch=None):
        a = float(cosine_abar(u))
        return (np.atleast_2d(X) + (1.0 - a) * self.score(X, u, branch)) / np.sqrt(a)

    def logp(self, X, u=0.0, branch=None):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form density")

    # ── reverse-diffusion sampling (DDIM), with optional analytic guidance ───
    def sample(self, n=16, steps=50, seed=0, u_start=1.0, x_init=None,
               guide_eps=None, guide_w=None, branch=None, clip=(-0.2, 1.2), eta=0.0,
               mcmc=None):
        """DDIM sampler, with `eta` interpolating to ancestral (stochastic) sampling.

        guide_eps(x, u) -> conditional epsilon (e.g. the PoE teacher of compose.py, their Eq. 12);
        guide_w(u) -> guidance weight.  Composed as classifier-free guidance, their Eq. 13:
        eps_hat = eps_uncond + w(u) * (eps_cond - eps_uncond).  `branch` selects a conditional
        branch of the unconditional model itself (a consolidated concept).

        `eta=0` (the paper's setting, and the default) is fully deterministic: the ONLY randomness in
        a trajectory is its starting noise.  That matters here because guidance strong enough to hold
        a composition also contracts trajectories, so different starting points can be funnelled to
        the same output and the pool collapses -- which is what our diversity investigation kept
        running into.  `eta>0` re-injects noise at every step, so variety no longer has to survive
        the whole trajectory from x_T alone.

        `mcmc` = dict(steps=k, lam=step_scale) turns on ANNEALED LANGEVIN CORRECTION of the composed
        score, i.e. Du et al., "Reduce, Reuse, Recycle" (ICML 2023, arXiv:2302.11552).  Their result
        is the sharpest criticism of every score-addition composition method, this repository's
        included:

            grad log q~_t^prod(x_t)  !=  grad log q_t^1(x_t) + grad log q_t^2(x_t)

        The sum of diffused scores is the score of the PRODUCT OF DIFFUSED MARGINALS, not the
        DIFFUSION OF THE PRODUCT.  So running a composed score through plain ancestral/DDIM sampling
        targets the wrong distribution, and the failure is attributable to the SAMPLER rather than to
        the model.  Their fix is to spend extra MCMC transitions at each noise level, which are
        invariant to the correct annealed target; it recovers CLEVR 5-cube composition from 70.8% to
        91.6% and ImageNet classifier guidance from 18.6% to 94.6%.

        Implemented as unadjusted Langevin (ULA), the variant that needs no density: with
        s(x,u) = -e(x,u)/sqrt(1-abar), each transition is

            x <- x + (sig^2/2) s(x,u) + sig z,     sig^2 = 2 * lam * (1 - abar_u)

        so the step size is annealed with the noise level exactly as they prescribe, and lam is
        dimensionless (lam=1 moves on the scale of the marginal's own width).  MALA's accept/reject
        would need an energy parameterization the pretrained eps-net does not provide.

        Costs (steps+1)x more network evaluations per denoising step; they report ~5x overall.
        """
        rng = np.random.default_rng(seed)
        D = self.dim
        a0 = float(cosine_abar(u_start))
        if x_init is None:
            x = rng.standard_normal((n, D))
        else:
            x = np.sqrt(a0) * np.atleast_2d(x_init) + np.sqrt(1 - a0) * rng.standard_normal((n, D))
        us = np.linspace(u_start, 0.0, steps + 1)
        mc_steps = 0 if not mcmc else int(mcmc.get("steps", 0))
        mc_lam = 0.1 if not mcmc else float(mcmc.get("lam", 0.1))

        def _guided_eps(xx, uu):
            ee = self.eps(xx, uu, branch)
            if guide_eps is not None:
                ww = 1.0 if guide_w is None else np.asarray(guide_w(uu), dtype=np.float64)
                ee = ee + ww * (guide_eps(xx, uu) - ee)
            return ee

        for i in range(steps):
            u, un = float(us[i]), float(us[i + 1])
            a, an = float(cosine_abar(u)), float(cosine_abar(un))
            # Du et al.'s correction: before taking the reverse step, let the chain equilibrate to
            # the composed target AT THIS noise level.  Corrects the composed score, not the base
            # model, which is why it only ever runs when something is actually being composed.
            if mc_steps and guide_eps is not None:
                sig = np.sqrt(2.0 * mc_lam * max(1.0 - a, 1e-12))
                for _ in range(mc_steps):
                    s = -_guided_eps(x, u) / np.sqrt(max(1.0 - a, 1e-12))
                    x = x + 0.5 * sig ** 2 * s + sig * rng.standard_normal(x.shape)
            e = self.eps(x, u, branch)
            if guide_eps is not None:
                # guide_w may return a scalar OR a [N, 1] column -- the paper's Eq. 14 makes the
                # weight a function of the teacher's covariance, so a batched run with one teacher
                # per block of rows needs one weight per row.  Keep it as an array either way.
                w = 1.0 if guide_w is None else np.asarray(guide_w(u), dtype=np.float64)
                e = e + w * (guide_eps(x, u) - e)
            x0 = np.clip((x - np.sqrt(1 - a) * e) / np.sqrt(a), *clip)
            if eta > 0.0 and un > 0.0:
                sig = eta * np.sqrt(max(1 - an, 0) / max(1 - a, 1e-12)) * np.sqrt(max(1 - a / an, 0)) \
                    if an > a else 0.0
                x = (np.sqrt(an) * x0 + np.sqrt(max(1 - an - sig ** 2, 0.0)) * e
                     + sig * rng.standard_normal(x.shape))
            else:
                x = np.sqrt(an) * x0 + np.sqrt(1 - an) * e
        return np.clip(x, clip[0], clip[1])

    # ── mode ascent on p_t (the paper's Eq. 4) ──────────────────────────────
    def mode_ascent(self, xq, u, n_starts=16, iters=100, lr=0.05, seed=0, init="query",
                    clamp=True, k_sigma=3.0, tweedie=True, adam=False):
        """Adam ascent on log p_t from noised copies of the query -> (modes in x_t space, x0-space).

        `adam=False` -- PLAIN gradient ascent -- is the default, and it is a measured deviation from
        the paper's Table 6, which specifies Adam.  Adam's update is normalized: every coordinate
        moves by about `lr` per iteration however small its gradient is, so near a mode, where the
        gradient vanishes, it keeps taking full-sized steps and marches past.  Measured on the
        pretrained CelebA backbone at u=0.15, against the query's own eps error of 0.0260:

            Adam  lr 0.09 100it (their column)   11.1% of x0 out of range   eps 0.0113
            Adam  lr 0.02 100it                   0.0%                      eps 0.0056
            plain lr 0.09 100it                   0.0%                      eps 0.0046
            plain lr 1.00 100it                  32.9%                      eps 0.0469 (worse
                                                                            than the query)

        Plain ascent is better on BOTH axes: it stays entirely in gamut and finds a lower-error
        mode.  That matters because out-of-range coordinates get clipped for display, and a large
        fraction of them clipped to exactly +-1 is what made the prototypes posterize into flat
        saturated colour -- clamping alone changed the numbers and not the pictures, because the
        values still arrived at the edge.

        `clamp` keeps the ascent on the DATA'S SUPPORT, and without it this walks off into nonsense.
        Adam's update is normalized, so every coordinate moves by about `lr` per iteration whatever
        the gradient is: at the paper's CelebA settings (lr 0.09, 100 iterations) a pixel can travel
        9.0 in a space where clean images live in [-1, 1], and nothing pulls it back.

        The constraint is not a hack, it is the support.  A clean image x0 lies in [-1, 1]; at noise
        level u a valid x_t is `sqrt(abar)*x0 + sqrt(1-abar)*eps`, so x_t is concentrated inside
        `+-(sqrt(abar) + k*sqrt(1-abar))`.  p_t has vanishing mass outside that box, so a "mode"
        found there is an artefact of unconstrained optimization rather than a mode of the
        distribution -- and the x0-space value returned (`x / sqrt(abar)`) is out of gamut, which is
        what saturated neon prototypes are.  `clamp=False` reproduces the old behaviour.
        """
        rng = np.random.default_rng(seed)
        a = float(cosine_abar(u))
        D = self.dim
        if init == "query" and xq is not None:
            Q = np.atleast_2d(np.asarray(xq, np.float64))
            Q = Q[np.arange(n_starts) % len(Q)]
            x = np.sqrt(a) * Q + np.sqrt(1 - a) * rng.standard_normal((n_starts, D))
        else:
            x = rng.standard_normal((n_starts, D))
        m = np.zeros_like(x); v = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        bound = np.sqrt(a) + k_sigma * np.sqrt(max(1.0 - a, 0.0))
        for i in range(1, iters + 1):
            g = self.score(x, u)
            if adam:
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                x = x + lr * (m / (1 - b1 ** i)) / (np.sqrt(v / (1 - b2 ** i)) + eps)
            else:
                # plain gradient ascent, scaled by the noise level so `lr` means the same thing at
                # every u: the score has magnitude ~1/sqrt(1-abar), so multiplying it back by
                # (1-abar) puts the step in x-units.  Unlike Adam this step SHRINKS as the gradient
                # vanishes, which is what makes it settle at a mode instead of marching past it.
                x = x + lr * (1.0 - a) * g
            if clamp:
                x = np.clip(x, -bound, bound)
        # x0-space mode by TWEEDIE, not by rescaling.  `x / sqrt(abar)` is the noisy point divided
        # by sqrt(abar): it still CONTAINS the noise, amplified by 1/sqrt(abar).  Measured, that is
        # 25% of the image's dynamic range at u=0.15, 52% at u=0.3 and 101% at u=0.5 -- and the
        # paper's CelebA grid runs to u=0.5.  So the "prototypes" it returned were mostly noise,
        # which is what made them saturated posterized nonsense once clipped for display.
        #     x0_hat = (x_t + (1-abar) * score(x_t)) / sqrt(abar)
        #            = (x_t - sqrt(1-abar) * eps_hat) / sqrt(abar)
        # `tweedie=False` restores the old rescaling for comparison.
        if tweedie:
            x0 = (x + (1.0 - a) * self.score(x, u)) / np.sqrt(a)
        else:
            x0 = x / np.sqrt(a)
        return x, np.clip(x0, -1.0, 1.0) if clamp else x0

    def laplace_diag(self, xt, u, probes=4, delta=5e-3, seed=0, var_floor=1e-4, var_cap=1.0):
        """Diagonal Laplace covariance of p_t at x_t via Hutchinson finite differences on the score,
        pulled back to x0-space:  Sigma_t = -1/diag(H),  Sigma_0 = (Sigma_t - (1-abar))/abar."""
        rng = np.random.default_rng(seed)
        xt = np.atleast_2d(xt)
        a = float(cosine_abar(u))
        h = np.zeros_like(xt)
        for _ in range(probes):
            vsg = rng.choice([-1.0, 1.0], size=xt.shape)
            h += vsg * (self.score(xt + delta * vsg, u) - self.score(xt - delta * vsg, u)) / (2 * delta)
        h /= probes
        # only strictly-negative curvature defines a variance; elsewhere fall back to the cap.
        # np.divide with a `where` mask rather than `np.where(cond, -1/h, cap)`, which would still
        # evaluate -1/h at h == 0 and emit a divide-by-zero warning.
        neg = h < -1e-9
        st = np.full_like(h, var_cap)
        np.divide(-1.0, h, out=st, where=neg)
        return np.clip((st - (1.0 - a)) / a, var_floor, var_cap)


class ScoreField(BaseScoreField):
    def __init__(self, snap, mode="leaves", level_gamma=1.0, chunk=128):
        self.snap = snap
        self.mode = mode
        self.level_gamma = float(level_gamma)
        self.chunk = int(chunk)
        self.max_depth = int(snap["depth"].max())
        self.dim = int(snap["M"].shape[1])
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

    # `eps`, `tweedie`, `sample`, `mode_ascent` and `laplace_diag` are inherited from
    # BaseScoreField -- they are written against `score`/`eps` alone, so the analytic field and the
    # neural one run the identical algorithms.  Only the closed-form pieces are overridden above.
