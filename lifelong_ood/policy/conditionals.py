"""Composition of a frozen flow policy's OWN conditionals (the overhaul's step 3: Read -> Compose -> Fold -> Consolidate).

The user's ICLR paper restores composition by summing single-object predictions, D(empty) + w sum_k [D(c_k) - D(empty)]
(Liu et al. 2022).  A flow policy conditions on images + a prompt, so a composition is a set of prompts and one coefficient
each, applied to the policy's velocities at the SAME noisy chunk on every Euler step:

    v_comp(x, t) = sum_k a_k v(x, t | o, prompt_k),   sum_k a_k = 1

Affine coefficients make composing velocities, noise predictions or clean-action estimates the same thing (each is affine in
v at fixed x and t), and a_0 = 1 with every other a_k = 0 is the policy itself.  The usual factor form
base + sum_j lam_j [v(c_j+) - v(c_j-)] has a_base = 1, a_{j+} = +lam_j, a_{j-} = -lam_j.  No term is an external target: the
policy is never overridden, unlike a guide toward a plan (ledger p0_audit guide_override).  The K prefixes are embedded once
per query in one batched VLM pass; each Euler step is one batched action-expert pass over K copies of the chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Composition:
    """prompts[0] is the base.  coef: [K] (one coefficient per prompt) or [K, H*7] (per step and action dim, for gates
    inside a long chunk).  Coefficients over the prompts must sum to 1 at every entry."""
    prompts: list
    coef: np.ndarray
    names: list = field(default_factory=list)

    def check(self):
        c = np.asarray(self.coef, np.float64)
        assert c.shape[0] == len(self.prompts), (c.shape, len(self.prompts))
        assert np.allclose(c.sum(0), 1.0, atol=1e-6), "composition coefficients must sum to 1 (affine)"
        return self

    @staticmethod
    def factor(base, pairs, lams, names=None):
        """base prompt + sum_j lam_j [v(plus_j) - v(minus_j)]; pairs = [(plus, minus)], lams = [lam_j] (scalars or [H*7])."""
        prompts = [base]
        coef = [1.0]
        for (plus, minus), lam in zip(pairs, lams):
            prompts += [plus, minus]
            coef += [lam, -np.asarray(lam) if not np.isscalar(lam) else -lam]
        K = len(prompts)
        shape = max((np.shape(c) for c in coef), key=len)
        C = np.zeros((K,) + tuple(shape), np.float64)
        for k, c in enumerate(coef):
            C[k] = c
        return Composition(prompts, C.astype(np.float32), names or []).check()


@torch.no_grad()
def prefixes(bk, obs, prompts):
    """One batched prefix for the observation under each prompt (the VLM sees images and text together, so each prompt needs
    its own pass; they run as one batch)."""
    return bk.prefix(bk.observations([dict(obs, prompt=p) for p in prompts]))


@torch.no_grad()
def sample_composed(bk, pre, coef, seed=0, num_steps=None, noise=None, full=False):
    """Euler on the composed field from the policy's own noise path.  pre: a batched Prefix of K prompts (prefixes()); coef:
    [K] or [K, dim] over the real dims.  -> real rows [1, dim] (or [1, dim_full] with `full`).  With coef = e_0 this is
    PolicyBackbone.sample(1, prefix of prompt 0, seed, noise) up to the batching of the VLM pass."""
    K = int(pre.n)
    c = torch.as_tensor(np.asarray(coef, np.float32), device=bk.device)
    if c.ndim == 1:
        cfull = c[:, None].expand(K, bk.dim_full).clone()
    else:                                                    # per real dim; padded dims follow the base
        cfull = torch.zeros((K, bk.dim_full), device=bk.device)
        cfull[0] = 1.0
        cfull[:, bk.real] = c
    steps = int(num_steps or bk.num_steps)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = (torch.randn((1, bk.dim_full), generator=g).to(bk.device) if noise is None
         else torch.as_tensor(np.asarray(noise, np.float32), device=bk.device).reshape(1, bk.dim_full))
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=bk.device)
    t = torch.tensor(1.0, dtype=torch.float32, device=bk.device)
    while t >= -dt / 2:
        V = bk.v_full(x.expand(K, -1), t.expand(K), pre)       # [K, dim_full]: every prompt at the same noisy chunk
        v = (cfull * V).sum(0, keepdim=True)
        x = x + dt * v
        t = t + dt
    out = x if full else x[:, bk.real]
    return out.float().cpu().numpy()


@torch.no_grad()
def tweedie_means(bk, pre, noise):
    """Each prompt's one-step conditional mean at t = 1 (x - v(x, 1)): the cheap read of its chunk for offline probes
    (0.034 rms from the sample mean on pi0.5, against a within-observation spread of 0.020)."""
    K = int(pre.n)
    x = torch.as_tensor(np.asarray(noise, np.float32), device=bk.device).reshape(1, bk.dim_full).expand(K, -1)
    t = torch.ones(K, device=bk.device)
    V = bk.v_full(x, t, pre)
    return (x - V)[:, bk.real].float().cpu().numpy()


def contributions(V_rows, coef):
    """Per prompt, its share of the composed displacement: |a_k| * ||row_k - row_0|| (interpretability: which factor
    moved this chunk, and by how much)."""
    R = np.asarray(V_rows, np.float64)
    c = np.asarray(coef, np.float64)
    c = c if c.ndim == 1 else c.mean(-1)
    return [float(abs(c[k]) * np.linalg.norm(R[k] - R[0])) for k in range(len(R))]
