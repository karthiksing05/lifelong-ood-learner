"""DSRL (Wagenmaker et al., CoRL 2025, "Steering Your Diffusion Policy with Latent Space Reinforcement Learning"), for the flow
policy pi0.5: the policy's weights stay frozen and a small SAC agent chooses the INITIAL NOISE of each sampled action chunk.

Latent action z in [-beta, beta]^(H x A) (tanh-squashed Gaussian; H x A = 10 x 32, the padded chunk the flow integrates from), the
observation = frozen ResNet-18 features of both cameras + the robot's 8-D state, sparse reward 1 when the episode succeeds (on its
last chunk) else 0, terminal at episode end.  The entropy term is per dimension (initial temperature 0.01, target -0.5 per dimension):
summed over 320 dimensions it is ~480 nats per step and the critic diverges on a 0/1 reward (seen in the smoke run).  The replay buffer starts with `warmup` episodes of ordinary Gaussian noise (the base
policy); after each later episode the agent takes utd x (that episode's chunks) gradient steps.  The client reports each episode's
outcome at the next episode's start (meta prev_success), so the last training episode's outcome is not used.

    hooks = DSRLHooks(feat, dim=320, train=True, save=path)       # GuidedPolicy asks hooks.noise_for(obs, meta) for every query
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, sdim, dim, beta, h=256):
        super().__init__()
        self.net, self.beta = mlp(sdim, 2 * dim, h), beta

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-5, 2)
        if deterministic:
            return self.beta * torch.tanh(mu), None
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        z = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (torch.log(self.beta * (1 - z.pow(2)) + 1e-6)).sum(-1)
        return self.beta * z, logp / z.shape[-1]                       # per dimension: 320 dims' entropy would swamp a 0/1 reward


class DSRLHooks:
    agree = None
    needs_prior = False

    def __init__(self, feat, dim, train=True, save=None, load=None, warmup=10, utd=10, batch=256, gamma=0.99, lr=3e-4, beta=2.0,
                 device="cuda", seed=0, log=print):
        self.feat, self.dim, self.train, self.save_path = feat, int(dim), bool(train), save
        self.warmup, self.utd, self.batch, self.gamma, self.beta = int(warmup), int(utd), int(batch), float(gamma), float(beta)
        self.device, self.log = device, log
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.sdim = None
        self.cur = []
        self.buf = dict(s=[], z=[], r=[], s2=[], d=[])
        self.n_eps = 0
        self.outcomes = []
        self.load_path = load
        self.lr = lr

    def _build(self, sdim):
        self.sdim = sdim
        d = self.device
        self.actor = Actor(sdim, self.dim, self.beta).to(d)
        self.q = nn.ModuleList([mlp(sdim + self.dim, 1) for _ in range(2)]).to(d)
        self.qt = nn.ModuleList([mlp(sdim + self.dim, 1) for _ in range(2)]).to(d)
        self.qt.load_state_dict(self.q.state_dict())
        self.log_alpha = torch.full((1,), math.log(0.01), device=d, requires_grad=True)
        self.target_entropy = -0.5                                          # per dimension
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=self.lr)
        if self.load_path:
            z = torch.load(self.load_path, map_location=d)
            self.actor.load_state_dict(z["actor"])
            self.log(f"[dsrl] loaded actor {self.load_path} (trained on {z.get('episodes')} episodes)")

    # ── the episode loop ──
    def reset(self, meta=None):
        meta = meta or {}
        if self.cur and "prev_success" in meta and meta["prev_success"] is not None:
            r = float(bool(meta["prev_success"]))
            self.outcomes.append(r)
            for k, (s, z) in enumerate(self.cur):
                last = k == len(self.cur) - 1
                self.buf["s"].append(s); self.buf["z"].append(z)
                self.buf["r"].append(r if last else 0.0)
                self.buf["s2"].append(s if last else self.cur[k + 1][0])
                self.buf["d"].append(1.0 if last else 0.0)
            self.n_eps += 1
            if self.train and self.n_eps >= self.warmup:
                info = self.update(self.utd * len(self.cur))
                self.log(f"[dsrl] episode {self.n_eps}: success {r:.0f} (last 10: {np.mean(self.outcomes[-10:]):.2f}); {info}")
            elif self.train:
                self.log(f"[dsrl] warm-up episode {self.n_eps}: success {r:.0f}")
            if self.train and self.save_path:
                torch.save(dict(actor=self.actor.state_dict(), episodes=self.n_eps, outcomes=self.outcomes, dim=self.dim,
                                beta=self.beta, sdim=self.sdim), self.save_path)
        self.cur = []

    def noise_for(self, obs, meta=None):
        s = np.asarray(self.feat(obs), np.float32)
        if self.sdim is None:
            self._build(len(s))
        if self.train and self.n_eps < self.warmup:
            z = np.clip(self.rng.standard_normal(self.dim), -self.beta, self.beta).astype(np.float32)
        else:
            with torch.no_grad():
                zt, _ = self.actor(torch.as_tensor(s, device=self.device)[None], deterministic=not self.train)
            z = zt[0].cpu().numpy().astype(np.float32)
        self.cur.append((s, z))
        return z.reshape(1, -1)

    def __call__(self, obs, pre, meta=None):
        return None, None

    # ── SAC ──
    def update(self, n):
        B = len(self.buf["r"])
        if B < 2:
            return {}
        S = torch.as_tensor(np.stack(self.buf["s"]), device=self.device)
        Z = torch.as_tensor(np.stack(self.buf["z"]), device=self.device)
        Rw = torch.as_tensor(np.asarray(self.buf["r"], np.float32), device=self.device)
        S2 = torch.as_tensor(np.stack(self.buf["s2"]), device=self.device)
        Dn = torch.as_tensor(np.asarray(self.buf["d"], np.float32), device=self.device)
        for _ in range(n):
            i = torch.as_tensor(self.rng.integers(0, B, min(self.batch, B)), device=self.device)
            s, z, r, s2, d = S[i], Z[i], Rw[i], S2[i], Dn[i]
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                z2, lp2 = self.actor(s2)
                qn = torch.min(*[qt(torch.cat([s2, z2], -1)).squeeze(-1) for qt in self.qt]) - alpha * lp2
                y = r + self.gamma * (1 - d) * qn
            ql = sum(F.mse_loss(q(torch.cat([s, z], -1)).squeeze(-1), y) for q in self.q)
            self.opt_q.zero_grad(); ql.backward(); self.opt_q.step()
            za, lp = self.actor(s)
            qa = torch.min(*[q(torch.cat([s, za], -1)).squeeze(-1) for q in self.q])
            al = (alpha * lp - qa).mean()
            self.opt_a.zero_grad(); al.backward(); self.opt_a.step()
            el = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
            self.opt_al.zero_grad(); el.backward(); self.opt_al.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                    pt.mul_(0.995).add_(0.005 * p)
        return dict(updates=n, buffer=B, q_loss=round(float(ql), 4), alpha=round(float(self.log_alpha.exp()), 4))
