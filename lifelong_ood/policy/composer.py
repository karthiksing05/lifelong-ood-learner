"""The COMPOSER: a learned product of experts that BORROWS parts of library prototypes.

A new task arrives with ONE demonstration.  Instead of storing each frame's chunk (replay) or per-frame mixing weights
fitted to it (the retracted "codes"), a tiny model learns how to borrow:

    w_k(h, d | z) = softmax_k( gate_k(z) + beta_k[h] + gamma_k[d] )            k over the task's vocabulary of M prototypes
    mu(h, d | z)  = sum_k w_k m_k[h, d] / v_k[h, d]  /  sum_k w_k / v_k[h, d]  (precision-weighted product of experts)

gate(z) says WHICH prototypes the current moment draws on (z = the normalised robot state and the progress through the
demo), beta_k says WHEN within the 10-step chunk prototype k is borrowed, gamma_k WHICH action dimensions (a prototype may
lend its arm path and not its gripper, or the reverse).  The composed chunk is a smooth blend of real library motions.
Everything the task stores: the M prototype ids, the model (~3k parameters, fp16), and the demo's states every 5 frames
(alignment and progress).  The library is shared and charged once.
"""
from __future__ import annotations

import collections
import json

import numpy as np
import torch
import torch.nn as nn

from ..operators.decompose import candidates, loglik_perdim, submodular_select
from .compose import ALIGN_DIMS, ComposeHooks, state_norm

H, A7 = 10, 7


def vocabulary(tree, rows, levels, M=32, K=4):
    """The M library prototypes most often selected (submodular per-dimension coverage, K per frame) across the demo."""
    ids, lv, Mm, V = candidates(tree, levels)
    count = collections.Counter()
    for r in np.asarray(rows, np.float32):
        count.update(submodular_select(loglik_perdim(r, Mm, V), K).tolist())
    top = [i for i, _ in count.most_common(M)]
    return [int(ids[i]) for i in top], float(sum(c for _, c in count.most_common(M)) / max(sum(count.values()), 1))


class Composer(nn.Module):
    def __init__(self, M, zdim=9, hidden=32):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(zdim, hidden), nn.Tanh(), nn.Linear(hidden, M))
        self.beta = nn.Parameter(torch.zeros(M, H))
        self.gamma = nn.Parameter(torch.zeros(M, A7))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def weights(self, z):
        a = self.gate(z)                                                      # [n, M]
        logits = (a[:, :, None, None] + self.beta[None, :, :, None] + self.gamma[None, :, None, :]) / self.log_tau.exp()
        return torch.softmax(logits, dim=1)                                   # [n, M, H, D]

    def forward(self, z, Mk, Vk):
        """z [n, zdim]; Mk, Vk [M, H, D] prototype means / variances (normalised).  -> mu [n, H*D], var [n, H*D], w."""
        w = self.weights(z)
        prec = (w / Vk[None]).sum(1)
        mu = (w * Mk[None] / Vk[None]).sum(1) / prec
        return mu.reshape(len(z), -1), (1.0 / prec).reshape(len(z), -1), w

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))


class PriorComposer(Composer):
    """M library prototypes + ONE MORE expert: pi0.5's own chunk on the CURRENT observation (visually grounded, closed-loop).
    The learned borrowing decides, per step and dimension, what to take from the frozen policy and what from the library;
    the prior expert's variance is learned per (step, dimension)."""

    def __init__(self, M, zdim=9, hidden=32):
        super().__init__(M + 1, zdim, hidden)
        self.log_vp = nn.Parameter(torch.full((H, A7), float(np.log(1e-2))))

    def forward(self, z, Mk, Vk, prior):
        w = self.weights(z)                                                   # [n, M+1, H, D]
        n = len(z)
        P = torch.as_tensor(prior, dtype=torch.float32).reshape(n, H, A7)
        Vp = self.log_vp.exp()
        num = (w[:, :-1] * Mk[None] / Vk[None]).sum(1) + w[:, -1] * P / Vp
        prec = (w[:, :-1] / Vk[None]).sum(1) + w[:, -1] / Vp
        return (num / prec).reshape(n, -1), (1.0 / prec).reshape(n, -1), w


def features(states, phase):
    """z = the normalised 8-D robot state and the progress through the demo (0..1)."""
    s = state_norm()(np.atleast_2d(states)).astype(np.float32)
    return np.concatenate([s, np.asarray(phase, np.float32).reshape(-1, 1)], 1)


def prototypes(tree, ids):
    Mk = np.stack([tree.nodes[i].mode for i in ids]).reshape(len(ids), H, A7).astype(np.float32)
    Vk = np.stack([tree.nodes[i].var for i in ids]).reshape(len(ids), H, A7).astype(np.float32)
    return torch.as_tensor(Mk), torch.as_tensor(np.maximum(Vk, 1e-4))


def fit(comp, Z, A, Mk, Vk, steps=3000, lr=3e-3, lam_ent=1e-3, lam_smooth=1e-2, seed=0, log=None, prior=None):
    """Reconstruct the demo chunks A [T, 70] from z [T, 9]; entropy keeps the borrowing sparse, the smoothness term keeps
    consecutive frames' borrowing close.  -> dict of curves."""
    torch.manual_seed(seed)
    Z = torch.as_tensor(np.asarray(Z, np.float32))
    A = torch.as_tensor(np.asarray(A, np.float32))
    opt = torch.optim.Adam(comp.parameters(), lr=lr)
    hist = []
    Pr = None if prior is None else torch.as_tensor(np.asarray(prior, np.float32))
    for s in range(1, steps + 1):
        mu, var, w = comp(Z, Mk, Vk) if Pr is None else comp(Z, Mk, Vk, Pr)
        rec = ((mu - A) ** 2).mean()
        ent = -(w * (w + 1e-9).log()).sum(1).mean()
        smooth = ((w[1:] - w[:-1]) ** 2).sum(1).mean() if len(w) > 1 else torch.zeros(())
        loss = rec + lam_ent * ent + lam_smooth * smooth
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == 1:
            hist.append(dict(step=s, rms=float(rec.sqrt()), entropy=float(ent), smooth=float(smooth)))
            if log:
                log(f"[composer] step {s}: rms {float(rec.sqrt()):.4f} entropy {float(ent):.3f} smooth {float(smooth):.4f}")
    return hist


class ComposerProgram:
    """What a task stores: vocabulary ids, the composer's parameters (fp16), the demo's states every `stride` frames."""

    def __init__(self, ids, state_dict, states, T, stride=5, zdim=9, hidden=32, library="", kind="plain"):
        self.ids, self.sd, self.states, self.T, self.stride = list(ids), state_dict, np.asarray(states, np.float32), int(T), int(stride)
        self.zdim, self.hidden, self.library, self.kind = zdim, hidden, library, kind

    def model(self):
        c = (PriorComposer if self.kind == "prior" else Composer)(len(self.ids), self.zdim, self.hidden)
        c.load_state_dict({k: torch.as_tensor(np.asarray(v, np.float32)) for k, v in self.sd.items()})
        return c.eval()

    def bytes(self):
        params = sum(int(np.asarray(v).size) for v in self.sd.values())
        return int(2 * len(self.ids) + 2 * params + 2 * self.states.size)

    def save(self, path):
        np.savez_compressed(path, ids=np.asarray(self.ids), states=self.states, meta=json.dumps(dict(T=self.T, stride=self.stride,
                            zdim=self.zdim, hidden=self.hidden, library=self.library, kind=self.kind)),
                            **{f"p__{k}": np.asarray(v, np.float16) for k, v in self.sd.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        sd = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith("p__")}
        return cls(z["ids"].tolist(), sd, z["states"], meta["T"], meta["stride"], meta["zdim"], meta["hidden"], meta["library"],
                   meta.get("kind", "plain"))


class _KeyProg:
    """The minimal program interface ComposeHooks needs for alignment (keys / states / length)."""

    def __init__(self, states):
        self.states = np.asarray(states, np.float32)
        self.key = state_norm()(self.states)[:, list(ALIGN_DIMS)]
        self.mu = np.zeros((len(self.states), H * A7), np.float32)
        self.var = np.ones_like(self.mu)
        self.fold = None
        self.steer = None
        self.prompt_id = None


class ComposerHooks(ComposeHooks):
    """Align to the demo (its states every `stride` frames) for progress and stall detection, then guide the frozen
    policy toward the composed chunk for the CURRENT state: z = (live robot state, progress)."""

    def __init__(self, prog, tree, **kw):
        super().__init__(_KeyProg(prog.states), **kw)
        self.cprog = prog
        self.comp = prog.model()
        self.Mk, self.Vk = prototypes(tree, prog.ids)
        self._state = None
        self.needs_prior = prog.kind == "prior"
        self._prior = None
        self.w_prior = []

    def set_prior(self, row):
        self._prior = np.asarray(row, np.float32)

    def reset(self, meta=None):
        if self.w_prior:
            import logging
            logging.info(f"[composer] episode: mean weight on the prior expert {np.mean(self.w_prior):.2f}")
        self.w_prior = []
        super().reset(meta)

    def rewrite(self, obs, meta=None):
        self._state = np.asarray(obs["observation/state"], np.float32)
        return super().rewrite(obs, meta)

    def guide_at(self, j):
        from ..operators.decompose import guide
        self._jg = j
        phase = min(1.0, j / max(len(self.prog.key) - 1, 1))
        z = torch.as_tensor(features(self._state, [phase]))
        with torch.no_grad():
            if self.needs_prior:
                mu, var, w = self.comp(z, self.Mk, self.Vk, self._prior[None])
                self.w_prior.append(float(w[0, -1].mean()))
            else:
                mu, var, _ = self.comp(z, self.Mk, self.Vk)
        mu = mu[0].numpy()
        v = np.full_like(mu, self.var if self.var is not None else 1e-3)
        self.prog.mu[j] = mu                       # for logging / agreement reads
        if self.schedule == "const":
            return guide(mu, v, wmax=self.w0, schedule="const", path="flow")
        return guide(mu, v, w0=self.w0, wmax=self.wmax, path="flow")


class ComposerHybridHooks(ComposerHooks):
    """COMPOSE with the learned borrowing everywhere, FOLD where the demo needs closed-loop precision: inside the fold
    windows (approach and contact around the demo's gripper events, `fold_mask` over demo frames) the gated expert is
    switched on and the program is silent."""

    def __init__(self, prog, tree, state, key, fold_mask, **kw):
        super().__init__(prog, tree, **kw)
        from ..experts.lora import keys_on
        self.on = keys_on(state, {key})
        self.fold_mask = np.asarray(fold_mask, bool)

    def __call__(self, obs, pre, meta=None):
        j = self._frame(obs)
        f = min(j * self.cprog.stride, len(self.fold_mask) - 1)
        if self.fold_mask[f]:
            return None, (lambda x, t: self.on)
        if self._released_now():
            return None, None
        return self.guide_at(j), None
