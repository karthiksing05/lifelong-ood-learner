"""PolicyBackbone: openpi's PyTorch pi0 / pi0.5 action expert as a score field over action chunks.

openpi trains the action expert by flow matching on x_t = t * eps + (1 - t) * a (t = 1 noise, t = 0 data)
with target u = eps - a and samples by Euler x <- x + dt * v, dt = -1/num_steps.  The identities this
module rests on (docs/PLAN_2026-09.md Section 3):

    E[a | x_t]   = x_t - t * v            (Tweedie)
    E[eps | x_t] = x_t + (1 - t) * v
    score(x_t)   = grad log p_t(x_t) = -E[eps | x_t] / t

So every operator that only needs eps / score / a sampler (tree.build, score_ops.mode_ascent,
laplace_diag, operators.decompose's guide, operators.fold's gate) runs on a policy through this handle.
Rows are FLATTENED NORMALISED action chunks [H * A] (H = action_horizon, A = action_dim, 32 padded);
the policy's own Normalize / Unnormalize transforms map to and from the dataset's 7-D actions.

Conditioning is a PREFIX: images + language (+ state) embedded once into the VLM's KV cache
(`prefix()`); every call at (x_t, t) then costs one action-expert forward.  A null prefix (for the
motor-vocabulary view) is the prefix of a neutral prompt with masked images; whether the fine-tune ever
saw such inputs is an empirical question M2 reports.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

import numpy as np
import torch

OPENPI = "/coc/flash10/ksingara3/robot/openpi"


def _import_openpi():
    if OPENPI + "/src" not in sys.path:
        sys.path.insert(0, OPENPI + "/src")
    import openpi.training.config as _config
    import openpi.policies.policy_config as _policy_config
    from openpi.models import model as _model
    return _config, _policy_config, _model


@dataclasses.dataclass
class Prefix:
    """The cached conditioning of one observation (or one batch of observations)."""
    state: torch.Tensor
    pad_masks: torch.Tensor
    kv: object
    n: int


class PolicyBackbone:
    kind = "pi0"
    space = "action"

    def __init__(self, config_name="pi05_libero", checkpoint_dir=None, device="cuda", num_steps=10):
        _config, _policy_config, _model = _import_openpi()
        ck = checkpoint_dir or "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch"
        self.train_config = _config.get_config(config_name)
        t0 = time.time()
        self.policy = _policy_config.create_trained_policy(self.train_config, ck, pytorch_device=device)
        assert self.policy._is_pytorch_model, "PolicyBackbone needs openpi's PyTorch model (model.safetensors)"
        self.model = self.policy._model.eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.device = device
        self.H = int(self.train_config.model.action_horizon)
        self.A = int(self.train_config.model.action_dim)
        self.dim = self.H * self.A
        self.num_steps = int(num_steps)
        self.clip = (-1e9, 1e9)                  # chunks are unbounded (normalised); no clamp
        self.layout = "chunk"
        self.img = None
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.expert = self.model.paligemma_with_expert.gemma_expert   # the action expert: where a fold lives
        self.load_secs = time.time() - t0
        self._Observation = _model.Observation

    # ── conditioning ────────────────────────────────────────────────────────
    @torch.no_grad()
    def observation(self, obs_dict):
        """A raw LIBERO observation dict (observation/image, observation/wrist_image, observation/state,
        prompt) -> the model's Observation (batch of 1), through the policy's own input transforms."""
        inputs = self.policy._input_transform(dict(obs_dict))
        import jax
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...], inputs)
        return self._Observation.from_dict(inputs)

    @torch.no_grad()
    def prefix(self, observation):
        """Embed the prefix once and cache its keys / values.  -> Prefix."""
        m = self.model
        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        att = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, kv = m.paligemma_with_expert.forward(attention_mask=m._prepare_attention_masks_4d(att), position_ids=pos,
                                                past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        return Prefix(state=state, pad_masks=prefix_pad_masks, kv=kv, n=int(state.shape[0]))

    # ── the field ───────────────────────────────────────────────────────────
    def _expand(self, prefix, n):
        """Broadcast a batch-1 prefix to n rows (the KV cache is re-used by index)."""
        if prefix.n == n:
            return prefix
        if prefix.n != 1:
            raise ValueError("prefix batch must be 1 or n")
        rep = lambda t: t.expand(n, *t.shape[1:]) if torch.is_tensor(t) else t   # noqa: E731
        kv = prefix.kv
        try:                                   # transformers DynamicCache
            kv2 = type(kv)()
            for k, v in zip(kv.key_cache, kv.value_cache):
                kv2.key_cache.append(k.expand(n, *k.shape[1:]))
                kv2.value_cache.append(v.expand(n, *v.shape[1:]))
        except Exception:
            kv2 = tuple((k.expand(n, *k.shape[1:]), v.expand(n, *v.shape[1:])) for k, v in kv)
        return Prefix(state=rep(prefix.state), pad_masks=rep(prefix.pad_masks), kv=kv2, n=n)

    def v(self, x_rows, t, prefix, gate=None):
        """The velocity field on flat rows at time t (float or [n]).  -> torch [n, dim]."""
        X = torch.as_tensor(x_rows, dtype=torch.float32, device=self.device)
        n = len(X)
        pre = self._expand(prefix, n)
        tt = torch.as_tensor(np.asarray(t, np.float32), device=self.device).reshape(-1)
        tt = tt.expand(n) if tt.numel() == 1 else tt
        x = X.reshape(n, self.H, self.A)
        import contextlib
        with (gate() if gate is not None else contextlib.nullcontext()):
            vt = self.model.denoise_step(pre.state, pre.pad_masks, pre.kv, x, tt)
        return vt.reshape(n, self.dim).float()

    def eps(self, X, t, prefix=None, gate=None, **_):
        """E[eps | x_t] = x_t + (1 - t) v.  Same name as the image Backbone so score_ops works unchanged."""
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        tt = torch.as_tensor(np.asarray(t, np.float32), device=self.device).reshape(-1, 1)
        return X + (1.0 - tt) * self.v(X, t, prefix, gate)

    def score(self, X, t, prefix=None, gate=None):
        tt = torch.as_tensor(np.asarray(t, np.float32), device=self.device).reshape(-1, 1)
        return -self.eps(X, t, prefix, gate) / tt

    def tweedie(self, X, t, prefix=None, gate=None):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        tt = torch.as_tensor(np.asarray(t, np.float32), device=self.device).reshape(-1, 1)
        return X - tt * self.v(X, t, prefix, gate)

    # ── sampling (Euler on the flow, with the guide / gate hooks of backbones.sampler) ───
    @torch.no_grad()
    def sample(self, n, prefix, seed=0, num_steps=None, guide=None, step_gate=None, noise=None):
        """-> flat normalised chunks [n, dim].  `guide=(eps_fn, w_fn)` mixes an analytic eps into the
        prediction (v = (eps_hat - x_t) / (1 - t) inverts the identity); `step_gate(x_rows, t)` sets an
        expert's weight per step."""
        K = int(num_steps or self.num_steps)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn((n, self.dim), generator=g).to(self.device) if noise is None else torch.as_tensor(noise, device=self.device).reshape(n, self.dim).float()
        dt = -1.0 / K
        t = 1.0
        for _ in range(K):
            gate = None
            if step_gate is not None:
                sg = step_gate(x, t)
                gate = sg[0] if isinstance(sg, tuple) else sg
            vt = self.v(x, t, prefix, gate)
            if guide is not None and t < 1.0 - 1e-6:
                eps_fn, w_fn = guide
                e0 = x + (1.0 - t) * vt
                et = torch.as_tensor(eps_fn(x, t), dtype=torch.float32, device=self.device)
                w = w_fn(x, t)
                w = w.to(e0.dtype).reshape(-1, 1) if torch.is_tensor(w) else float(w)
                e = e0 + w * (et - e0)
                vt = (e - x) / (1.0 - t)
            x = x + dt * vt
            t = t + dt
        return x.float().cpu().numpy()

    # ── normalisation to / from the dataset's action units ──────────────────
    def unnormalize(self, rows):
        """flat normalised chunks -> dataset actions [n, H, 7] through the policy's output transform."""
        out = []
        for r in np.asarray(rows, np.float32).reshape(-1, self.H, self.A):
            out.append(self.policy._output_transform({"actions": r})["actions"])
        return np.stack(out)

    def describe(self):
        return (f"PolicyBackbone({self.train_config.name}: H={self.H}, A={self.A}, dim={self.dim}, "
                f"n_params={self.n_params:,}, expert={sum(p.numel() for p in self.expert.parameters()):,}, "
                f"device={self.device}, load {self.load_secs:.0f}s)")
