"""PolicyBackbone: openpi's PyTorch pi0 / pi0.5 action expert as a score field over action chunks.

openpi trains the action expert by flow matching on x_t = t * eps + (1 - t) * a (t = 1 noise, t = 0 data)
with target u = eps - a and samples by Euler x <- x + dt * v, dt = -1/num_steps.  The identities this
module rests on (docs/PLAN_2026-09.md Section 3):

    E[a | x_t]   = x_t - t * v            (Tweedie)
    E[eps | x_t] = x_t + (1 - t) * v
    score(x_t)   = grad log p_t(x_t) = -E[eps | x_t] / t

So every operator that only needs eps / score / a sampler (tree.build, score_ops.mode_ascent,
laplace_diag, operators.decompose's guide, operators.fold's gate) runs on a policy through this handle.
Rows are FLATTENED NORMALISED action chunks over the REAL action dims, [H * 7] (H = action_horizon); the
model's 32-D padded dims (trained on zeros, so the expert predicts v = eps there and Tweedie = 0) are filled
with N(0, t^2) noise on the way in -- the exact marginal of x_t in those dims, independent of the real ones,
so the real components of the score are unaffected -- and dropped on the way out.  The policy's own
Normalize / Unnormalize transforms map to and from the dataset's 7-D actions.

Conditioning is a PREFIX: images + language (+ state) embedded once into the VLM's KV cache
(`prefix()`); every call at (x_t, t) then costs one action-expert forward.  A null prefix (for the
motor-vocabulary view) is the prefix of a neutral prompt with masked images; whether the fine-tune ever
saw such inputs is an empirical question M2 reports.
"""
from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np
import torch

OPENPI = "/coc/flash10/ksingara3/robot/openpi"


def _tt(t, device):
    """A time (float, numpy or tensor, scalar or [n]) -> float32 tensor on `device`, flattened."""
    if torch.is_tensor(t):
        return t.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.as_tensor(np.asarray(t, np.float32), device=device).reshape(-1)


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
    path = "flow"                # x_t = (1 - t) a + t eps: backbones.schedule.path_coeffs

    def __init__(self, config_name="pi05_libero", checkpoint_dir=None, device="cuda", num_steps=10, compile=False):
        _config, _policy_config, _model = _import_openpi()
        ck = checkpoint_dir or "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch"
        self.train_config = _config.get_config(config_name)
        if getattr(self.train_config.model, "pytorch_compile_mode", None) is not None and not compile:
            # openpi wraps sample_actions in torch.compile(mode="max-autotune"): minutes of autotuning on the
            # first call and a recompile per new shape.  Our workload changes shapes constantly and calls
            # denoise_step directly, so the backbone runs eager (the reference sampler too, for parity).
            self.train_config = dataclasses.replace(self.train_config, model=dataclasses.replace(self.train_config.model, pytorch_compile_mode=None))
        t0 = time.time()
        self.policy = _policy_config.create_trained_policy(self.train_config, ck, pytorch_device=device)
        assert self.policy._is_pytorch_model, "PolicyBackbone needs openpi's PyTorch model (model.safetensors)"
        self.model = self.policy._model.eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.device = device
        self.H = int(self.train_config.model.action_horizon)
        self.A = int(self.train_config.model.action_dim)          # 32, padded
        self.a = 7                                                # real action dims (LIBERO)
        self.dim = self.H * self.a                                # the row space: 70 for H = 10
        self.dim_full = self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
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
    def observations(self, obs_dicts):
        """A list of raw observation dicts -> one batched Observation (for per-row conditioning)."""
        import jax
        ins = [self.policy._input_transform(dict(o)) for o in obs_dicts]
        stacked = jax.tree.map(lambda *xs: torch.from_numpy(np.stack([np.array(x) for x in xs])).to(self.device), *ins)
        return self._Observation.from_dict(stacked)

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

    def pad(self, X, t, seed=0):
        """[n, dim] real rows -> [n, H, A] model input, padded dims ~ N(0, t^2) (their exact marginal)."""
        n = len(X)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        full = torch.randn((n, self.dim_full), generator=g).to(self.device)
        tt = _tt(t, self.device).reshape(-1, 1)
        full = full * tt
        full[:, self.real] = X
        return full.reshape(n, self.H, self.A)

    def repeat(self, prefix, k):
        """Repeat every row of a batched prefix k times (row i -> rows i*k .. i*k+k-1).  The expert forward reads the
        cache without mutating it (use_cache=False concatenates), so the copies are safe to share across steps."""
        k = int(k)
        rep = lambda t: t.repeat_interleave(k, dim=0)          # noqa: E731
        kv = prefix.kv
        kv2 = type(kv)()
        for layer, (kk, vv) in enumerate(zip(kv.key_cache, kv.value_cache)):
            kv2.update(rep(kk), rep(vv), layer)
        return Prefix(state=rep(prefix.state), pad_masks=rep(prefix.pad_masks), kv=kv2, n=prefix.n * k)

    def v_full(self, x_full, t, prefix, gate=None):
        """The velocity field on FULL [n, H, A] (or [n, H * A]) inputs.  -> torch [n, H * A]."""
        x = torch.as_tensor(x_full, dtype=torch.float32, device=self.device).reshape(-1, self.H, self.A)
        n = len(x)
        pre = self._expand(prefix, n)
        tt = _tt(t, self.device).reshape(-1)
        tt = tt.expand(n) if tt.numel() == 1 else tt
        import contextlib
        with (gate() if gate is not None else contextlib.nullcontext()):
            vt = self.model.denoise_step(pre.state, pre.pad_masks, pre.kv, x, tt)
        return vt.reshape(n, self.dim_full).float()

    def v(self, x_rows, t, prefix=None, gate=None):
        """The velocity field on real rows [n, dim] at time t (float or [n]).  -> torch [n, dim]."""
        X = torch.as_tensor(x_rows, dtype=torch.float32, device=self.device)
        return self.v_full(self.pad(X, t), t, self._pre(prefix), gate)[:, self.real]

    def bind(self, prefix):
        """Set the conditioning used when a caller (score_ops, tree.build) passes no prefix."""
        self._bound = prefix
        return self

    def _pre(self, prefix):
        pre = prefix if prefix is not None else getattr(self, "_bound", None)
        if pre is None:
            raise ValueError("PolicyBackbone: pass a prefix or bind() one first")
        return pre

    def eps(self, X, t, prefix=None, gate=None, cls=None, emb=None, batch=None, **_):
        """E[eps | x_t] = x_t + (1 - t) v on real rows.  Same signature as the image Backbone (cls / emb
        ignored) so score_ops runs unchanged with a bound prefix."""
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        tt = _tt(t, self.device).reshape(-1, 1)
        return X + (1.0 - tt) * self.v(X, t, self._pre(prefix), gate)

    def score(self, X, t, prefix=None, gate=None):
        tt = _tt(t, self.device).reshape(-1, 1)
        return -self.eps(X, t, prefix, gate) / tt

    def tweedie(self, X, t, prefix=None, gate=None):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        tt = _tt(t, self.device).reshape(-1, 1)
        return X - tt * self.v(X, t, self._pre(prefix), gate)

    # ── sampling (Euler on the flow, with the guide / gate hooks of backbones.sampler) ───
    @torch.no_grad()
    def sample(self, n, prefix, seed=0, num_steps=None, guide=None, step_gate=None, noise=None, full=False):
        """Euler on the flow over the FULL padded chunk, exactly openpi's sample_actions when no guide / gate
        is given (the same noise gives the same actions).  -> real rows [n, dim] (or [n, H * A] with `full`).
        `guide=(eps_fn, w_fn)` mixes an analytic eps (on real rows) into the prediction, v = (eps_hat - x_t) /
        (1 - t) inverting the identity; `step_gate(x_rows, t)` sets an expert's weight per step."""
        K = int(num_steps or self.num_steps)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = (torch.randn((n, self.dim_full), generator=g).to(self.device) if noise is None
             else torch.as_tensor(noise, device=self.device).reshape(n, self.dim_full).float())
        dt = torch.tensor(-1.0 / K, dtype=torch.float32, device=self.device)
        t = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        while t >= -dt / 2:
            tf = float(t)
            gate = None
            if step_gate is not None:
                sg = step_gate(x[:, self.real], tf)
                gate = sg[0] if isinstance(sg, tuple) else sg
            vt = self.v_full(x, t.expand(n), prefix, gate)
            if guide is not None and tf < 1.0 - 1e-6:
                eps_fn, w_fn = guide
                xr = x[:, self.real]
                e0 = x + (1.0 - t) * vt
                et = torch.as_tensor(eps_fn(xr, tf), dtype=torch.float32, device=self.device)
                w = w_fn(xr, tf)
                if isinstance(w, np.ndarray):
                    w = torch.as_tensor(w, dtype=e0.dtype, device=e0.device)
                if torch.is_tensor(w):                  # per row [n] or per dimension [dim]
                    w = w.to(e0.dtype).reshape(1, -1) if w.numel() == self.dim else w.to(e0.dtype).reshape(-1, 1)
                else:
                    w = float(w)
                e = e0.clone()
                e[:, self.real] = e0[:, self.real] + w * (et - e0[:, self.real])
                vt = (e - x) / (1.0 - t)
            x = x + dt * vt
            t = t + dt
        out = x if full else x[:, self.real]
        return out.float().cpu().numpy()

    # ── normalisation to / from the dataset's action units ──────────────────
    def unnormalize(self, rows):
        """real rows [n, dim] -> dataset actions [n, H, 7] through the policy's own output transform."""
        R = np.asarray(rows, np.float32).reshape(-1, self.dim)
        out = []
        for r in R:
            full = np.zeros((self.H, self.A), np.float32)
            full.reshape(-1)[self.real] = r
            # openpi's Unnormalize walks every key of its norm stats (state and actions): give it a state
            out.append(self.policy._output_transform({"actions": full, "state": np.zeros(self.A, np.float32)})["actions"])
        return np.stack(out)

    def describe(self):
        return (f"PolicyBackbone({self.train_config.name}: H={self.H}, A={self.A}, real={self.a}, dim={self.dim}, "
                f"n_params={self.n_params:,}, expert={sum(p.numel() for p in self.expert.parameters()):,}, "
                f"device={self.device}, load {self.load_secs:.0f}s)")
