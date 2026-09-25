"""A Cobweb-form weight update: absorb a genuinely new concept into the backbone's WEIGHTS, sparsely,
instead of adding a block.

WHY THIS EXISTS.  RESULTS 5q Phase B: on a backbone that never rendered background 3, conditioning
alone scores bg 0.000 and only an adapter trained on REAL images works (0.862, 551K parameters per
concept).  Every cheap tier re-addresses something the backbone already renders; genuinely new
content has so far cost added structure.  This module is the other road -- change the backbone
itself, but only where the new concept lives.

THE CORRESPONDENCE (derived in docs/COBWEB_UPDATE.md).  For the Bayes-optimal denoiser of an
empirical distribution, D_n(x_t) = sum_i w_i(x_t) x_i with softmax kernel responsibilities w_i,
adding one instance x_{n+1} gives EXACTLY

    D_{n+1} = D_n + r(x_t) (x_{n+1} - D_n),        r = k_{n+1} / (K_n + k_{n+1}),

and the posterior covariance moves as (1-r) S + r(1-r)(x-D)(x-D)^T.  That is Cobweb's incremental
mean/variance update with the hard share 1/(c+1) replaced by a soft responsibility, applied at every
noise level at once: at high noise r ~ 1/(n+1) (the root -- a minuscule change everything shares),
near the new instance at low noise r -> 1 (its own leaf).  A network cannot be edited per input, so
the three properties that make the ideal update Cobweb-like are carried into weight space as:

  * SPARSITY -- only a small set of parameters moves, chosen as the ones the new concept needs and
    the old concepts do not use:  score_j = E[g_new_j^2] / (F_old_j + delta), top rho globally.
    The weight-space reading of "rewrite the leaf, leave the ancestors".
  * ANCHORING -- old concepts' denoiser is held at its pre-update value on the model's OWN samples of
    them (no stored data): the r ~ 0 half of the update, written as a loss.
  * ABSORB UNTIL IN-DISTRIBUTION -- the update stops as soon as the concept's denoising loss on its
    own held-back instances reaches the level of a trained class.  The statistic the router
    thresholds decides when the update is done, so a concept that is still largely novel moves the
    weights more and one that is nearly addressable moves them less.

Everything here takes its conditioning EXPLICITLY -- a vector or a trained class id -- and every
forward pass that should see an adapter runs inside its `gate`.  `sample_concept`'s VE branch
dropped the adapter path silently (RESULTS 7kk); an explicit gate is how that cannot recur here.
"""
import contextlib

import numpy as np
import torch
import torch.nn as nn

from ..score import cosine_abar


# ── layout: MUST match ddpm._as_img / ddpm._as_flat ──────────────────────────
def to_img(T, img, layout):
    """Flat rows (HWC convention) -> the (N, 3, H, W) tensor the checkpoint was trained on."""
    if layout == "hwc":
        return T.reshape(-1, img, img, 3).permute(0, 3, 1, 2).contiguous()
    return T.reshape(-1, 3, img, img)


def to_flat(T, layout):
    """Inverse of `to_img`."""
    if layout == "hwc":
        return T.permute(0, 2, 3, 1).reshape(len(T), -1)
    return T.reshape(len(T), -1)


def abar_of(u):
    """torch u -> torch abar on the same device, through the repository's cosine schedule."""
    a = cosine_abar(u.detach().cpu().numpy().astype(np.float64))
    return torch.as_tensor(a, dtype=torch.float32, device=u.device)


def _gate(gate):
    return gate() if gate is not None else contextlib.nullcontext()


def _cond(model, m, device, emb=None, cls=None, i0=0):
    """-> (c, emb_override) for rows i0:i0+m.  `emb` is [d] or [N, d]; `cls` an int or [N] ids."""
    if emb is not None:
        E = emb if emb.dim() == 2 else emb[None].expand(m, -1)
        return None, E[i0:i0 + m] if emb.dim() == 2 else E
    if cls is None:
        return torch.full((m,), model.null_id, device=device, dtype=torch.long), None
    c = torch.as_tensor(cls, device=device, dtype=torch.long)
    if c.dim() == 0:
        return c.expand(m).clone(), None
    return c[i0:i0 + m], None


# ── sampling ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_sample(model, n, steps=50, seed=0, emb=None, cls=None, layout="permuted", img=32,
                device="cuda", batch=256, gate=None, clip=(-1.0, 1.0)):
    """DDIM (eta = 0) on the cosine schedule -> flat MODEL-space rows [n, 3*img*img] in [-1, 1].

    Mirrors `BaseScoreField.sample`: the starting noise is `np.random.default_rng(seed)` drawn in the
    flat row space, the u grid is linspace(1, 0, steps + 1), and x0 is clipped to the model's space
    at every step.  So a number sampled here is comparable with a table sampled through the field,
    but each step is ONE forward pass and the conditioning is explicit.

    `clip` is the space the model's x0 lives in: (-1, 1) for this repository's UNets, (0, 1) for the
    VE bridge, whose rows are data-space.  Passing the wrong one clamps half the image away at every
    step -- the fault behind RESULTS 7bb's first false negative.
    """
    D = 3 * img * img
    noise = np.random.default_rng(seed).standard_normal((n, D)).astype(np.float32)
    us = np.linspace(1.0, 0.0, steps + 1)
    ab = cosine_abar(us)
    E = None if emb is None else torch.as_tensor(np.asarray(emb, np.float32), device=device)
    out = np.empty((n, D), np.float32)
    model.eval()
    with _gate(gate):
        for i0 in range(0, n, batch):
            x = to_img(torch.from_numpy(noise[i0:i0 + batch]).to(device), img, layout)
            m = len(x)
            c, eo = _cond(model, m, device, emb=E, cls=cls, i0=i0)
            for i in range(steps):
                a, an = float(ab[i]), float(ab[i + 1])
                e = model(x, torch.full((m,), float(us[i]), device=device), c=c, emb_override=eo)
                x0 = ((x - (1.0 - a) ** 0.5 * e) / a ** 0.5).clamp(*clip)
                x = an ** 0.5 * x0 + (1.0 - an) ** 0.5 * e
            out[i0:i0 + m] = to_flat(x.clamp(*clip), layout).cpu().numpy()
    return out


# ── the denoising-error profile, on common random numbers ────────────────────
@torch.no_grad()
def eps_profile(model, X, u_grid, emb=None, cls=None, n_noise=4, seed=0, layout="permuted",
                img=32, device="cuda", batch=256, gate=None, per_image=False):
    """Mean eps-prediction error of MODEL-space rows `X` at each u -> [len(u)] or [n, len(u)].

    The (u, epsilon) draws are seeded HERE, independent of the caller's RNG, so two calls that differ
    only in the model (before / after an update) or the conditioning see identical noise, and their
    difference is the model's rather than the draw's.
    """
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    n = len(Xt)
    E = None if emb is None else torch.as_tensor(np.asarray(emb, np.float32), device=device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.zeros(n, len(u_grid), dtype=torch.float64)
    model.eval()
    with _gate(gate):
        for j, u in enumerate(u_grid):
            a = float(cosine_abar(float(u)))
            for _ in range(n_noise):
                eps = torch.randn(tuple(Xt.shape), generator=g).to(device)
                for i0 in range(0, n, batch):
                    xb, eb = Xt[i0:i0 + batch], eps[i0:i0 + batch]
                    m = len(xb)
                    c, eo = _cond(model, m, device, emb=E, cls=cls, i0=i0)
                    xt = to_img(a ** 0.5 * xb + (1.0 - a) ** 0.5 * eb, img, layout)
                    pred = to_flat(model(xt, torch.full((m,), float(u), device=device), c=c,
                                         emb_override=eo), layout)
                    out[i0:i0 + m, j] += ((pred - eb) ** 2).mean(1).double().cpu()
    out /= n_noise
    return out.numpy() if per_image else out.mean(0).numpy()


@torch.no_grad()
def candidate_losses(model, X, u_grid, embs=None, cls=None, n_noise=2, seed=0, layout="permuted",
                     img=32, device="cuda", batch=1024, gate=None, per_u=False):
    """Mean eps-loss of the SAME rows `X` under each of C candidate conditionings -> [C]
    (or [C, len(u_grid)] with `per_u`, the profile a per-noise-level criterion needs).

    Every candidate sees the identical (u, epsilon) draw for each image -- the router's correctness
    requirement: with independent draws the differences between conditionings are swamped by the
    variance of the draw.  `embs` [C, d] or `cls` [C] ids.
    """
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    n = len(Xt)
    E = None if embs is None else torch.as_tensor(np.asarray(embs, np.float32), device=device)
    Cc = None if cls is None else torch.as_tensor(np.asarray(cls), device=device, dtype=torch.long)
    C = len(E) if E is not None else len(Cc)
    per = max(1, batch // max(n, 1))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    tot = torch.zeros(C, len(u_grid), dtype=torch.float64)
    model.eval()
    with _gate(gate):
        for ju, u in enumerate(u_grid):
            a = float(cosine_abar(float(u)))
            for _ in range(n_noise):
                eps = torch.randn(tuple(Xt.shape), generator=g).to(device)
                xt = to_img(a ** 0.5 * Xt + (1.0 - a) ** 0.5 * eps, img, layout)
                for c0 in range(0, C, per):
                    m = min(per, C - c0)
                    xb = xt.repeat(m, 1, 1, 1)                       # candidate-major blocks
                    eo = None if E is None else E[c0:c0 + m].repeat_interleave(n, 0)
                    cb = None if Cc is None else Cc[c0:c0 + m].repeat_interleave(n)
                    pred = to_flat(model(xb, torch.full((m * n,), float(u), device=device), c=cb,
                                         emb_override=eo), layout)
                    err = ((pred - eps.repeat(m, 1)) ** 2).mean(1).view(m, n).mean(1)
                    tot[c0:c0 + m, ju] += err.double().cpu()
    prof = (tot / n_noise).numpy()
    return prof if per_u else prof.mean(1)


def eps_loss(model, x0, u_lo=1e-3, u_hi=1.0, emb=None, cls=None, gen=None):
    """The training objective on image-shaped x0: E || eps_theta(x_u, u, c) - eps ||^2."""
    dev = x0.device
    u = torch.rand(len(x0), generator=gen, device="cpu").to(dev) * (u_hi - u_lo) + u_lo
    a = abar_of(u).view(-1, 1, 1, 1)
    e = torch.randn(x0.shape, generator=gen, device="cpu").to(dev)
    xt = a.sqrt() * x0 + (1.0 - a).sqrt() * e
    c, eo = _cond(model, len(x0), dev, emb=emb, cls=cls)
    return ((model(xt, u, c=c, emb_override=eo) - e) ** 2).mean()


def anchor_loss(model, teacher, x0, cls=None, emb=None, gen=None, teacher_gate=None, gate=None,
                t_emb=None):
    """|| eps_theta(x_u) - eps_teacher(x_u) ||^2 on the same draw: holding an old denoiser where it
    was, on inputs the model itself generated (no stored data).

    `t_emb` is the TEACHER's conditioning when it differs from the student's.  A concept conditioned
    through a shared direction that has since moved must still produce what it produced when it was
    absorbed: the student sees its CURRENT vector, the teacher the vector it had then.
    """
    dev = x0.device
    u = torch.rand(len(x0), generator=gen, device="cpu").to(dev) * (1.0 - 1e-3) + 1e-3
    a = abar_of(u).view(-1, 1, 1, 1)
    e = torch.randn(x0.shape, generator=gen, device="cpu").to(dev)
    xt = a.sqrt() * x0 + (1.0 - a).sqrt() * e
    c, eo = _cond(model, len(x0), dev, emb=emb, cls=cls)
    tc, teo = (c, eo) if t_emb is None else _cond(model, len(x0), dev, emb=t_emb)
    with torch.no_grad(), _gate(teacher_gate):
        tgt = teacher(xt, u, c=tc, emb_override=teo)
    with _gate(gate):
        pred = model(xt, u, c=c, emb_override=eo)
    return ((pred - tgt) ** 2).mean()


# ── which parameters a concept may move ──────────────────────────────────────
def sq_grad(params, loss_fn, n):
    """E[g^2] per entry of every tensor in `params`, over `n` single-example losses `loss_fn(i)`.

    Single-example gradients, not a minibatch gradient squared: the latter is the square of a MEAN,
    which cancels exactly the per-example disagreement a Fisher is meant to measure.
    """
    acc = [torch.zeros_like(p) for p in params]
    for i in range(n):
        gs = torch.autograd.grad(loss_fn(i), params, allow_unused=True)
        for a, g in zip(acc, gs):
            if g is not None:
                a.add_(g.detach() ** 2)
    return [a / max(n, 1) for a in acc]


def topk_mask(scores, rho):
    """Global top-`rho` fraction over a list of score tensors -> list of bool masks (same shapes).

    Exactly k = round(rho * total) entries are selected, by index, so ties at the threshold cannot
    inflate the budget -- with most entries of an unused layer at score 0, a threshold rule would.
    """
    flat = torch.cat([s.reshape(-1) for s in scores])
    k = max(1, int(round(float(rho) * flat.numel())))
    keep = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    keep[torch.topk(flat, k, largest=True, sorted=False).indices] = True
    out, off = [], 0
    for s in scores:
        out.append(keep[off:off + s.numel()].reshape(s.shape))
        off += s.numel()
    return out


# ── a gated low-rank delta: the added-structure baseline for a new VALUE ─────
class GatedLoRA(nn.Module):
    """A low-rank delta on a frozen layer, applied only while `state["on"]` is True.

    A is Kaiming, B is zero, so a fresh delta is an EXACT no-op -- the same initialisation the
    adapter bank uses.  Gated rather than path-keyed: one delta per new factor VALUE, switched on
    for every concept that uses the value and off for everything else.
    """

    def __init__(self, base, rank, state):
        super().__init__()
        self.base, self.state = base, state
        if isinstance(base, nn.Conv2d):
            self.A = nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding,
                               dilation=base.dilation, bias=False)
            self.B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        else:
            self.A = nn.Linear(base.in_features, rank, bias=False)
            self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        out = self.base(x)
        return out + self.B(self.A(x)) if self.state["on"] else out


def inject_gated_lora(model, rank, state, exclude=()):
    """Wrap every nn.Linear / nn.Conv2d of `model` in place, except modules whose qualified name
    contains any of `exclude` (on a pretrained UNet, adapting the noise-level path -- `time_emb` --
    destroys the model, RESULTS 12).  -> the list of new parameters."""
    params = []
    for mname, mod in list(model.named_modules()):
        if isinstance(mod, GatedLoRA):
            continue
        for cname, child in list(mod.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if any(e in full for e in exclude):
                continue
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                w = GatedLoRA(child, rank, state).to(child.weight.device)
                setattr(mod, cname, w)
                params += list(w.A.parameters()) + list(w.B.parameters())
    return params


def gate_on(state):
    """-> a zero-argument callable returning a context manager that switches `state` on."""
    @contextlib.contextmanager
    def _on():
        prev = state["on"]
        state["on"] = True
        try:
            yield
        finally:
            state["on"] = prev
    return _on


# ── bookkeeping ──────────────────────────────────────────────────────────────
@torch.no_grad()
def delta_stats(named_params, ref):
    """How much of the backbone moved: entries changed, fraction, and L2 norm of the change."""
    changed = total = 0
    sq = 0.0
    for k, p in named_params:
        d = p.detach().float() - ref[k].float()
        changed += int((d != 0).sum())
        total += d.numel()
        sq += float((d ** 2).sum())
    return dict(changed=changed, total=total, frac=changed / max(total, 1), l2=sq ** 0.5)
