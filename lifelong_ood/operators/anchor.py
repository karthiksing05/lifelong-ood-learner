"""Anchoring and the sparse-support machinery: the r ~ 0 half of Proposition 2 written as a loss.

C1 measured that a top-1% g^2/F mask WITHOUT anchoring drifts as much as full fine-tuning, that a random
mask with anchoring ties the g^2/F mask, and that the gate + anchor is what preserves.  Parameter sparsity
is not functional sparsity; these helpers are kept for the frontier's zero-structure arms and for the
anchor term every fold uses.
"""
import numpy as np
import torch

from ..backbones.sampler import _cond, _gate
from ..backbones.schedule import abar_of, cosine_abar


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


