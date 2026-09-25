"""FOLD: add a genuinely new prototype to the manifold with a responsibility-gated minimal expert.

The two halves of Proposition 2, made explicit for a network (docs/REVIEW_2026-09.md Part 2):

    L = E_{x ~ arrival, u} [ r_k(x_u) || eps(x_u; w_k = r_k) - eps ||^2 ]                    (r ~ 1 half)
      + lam E_{x ~ the model's OWN samples of other nodes, u} [ || eps(x_u; w_k = r_k(x_u)) - eps_base(x_u) ||^2 ]   (r ~ 0 half)

where r_k is the tree's responsibility for the new node at the level nearest u (the label-free
gate, `operators.address` / `tree.responsibility`), and eps_base is the SAME model with the expert
off -- exact, so no teacher copy exists or is charged.  With gate="label" the weight is 1 on the
arrival and 0 elsewhere: the C1 regime, kept as the control the frontier must reproduce.

The stop rule replaces C1's loss criterion (which certified denoising, not generation):
  S1 generative membership  samples from the forced gate, noised to each new node's level, are
                            claimed by the node at >= the calibrated own-node rate;
  S2 no leakage             the model's own samples of OTHER nodes have mean r_k <= leak_max and the
                            gated function on them moves the base by <= drift_max (relative eps);
  S3 coding gain            the gated model's eps-loss on held-back arrival images is at the level
                            the base reaches on its own samples, in the read band;
  plus a memorisation kill (`evals.memorization` > mem_max).
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from ..backbones.layout import to_flat, to_img
from ..backbones.reads import eps_profile
from ..backbones.sampler import _cond, sample
from ..backbones.schedule import abar_of, cosine_abar
from ..tree.responsibility import responsibility
from ..tree.update import incorporate as _incorporate
from .address import explained


# ── the gate ─────────────────────────────────────────────────────────────────
class Gate:
    """Weights for an expert's keys given noisy rows and a level.  mode: "analytic" (the tree's
    responsibility for the new nodes), "label" (1 on flagged rows, 0 otherwise), "forced" (1)."""

    def __init__(self, tree, node_ids, mode="analytic"):
        self.tree, self.node_ids, self.mode = tree, list(node_ids), mode
        self.level_of_node = {i: tree.nodes[i].level for i in self.node_ids}
        self.levels = sorted(set(self.level_of_node.values()))

    def weights(self, xt_rows, u, flag=None):
        """-> numpy [n] in [0, 1]."""
        n = len(xt_rows)
        if self.mode == "forced":
            return np.ones(n, np.float32)
        if self.mode == "label":
            return np.ones(n, np.float32) if flag is None else np.asarray(flag, np.float32)
        level = self.tree.level_of(u)
        mine = [i for i in self.node_ids if self.level_of_node[i] == level]
        if not mine:
            return np.zeros(n, np.float32)
        X = xt_rows.detach().cpu().numpy() if torch.is_tensor(xt_rows) else np.asarray(xt_rows)
        ids, r = responsibility(self.tree, X, self.tree.u_grid[level])
        cols = [ids.index(i) for i in mine if i in ids]
        return r[:, cols].sum(1).astype(np.float32) if cols else np.zeros(n, np.float32)

    def context(self, expert, xt_rows, u, flag=None):
        """-> (zero-arg context factory, emb_override or None) for one forward."""
        w = torch.as_tensor(self.weights(xt_rows, u, flag), dtype=torch.float32, device=xt_rows.device)
        tok = expert.emb()
        emb = None if tok is None else w[:, None] * tok[None]
        return expert.on(w), emb, w

    def step_gate(self, expert):
        def _sg(x_rows, u):
            ctx, emb, _ = self.context(expert, x_rows, u)
            return (ctx, emb) if emb is not None else ctx
        return _sg


# ── losses ───────────────────────────────────────────────────────────────────
def _draw(x0, gen, u_lo=1e-3, u_hi=1.0):
    dev = x0.device
    u = torch.rand(len(x0), generator=gen, device="cpu").to(dev) * (u_hi - u_lo) + u_lo
    a = abar_of(u).view(-1, 1, 1, 1)
    e = torch.randn(x0.shape, generator=gen, device="cpu").to(dev)
    return u, a.sqrt() * x0 + (1.0 - a).sqrt() * e, e


def arrival_loss(model, expert, gate, x0, layout, emb=None, gen=None, flag=None, weight_by_r=True):
    """The r ~ 1 half, with one noise level per row: every row is gated at its own u."""
    u, xt, e = _draw(x0, gen)
    dev = x0.device
    tot, cnt = torch.zeros((), device=dev), 0.0
    for uu in torch.unique(torch.round(u * 20) / 20):          # group rows by noise bin for the gate
        m = torch.abs(torch.round(u * 20) / 20 - uu) < 1e-6
        rows = to_flat(xt[m], layout)
        ctx, emb_g, w = gate.context(expert, rows, float(uu), None if flag is None else flag[m])
        eo = emb_g if emb is None else (emb[m] if emb.dim() == 2 else emb) + (0 if emb_g is None else emb_g)
        c, eo2 = _cond(model, int(m.sum()), dev, emb=eo, cls=None)
        with ctx():
            pred = model(xt[m], u[m], c=c, emb_override=eo2)
        per = ((pred - e[m]) ** 2).mean((1, 2, 3))
        wt = w if weight_by_r else torch.ones_like(w)
        tot = tot + (wt * per).sum()
        cnt += float(wt.sum()) if weight_by_r else float(len(per))
    return tot / max(cnt, 1e-6)


def anchor_loss_natural(model, expert, gate, x0, layout, emb=None, cls=None, gen=None):
    """The r ~ 0 half: hold the model's own samples of OTHER nodes where the base leaves them, with
    the gate at its natural value.  The target is the same model with the expert OFF."""
    u, xt, _ = _draw(x0, gen)
    dev = x0.device
    tot, cnt = torch.zeros((), device=dev), 0
    for uu in torch.unique(torch.round(u * 20) / 20):
        m = torch.abs(torch.round(u * 20) / 20 - uu) < 1e-6
        rows = to_flat(xt[m], layout)
        ctx, emb_g, w = gate.context(expert, rows, float(uu))
        c, eo_base = _cond(model, int(m.sum()), dev, emb=None if emb is None else (emb[m] if emb.dim() == 2 else emb),
                           cls=None if cls is None else cls[m])
        with torch.no_grad():
            tgt = model(xt[m], u[m], c=c, emb_override=eo_base)            # expert off: the base, exactly
        eo = eo_base if emb_g is None else (emb_g if eo_base is None else eo_base + emb_g)
        with ctx():
            pred = model(xt[m], u[m], c=c, emb_override=eo)
        tot = tot + ((pred - tgt) ** 2).mean((1, 2, 3)).sum()
        cnt += int(m.sum())
    return tot / max(cnt, 1)


# ── node creation ────────────────────────────────────────────────────────────
def create_nodes(tree, cal, X_rows, levels=None, key=None, count=None):
    """New nodes at the failing band levels of X (or `levels`): mode = the mean of X, extent = the
    level's sibling median.  -> node ids (finest first)."""
    X = np.atleast_2d(np.asarray(X_rows, np.float32))
    if levels is None:
        ok, depth, coarsest, maxr = explained(tree, cal, X)
        thr = np.asarray([cal.thresholds[l] for l in cal.levels])
        fail = (maxr.mean(0) < thr)
        levels = [cal.levels[j] for j in np.where(fail)[0]]
        if not levels:
            levels = [cal.levels[int(np.argmin(maxr.mean(0) - thr))]]
    mean = X.mean(0)
    ids = []
    for l in sorted(levels):
        _, _, V, _ = tree.stats(l)
        ext = np.median(V.reshape(len(V), tree.img, tree.img, 3).mean(axis=(1, 2)), axis=0) if len(V) else np.full(3, 0.01, np.float32)
        nid = tree.add(l, mean, extent_channels=ext.astype(np.float32), count=float(count or len(X)))
        tree.nodes[nid].expert = key
        ids.append(nid)
    from ..tree.build import link_parents
    link_parents(tree)
    return ids


def incorporate_rows(tree, node_ids, X_rows):
    X = np.atleast_2d(np.asarray(X_rows, np.float32))
    for nid in node_ids:
        for x in X:
            _incorporate(tree, nid, x, r=1.0)


# ── the stop rule ────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Stop:
    s1: float
    s1_ok: bool
    s2_leak: float
    s2_drift: float
    s2_ok: bool
    s3_ratio: float
    s3_ok: bool
    memorized: float
    mem_ok: bool

    @property
    def absorbed(self):
        return self.s1_ok and self.s2_ok and self.s3_ok and self.mem_ok


def membership_of_generated(tree, node_ids, S_rows, seed=0, n_noise=4):
    """S1: mean responsibility of the new nodes on rows S, at each new node's level."""
    vals = []
    for nid in node_ids:
        l = tree.nodes[nid].level
        a = float(cosine_abar(tree.u_grid[l]))
        rng = np.random.default_rng(seed)
        acc = np.zeros(len(S_rows))
        for _ in range(n_noise):
            xt = np.sqrt(a) * S_rows + np.sqrt(1 - a) * rng.standard_normal(S_rows.shape).astype(np.float32)
            ids, rr = responsibility(tree, xt, tree.u_grid[l])
            acc += rr[:, ids.index(nid)]
        vals.append(float((acc / n_noise).mean()))
    return float(np.mean(vals)), vals


def stop_rule(backbone, tree, cal, expert, gate, node_ids, X_val, X_anchor, ref_profile, u_band, *,
              n_gen=32, steps=50, seed=0, leak_max=0.05, drift_max=0.02, tol=0.05, mem_max=0.2, X_train=None,
              emb_val=None):
    """Evaluate S1-S3 and memorisation.  `ref_profile` is the base's eps-loss on its own samples at
    u_band (the trained-node reference); `emb_val` conditions the S3 read (None = null token)."""
    forced = Gate(tree, node_ids, "forced")
    S = sample(backbone, n_gen, steps=steps, seed=seed, step_gate=forced.step_gate(expert))
    s1, per_node = membership_of_generated(tree, node_ids, S, seed=seed)
    s1_ok = all(v >= min(cal.thresholds.get(tree.nodes[n].level, 0.5), 0.9) for v, n in zip(per_node, node_ids))
    ids_w = gate.weights(torch.as_tensor(X_anchor), tree.u_grid[tree.nodes[node_ids[0]].level])
    leak = float(np.mean(ids_w))
    xa = to_img(torch.as_tensor(X_anchor, device=backbone.device), backbone.img, backbone.layout)
    g = torch.Generator(device="cpu").manual_seed(seed + 5)
    with torch.no_grad():
        drift_num = float(anchor_loss_natural(backbone.model, expert, gate, xa, backbone.layout, gen=g))
        u, xt, e = _draw(xa, torch.Generator(device="cpu").manual_seed(seed + 5))
        c, eo = _cond(backbone.model, len(xa), backbone.device)
        base_err = float(((backbone.model(xt, u, c=c, emb_override=eo) - e) ** 2).mean())
    drift = drift_num / max(base_err, 1e-8)
    s2_ok = leak <= leak_max and drift <= drift_max
    natural = gate.step_gate(expert)
    prof = eps_profile_gated(backbone, expert, gate, X_val, u_band, emb=emb_val, seed=seed)
    ratio = float(np.max(prof / np.asarray(ref_profile)))
    s3_ok = ratio <= 1.0 + tol
    mem = 0.0
    if X_train is not None:
        from ..evals import memorization
        mem = float(memorization(backbone.rows_to_data(S), backbone.rows_to_data(X_train))["frac_memorized"])
    return Stop(s1, s1_ok, leak, drift, s2_ok, ratio, s3_ok, mem, mem <= mem_max), S


@torch.no_grad()
def eps_profile_gated(backbone, expert, gate, X, u_grid, emb=None, seed=0, n_noise=2, batch=256):
    """eps-loss of rows X per level with the expert at its natural gate (CRN)."""
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=backbone.device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = np.zeros(len(u_grid))
    model = backbone.model
    E = None if emb is None else torch.as_tensor(np.asarray(emb, np.float32), device=backbone.device)
    for j, u in enumerate(u_grid):
        a = float(cosine_abar(float(u)))
        for _ in range(n_noise):
            eps = torch.randn(tuple(Xt.shape), generator=g).to(backbone.device)
            xt_rows = a ** 0.5 * Xt + (1.0 - a) ** 0.5 * eps
            ctx, emb_g, _ = gate.context(expert, xt_rows, float(u))
            eo = emb_g if E is None else (E if emb_g is None else E + emb_g)
            c, eo2 = _cond(model, len(Xt), backbone.device, emb=eo, cls=None)
            with ctx():
                pred = to_flat(model(to_img(xt_rows, backbone.img, backbone.layout), torch.full((len(Xt),), float(u), device=backbone.device),
                                     c=c, emb_override=eo2), backbone.layout)
            out[j] += float(((pred - eps) ** 2).mean())
    return out / n_noise


def base_profile(backbone, X, u_grid, emb=None, seed=0, n_noise=2):
    """The base's eps-loss per level on rows X (the trained-node reference when X = own samples)."""
    return eps_profile(backbone.model, X, u_grid, emb=emb, cls=None, n_noise=n_noise, seed=seed,
                       layout=backbone.layout, img=backbone.img, device=backbone.device)
