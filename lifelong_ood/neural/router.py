"""The router: deciding, from the query alone, WHICH update a new observation needs.

Everything else in this repository assumes that decision has already been made by the experimenter.
RESULTS 5q established the rule --

    new COMBINATION of known factor values -> compose it, zero parameters
    new VOCABULARY item                    -> a weight update from real data, never from samples

-- and both halves are measured, but a lifelong learner has to apply the rule to an observation
whose category nobody told it.  That is also the half of the gap the source paper names and does not
close: "detecting, consolidating, and reusing genuinely new primitives".  Consolidating and reusing
are done; this module is detecting.

THE SIGNAL IS THE MODEL'S OWN, and no probe or label is used anywhere: the epsilon-prediction error
of the query under the BEST conditioning the model can offer it, at a fixed noise level.  It is
monotone in how novel the query is, which is what makes one quantity enough for a three-way
decision (`tests/diag_router_signals.py`, 40/30/40 queries):

    u = 0.20   in-distribution 0.0147   new-combination 0.0256   new-vocabulary 0.0666

so a low threshold separates "known" from "novel" and a higher one separates "new arrangement of
known parts" from "part I do not have".  Measured AUROC 0.985 (novel vs in-distribution, u=0.10) and
0.995 (new-vocabulary vs new-combination, u=0.45).

TWO NEGATIVE RESULTS FROM THAT DIAGNOSTIC ARE KEPT HERE, because both were things this module first
asserted and neither survived measurement.

  * NORMALISING BY THE UNCONDITIONAL LOSS INVERTS THE SIGNAL.  The first version divided the best
    conditional loss by the unconditional one, reasoning that this divides out per-image denoising
    difficulty.  It scores AUROC 0.199 -- not weak, REVERSED -- because the denominator is itself
    sensitive to novelty: on a query carrying a factor value the backbone never saw, the
    unconditional branch is also bad, and it degrades by more than the numerator does.  A ratio to a
    quantity that responds to novelty cannot measure novelty.
  * THE CONDITIONING IS NOT WHAT CARRIES THE DECISION.  `uncond_abs` -- the plain unconditional
    error, using no class and no basis at all -- scores 0.985 and 0.998, matching the conditional
    signal.  So this router is an OOD detector on the model's own density, and it must NOT be
    reported as evidence that the FACTOR BASIS is what decides the route.  `n18` runs it as an
    explicit ablation row for exactly this reason.

Thresholds: `t_novel` is calibrated on TRAINED slots, which is label-free (it asks how well the
backbone explains data it was trained on).  `t_vocab` cannot be -- it separates two categories that
are both absent from training data -- so it is calibrated against the model's OWN samples from
untrained basis combinations, which is the closest thing to a new-combination example the system can
manufacture without being given one.  That substitution is an assumption, `calibrate` says so, and
n18 reports the sensitivity sweep alongside the calibrated operating point.
"""
import numpy as np
import torch

from ..score import cosine_abar
from .ddpm import CH


@torch.no_grad()
def conditional_losses(field, xq, conds, u_grid=(0.15, 0.3, 0.45, 0.6), n_noise=2, seed=0,
                       batch=256):
    """Epsilon-prediction loss of ONE query under MANY conditioning vectors.

    -> [M] mean loss, one per row of `conds` [M, edim].

    Every candidate sees the IDENTICAL noise draw at every level, which is what makes the losses
    comparable across candidates: with independent noise the differences between conditionings are
    swamped by the variance of the draw.
    """
    model, dev = field.model, field.device
    x0 = torch.from_numpy(np.asarray(xq, np.float32).reshape(1, CH, field.img, field.img)).to(dev)
    C = torch.as_tensor(np.asarray(conds), dtype=torch.float32, device=dev).reshape(len(conds), -1)
    M = len(C)
    tot = torch.zeros(M, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    for u in u_grid:
        a = float(cosine_abar(np.array([u]))[0])
        for _ in range(n_noise):
            e = torch.randn(x0.shape, generator=g, device=dev)
            xt = (a ** 0.5) * x0 + ((1 - a) ** 0.5) * e
            for i0 in range(0, M, batch):
                cb = C[i0:i0 + batch]
                xb = xt.expand(len(cb), -1, -1, -1)
                ub = torch.full((len(cb),), float(u), device=dev)
                pred = model(xb, ub, emb_override=cb)
                tot[i0:i0 + batch] += ((pred - e) ** 2).flatten(1).mean(1)
    return (tot / (len(u_grid) * n_noise)).cpu().numpy()


def basis_assignments(fac, gains=None, sizes=None):
    """Every assignment of known factor values -> (conds [M, edim], labels [M, F]).

    The full Cartesian product is enumerated rather than searched.  For ColorMNIST that is 160 rows
    scored in one batched forward pass, which is cheaper and less fragile than a coordinate descent
    that can stop in a local minimum and report a spuriously high `rel_basis`.
    """
    from .idealwords import compose
    sizes = sizes or [len(v) for v in fac["V"]]
    grids = np.meshgrid(*[np.arange(s) for s in sizes], indexing="ij")
    labels = np.stack([g.ravel() for g in grids], 1)
    conds = np.stack([compose(fac, tuple(int(v) for v in row), gains=gains) for row in labels])
    return conds, labels


def attr_assignments(model, combos):
    """Factored-conditioning analogue: conditioning vectors for a list of attribute rows."""
    with torch.no_grad():
        return model.attr_embedding(torch.as_tensor(np.asarray(combos), dtype=torch.long)).cpu().numpy()


def null_conditioning(model):
    """The vector that means 'no class signal' FOR THIS MODEL.

    Not simply zeros.  In `class` mode the null row starts at zero but is a trainable parameter that
    receives a gradient every time CFG dropout selects it, so after training it is near zero and not
    at it.  `rel_*` divides by the loss under this vector, so using the wrong one biases every ratio
    in the same direction and would quietly move the calibrated thresholds.
    """
    with torch.no_grad():
        if getattr(model, "cond_mode", "class") == "factored":
            return np.zeros(model.emb_dim, np.float64)
        return model.cemb.weight[model.null_id].detach().cpu().numpy().astype(np.float64)


U_NOVEL = 0.10        # separates novel from in-distribution best (AUROC 0.985)
U_VOCAB = 0.45        # separates new-vocabulary from new-combination best (AUROC 0.995)


def route_signals(field, xq, class_conds, basis_conds=None, null_cond=None, n_noise=4, seed=0,
                  u_novel=U_NOVEL, u_vocab=U_VOCAB):
    """-> the two decision signals plus the ablation and diagnostic quantities.

    `s_novel` / `s_vocab` are the best conditional error at the two noise levels the diagnostic
    chose.  `uncond_novel` / `uncond_vocab` are the same measurement with NO conditioning and exist
    to keep the ablation honest -- they are as discriminative, so the route is decided by the
    model's density, not by the basis.
    """
    null = (null_conditioning(field.model) if null_cond is None
            else np.asarray(null_cond)).reshape(1, -1)
    parts = [null, np.asarray(class_conds)]
    if basis_conds is not None and len(basis_conds):
        parts.append(np.asarray(basis_conds))
    allc = np.concatenate(parts)
    n_cls = len(class_conds)
    out = {}
    for tag, u in (("novel", u_novel), ("vocab", u_vocab)):
        L = conditional_losses(field, xq, allc, u_grid=(u,), n_noise=n_noise, seed=seed)
        cls = L[1:1 + n_cls]
        out[f"s_{tag}"] = float(cls.min())
        out[f"uncond_{tag}"] = float(L[0])
        out[f"argmin_class_{tag}"] = int(cls.argmin())
        if len(L) > 1 + n_cls:
            bas = L[1 + n_cls:]
            out[f"basis_{tag}"] = float(bas.min())
            out[f"argmin_basis_{tag}"] = int(bas.argmin())
    return out


def calibrate(sig_trained, sig_selfcombo=None, q_novel=0.90, q_vocab=0.90, vocab_mult=2.0,
              key_novel="s_novel", key_vocab="s_vocab"):
    """Thresholds without using a held-out label anywhere.

    `t_novel` is a quantile of the novelty signal over TRAINED-slot queries: "more unusual than
    nine-tenths of what I was trained on".  That is genuinely label-free.

    `t_vocab` cannot be, because it separates two categories that are BOTH absent from training.
    Two options, in order of preference:
      * `sig_selfcombo` -- signals measured on the model's own samples from basis combinations it
        was never trained on.  The system can manufacture these unaided: it knows which slots it
        trained on and it can compose the rest.  This is the honest substitute for a labelled
        new-combination example, and its bias is stated in n18 (self-samples are on-manifold, so
        the threshold it yields is if anything too low).
      * failing that, `vocab_mult` x the trained-slot quantile -- an explicit assumption, swept in
        n18 rather than asserted.
    """
    t_novel = float(np.quantile([s[key_novel] for s in sig_trained], q_novel))
    if sig_selfcombo:
        t_vocab = float(np.quantile([s[key_vocab] for s in sig_selfcombo], q_vocab))
        how = "self-generated combinations"
    else:
        t_vocab = vocab_mult * float(np.quantile([s[key_vocab] for s in sig_trained], q_novel))
        how = f"{vocab_mult}x trained-slot quantile"
    return dict(t_novel=t_novel, t_vocab=t_vocab, calibrated_by=how)


def route(sig, thr, key_novel="s_novel", key_vocab="s_vocab"):
    if sig[key_novel] <= thr["t_novel"]:
        return "in-distribution"
    if sig[key_vocab] <= thr["t_vocab"]:
        return "new-combination"
    return "new-vocabulary"


def auroc(pos, neg):
    """Threshold-free separation, so the result does not depend on the calibration quantile."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties, else a signal that is constant on one side reports a spurious 1.0
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
