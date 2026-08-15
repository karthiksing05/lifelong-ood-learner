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
from .ddpm import CH, _as_img


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
    # LAYOUT-AWARE.  This used to hardcode `.reshape(1, CH, img, img)`, i.e. it read every flat row
    # as CHW.  That is correct for a `permuted`-layout backbone and WRONG for an `hwc` one, where the
    # row is H*W*C -- so the model was handed a scrambled image and no conditioning could explain it.
    # It is the same fault class as RESULTS 30d ("the bank renders nothing" was a renderer reading
    # HWC rows as CHW), and here it silently flattened the Shapes3D conditioning gap to +0.00007 and
    # its router AUROC to 0.65, which read as "the dataset does not condition".
    x0 = _as_img(np.asarray(xq, np.float32), dev, field.img,
                 getattr(field, "layout", None))[:1]
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


U_NOVEL = 0.10        # ColorMNIST's operating point (AUROC 0.985) -- see calibrate_u_novel
U_VOCAB = 0.45        # separates new-vocabulary from new-combination best (AUROC 0.995)


def calibrate_u_novel(field, class_conds, X, slots, seen_slots, u_grid=None, n_noise=4, seed=0,
                      n_q=24, null_cond=None, prep=None, verbose=False):
    """Choose the novelty operating point for THIS backbone, using SEEN slots only.  -> (u, curve).

    `U_NOVEL = 0.10` was tuned on a 32 px ColorMNIST backbone and does not transfer.  Carried over
    unexamined to a 64 px pretrained Shapes3D backbone it produced AUROC **0.354** -- below chance --
    and that number was recorded as "the router does not transfer" (5t-bis).  It was the constant,
    not the method: at u=0.85 the same checkpoint scores **0.861** on held-out queries (N62).

    The criterion is the noise level at which the CLASS LABEL IS MOST LEGIBLE -- AUROC separating a
    query's loss under its own trained class from its loss under the best wrong class.  Label-free
    in the sense `experiments/common.py:backbone_gains` established: only training images and the
    labels they were trained with, never a held-out combination.  A deployed router cannot use
    held-out examples to pick its own operating point; if it had them it would not need detecting.

    MEASURED, INCLUDING WHERE IT FAILS (N65, 3 seeds):

      | backbone | label-free u | AUROC there | oracle u | AUROC | shipped u=0.10 |
      |---|---|---|---|---|---|
      | Shapes3D (pretrained) | **0.85** | **0.929** | 0.85 | 0.929 | 0.662 |
      | ColorMNIST | 0.82 | 0.826 | 0.05 | 1.000 | 0.988 |

    On Shapes3D it recovers the oracle's choice EXACTLY, on every seed.  On ColorMNIST it does not,
    and the reason is 6c: that router runs on the model's DENSITY, not on its conditioning -- its
    class legibility sits at chance (0.46-0.50) precisely where its novelty AUROC is 1.000, so a
    legibility criterion has nothing to track.  The two results agree rather than conflict, and the
    practical rule falls out of them: **calibrate where conditioning carries the decision, which is
    the case where the inherited constant fails.**  Compare `uncond_novel` against `s_novel` to tell
    which regime a backbone is in before trusting this.
    """
    grid = (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 0.95) if u_grid is None else u_grid
    prep = (lambda x: x) if prep is None else prep
    null = (null_conditioning(field.model) if null_cond is None
            else np.asarray(null_cond)).reshape(1, -1)
    allc = np.concatenate([null, np.asarray(class_conds)])
    seen_sorted = np.asarray(sorted(int(v) for v in seen_slots))
    rng = np.random.default_rng(seed)
    picks = rng.permutation(seen_sorted)[:n_q]
    curve = []
    for u in grid:
        corr, wrong = [], []
        for s in picks:
            idx = np.where(np.asarray(slots) == int(s))[0]
            if not len(idx):
                continue
            L = conditional_losses(field, prep(X[int(idx[0])]), allc, u_grid=(u,),
                                   n_noise=n_noise, seed=seed)
            cls = L[1:]                                  # row 0 is the null token
            j = int(np.searchsorted(seen_sorted, int(s)))
            corr.append(float(cls[j]))
            wrong.append(float(np.min(np.delete(cls, j))))
        curve.append((float(u), auroc(np.array(wrong), np.array(corr))))
        if verbose:
            print(f"[calibrate_u_novel] u={u:<5g} class legibility {curve[-1][1]:.3f}", flush=True)
    return max(curve, key=lambda t: t[1])[0], curve


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


def route_by_trial(sig_after, t_novel, key_novel="s_novel"):
    """Route by TRYING the cheap tier and checking whether it worked.  -> "ok" | "escalate".

    Three attempts to calibrate `t_vocab` in advance all failed, and measurement says why:

        real new-combination    0.00977      manufactured combination      0.00725
        real new-vocabulary     0.02359      manufactured vocab (any scale) 0.00893 - 0.00966
        real in-distribution    0.00780      manufactured in-distribution   0.00957

    Every population the model can MANUFACTURE clusters near 0.009 whatever we do to it, while real
    new-vocabulary sits at 0.024.  That is close to tautological in hindsight: a generator cannot
    synthesise an image containing a feature it has never seen, so no self-generated population can
    stand in for "a value outside my vocabulary".  The positive class is not manufacturable, and a
    threshold placed between two populations needs one.

    So stop placing it in advance.  The cheap tier is 72 bytes and its outcome is **checkable without
    a label**: compose the concept via ADDRESS, then ask the model how well it now explains the query.
    If the residual novelty is still above `t_novel` -- the one threshold that IS honestly calibrable,
    being a quantile over TRAINED slots -- the cheap route did not capture the concept and the query
    is escalated to a kept delta.

    This is Cobweb's actual character rather than an analogy to it: an incremental learner does not
    pre-compute where the category boundaries are, it commits, observes, and revises.  It also makes
    the error self-correcting, and turns the cost asymmetry around:

        current rule   a true combination is misrouted 33% of the time and pays 69K it never needed
        by trial       a true vocabulary item pays 72 B of wasted trial before its 69K -- 0.1% over

    `t_vocab` disappears entirely, and with it the substitution assumption `calibrate` had to flag.
    """
    return "ok" if sig_after[key_novel] <= t_novel else "escalate"


def leave_one_value_out(C, labels, factor=None, value=None, seed=0, gains=None):
    """Simulate "a value I do not have" by HIDING one the system does have.  -> dict.

    This replaces `synthetic_vocab_conds`, which manufactured the positive class by conditioning on
    an off-basis direction and did not work: N18 measured overspend 0.900 against the quantile rule's
    0.333, because a vector the model never saw decodes to a generic blur, and a blur is EASY to
    denoise -- so the manufactured "new vocabulary" scored LOWER on `s_vocab` than real combinations
    and dragged the threshold underneath them.

    The fix is to stop manufacturing images at all.  The system already knows its own factor values,
    so it can refit the basis with one value REMOVED and ask what the router sees for real images
    carrying that value.  Those are genuine positives -- real photographs whose factor value the
    (reduced) basis provably cannot express -- and choosing which value to hide is a decision about
    its own representation, not a held-out label.

    This is the same counterfactual Cobweb runs when it evaluates creating a node: ask what the
    partition looks like with and without, and price the difference.  Here the counterfactual is over
    a factor VALUE rather than a node, and the price is bytes.

    -> dict(fac, conds, labels, pos, neg) where `pos` indexes rows of `C`/`labels` carrying the
       hidden value (new-vocabulary under the reduced basis) and `neg` indexes rows that do not
       (expressible, so new-combination at worst).
    """
    from .idealwords import compose, factorize
    C = np.asarray(C, float)
    L = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    sizes = [int(L[:, f].max()) + 1 for f in range(L.shape[1])]
    if factor is None:
        # prefer a factor with enough values that dropping one still leaves a usable basis
        cand = [f for f, s in enumerate(sizes) if s >= 3]
        factor = int(rng.choice(cand)) if cand else int(np.argmax(sizes))
    if value is None:
        value = int(rng.integers(sizes[factor]))
    pos = np.where(L[:, factor] == value)[0]
    neg = np.where(L[:, factor] != value)[0]
    if len(neg) < 2 or len(pos) < 1:
        return None
    # refit WITHOUT the hidden value, and relabel so the reduced factor is contiguous
    Lr = L[neg].copy()
    Lr[:, factor] = np.where(Lr[:, factor] > value, Lr[:, factor] - 1, Lr[:, factor])
    fac_r = factorize(C[neg], Lr)
    conds, clabels = basis_assignments(fac_r, gains=gains, sizes=fac_r["sizes"])
    return dict(fac=fac_r, conds=conds, labels=clabels, pos=pos, neg=neg,
                factor=int(factor), value=int(value))


def synthetic_vocab_conds(fac, n=16, seed=0, scale=1.0):
    """REFUTED (2026-08-13).  Kept as the record of an approach that cannot work; use
    `leave_one_value_out` instead.

    The idea was to manufacture the positive class by conditioning on a direction orthogonal to a
    factor's known value-vectors.  `tests/diag_synth_vocab.py` measured what it actually produces,
    against the real populations on the same axis:

        real new-combination            0.00977      <- the negatives
        real new-vocabulary             0.02359      <- what a positive should look like
        manufactured vocab x0.5 .. x4   0.00893 .. 0.00966

    The manufactured population lands on top of the real NEGATIVES and is **flat across an 8x range
    of off-basis magnitude** -- scale does not move it at all.  The reason is that `s_vocab` is a
    MINIMUM of denoising error over conditionings: an embedding the model never saw decodes to a
    generic, low-detail image, and a low-detail image is easy to denoise under almost any
    conditioning, so it scores LOW however far off-basis the conditioning was.  A positive class
    cannot be built this way at any setting.

    Used in N18 it gave overspend 0.900 against the quantile rule's 0.333.

    Conditionings for factor values the basis CANNOT express.  -> [n, d] embeddings.

    The system can already manufacture the negative class -- new COMBINATIONS are compositions of
    values it has, over slots it never trained on.  It could not manufacture the positive class, so
    `calibrate` had only one population and had to take a quantile inside it.  This supplies the
    other side.

    A new VOCABULARY item is, by construction, a factor value whose direction is not in the span of
    the values that factor already has.  So: draw a direction orthogonal to that span, scale it to
    the typical magnitude of the factor's existing vectors, and substitute it for one value in an
    otherwise ordinary composition.  The result is an embedding the additive basis provably cannot
    reach, generated without a single held-out label -- the same trick as the negative class, applied
    to the axis the negative class holds fixed.
    """
    rng = np.random.default_rng(seed)
    mu, V = np.asarray(fac["mu"], float), [np.asarray(v, float) for v in fac["V"]]
    d = mu.shape[0]
    out = []
    for i in range(n):
        f = int(rng.integers(len(V)))
        Vf = V[f]
        u = rng.standard_normal(d)
        # project out the span of this factor's known value-vectors: what remains is a direction the
        # basis has no coordinate for, which is exactly what "a new value of a known factor" means
        B = Vf - Vf.mean(0, keepdims=True)
        if len(B) and np.linalg.norm(B) > 0:
            Q, _ = np.linalg.qr(B.T)
            u = u - Q @ (Q.T @ u)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        mag = float(np.median(np.linalg.norm(Vf - Vf.mean(0, keepdims=True), axis=1))) or 1.0
        u = scale * mag * u / nu
        c = mu.copy()
        for g, Vg in enumerate(V):
            c = c + (u if g == f else Vg[int(rng.integers(len(Vg)))])
        out.append(c)
    return np.asarray(out, np.float32)


def calibrate_cost(sig_combo, sig_vocab, bytes_free=72.0, bytes_paid=69_000.0,
                   underspend_penalty=None, key_vocab="s_vocab", grid=512):
    """Place `t_vocab` where it minimizes EXPECTED BYTES, not where it maximizes accuracy.

    This exists because of a specific measured failure.  N18 separates new-vocabulary from
    new-combination at **AUROC 0.995** -- the signal is very nearly perfect -- and yet the calibrated
    operating point still sends **30% of new combinations down the paid route**, which is the entire
    oracle-vs-measured gap in N24's stream accounting (3.8M against 10.1M).  Nothing is wrong with the
    detector; the threshold is in the wrong place.

    Two reasons it lands there, and this function fixes both:

      1. `calibrate` takes a QUANTILE, which is an accuracy criterion.  Accuracy weighs the two
         errors equally, and here they are not remotely equal: routing a combination to the paid path
         wastes ~69 KB, while routing a vocabulary item to the free path produces a concept the basis
         cannot express.  The right operating point minimizes cost, so cost has to be in the rule.
      2. It calibrates on ONE manufactured population (self-generated combinations) and picks a
         quantile of it.  A quantile of the negative class alone cannot know where the positive class
         begins -- and these self-samples are on-manifold, so their signal runs LOW, which drags the
         threshold down onto the real combinations.  Given both populations the threshold can sit
         BETWEEN them instead of inside one.

    `underspend_penalty` is the byte-equivalent charged for sending a genuine new-vocabulary item to
    the free route.  It is not a byte cost at all -- it is a capability failure -- so it defaults to
    100x the paid route, which keeps the error one-sided (N18 measures underspend at 0.000 and that
    property is worth preserving) while still letting cost, rather than a quantile, choose the point.
    """
    c = np.asarray([s[key_vocab] for s in sig_combo], float)
    v = np.asarray([s[key_vocab] for s in sig_vocab], float)
    pen = float(underspend_penalty if underspend_penalty is not None else 100.0 * bytes_paid)
    lo, hi = float(min(c.min(), v.min())), float(max(c.max(), v.max()))
    ts = np.linspace(lo, hi, grid)
    # expected bytes per query, assuming the two populations are equally likely: a combination above
    # the threshold buys `bytes_paid` it did not need; a vocabulary item below it incurs `pen`.
    over = np.array([(c > t).mean() for t in ts]) * bytes_paid
    under = np.array([(v <= t).mean() for t in ts]) * pen
    j = int(np.argmin(over + under))
    return dict(t_vocab=float(ts[j]), expected_bytes=float((over + under)[j]),
                overspend=float((c > ts[j]).mean()), underspend=float((v <= ts[j]).mean()),
                calibrated_by="cost-minimizing over both manufactured populations")


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


# ══════════════════════════════════════════════════════════════════════════════
# The MDL router: choose the tier by DESCRIPTION LENGTH, not by a novelty threshold.
# ══════════════════════════════════════════════════════════════════════════════
#
# Everything above this line asks a classification question -- "IS this query out of vocabulary?" --
# and calibrates a threshold to answer it.  Five label-free routes to that threshold are closed
# (RESULTS 6c-bis) and four of them for one reason: the positive class of that question cannot be
# manufactured by the generator that has to answer it.
#
# Cobweb never asks that question.  Presented with an instance it does not classify it as
# novel-or-not; it *hypothetically applies each operator* -- insert into a child, make a new child,
# merge, split -- and scores the resulting partition by category utility, an intrinsic objective with
# a complexity penalty built into it (utility is normalized by the number of children, so a partition
# does not get to buy fit with structure for free).  There is no threshold and no positive class, only
# a comparison between futures of the same instance.
#
# That is directly transplantable, and the storage ladder supplies the complexity penalty for free --
# the tiers already have honest costs in BITS:
#
#     NOTHING          0 bits   the model already expresses this
#     ADDRESS        576 bits   72 B of factor-value indices, composed through the frozen basis
#     EXTEND BASIS  8192 bits   1 KB: one new direction for one factor value
#     BLOCK        2.2 Mbit     69K rank-1 parameters
#
# So encode the query under each tier and pay for the tier:
#
#     L(t) = C_t  +  N * (D_eff/2) * log2( m_t )                                              (*)
#
# with `m_t` the epsilon-prediction residual under tier t's conditioning, N the number of query
# images, and D_eff the effective dimension (below).  The additive constant of the Gaussian
# codelength is tier-independent and cancels in the argmin.  Route = argmin_t L(t).
#
# Three properties follow, and each was a separate open problem:
#
#   1. NO THRESHOLD IS CALIBRATED.  `t_novel` and `t_vocab` both disappear.  The bar a tier must clear
#      is its own cost in bits, which is known exactly rather than estimated from a population.
#
#   2. ATTEMPT 5's CONFOUND CANCELS.  `route_by_trial` failed because for *any* OOD query the composed
#      ADDRESS conditioning explains the query worse than a trained slot explains its own images, so
#      80% of genuine new-combinations tripped the bar.  Here that common term appears on both sides
#      of a DIFFERENCE -- the question is whether EXTEND explains the query enough *better than
#      ADDRESS* to be worth 7616 further bits -- and a term common to both tiers has no effect on
#      which is smaller.
#
#   3. THE DECISION SCALES WITH EVIDENCE.  N multiplies the fit term and not the cost term, so one
#      novel image warrants ADDRESS and a hundred of the same novelty warrant EXTEND.  "How much does
#      an OOD query warrant addition" stops being a threshold and becomes an exchange rate.  A
#      threshold router is blind to N by construction.
#
# What this does NOT get for free is D_eff, the number of bits of evidence one image carries.  The
# nominal 3*32*32 is wrong because residual pixels are not independent.  It is estimated label-free
# from TRAINED slots -- the one population that is always available -- by the chi-squared moment
# relation Var[log m] = 2/D_eff over independent noise draws of a fixed image under a fixed
# conditioning.  That is the single assumption in the rule and it is swept, not asserted.

TIER_BITS = {
    "none":    0.0,
    "address": 72.0 * 8,
    "extend":  1024.0 * 8,
    "block":   69_000.0 * 32,
}

# which lifelong route each tier corresponds to, so the MDL rule can be scored against 6c's matrix
TIER_ROUTE = {"none": "in-distribution", "address": "new-combination", "extend": "new-vocabulary",
              "block": "new-vocabulary"}


@torch.no_grad()
def batch_losses(field, X, conds, u_grid=(U_VOCAB,), n_noise=4, seed=0, batch=256):
    """`conditional_losses` for a BATCH of images that share a concept.  -> [M] mean over the batch.

    The batch is the unit of decision: a concept arriving as N examples is one thing to store, so it
    gets one conditioning, chosen to minimize the loss over all N rather than per image.
    """
    X = np.asarray(X, np.float32).reshape(len(X), -1)
    tot = None
    for i, x in enumerate(X):
        L = conditional_losses(field, x, conds, u_grid=u_grid, n_noise=n_noise,
                               seed=seed + 1000 * i, batch=batch)
        tot = L if tot is None else tot + L
    return tot / len(X)


@torch.no_grad()
def effective_dim(field, X, cond, n_rep=12, u_grid=(U_VOCAB,), seed=0, clip=(4.0, 4096.0)):
    """D_eff from the model's own residual noise on data it already explains.  -> float.

    For an isotropic Gaussian residual over D independent dimensions, the mean squared error over a
    draw is chi-squared/D, so Var[log m] = 2/D.  Inverting that on repeated *noise draws* of a fixed
    image under a fixed conditioning measures how many independent dimensions the residual actually
    behaves like -- far below 3*32*32, because neighbouring pixels are not independent.

    Repeated NOISE DRAWS rather than repeated images, deliberately: across images the spread also
    contains genuine difficulty variation, which would inflate Var, deflate D_eff, and bias the rule
    toward the cheap tier -- i.e. toward under-spending, which 6c establishes is the worse error.
    This estimator asks only "how precisely do I know m", which is the quantity the comparison in (*)
    actually depends on.

    Label-free: it reads trained slots under their own conditionings and nothing else.
    """
    c = np.asarray(cond).reshape(1, -1)
    logs = []
    for i, x in enumerate(np.asarray(X, np.float32).reshape(len(X), -1)):
        m = [float(conditional_losses(field, x, c, u_grid=u_grid, n_noise=1,
                                      seed=seed + 7919 * i + 13 * r)[0]) for r in range(n_rep)]
        m = np.asarray(m, float)
        if np.all(m > 0):
            logs.append(np.var(np.log(m), ddof=1))
    if not logs:
        return float(clip[0])
    v = float(np.mean(logs))
    return float(np.clip(2.0 / v if v > 0 else clip[1], *clip))


def mdl_lengths(resid, n, d_eff, bits=None):
    """Total description length in bits per tier.  -> {tier: bits}

    `resid` {tier: m_t}; `n` images in the batch.  Equation (*) above.  The Gaussian constant is
    tier-independent, so these are comparable to each other and not interpretable on their own.
    """
    bits = TIER_BITS if bits is None else bits
    return {t: float(bits[t] + n * 0.5 * d_eff * np.log2(max(m, 1e-12)))
            for t, m in resid.items() if t in bits}


def mdl_route(resid, n, d_eff, bits=None):
    """Cobweb's operator comparison, with storage cost as the complexity term.  -> (tier, lengths).

    No threshold, no calibrated novelty statistic, no manufactured positive class: the tier is simply
    whichever encodes the query batch in fewest total bits.
    """
    L = mdl_lengths(resid, n, d_eff, bits=bits)
    return min(L, key=L.get), L


def mdl_crossover(resid, d_eff, bits=None, lo="address", hi="extend"):
    """How many examples this query must arrive as before `hi` is worth its extra bits.  -> float.

    Solving (*) for N at the point where the two tiers cost the same:

        N* = (C_hi - C_lo) / ( (D_eff/2) * log2( m_lo / m_hi ) )

    `inf` when the expensive tier does not fit better at all, in which case no amount of evidence
    justifies it.

    This is the estimate to read when D_eff is in doubt, and the reason is structural: **N and D_eff
    appear in (*) only as their product**, so any error in D_eff is exactly a rescaling of N.  The
    ORDER of two queries' N* is therefore independent of the estimate entirely, and a separation
    measured on N* (AUROC, say) is a property of the signal rather than of the calibration -- which is
    what every attempt in 6c-bis was missing.

    It is also the direct answer to "how much does an OOD query warrant addition": not a yes/no from
    a threshold, but an exchange rate between examples and bytes.
    """
    bits = TIER_BITS if bits is None else bits
    gain = np.log2(max(resid[lo], 1e-12)) - np.log2(max(resid[hi], 1e-12))
    if gain <= 0:
        return float("inf")
    return float((bits[hi] - bits[lo]) / (0.5 * d_eff * gain))


def utility_route(resid, bits=None, base="none"):
    """Cobweb's category utility, transplanted literally: fit gain PER BIT SPENT.  -> (tier, U).

    Category utility scores a partition by how much it improves prediction and then divides by the
    number of children, so structure is never free.  The same shape here:

        U(t) = [ log2 m_base - log2 m_t ] / C_t                                              (**)

    argmax over the paid tiers, and `base` (cost 0, "the model already expresses this") wins when no
    tier's gain is positive.

    Two differences from `mdl_route`, both deliberate, so the pair brackets the answer:

      * **No D_eff.**  (**) is a ratio of two quantities in the same units, so the exchange rate
        between residual and bits cancels entirely.  Nothing is calibrated and nothing is estimated;
        the one assumption `mdl_route` makes is gone.
      * **No batch-size sensitivity.**  N cancels along with it, so this rule cannot answer "how much
        does an OOD query warrant addition" -- it answers only "which tier is the best buy".

    So: read `utility_route` for the ranking that survives any calibration, and `mdl_route` (or
    `mdl_crossover`) when the question is how much evidence is on the table.
    """
    bits = TIER_BITS if bits is None else bits
    lb = np.log2(max(resid[base], 1e-12))
    U = {t: (lb - np.log2(max(m, 1e-12))) / bits[t]
         for t, m in resid.items() if t != base and bits.get(t, 0) > 0}
    if not U or max(U.values()) <= 0:
        return base, U
    return max(U, key=U.get), U


# ── the codelength the rule actually needs ────────────────────────────────────────────────────────
#
# `batch_losses` returns the SIMPLE loss -- E‖eps - eps_theta‖^2, unweighted -- and the first version
# of this router used it directly as a Gaussian codelength.  That is wrong, and wrong by orders of
# magnitude in a knowable direction.  The simple loss is the diffusion variational bound with its
# per-timestep weights DIVIDED OUT (Ho et al. §3.4 adopt it precisely because dropping w_t improves
# sample quality); the weights are largest exactly where the bound is most informative, at low noise,
# and they are not a constant.  Coding a query with the unweighted residual therefore under-counts
# the evidence one image carries, and the symptom is unmistakable: every query, including a plainly
# out-of-vocabulary one, routes to "store nothing".
#
# The bound itself has no such problem, and for a DIFFERENCE between two conditionings on the same
# data almost all of it cancels -- the prior term, the reconstruction term and the entropy constants
# are conditioning-independent -- leaving
#
#     Delta NLL  =  sum_t  w_t * D * ( m_t^A - m_t^B )   nats,     w_t = beta_t / (2 alpha_t (1-abar_t))
#
# with D the true pixel count.  No effective dimension is needed: the bound is already a sum over all
# D coordinates, and any correlation between them is the model's business, not the accountant's.
# `effective_dim` is kept for the record and for `mdl_route`'s simple-loss mode, but `elbo_bits` is
# what the rule should be read on.

@torch.no_grad()
def elbo_bits(field, X, cond, n_t=96, n_noise=2, seed=0, u_lo=0.02, u_hi=0.98, d=None):
    """Conditioning-dependent part of the variational bound, in BITS per image.  -> float.

    Only the eps-matching terms are computed, so the value is an offset codelength: differences
    between conditionings on the same `X` are exact, absolute values are not interpretable.  That is
    all `mdl_route` ever uses.

    The noise draw at each t is shared across every call with the same `seed`, which is what makes
    two tiers comparable -- unpaired draws swamp a 10% residual difference with sampling noise.
    Pairing is doing real work here: `tests/diag_elbo_stability.py` measures |mean|/sd of the paired
    difference at 17.3 with two draws and 93.6 with eight, on the *hardest* case (an in-distribution
    query, where the true difference is smallest).  The concern it was written to test -- that the
    bound puts half its weight above u = 0.64 and its five heaviest grid points at u = 0.98, 0.97,
    0.02, 0.96, 0.03, i.e. where x_t is nearly pure noise and the conditioning barely enters -- turns
    out not to destroy measurability.

    IT DOES, HOWEVER, DECIDE THE ANSWER.  The same diagnostic finds the sign of the difference stable
    across seeds but **not across bands**: on a new-vocabulary query the full bound says ADDRESS codes
    17.2 +/- 1.5 bits WORSE than the nearest trained conditioning, while the mid-noise band u in
    [0.10, 0.70] says 7.6 +/- 0.8 bits BETTER.  Both are stable; they disagree.  The full bound is the
    principled choice -- it is the actual variational bound, and its answer is the semantically right
    one, since the basis holds no direction for the missing value and composing therefore should not
    beat the nearest trained slot -- but "ADDRESS never pays for a new vocabulary item" is a statement
    about how the schedule is weighted, not a schedule-free fact.  Any conclusion drawn from the sign
    of this quantity has to say which band it was measured in.
    """
    model, dev = field.model, field.device
    Xt = torch.from_numpy(np.asarray(X, np.float32).reshape(len(X), CH, field.img, field.img)).to(dev)
    c = torch.as_tensor(np.asarray(cond), dtype=torch.float32, device=dev).reshape(1, -1)
    D = float(d if d is not None else CH * field.img * field.img)
    us = np.linspace(u_lo, u_hi, n_t)
    ab = cosine_abar(us)
    # discrete-time weights from the continuous schedule: alpha_t = abar_t / abar_{t-1}
    ab_prev = cosine_abar(np.clip(us - (u_hi - u_lo) / max(n_t - 1, 1), 0.0, 1.0))
    alpha = np.clip(ab / np.maximum(ab_prev, 1e-8), 1e-6, 1.0)
    beta = 1.0 - alpha
    w = beta / (2.0 * alpha * np.maximum(1.0 - ab, 1e-6))
    tot = 0.0
    g = torch.Generator(device=dev).manual_seed(seed)
    for i, u in enumerate(us):
        a = float(ab[i])
        acc = 0.0
        for _ in range(n_noise):
            e = torch.randn(Xt.shape, generator=g, device=dev)
            xt = (a ** 0.5) * Xt + ((1 - a) ** 0.5) * e
            uu = torch.full((len(Xt),), float(u), device=dev)
            pred = model(xt, uu, emb_override=c.expand(len(Xt), -1))
            acc += float(((pred - e) ** 2).flatten(1).mean(1).mean())
        tot += float(w[i]) * D * (acc / n_noise)
    # No further scaling: `beta` is derived from THIS grid's spacing, so the sum is already the bound
    # for an n_t-step chain.  Multiplying by an extra du/n_t here would discount the step size twice.
    # It converges from ABOVE as the grid refines -- the weight per unit MSE runs 44679 / 35312 /
    # 32024 / 30647 bits at n_t = 24 / 48 / 96 / 192 -- so a coarse grid OVER-states the evidence and
    # biases toward over-spending, which is the safe direction (6c) but is still a bias.  n_t=96 is
    # within 5% of the next refinement and is the default for that reason.
    return tot / np.log(2.0)


def mdl_route_bits(code, n, bits=None):
    """`mdl_route` on a real codelength.  `code` {tier: bits per image}.  -> (tier, lengths)."""
    bits = TIER_BITS if bits is None else bits
    L = {t: float(bits[t] + n * c) for t, c in code.items() if t in bits}
    return min(L, key=L.get), L


def mdl_crossover_bits(code, bits=None, lo="address", hi="extend"):
    """`mdl_crossover` on a real codelength: examples needed before `hi` pays for itself."""
    bits = TIER_BITS if bits is None else bits
    gain = code[lo] - code[hi]
    return float("inf") if gain <= 0 else float((bits[hi] - bits[lo]) / gain)
