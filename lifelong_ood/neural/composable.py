"""Composable Diffusion (Liu et al., ECCV 2022) -- composition by ADDING conditional scores.

This is the canonical compositional-generation baseline in diffusion, and the operator Wang et al.
themselves use for their `top-3` row, so a paper that claims a better composition operator has to
report it.  We had only `top-1` retrieval, which is not the same thing: retrieval picks one trained
conditional, while composable diffusion combines several.

The operator, for concepts c_1..c_n with weights w_i, is a sum of guidance residuals against the
unconditional branch:

    eps_hat(x_t, t) = eps(x_t, t, null) + sum_i  w_i * ( eps(x_t, t, c_i) - eps(x_t, t, null) )

which is the score of a PRODUCT of the conditionals (their Eq. 3-4).  `AND` is this with positive
weights; `NOT` negates a weight.  There is no teacher distribution and nothing is sampled from a
Gaussian -- the model's own conditionals do all the work, which is exactly why it is a strong
baseline here: unlike the analytic PoE teacher it cannot collapse to a prototype mean, so it is the
fairest available test of whether OUR composition beats simply adding the scores of the nearest
trained classes.

Known limitation, and the reason it is not trivially the answer: the concepts must already BE trained
conditionals of the model.  For a held-out (fg, bg) pairing there is no trained conditional for
"cyan digit" and one for "deep-red background" -- there are only 120 entangled slot conditionals.
So the honest instantiation on this benchmark composes the k nearest trained SLOTS, and that is what
`compose_classes` does.  Where a factorized conditioning head exists (CelebA, unet cond_mode
"factored") the per-attribute conditionals ARE available and `compose_attributes_scores` composes
those instead, which is composable diffusion in the setting it was designed for.

Liu, Li, Du, Torralba, Tenenbaum. "Compositional Visual Generation with Composable Diffusion
Models." ECCV 2022. arXiv:2206.01714
"""
import numpy as np
import torch


def composable_eps(field, conds, weights=None, null_emb=None, scale=1.0):
    """-> eps(x, u) implementing Liu et al. Eq. 11 over a list of conditioning vectors.

    `conds` are embedding vectors (not class ids), so this works for a class table and for a
    factored head alike.

    `weights` default to `scale` PER CONCEPT, not to 1/n.  This matters and it was wrong here
    before: Liu et al.'s w_i are per-concept GUIDANCE SCALES that are meant to add (they run values
    around 5-7.5), because the sum of residuals is what makes the result a PRODUCT of the
    conditionals.  Dividing by n instead makes the total guidance 1.0 however many concepts are
    composed, which is approximately their AVERAGE -- a strictly weaker operator, and one that gets
    weaker as more concepts are added.  Running the canonical composition baseline in that
    configuration would understate it in exactly the regime the comparison is about.
    """
    n = len(conds)
    w = np.full(n, float(scale)) if weights is None else np.asarray(weights, np.float64)
    C = [torch.as_tensor(np.asarray(c), dtype=torch.float32, device=field.device).reshape(-1)
         for c in conds]
    ne = (None if null_emb is None else
          torch.as_tensor(np.asarray(null_emb), dtype=torch.float32, device=field.device).reshape(-1))

    def eps(x, u):
        e0 = field.eps(x, u, None) if ne is None else field.eps(x, u, None, emb=ne)
        out = e0.copy()
        for wi, ci in zip(w, C):
            out = out + float(wi) * (field.eps(x, u, None, emb=ci) - e0)
        return out

    return eps


def compose_classes(field, class_embs, ids, weights=None, n=16, steps=50, seed=0, clip=(-1, 1),
                    null_emb=None, eta=0.0, scale=1.0, mcmc=None):
    """Composable diffusion over k TRAINED class conditionals -- the honest instantiation here.

    Samples from pure noise through the full reverse process with the composed score, so diversity
    comes from the unconditional branch and the initial draw exactly as in the source method.
    """
    conds = [class_embs[int(i)] for i in ids]
    ce = composable_eps(field, conds, weights=weights, null_emb=null_emb, scale=scale)
    return field.sample(n=n, steps=steps, seed=seed, clip=clip, eta=eta, mcmc=mcmc,
                        guide_eps=lambda x, u: ce(x, u), guide_w=lambda u: 1.0)


def compose_attributes_scores(field, model, combo, n=16, steps=50, seed=0, clip=(-1, 1), eta=0.0,
                              weights=None, scale=1.0, mcmc=None):
    """Composable diffusion in the setting it was designed for: one conditional per ATTRIBUTE.

    Only available on a factored conditioning head, where `attr[k, v]` is a genuine per-attribute
    conditional rather than one entangled vector per combination.  Each attribute contributes its own
    guidance residual, so an unseen combination is a sum of scores the model already knows -- the
    direct score-space analogue of what we do in embedding space.
    """
    if getattr(model, "cond_mode", "class") != "factored":
        raise RuntimeError("compose_attributes_scores needs cond_mode='factored'")
    with torch.no_grad():
        conds = [model.attr[k, int(v)].detach().cpu().numpy()
                 for k, v in enumerate(np.asarray(combo).ravel())]
    ce = composable_eps(field, conds, weights=weights, scale=scale,
                        null_emb=np.zeros(model.emb_dim, np.float32))
    return field.sample(n=n, steps=steps, seed=seed, clip=clip, eta=eta, mcmc=mcmc,
                        guide_eps=lambda x, u: ce(x, u), guide_w=lambda u: 1.0)
