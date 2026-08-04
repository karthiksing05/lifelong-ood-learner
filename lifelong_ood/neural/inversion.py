"""Learn a concept as a POINT IN THE MODEL'S OWN CONDITIONING SPACE, not as a Gaussian to sample.

The architecture this implements: **the manifold follows the teacher, and sampling follows the
manifold.**  The analytic PoE teacher says where the concept is; it never generates.  Generation
comes from the model's own class conditional, which is the part measured to be diverse.

WHY.  RESULTS 5e established, by elimination, that classifier-free guidance toward the PoE teacher
cannot produce a covering sample set: the teacher is a unimodal Gaussian, so guidance pulls every
trajectory to its single mean, and the guidance knobs only trade collapse against divergence
(recall 0.00 in every configuration tried).  Meanwhile the base DDPM sampled on a trained class
embedding reaches precision 0.89 / recall 0.90 / content diversity 0.97x real -- the ceiling.  The
diversity exists; it lives in the model's learned conditional, and the way to reach it is to
CONDITION rather than to guide.

HOW.  A new concept is a vector in the class-embedding space, learned by textual inversion (Gal et
al.) but with the crucial restriction that it is built from the embeddings the model already has:

    c_new = sum_k alpha_k * c_k        over the trained class embeddings

Optimizing `alpha` rather than a free vector matters for three reasons.  It is ~120 parameters
against a 3072-dimensional image, so a single query cannot be memorized.  It keeps c_new inside (or
near) the region of embedding space the model was actually trained on, every point of which is known
to decode to a diverse conditional -- so diversity is inherited rather than hoped for.  And it makes
the result a genuine COMPOSITION: alpha says which seen compositions the unseen one is built from,
in the model's own vocabulary, which is the same claim the PoE makes in pixel space but expressed
where the model can act on it.

The teacher enters as initialization and as an optional anchor, so composition still decides what is
learned -- but it is not what gets sampled.
"""
import numpy as np
import torch

from ..score import cosine_abar
from .ddpm import CH, IMG


def _basis(model, ids=None):
    """The trained class embeddings, excluding the null row (which is 'no class', not a class)."""
    W = model.cemb.weight.detach()
    keep = [i for i in range(model.n_classes)] if ids is None else list(ids)
    return W[keep], keep


def learn_embedding(field, targets, n_steps=300, lr=0.05, batch=8, seed=0, init_alpha=None,
                    weight_decay=1e-3, simplex=False, u_range=(0.05, 0.95), verbose=False,
                    ids=None, log_every=0, anchor=0.0, anchor_alpha=None):
    """Fit `c_new = sum_k alpha_k c_k` so the model explains `targets` when conditioned on it.

    `targets`: [N, D] images in the model's [-1, 1] space.  One image is the intended setting (a
    single OOD query) and is why the parameterization is restricted -- see the module docstring.

    `simplex=True` constrains alpha to a convex combination via softmax, which cannot leave the hull
    of trained embeddings.  `False` allows extrapolation past them, which an out-of-distribution
    composition may genuinely need; N7 compares the two.

    `anchor` is how the teacher keeps hold of the manifold.  Initializing alpha from the teacher and
    then optimizing freely lets a single query pull it anywhere -- measured: recall recovers to 0.89
    but composition accuracy falls to 0.24, because 120 parameters are underdetermined by one image.
    A penalty `anchor * ||alpha - anchor_alpha||^2` makes the teacher a PRIOR rather than only a
    starting point, so composition constrains the solution throughout and `anchor` is the dial
    between "what the teacher composed" and "what this particular query looks like".

    Using the teacher's sampled POOL as the target instead is the obvious alternative and it does
    not work: the pool is collapsed, so inverting on it teaches the embedding one image (measured
    recall 0.02 against 0.89 for the query).  The teacher has to constrain through alpha, not
    through pixels.

    Returns (c_new [d_emb], alpha [K], history).
    """
    model = field.model
    dev = field.device
    C, keep = _basis(model, ids)
    K = len(keep)
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    a = torch.zeros(K, device=dev) if init_alpha is None \
        else torch.as_tensor(np.asarray(init_alpha, np.float32), device=dev).clone()
    if not simplex and init_alpha is None:
        a += 1.0 / K
    a_ref = None
    if anchor > 0.0:
        ref = anchor_alpha if anchor_alpha is not None else init_alpha
        if ref is not None:
            a_ref = torch.as_tensor(np.asarray(ref, np.float32), device=dev)
            if simplex:
                a_ref = torch.softmax(a_ref, 0)
    a.requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr, weight_decay=(0.0 if simplex else weight_decay))

    X = torch.from_numpy(np.asarray(targets, np.float32)).to(dev).reshape(-1, CH, IMG, IMG)
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hist = []
    for it in range(n_steps):
        idx = torch.randint(0, len(X), (min(batch, len(X)),), generator=g).to(dev)
        x0 = X[idx]
        u = torch.rand(len(x0), device=dev) * (u_range[1] - u_range[0]) + u_range[0]
        ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
        e = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        w = torch.softmax(a, 0) if simplex else a
        c_new = w @ C
        loss = torch.nn.functional.mse_loss(model(xt, u, emb_override=c_new), e)
        if anchor > 0.0 and a_ref is not None:
            loss = loss + anchor * ((w - a_ref) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if log_every and (it + 1) % log_every == 0:
            print(f"      inversion {it + 1}/{n_steps}: {np.mean(hist[-log_every:]):.4f}", flush=True)
    if was_training:
        model.train()
    with torch.no_grad():
        w = torch.softmax(a, 0) if simplex else a
        c_new = (w @ C).detach()
    return c_new, w.detach().cpu().numpy(), hist


def teacher_init_alpha(field, teacher, xq, ids=None, temperature=0.02, reps=2, seed=0):
    """Seed `alpha` from the teacher: score every trained class by how well it explains mu_T.

    This is where composition enters the model's vocabulary -- the same epsilon-loss ranking the
    paper uses for its top-k baseline, but applied to the COMPOSED prototype rather than the raw
    query, and kept as a soft weighting over all classes rather than a hard top-k.  So the manifold
    starts from what the teacher decided and is refined from there.
    """
    C, keep = _basis(field.model, ids)
    rng = np.random.default_rng(seed)
    mu = np.atleast_2d(teacher["mu"])
    losses = []
    for cid in keep:
        tot = 0.0
        for u in (0.2, 0.4, 0.6):
            a = float(cosine_abar(u))
            for _ in range(reps):
                e = rng.standard_normal(mu.shape)
                xt = np.sqrt(a) * mu + np.sqrt(1 - a) * e
                tot += float(np.mean((field.eps(xt, u, int(cid)) - e) ** 2))
        losses.append(tot / (3 * reps))
    losses = np.asarray(losses)
    z = -(losses - losses.min()) / max(temperature, 1e-9)
    alpha = np.exp(z - z.max())
    return alpha / alpha.sum()


@torch.no_grad()
def sample_embedding(field, c_new, n=32, steps=50, seed=0, guidance=1.0, eta=0.0, clip=(-1, 1)):
    """Sample the MODEL conditioned on `c_new` -- the manifold generating, not a Gaussian.

    `guidance=1.0` means plain conditional sampling, which is exactly how the base model reaches
    recall 0.90 on its own classes.  Values above 1 apply ordinary classifier-free guidance against
    the null token; that sharpens and, as always with CFG, costs diversity, so it is off by default.
    """
    def cond(x, u):
        return field.eps(x, u, None, emb=c_new)

    if guidance == 1.0:
        return field.sample(n=n, steps=steps, seed=seed, clip=clip, eta=eta,
                            guide_eps=cond, guide_w=lambda u: 1.0)
    return field.sample(n=n, steps=steps, seed=seed, clip=clip, eta=eta,
                        guide_eps=cond, guide_w=lambda u: guidance)
