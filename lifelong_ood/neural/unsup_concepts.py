"""Liu, Du, Li, Tenenbaum & Torralba, "Unsupervised Compositional Concepts Discovery with
Text-to-Image Generative Models", ICCV 2023 -- implemented as a baseline.

NOT the same Liu et al. this repository already compares against.  `composable-diff` /
`composable-mcmc` are **Liu et al. 2022** (Composable Diffusion, ECCV), which COMPOSES concepts it
is GIVEN.  This is the 2023 follow-up, which DISCOVERS the concepts unsupervised, and it is the
closer baseline for us: like our factor basis it learns a shared library plus a small per-image
coordinate, so the two can be compared on the same axis this project cares about -- what a concept
costs to store.

Their Algorithm 1, verbatim in structure:

    a library of K concept embeddings {c_1..c_K}, SHARED across all images
    a per-image weight vector w_i in R^K on the simplex (sum_k w_i^k = 1)

    eps_unsup(x_t, t) = eps_phi(x_t, t) + sum_k w_i^k [ eps(x_t, t | c_k) - eps_phi(x_t, t) ]   (11)
    L = || eps - eps_unsup ||^2                                                                 (10)

    both w_i and c_k updated by gradient descent on L                                           (13)

Equation 11 is Composable Diffusion's score composition (their Eq. 5) with learned weights, so the
composition operator is one this repository already implements and agrees with; what is new is that
nothing is labelled -- the decomposition is discovered from pixels alone.

Two honest notes about the port:

  * They parameterize c_k as a word embedding in a text-to-image model.  Our backbones have a class
    embedding table instead, so c_k is a free vector in that table's space.  The paper explicitly
    permits this ("we can parameterize these K different denoising networks in other ways"), and it
    is the same space our own factor basis lives in, which is what makes the storage comparison
    meaningful rather than an apples-to-oranges one.
  * The simplex constraint is theirs and is load-bearing: it is what makes w a *mixture* and lets a
    one-hot w recover clustering.  Kept via softmax, with `simplex=False` available as an
    ABLATION, because the constraint has a consequence worth testing rather than assuming.
    Under sum_k w_k = 1 the unconditional term of Eq. 11 cancels exactly:

        eps_phi + sum_k w_k (eps_k - eps_phi) = sum_k w_k eps_k

    so the composed score is a CONVEX COMBINATION -- a weighted AVERAGE of conditionals, not
    a product.  That suits what the paper does with it (deciding which of K concepts an image
    is, or splitting a scene into complementary global/local aspects) and it cannot express a
    CONJUNCTION: on Color-MNIST a composition is digit AND foreground AND background at once,
    and weights summing to one cannot hold three factors at full strength simultaneously.
    Dropping the constraint tests whether that is what limits the method here.
"""
import numpy as np
import torch

from ..score import cosine_abar
from .ddpm import _as_flat, _as_img


def _eps_unsup(model, xt, u, C_lib, w, null_id):
    """Their Eq. 11.  `w` is [B, K] on the simplex, `C_lib` is [K, d].  -> [B, C, H, W]."""
    n = len(xt)
    cnull = torch.full((n,), null_id, device=xt.device, dtype=torch.long)
    e_phi = model(xt, u, cnull)
    out = e_phi
    for k in range(len(C_lib)):
        ek = model(xt, u, emb_override=C_lib[k])
        out = out + w[:, k].view(-1, 1, 1, 1) * (ek - e_phi)
    return out


def discover_concepts(field, X01, K=8, n_steps=400, lr=0.05, batch=8, seed=0,
                      u_range=(0.05, 0.95), verbose=False, simplex=True):
    """Algorithm 1: learn a shared library of K concepts and per-image simplex weights.

    `X01` are UNLABELLED images in [0,1], flat.  Nothing about their classes is used -- that is the
    whole point of the method, and it is why this is a fair baseline for a setting where the stream
    arrives without labels.

    -> dict(C=[K, d] library, W=[N, K] weights, hist=[losses])
    """
    from .ddpm import to_model
    model, dev = field.model, field.device
    was = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d = model.cemb.weight.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Initialize each concept at the class table's MEAN plus a small perturbation to break symmetry
    # between the K of them.  NOT at a random point: `inversion.learn_embedding(free=True)` records
    # why -- a free vector in conditioning space has to start "somewhere the model decodes sensibly
    # rather than at an arbitrary origin", and the paper's own setting hides this, because a
    # pretrained text encoder's embedding space is dense with meaning almost everywhere while a
    # 120-row class table trained from scratch is not.  The first version of this initialized at
    # `randn * table_std` and the SEEN control came out at chance (joint 0.000-0.062 against 0.006),
    # i.e. the port could not reproduce even in-distribution compositions.
    mu = model.cemb.weight.detach().mean(0)
    scale = float(model.cemb.weight.detach().std())
    C_lib = (mu.unsqueeze(0) + 0.1 * scale * torch.randn(K, d, generator=g).to(dev)
             ).clone().requires_grad_(True)
    X = _as_img(to_model(np.asarray(X01)), dev, field.img, getattr(field, 'layout', None))
    N = len(X)
    logits = torch.zeros(N, K, device=dev).requires_grad_(True)   # softmax -> the simplex
    opt = torch.optim.Adam([C_lib, logits], lr=lr)

    hist = []
    for it in range(n_steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g).to(dev)
        x0 = X[idx]
        u = torch.rand(len(x0), device=dev) * (u_range[1] - u_range[0]) + u_range[0]
        ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
        e = torch.randn(x0.shape, generator=g).to(dev)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        w = torch.softmax(logits[idx], dim=1) if simplex else logits[idx]
        loss = ((e - _eps_unsup(model, xt, u, C_lib, w, model.null_id)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if verbose and (it % max(1, n_steps // 8) == 0):
            print(f"      [liu] step {it:4d}  loss {float(loss):.4f}", flush=True)
    if was:
        model.train()
    W = (torch.softmax(logits, 1) if simplex else logits).detach().cpu().numpy()
    return dict(C=C_lib.detach().cpu().numpy(), W=W, hist=hist)


def fit_weights(field, x01, C_lib, n_steps=200, lr=0.05, seed=0, u_range=(0.05, 0.95),
                simplex=True):
    """Concepts FROZEN, fit one image's simplex weights.  This is what a new query costs them.

    Their Algorithm 1 updates w_i and c_k jointly over the training set; at test time on a held-out
    image only w is free, which is the analogue of our addressing step and the number the storage
    comparison turns on: K floats against our 72 bytes.
    """
    from .ddpm import to_model
    model, dev = field.model, field.device
    model.eval()
    K = len(C_lib)
    Cl = torch.as_tensor(np.asarray(C_lib, np.float32), device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = _as_img(to_model(np.atleast_2d(np.asarray(x01))), dev, field.img,
                 getattr(field, 'layout', None))
    logits = torch.zeros(1, K, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(n_steps):
        u = torch.rand(1, device=dev) * (u_range[1] - u_range[0]) + u_range[0]
        ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
        e = torch.randn(x0.shape, generator=g).to(dev)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        w = torch.softmax(logits, dim=1) if simplex else logits
        loss = ((e - _eps_unsup(model, xt, u, Cl, w, model.null_id)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (torch.softmax(logits, 1) if simplex else logits).detach().cpu().numpy()[0]


@torch.no_grad()
def sample_unsup(field, C_lib, w, n=16, steps=50, seed=0, clip=(-1, 1)):
    """DDIM sampling along their composed score (Eq. 11), eta=0 as everywhere else here."""
    model, dev = field.model, field.device
    Cl = torch.as_tensor(np.asarray(C_lib, np.float32), device=dev)
    wt = torch.as_tensor(np.asarray(w, np.float32), device=dev).reshape(1, -1).repeat(n, 1)

    lay = getattr(field, "layout", None)

    def guide(xflat, u):
        xt = _as_img(xflat, dev, field.img, lay)
        uu = torch.full((len(xt),), float(u), device=dev)
        e = _eps_unsup(model, xt, uu, Cl, wt[:len(xt)], model.null_id)
        return _as_flat(e, lay).double().cpu().numpy()

    # guide_w == 1 makes `guide_eps` a full REPLACEMENT of the score rather than a guidance term:
    # eps = eps + 1*(guide - eps) = guide.  That is what Eq. 11 is -- a composed score, not a
    # correction to the unconditional one -- so it must replace, not be blended in.
    return field.sample(n=n, steps=steps, seed=seed, clip=clip,
                        guide_eps=guide, guide_w=lambda _u: 1.0)
