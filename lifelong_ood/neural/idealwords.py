"""Ideal words: read the class-embedding table as a SUM OF FACTORS, and compose in that basis.

WHY THIS EXISTS.  The span oracle (RESULTS 5h) showed that fitting weights over the 120 trained class
embeddings cannot place a held-out (fg, bg) pairing even given 64 real images of it -- it recovers
one factor or the other, never both.  That is the textbook symptom of a model that has "not learned
to bind attributes to objects but merely memorized common object-attribute pairings", which is the
standard diagnosis for compositional failure in diffusion models [1,2].  Our conditioning is a
single global vector added to the timestep embedding, so there is nothing in the architecture that
forces the three factors apart: each of the 120 embeddings is free to be an arbitrary point, and a
weighted average of arbitrary points is not a composition.

THE FIX, from Trager et al., *Linear Spaces of Meanings* (ICCV 2023) [3].  Embeddings of composite
concepts are often approximately decomposable as a sum of per-factor vectors -- "ideal words".  Our
120 seen slots are a complete 10 x 4 x ... design over (digit, fg, bg) with every factor VALUE seen,
so the decomposition is heavily over-determined: 19 unknowns per embedding dimension fitted from 120
observations.  If the table admits it, a held-out combination is just a sum of vectors that all
appear in the training data:

    c(d, f, b)  ~  mu + D_d + F_f + B_b

`factorize` fits that by least squares and REPORTS ITS OWN R^2, so whether the space is additively
compositional is measured rather than assumed.  Two things follow:

  * `compose` builds a held-out concept directly, with no query and no optimization at all.
  * `learn_factored` runs the same textual inversion as before but over the ~19 FACTOR vectors
    instead of 120 entangled class vectors.  That is compositional by construction and vastly better
    conditioned from a single query, which is what the oracle said the class basis could never be.

[1] Improving Compositional Attribute Binding in Text-to-Image Generative Models (arXiv 2406.07844)
[2] Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis
[3] Trager et al., Linear Spaces of Meanings (arXiv 2302.14383)
"""
import numpy as np
import torch

from ..score import cosine_abar
from .ddpm import CH, IMG


def factorize(C, labels, ridge=1e-3):
    """Fit `C[i] ~ mu + sum_f V[f][labels[i][f]]` by ridge least squares.

    `C`  [N, d] the trained class embeddings.
    `labels` [N, F] integer factor values per row (e.g. digit, fg, bg).
    -> dict(mu, V, r2, r2_per_dim) where `V[f]` is [n_values_f, d].

    The intercept absorbs the mean, and each factor's vectors are centred so the decomposition is
    identifiable.  `r2` is the fraction of embedding variance the additive model explains, and it is
    the number that says whether composing in this basis can work at all.
    """
    C = np.asarray(C, np.float64)
    L = np.asarray(labels, int)
    N, d = C.shape
    F = L.shape[1]
    sizes = [int(L[:, f].max()) + 1 for f in range(F)]
    # one-hot design, dropping nothing (ridge handles the rank deficiency from the intercept)
    cols = [np.ones((N, 1))]
    for f in range(F):
        M = np.zeros((N, sizes[f]))
        M[np.arange(N), L[:, f]] = 1.0
        cols.append(M)
    X = np.concatenate(cols, 1)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ C)                     # [1+sum(sizes), d]
    pred = X @ W
    ss_res = ((C - pred) ** 2).sum()
    ss_tot = ((C - C.mean(0)) ** 2).sum()
    off = 1
    V = []
    for f in range(F):
        V.append(W[off:off + sizes[f]])
        off += sizes[f]
    return dict(mu=W[0], V=V, sizes=sizes,
                r2=float(1 - ss_res / max(ss_tot, 1e-12)),
                r2_per_dim=1 - ((C - pred) ** 2).sum(0) / np.maximum(((C - C.mean(0)) ** 2).sum(0), 1e-12))


def compose(fac, assignment, gains=None):
    """Build a concept vector from factor values, with no query and no optimization.

    `assignment` is one integer per factor.  For a held-out combination every summand is a vector
    estimated from data the model DID see -- only their sum is new.

    `gains` scales each factor's contribution, and it is not cosmetic.  The factor directions are
    correct -- a factored inversion reads (digit, fg, bg) exactly on 7 of 8 held-out slots -- but the
    model UNDER-RESPONDS to the weakest of them: background vectors have mean norm 0.55 against 0.79
    for digit, and composing at unit gain yields digit 0.87 / fg 0.95 / bg 0.38.  Doubling the
    background gain takes bg to 0.87 and joint accuracy from 0.29 to 0.38.  Calibrate on SEEN slots
    (`calibrate_gains`), never on held-out labels.
    """
    c = np.array(fac["mu"], np.float64)
    g = [1.0] * len(fac["V"]) if gains is None else list(gains)
    for f, v in enumerate(assignment):
        c = c + float(g[f]) * fac["V"][f][int(v)]
    return c


def learn_factored(field, targets, fac, n_steps=300, lr=0.05, batch=8, seed=0, init=None,
                   temperature=1.0, u_range=(0.05, 0.95), log_every=0, gains=None):
    """Textual inversion over the FACTOR vectors: one soft assignment per factor.

    Instead of ~120 weights over entangled class embeddings, this optimizes one softmax per factor
    (10 + 4 + 4 = 18 parameters here), and the concept is always a sum of one vector per factor.
    Composition is then structural rather than something the optimizer has to discover, and a single
    query is ample to determine 18 parameters where it could never determine 120.

    -> (c_new [d], list of per-factor weight vectors, history).
    """
    model = field.model
    dev = field.device
    F = len(fac["V"])
    Vt = [torch.as_tensor(np.asarray(v, np.float32), device=dev) for v in fac["V"]]
    mu = torch.as_tensor(np.asarray(fac["mu"], np.float32), device=dev)
    logits = []
    for f in range(F):
        z = torch.zeros(fac["sizes"][f], device=dev)
        if init is not None and init[f] is not None:
            z = torch.as_tensor(np.asarray(init[f], np.float32), device=dev).clone()
        z.requires_grad_(True)
        logits.append(z)
    opt = torch.optim.Adam(logits, lr=lr)
    gain = [1.0] * F if gains is None else [float(x) for x in gains]

    X = torch.from_numpy(np.asarray(targets, np.float32)).to(dev).reshape(-1, CH, field.img, field.img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
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
        c_new = mu.clone()
        for f in range(F):
            c_new = c_new + gain[f] * (torch.softmax(logits[f] / temperature, 0) @ Vt[f])
        loss = torch.nn.functional.mse_loss(model(xt, u, emb_override=c_new), e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if log_every and (it + 1) % log_every == 0:
            print(f"      factored inversion {it + 1}/{n_steps}: "
                  f"{np.mean(hist[-log_every:]):.4f}", flush=True)
    with torch.no_grad():
        c_new = mu.clone()
        ws = []
        for f in range(F):
            w = torch.softmax(logits[f] / temperature, 0)
            ws.append(w.detach().cpu().numpy())
            c_new = c_new + gain[f] * (w @ Vt[f])
    return c_new.detach(), ws, hist


def slot_factors(slots):
    """ColorMNIST slot ids -> (digit, fg, bg) label rows, the design the factorization needs."""
    s = np.asarray(slots, int)
    return np.stack([s // 16, (s % 16) // 4, s % 4], 1)


def calibrate_gains(field, fac, probes, data, seen_slots, grid=(1.0, 1.5, 2.0, 3.0), factor=2,
                    n_slots=6, n=16, steps=50, seed=0, score_fn=None, holdout_basis=True,
                    C=None, labels=None):
    """Choose the per-factor gain using SEEN slots only -- never held-out labels.

    Composing a seen slot and asking whether the model reproduces it is label-free, because the slots
    are training data.  But it measures the wrong thing by default: the calibration slot is IN the
    basis fit, so the model is being asked to reproduce a combination it has already memorised, and
    gain 1.0 suffices.  Applied to a HELD-OUT combination the same gain under-delivers the weak
    factor -- measured on ColorMNIST as backgrounds coming out the wrong colour while the seen-slot
    score reads 0.82.

    `holdout_basis=True` fixes the mismatch without touching a held-out label: the basis is refit
    with the calibration slots REMOVED, so reproducing them is a generalisation question of exactly
    the kind a held-out combination poses.  This is cross-validation inside the seen set, and it is
    what the gain is actually for.
    """
    from .ddpm import to_data
    from .inversion import sample_embedding
    rng = np.random.default_rng(seed)
    picks = rng.choice(np.asarray(seen_slots), min(n_slots, len(seen_slots)), replace=False)

    fac_cal = fac
    if holdout_basis and C is not None and labels is not None:
        keep = ~np.isin(np.asarray(seen_slots), picks)
        if keep.sum() > len(fac["V"]) * 4:            # enough rows left to identify the basis
            fac_cal = factorize(np.asarray(C)[keep], np.asarray(labels)[keep])

    best, best_g = None, 1.0
    for g in grid:
        gains = [1.0] * len(fac_cal["V"])
        gains[factor] = g
        sc = []
        for s_ in picks:
            d, f, b = int(s_) // 16, (int(s_) % 16) // 4, int(s_) % 4
            c = compose(fac_cal, (d, f, b), gains=gains)
            S = to_data(sample_embedding(field, torch.as_tensor(c, dtype=torch.float32,
                                                                device=field.device),
                                         n=n, steps=steps, seed=seed)).astype(np.float32)
            sc.append(score_fn(S, d, f, b) if score_fn else 0.0)
        m = float(np.mean(sc))
        if best is None or m > best:
            best, best_g = m, g
    return best_g, best


@torch.no_grad()
def sample_factored_pool(field, c_new, n=64, steps=50, seed=0, guidance=1.0, eta=0.0):
    """Sample the model conditioned on a composed concept vector -- the manifold generating."""
    from .inversion import sample_embedding
    return sample_embedding(field, c_new, n=n, steps=steps, seed=seed, guidance=guidance, eta=eta)


def extend_basis(field, targets, labels, fac, factor, n_steps=400, lr=0.05, batch=8, seed=0,
                 gains=None, u_range=(0.05, 0.95), init=None, log_every=0,
                 norm_match=False):
    """Learn ONE new direction for a new value of `factor`, shared across all its instances.

    This is the alternative to giving every concept its own adapter.  A LoRA adapter is a private
    function: concept 41 gains nothing from concepts 1-40, which is why N3 measures transfer at
    +0.0013 and why `independent` ties the hierarchy.  A basis direction is shared vocabulary --
    learn "deep_red background" once from a handful of OOD instances and every (digit, foreground)
    combination with it becomes expressible immediately, none of which were trained.

    `targets` [N, D] instances that all share the NEW factor value.
    `labels`  [N, F] their values on the OTHER factors (the entry for `factor` is ignored).
    Only the new vector is optimized; the existing basis is read-only, so every previously
    expressible composition is bit-identical afterwards by construction rather than by freezing.

    `norm_match` holds ‖v‖ at the mean norm of the factor's EXISTING directions.  Left free, the
    optimizer returns a vector 15-20x larger than the basis it is joining (‖v‖ 4.7-5.4 against 0.26-
    0.33 for the trained directions), which is then multiplied by the gain on top; across three seeds
    that produced unseen-combination F1 of 0.65 / 0.35 / 0.71, and the failing seed was the one with
    the largest norm. A new value of a factor should live at the same scale as that factor's other
    values, so this makes the constraint explicit rather than hoping the loss enforces it.

    -> (v [d], history)
    """
    model = field.model
    dev = field.device
    F = len(fac["V"])
    Vt = [torch.as_tensor(np.asarray(v, np.float32), device=dev) for v in fac["V"]]
    mu = torch.as_tensor(np.asarray(fac["mu"], np.float32), device=dev)
    gain = [1.0] * F if gains is None else [float(x) for x in gains]

    d = Vt[0].shape[1]
    v = (torch.zeros(d, device=dev) if init is None
         else torch.as_tensor(np.asarray(init, np.float32), device=dev).clone())
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    target_norm = float(np.linalg.norm(np.asarray(fac["V"][factor], np.float64), axis=1).mean())

    X = torch.from_numpy(np.asarray(targets, np.float32)).to(dev).reshape(-1, CH, field.img, field.img)
    L = torch.as_tensor(np.asarray(labels, int), device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hist = []
    for it in range(n_steps):
        idx = torch.randint(0, len(X), (min(batch, len(X)),), generator=g).to(dev)
        x0, lab = X[idx], L[idx]
        u = torch.rand(len(x0), device=dev) * (u_range[1] - u_range[0]) + u_range[0]
        ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
        e = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        # per-instance conditioning: the KNOWN factors come from the frozen basis, the new value is
        # the single vector being learned, so the gradient sees it under many different contexts
        c = mu.unsqueeze(0).repeat(len(x0), 1)
        for f in range(F):
            if f == factor:
                c = c + gain[f] * v.unsqueeze(0)
            else:
                c = c + gain[f] * Vt[f][lab[:, f]]
        loss = torch.nn.functional.mse_loss(model(xt, u, emb_override=c), e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if norm_match:
            with torch.no_grad():
                n = v.norm()
                if float(n) > 0:
                    v.mul_(target_norm / n)
        hist.append(float(loss))
        if log_every and (it + 1) % log_every == 0:
            print(f"      basis extension {it + 1}/{n_steps}: "
                  f"{np.mean(hist[-log_every:]):.4f}", flush=True)
    return v.detach().cpu().numpy(), dict(first=hist[0], final=float(np.mean(hist[-20:])),
                                          curve=hist, target_norm=target_norm,
                                          norm=float(v.detach().norm()))


def learn_attr_value(field, targets, combos, attr_idx, value=1, n_steps=400, lr=0.05, batch=8,
                     seed=0, u_range=(0.05, 0.95), log_every=0, norm_match=True):
    """`extend_basis` for a FACTORED conditioning head (unet.py `cond_mode='factored'`).

    Same claim, different substrate.  There the basis is a matrix recovered by ideal-words regression
    from an entangled class-embedding table, and a new factor value is a vector appended to it.  Here
    the head already IS a factor basis: the slot `attr[attr_idx, value]` exists but has never
    received a gradient, because no training image carried that attribute value.  Learning the new
    vocabulary item is therefore optimizing exactly one `edim` vector while every other parameter --
    the UNet and the other 27 attribute vectors -- stays frozen.

    Two consequences follow structurally rather than by engineering, and both are the point:
      * cost is one `edim` vector for the value, not one adapter per combination that uses it;
      * every composition NOT involving this attribute value is bit-identical afterwards, because
        nothing it reads was written.

    `targets` [N, D] real instances carrying the new value, in the model's [-1, 1] space.
    `combos`  [N, n_attrs] their FULL attribute rows, so the vector is learned under many different
              contexts at once.  (n11's lesson: teach a direction against varied everything-else or
              it absorbs whatever the teaching set held constant.)
    -> (v [edim], history)
    """
    model, dev = field.model, field.device
    if getattr(model, "cond_mode", "class") != "factored":
        raise RuntimeError("learn_attr_value needs a factored-conditioning backbone")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        others = model.attr[[k for k in range(model.n_attrs) if k != attr_idx]]
        target_norm = float(others.reshape(-1, model.edim).norm(dim=1).mean())
        v = model.attr[attr_idx, value].detach().clone()
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)

    X = torch.from_numpy(np.asarray(targets, np.float32)).to(dev)
    X = X.reshape(-1, CH, field.img, field.img)
    Cb = torch.as_tensor(np.asarray(combos, int), device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    hist = []
    for it in range(n_steps):
        idx = torch.randint(0, len(X), (min(batch, len(X)),), generator=g).to(dev)
        x0, lab = X[idx], Cb[idx]
        u = torch.rand(len(x0), device=dev) * (u_range[1] - u_range[0]) + u_range[0]
        ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float()
        ab = ab.view(-1, 1, 1, 1)
        e = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        # conditioning = frozen vectors for the other attributes + the one being learned, so the
        # gradient sees `v` under every context the teaching instances happen to carry
        c = torch.zeros(len(x0), model.edim, device=dev)
        for k in range(model.n_attrs):
            if k == attr_idx:
                c = c + torch.where((lab[:, k] == value).unsqueeze(1), v.unsqueeze(0),
                                    model.attr[k, 0].unsqueeze(0))
            else:
                c = c + model.attr[k, lab[:, k]]
        loss = torch.nn.functional.mse_loss(model(xt, u, emb_override=c), e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if norm_match:
            with torch.no_grad():
                nv = v.norm()
                if float(nv) > 0:
                    v.mul_(target_norm / nv)
        hist.append(float(loss))
        if log_every and (it + 1) % log_every == 0:
            print(f"      attr-value extension {it + 1}/{n_steps}: "
                  f"{np.mean(hist[-log_every:]):.4f}", flush=True)
    return v.detach(), dict(first=hist[0], final=float(np.mean(hist[-20:])), curve=hist,
                            target_norm=target_norm, norm=float(v.detach().norm()))


def compose_attr(model, combo, overrides=None):
    """Attribute row -> conditioning vector, with optional per-(attr, value) vector substitutions.

    `overrides` maps `(attr_idx, value) -> tensor[edim]`, which is how a newly learned vocabulary
    item is used before (or instead of) being written back into the model.
    """
    overrides = overrides or {}
    c = torch.zeros(model.edim, device=model.attr.device)
    for k in range(model.n_attrs):
        val = int(combo[k])
        c = c + overrides.get((k, val), model.attr[k, val])
    return c


def compose_with(fac, assignment, factor, v, gains=None):
    """`compose`, but substituting a learned direction `v` for one factor's value.

    Used to render combinations of a NEWLY added factor value with existing ones -- none of which
    were trained, and none of which need their own parameters.
    """
    gain = [1.0] * len(fac["V"]) if gains is None else [float(x) for x in gains]
    c = np.asarray(fac["mu"], np.float64).copy()
    for f, a in enumerate(assignment):
        c = c + gain[f] * (np.asarray(v, np.float64) if f == factor
                           else np.asarray(fac["V"][f][int(a)], np.float64))
    return c


def learn_attribute_basis(field, groups, n_steps=400, lr=0.05, batch=2, seed=0,
                          u_range=(0.05, 0.95), norm=None, log_every=0):
    """Build a factor basis from DATA, for a model that has no class-embedding table to factorize.

    `factorize` reads an additive structure out of embeddings the backbone already learned.  A
    pretrained UNCONDITIONAL model (google/ddpm-celebahq-256) has no such table -- there is nothing
    to factorize, so the basis has to be learned through the diffusion loss instead.

    `groups` = {name: (images, )} -- one entry per attribute VALUE, each holding real images that
    have it.  Each direction is optimized independently against its own images, and compositions are
    then formed by SUMMING the directions for the values requested.  That is the ideal-words
    hypothesis stated as a construction rather than as a fit: if it holds on a real pretrained model,
    combinations nobody trained are reachable by addition.

    `norm` rescales each direction to a fixed magnitude.  n11 found this matters: left free the
    optimizer returns directions an order of magnitude out of scale with each other, and summing
    directions of mismatched scale lets whichever is largest dominate the composition.

    -> {name: v}
    """
    out = {}
    for i, (name, X) in enumerate(groups.items()):
        d = field.emb_dim if hasattr(field, "emb_dim") else field.model.emb_dim
        v = torch.zeros(d, device=field.device, requires_grad=True)
        opt = torch.optim.Adam([v], lr=lr)
        Xt = torch.from_numpy(np.asarray(X, np.float32)).to(field.device)
        Xt = Xt.reshape(-1, CH, field.img, field.img)
        g = torch.Generator(device="cpu").manual_seed(seed + i)
        torch.manual_seed(seed + i)
        hist = []
        for it in range(n_steps):
            idx = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g).to(field.device)
            x0 = Xt[idx]
            u = (torch.rand(len(x0), device=field.device) * (u_range[1] - u_range[0]) + u_range[0])
            ab = np.array([field.abar(float(uu)) for uu in u.detach().cpu().numpy()]) \
                if hasattr(field, "abar") else cosine_abar(u.detach().cpu().numpy())
            ab = torch.from_numpy(np.asarray(ab)).to(field.device).float().view(-1, 1, 1, 1)
            e = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
            pred = _eps_grad(field, xt, u, v)
            loss = torch.nn.functional.mse_loss(pred, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if norm:
                with torch.no_grad():
                    nv = v.norm()
                    if float(nv) > 0:
                        v.mul_(float(norm) / nv)
            hist.append(float(loss))
            if log_every and (it + 1) % log_every == 0:
                print(f"      {name} {it + 1}/{n_steps}: {np.mean(hist[-log_every:]):.4f}",
                      flush=True)
        out[name] = v.detach().cpu().numpy()
        print(f"    [basis] {name}: loss {hist[0]:.4f} -> {np.mean(hist[-20:]):.4f}, "
              f"||v|| {float(v.detach().norm()):.3f}", flush=True)
    return out


def _eps_grad(field, xt, u, emb):
    """Epsilon WITH gradient to `emb`.  `field.eps` is no_grad and numpy, so it cannot be used here."""
    if hasattr(field, "model") and hasattr(field.model, "time_embedding") and \
            hasattr(field.model.time_embedding, "extra"):          # pretrained diffusers wrapper
        t = torch.full((len(xt),), field._timestep(float(u.mean())), device=field.device,
                       dtype=torch.long)
        field.model.time_embedding.extra = emb
        try:
            return field.model(xt, t).sample
        finally:
            field.model.time_embedding.extra = None
    return field.model(xt, u, emb_override=emb)                    # our own UNet


def compose_attributes(dirs, names, gains=None):
    """Sum the directions for a set of attribute values -- the composition, with no optimization."""
    v = None
    for i, nm in enumerate(names):
        d = np.asarray(dirs[nm], np.float64) * (1.0 if gains is None else float(gains[i]))
        v = d if v is None else v + d
    return v
