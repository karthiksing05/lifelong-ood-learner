"""Evaluation: attribute probes, composition accuracy, forgetting, and figure helpers.

Composition accuracy is the primary metric, following the logic (not the letter) of the poe-diffusion
paper's evaluation: three softmax probes (digit / fg-colour / bg-colour) are trained on the SEEN slots
only, then applied to generated samples; a sample counts as correct when all three probes recover the
queried combination.  This asks the compositional question directly -- did the sampler produce the
right digit in the right colours on the right background -- where FID at N=100 mostly measures noise.

Forgetting is measured the way continual-learning papers measure it, but on the generative object
itself: per-slot mean model log-likelihood and per-slot denoising R^2 on held-out SEEN data, tracked
across consolidation steps.  For strategy='graft' the base components are untouched, so any drift is
numerically zero up to the mixture-weight renormalization; strategy='ifit' is the destructive control.
"""
import json
import os

import numpy as np

from .datasets import IMG, rgb_img


# ── numpy softmax probe ───────────────────────────────────────────────────────
class SoftmaxProbe:
    def __init__(self, n_classes, lr=0.05, epochs=30, batch=256, seed=0, l2=1e-4):
        self.C, self.lr, self.epochs, self.batch, self.seed, self.l2 = n_classes, lr, epochs, batch, seed, l2
        self.Wb = None

    def fit(self, X, y):
        X = np.asarray(X, np.float32); y = np.asarray(y)
        N, D = X.shape
        rng = np.random.default_rng(self.seed)
        W = np.zeros((D + 1, self.C), np.float32)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        Xb = np.concatenate([X, np.ones((N, 1), np.float32)], 1)
        for ep in range(self.epochs):
            for i0 in range(0, N, self.batch):
                idx = rng.integers(0, N, min(self.batch, N))
                xb, yb = Xb[idx], y[idx]
                z = xb @ W
                z -= z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True)
                p[np.arange(len(yb)), yb] -= 1.0
                g = xb.T @ p / len(yb) + self.l2 * W
                t += 1
                mW = b1 * mW + (1 - b1) * g; vW = b2 * vW + (1 - b2) * g * g
                W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
        self.Wb = W
        return self

    def predict(self, X):
        X = np.asarray(X, np.float32)
        Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        return np.argmax(Xb @ self.Wb, 1)

    def acc(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def fit_probes(data, n_train=None, seed=0, epochs=30):
    """Train digit/fg/bg probes on the SEEN slots.  -> dict of fitted probes (+ heldout accuracies)."""
    seen = np.where(data["seen"])[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(seen)
    if n_train:
        seen = seen[:n_train]
    cut = int(0.9 * len(seen))
    tr, va = seen[:cut], seen[cut:]
    X = data["X"]
    out = {}
    for key, C in (("digit", 10), ("fg", 4), ("bg", 4)):
        p = SoftmaxProbe(C, seed=seed, epochs=epochs).fit(X[tr], data[key][tr])
        out[key] = p
        out[f"{key}_val_acc"] = p.acc(X[va], data[key][va])
    return out


def composition_accuracy(probes, samples, d, f, b):
    """Per-factor and joint accuracy of generated samples against the queried composition."""
    pd = probes["digit"].predict(samples) == d
    pf = probes["fg"].predict(samples) == f
    pb = probes["bg"].predict(samples) == b
    return dict(digit=float(pd.mean()), fg=float(pf.mean()), bg=float(pb.mean()),
                joint=float((pd & pf & pb).mean()))


def _rmse(A, B):
    return float(np.sqrt(np.mean((np.asarray(A, np.float64) - np.asarray(B, np.float64)) ** 2)))


def sample_geometry(samples, xq, reference=None):
    """The faithfulness/generalization split the accuracy metric cannot see.

    Composition accuracy alone has a degenerate optimum: copy the query.  It scores 1.0 while
    learning nothing about the held-out class, which is exactly the failure the poe-diffusion paper
    diagnoses in its query-only baseline (their Table 2: best faithfulness, collapsed
    generalization).  These three numbers separate the cases.

      novelty      RMSE from each sample to the QUERY.  Near zero = memorization.
      diversity    mean pairwise RMSE among samples.  Near zero = mode collapse onto one image.
      gen_rmse     mean over samples of the min RMSE to the slot's OTHER held-out instances.
                   Low = the samples look like real members of the OOD class, not just like x_q.

    A method that has genuinely recovered the composition has all three healthy: novelty and
    diversity comparable to the reference set's own internal spread, gen_rmse no worse than it.
    """
    S = np.asarray(samples, np.float64)
    out = dict(novelty=_rmse(S, np.ravel(xq)[None]))
    if len(S) > 1:
        d = np.sqrt(((S[:, None] - S[None]) ** 2).mean(2))
        out["diversity"] = float(d[np.triu_indices(len(S), 1)].mean())
    else:
        out["diversity"] = 0.0
    if reference is not None and len(reference):
        R = np.asarray(reference, np.float64)
        dm = np.sqrt(((S[:, None] - R[None]) ** 2).mean(2))
        out["gen_rmse"] = float(dm.min(1).mean())
        if len(R) > 1:
            dr = np.sqrt(((R[:, None] - R[None]) ** 2).mean(2))
            out["ref_spread"] = float(dr[np.triu_indices(len(R), 1)].mean())
    return out


# ── forgetting on the generative object itself ───────────────────────────────
def per_slot_metrics(field, X, slots, slot_ids, u_recon=0.05, max_per_slot=40):
    """-> {slot: dict(loglik, r2)} on held-out data, under the CURRENT field."""
    out = {}
    for s in slot_ids:
        idx = np.where(slots == s)[0][:max_per_slot]
        if len(idx) == 0:
            continue
        Xs = X[idx]
        ll = float(np.mean(field.logp(Xs, u=0.0)))
        rec = field.tweedie(np.sqrt(field_abar(u_recon)) * Xs, u_recon)
        ss_res = float(np.sum((rec - Xs) ** 2))
        ss_tot = float(np.sum((Xs - Xs.mean(0)) ** 2))
        out[int(s)] = dict(loglik=ll, r2=1.0 - ss_res / max(ss_tot, 1e-9))
    return out


def field_abar(u):
    from .score import cosine_abar
    return float(cosine_abar(u))


# ── figures ──────────────────────────────────────────────────────────────────
def save_grid(images, path, ncol=8, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = np.asarray(images)
    n = len(images)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.1 * ncol, 1.1 * nrow + (0.4 if title else 0)))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < n:
            ax.imshow(rgb_img(images[i]))
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_json(obj, path, seed=None):
    """Write `obj` to `path`, and -- when `seed` is given -- also to `<path>__seed<k>.json`.

    Writing both means multi-seed aggregation needs no flags and no discipline: the base path is
    always the most recent run, and `experiments/aggregate.py` finds every seed that has ever been
    run.  Without this, re-running with a new seed silently overwrites the old result and a
    single-seed table looks indistinguishable from a pooled one.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    payload = clean(obj)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    if seed is not None:
        stem = path[:-5] if path.endswith(".json") else path
        with open(f"{stem}__seed{int(seed)}.json", "w") as fh:
            json.dump(payload, fh, indent=1)
