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


def _feat(X, img=IMG):
    """A cheap feature space for the k-NN density metrics: 2x downsample, unit-normalized.

    The paper uses Inception-V3 features. We have no Inception here and it would be the wrong
    detector for 32x32 recoloured MNIST anyway; what matters for these metrics is a space where
    "same digit, same colours, different instance" is near and "different digit" is far, and a mild
    low-pass over pixels does that on this dataset while ignoring the speckle that raw pixels count
    as variety. Absolute values are therefore NOT comparable to the paper's; the ordering between
    methods, which is what the comparison rests on, is.
    """
    X = np.asarray(X, np.float64)
    n, d = X.shape[0], X.shape[1]
    ch = d // (img * img)
    A = X.reshape(n, img, img, ch)
    A = A.reshape(n, img // 2, 2, img // 2, 2, ch).mean((2, 4)).reshape(n, -1)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def speckle(samples, reference=None, img=IMG):
    """High-frequency energy relative to real data -- the "scattered pixels" artifact, quantified.

    Classifier-free guidance applied at LOW noise levels amplifies the model's high-frequency error:
    the trajectory is being pushed while the only thing left to decide is fine detail, and the push
    lands on individual pixels.  It shows up as isolated bright dots on an otherwise clean image, and
    no accuracy or coverage metric in this file can see it -- a speckled digit is still the right
    digit in the right colours.

    Two numbers, both scale-free:
      `tv_ratio`   total variation of the samples / total variation of the reference. 1.0 means the
                   samples are as smooth as real data; >1 means extra high-frequency energy.
      `isolated`   fraction of pixels that differ from ALL four neighbours by more than `thr` -- a
                   direct count of dots, which TV alone can be fooled about by legitimate texture.
    """
    def _stats(X):
        n = len(X)
        A = np.asarray(X, np.float64).reshape(n, img, img, -1)
        dh = np.abs(np.diff(A, axis=1)).mean()
        dw = np.abs(np.diff(A, axis=2)).mean()
        tv = 0.5 * (dh + dw)
        P = A.mean(-1)                                   # luminance, for the isolation test
        pad = np.pad(P, ((0, 0), (1, 1), (1, 1)), mode="edge")
        nb = np.stack([pad[:, :-2, 1:-1], pad[:, 2:, 1:-1],
                       pad[:, 1:-1, :-2], pad[:, 1:-1, 2:]])
        thr = 0.25
        iso = (np.abs(P[None] - nb) > thr).all(0).mean()
        return tv, iso

    tv, iso = _stats(samples)
    out = dict(tv=float(tv), isolated=float(iso))
    if reference is not None and len(reference):
        rtv, riso = _stats(reference)
        out["tv_ratio"] = float(tv / max(rtv, 1e-9))
        out["isolated_ratio"] = float(iso / max(riso, 1e-9))
    return out


def vendi_score(samples, q=1.0):
    """Diversity WITHOUT a reference set (Friedman & Dieng, TMLR 2023, arXiv:2210.02410).

    The exponential of the Shannon entropy of the eigenvalues of the normalized sample-similarity
    matrix; it reads as an effective number of distinct modes, so 1.0 means "every sample is the
    same image" and n means "all n are mutually dissimilar".

    This is here because every other metric in this file is anchored to real images of the target,
    and for a held-out combination those may not exist -- and where they do, precision and density
    are anchored to a manifold the correct answer is deliberately absent from.  Vendi needs no
    reference at all, so it is the one diversity number that stays well-defined in exactly the
    setting this repository is about, and it is what distinguishes a method that covers a new
    combination from one that emits eight copies of a prototype.
    """
    F = _feat(np.asarray(samples, np.float64))
    n = len(F)
    if n < 2:
        return 1.0
    K = (F @ F.T) / n                      # unit-norm rows, so this is cosine similarity / n
    w = np.linalg.eigvalsh((K + K.T) / 2)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 1.0
    w = w / s
    nz = w[w > 1e-12]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def memorization(samples, train):
    """MOSAIC's copying test: a sample is memorized iff d_NN1 / d_NN2 < 1/3.

    Reported because "generated the held-out combination" and "retrieved a training image that
    happens to look like it" are indistinguishable in every aggregate metric here -- a method that
    copies scores WELL on precision and Frechet.  Comparing each sample's nearest training neighbour
    to its SECOND nearest makes copying visible: a genuine sample sits at comparable distance from
    several training images, while a copy is dramatically closer to the one it came from.

    -> dict(frac_memorized, median_ratio).  `train` should be images the model actually saw.
    """
    S, T = _feat(np.asarray(samples, np.float64)), _feat(np.asarray(train, np.float64))
    if len(T) < 2 or not len(S):
        return dict(frac_memorized=float("nan"), median_ratio=float("nan"))
    D = np.sqrt(np.maximum(((S[:, None, :] - T[None, :, :]) ** 2).sum(-1), 0.0))
    part = np.sort(D, axis=1)[:, :2]
    ratio = part[:, 0] / np.maximum(part[:, 1], 1e-12)
    return dict(frac_memorized=float((ratio < 1.0 / 3.0).mean()),
                median_ratio=float(np.median(ratio)))


def _knn_radius(F, k):
    D = np.sqrt(((F[:, None] - F[None]) ** 2).sum(2))
    np.fill_diagonal(D, np.inf)
    return np.sort(D, 1)[:, min(k, D.shape[1] - 1) - 1]


def precision_recall(samples, reference, k=3):
    """Improved precision & recall (Kynkaanniemi et al. 2019) -- the paper's coverage metric.

    PRECISION = fraction of generated samples that fall inside the reference manifold (are they
    plausible members of the class?).  RECALL = fraction of REFERENCE images that fall inside the
    generated manifold (does the generation cover the class, or only one corner of it?).

    Recall is the number our composition-accuracy metric structurally cannot express, and it is the
    one that separates the paper's PoE (80.7%) from its query-only baseline (2.5%).  A method that
    emits near-copies of one image can score high precision and essentially zero recall -- which is
    exactly the diagnosis the case studies suggested for us.
    """
    S, R = np.asarray(samples, np.float64), np.asarray(reference, np.float64)
    if len(S) < 2 or len(R) < 2:
        return dict(precision=float("nan"), recall=float("nan"), f1=float("nan"))
    FS, FR = _feat(S), _feat(R)
    rR, rS = _knn_radius(FR, k), _knn_radius(FS, k)
    D = np.sqrt(((FS[:, None] - FR[None]) ** 2).sum(2))          # [n_samples, n_ref]
    precision = float((D <= rR[None, :]).any(1).mean())          # sample inside SOME ref ball
    recall = float((D.T <= rS[None, :]).any(1).mean())           # ref inside SOME sample ball
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=float(f1))


def density_coverage(samples, reference, k=3):
    """Density & Coverage (Naeem et al. 2020), the modern replacement for precision & recall.

    Precision/recall (Kynkaanniemi et al. 2019) are known to be unstable to outliers: a single
    reference point with a large k-NN radius swallows arbitrarily many generated samples and inflates
    precision.  Naeem et al. fix both halves:

      DENSITY  = (1/(k*|S|)) * sum_s |{r : ||s-r|| <= radius_k(r)}|  -- counts how many reference
                 balls each sample lands in, so one fat ball cannot certify a whole cloud.
      COVERAGE = fraction of REFERENCE points with at least one generated sample inside their own
                 k-NN radius -- the fraction of the real class that was actually reached.

    Coverage is the quantity this repository's whole argument rests on, and it is more robust than
    the recall it complements: recall asks whether a real point falls in the GENERATED manifold,
    whose radii a mode-collapsed generator can make meaningless, while coverage uses the REFERENCE
    radii, which the generator cannot influence.  Reported alongside precision/recall, not instead.
    """
    S, R = np.asarray(samples, np.float64), np.asarray(reference, np.float64)
    if len(S) < 2 or len(R) < 2:
        return dict(density=float("nan"), coverage=float("nan"))
    FS, FR = _feat(S), _feat(R)
    rR = _knn_radius(FR, k)
    D = np.sqrt(((FS[:, None] - FR[None]) ** 2).sum(2))          # [n_samples, n_ref]
    inside = D <= rR[None, :]
    density = float(inside.sum() / (k * len(FS)))
    coverage = float(inside.any(0).mean())
    return dict(density=density, coverage=coverage)


def frechet_distance(samples, reference):
    """Frechet distance in the same feature space the other coverage metrics use.

    NOT "FID": FID is defined on Inception-V3 pool3 features, and Inception is the wrong detector for
    32x32 recoloured MNIST -- it was trained on natural images at 299x299 and this repository's whole
    benchmark is out of its domain.  Using it anyway would produce a number that LOOKS comparable to
    the literature and is not.  This reports the same Frechet statistic on the low-pass pixel
    features `_feat` already defines, so it is internally comparable across arms and honestly
    labelled as not comparable to published FID values.
    """
    from scipy import linalg
    FS, FR = _feat(np.asarray(samples, np.float64)), _feat(np.asarray(reference, np.float64))
    if len(FS) < 4 or len(FR) < 4:
        return float("nan")
    # Project onto the reference's leading directions first.  With 24 samples in 768 dimensions the
    # covariances are singular and `sqrtm` of their product is both slow and numerically junk; the
    # statistic is only meaningful in a subspace the sample count can actually estimate.
    d = int(min(64, len(FS) - 1, len(FR) - 1))
    Rc = FR - FR.mean(0)
    _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    P = Vt[:d].T
    A, B = FS @ P, FR @ P
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    eps = 1e-8 * np.eye(d)                                   # shrinkage keeps sqrtm well-posed
    covmean, _ = linalg.sqrtm((s1 + eps).dot(s2 + eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def _content_rmse(X, img=IMG):
    """Mean pairwise RMSE after a 2x2 box downsample -- a crude low-pass.

    Raw pixel diversity cannot distinguish "a different instance of the class" from "the same image
    plus salt-and-pepper".  Inspecting the case-study panels made that concrete: rows whose measured
    diversity looked respectable were visibly the same digit eight times over, varying only in
    speckle.  Averaging 2x2 blocks destroys per-pixel noise while leaving stroke shape and colour
    intact, so comparing this against the raw number says how much of the apparent variety is
    CONTENT.  Measured on the neural PoE: raw diversity 0.61x the class spread, content 0.45x.
    """
    X = np.asarray(X, np.float64)
    n, d = X.shape[0], X.shape[1]
    ch = d // (img * img)
    A = X.reshape(n, img, img, ch)
    A = A.reshape(n, img // 2, 2, img // 2, 2, ch).mean((2, 4)).reshape(n, -1)
    if n < 2:
        return 0.0
    D = np.sqrt(((A[:, None] - A[None]) ** 2).mean(2))
    return float(D[np.triu_indices(n, 1)].mean())


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
    out["content_diversity"] = _content_rmse(S)
    # diversity / novelty separates "independent draws around the query" from "one rendering of it".
    # For samples = query + iid noise the ratio is sqrt(2) ~ 1.41; measured at ~1.04, which means the
    # samples share a common offset from the query and are near-copies of each other, not of a
    # distribution around it.  This number is what turned a plausible-looking grid into a red flag.
    out["div_over_nov"] = out["diversity"] / max(out["novelty"], 1e-9)
    if reference is not None and len(reference):
        R = np.asarray(reference, np.float64)
        dm = np.sqrt(((S[:, None] - R[None]) ** 2).mean(2))
        out["gen_rmse"] = float(dm.min(1).mean())
        if len(R) > 1:
            dr = np.sqrt(((R[:, None] - R[None]) ** 2).mean(2))
            out["ref_spread"] = float(dr[np.triu_indices(len(R), 1)].mean())
            out["content_ref_spread"] = _content_rmse(R)
            out["content_ratio"] = (out["content_diversity"]
                                    / max(out["content_ref_spread"], 1e-9))
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
def _auto_img(vec):
    """Infer the side length of a flat RGB image so figure helpers work at any resolution."""
    n = int(np.asarray(vec).size)
    side = int(round((n / 3) ** 0.5))
    return side if side * side * 3 == n else IMG


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
            side = _auto_img(images[i])
            ax.imshow(np.clip(np.asarray(images[i], np.float32).reshape(side, side, 3), 0, 1))
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


# ── CelebA: one binary probe per attribute ───────────────────────────────────
def fit_attr_probes(data, n_train=None, seed=0, epochs=30, attrs=None):
    """Train one binary probe per CelebA attribute on the SEEN classes only.

    ColorMNIST has three categorical factors and a natural "joint = all three correct".  CelebA has
    14 BINARY attributes that overlap spatially, so the analogue is one probe per attribute and a
    joint score over them.  Reporting the per-attribute validation accuracy alongside matters more
    here than it did there: several CelebA attributes are close to unlearnable at 64x64 from a few
    hundred examples, and an attribute the probe cannot read is not evidence about the generator.
    """
    from .celeba import ATTRS
    attrs = attrs or ATTRS
    seen = np.where(data["seen"])[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(seen)
    if n_train:
        seen = seen[:n_train]
    cut = int(0.9 * len(seen))
    tr, va = seen[:cut], seen[cut:]
    X, A = data["X"], data["attrs"]
    out = {}
    for k, a in enumerate(attrs):
        pr = SoftmaxProbe(2, seed=seed, epochs=epochs).fit(X[tr], A[tr, k].astype(int))
        out[a] = pr
        y = A[va, k].astype(int)
        p = pr.predict(X[va])
        out[f"{a}_val_acc"] = pr.acc(X[va], y)
        # TPR/TNR/balanced as well as plain accuracy, because plain accuracy is the WRONG
        # readability criterion for a rare attribute: Gray_Hair is ON in 4.3% of CelebA, so a probe
        # that answers "no" every time scores 0.957 while recognising the attribute 32% of the time.
        # Measured on real images, five of the twelve attributes the old >=0.65 accuracy filter
        # admitted cannot recognise their own attribute (Oval_Face TPR 0.066, Bald 0.206, Brown_Hair
        # 0.171, Gray_Hair 0.321, No_Beard TNR 0.303) -- and Wavy_Hair, which the filter REJECTED at
        # 0.533, is usable with TPR 0.908.  Scoring a generator on a blind probe measures the probe.
        out[f"{a}_tpr"] = float((p[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
        out[f"{a}_tnr"] = float((p[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
        out[f"{a}_bal"] = 0.5 * (out[f"{a}_tpr"] + out[f"{a}_tnr"])
        out[f"{a}_base"] = float(y.mean())
    out["_attrs"] = list(attrs)
    return out


def probe_usable(probes, a, min_bal=0.65, min_tpr=0.5, min_val_acc=0.65):
    """Can this probe be used to SCORE a generator on attribute `a`?

    Requires it to recognise the attribute when present (`tpr`) and to be meaningfully above chance
    on both classes (`bal`), not merely accurate on an imbalanced test set.  Falls back to plain
    accuracy for probe dicts pickled before these statistics were recorded.
    """
    bal, tpr = probes.get(f"{a}_bal"), probes.get(f"{a}_tpr")
    if bal is None or tpr is None or bal != bal or tpr != tpr:
        return probes.get(f"{a}_val_acc", 0.0) >= min_val_acc
    return bal >= min_bal and tpr >= min_tpr


def attr_accuracy(probes, samples, combo, attrs=None, min_val_acc=0.65):
    """Per-attribute and joint accuracy of samples against a target attribute combination.

    Attributes whose own probe cannot clear `min_val_acc` on real held-out data are EXCLUDED from
    the joint score and reported separately -- scoring a generator on an attribute no probe can read
    measures the probe, not the generator.
    """
    attrs = attrs or probes["_attrs"]
    per, usable = {}, []
    for k, a in enumerate(attrs):
        if not probe_usable(probes, a, min_val_acc=min_val_acc):
            continue
        hit = probes[a].predict(np.asarray(samples, np.float32)) == int(combo[k])
        per[a] = float(hit.mean())
        usable.append(hit)
    if not usable:
        return dict(joint=float("nan"), per_attr=per, n_scored=0)
    U = np.stack(usable)
    return dict(joint=float(U.all(0).mean()), mean_attr=float(U.mean()),
                per_attr=per, n_scored=len(usable))
