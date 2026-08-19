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

from .data.datasets import IMG, rgb_img


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

    The paper uses Inception-V3 features. This is a mild low-pass over pixels instead: a space where
    "same digit, same colours, different instance" is meant to be near and "different digit" far,
    while ignoring the speckle that raw pixels count as variety.

    **This docstring used to claim that although absolute values differ from the paper's, "the
    ordering between methods, which is what the comparison rests on" survives. That claim has since
    been measured and it is not safe for RECALL.** Inception-V3 is now available here
    (`lifelong_ood.inception`, weights already cached) and the two spaces disagree qualitatively on
    the quantity the composition argument turns on: on the same 1600 generated images, Generalization
    recall reads **0.000 in this space and 0.180 in Inception's** (RESULTS §5e-bis). Pixel distance
    is dominated by spatial alignment, so it calls two instances of the same digit far apart exactly
    when they differ in position -- which is the failure mode a recall metric exists to detect.

    Kept because every historical number in this repository was computed here and changing it in
    place would silently rewrite results whose provenance is recorded elsewhere. New work that
    reports precision/recall against the paper should call `paper_metrics`, which names its space.
    """
    X = np.asarray(X, np.float64)
    n, d = X.shape[0], X.shape[1]
    ch = d // (img * img)
    A = X.reshape(n, img, img, ch)
    A = A.reshape(n, img // 2, 2, img // 2, 2, ch).mean((2, 4)).reshape(n, -1)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def _infer_img(X, img=None):
    """Side length of a flat RGB row.  -> int.

    The module default `IMG` is ColorMNIST's 32, and passing it for a 64px image reinterprets that
    image as 32x32 with TWELVE channels -- `_feat` then low-passes over the wrong neighbourhood and
    Inception rejects the tensor outright. Inferring from the row length is the only thing that is
    right for every dataset here (ColorMNIST 32, the CelebA grid 64, CelebA 128).
    """
    if img:
        return int(img)
    d = int(np.asarray(X).shape[1])
    side = int(round((d / 3.0) ** 0.5))
    return side if side * side * 3 == d else IMG


def metric_features(X, space="pixel", img=None, device=None):
    """Feature space for the k-NN density metrics, as a NAMED, RECORDED parameter.  -> (F, used).

    `_feat` above is a pixel space and its own docstring records the retraction: Generalization
    recall reads **0.000 in it and 0.180 in Inception's** on the same 1600 images (RESULTS 5e-bis).
    Fifteen pinned claims are computed through it, four of which read exactly 0.000 -- and a k-NN
    metric reading exactly zero is precisely what a wrong feature space produces.

    So the space becomes a parameter rather than a decision.  **The default does not change in the
    commit that introduces the parameter**: every existing number stays bit-identical, both spaces
    can be reported side by side, and which one is right gets settled by
    `tests/test_feature_space.py`, which calibrates them against arms whose answers are known.

    Rows are L2-normalised in every space.  `_feat` normalises internally and `vendi_score` relies
    on that -- its kernel `F @ F.T / n` is only a cosine kernel for unit rows, so an un-normalised
    space would silently stop its eigenvalues being a probability vector.

    `"inception"` is the torchvision port (`lifelong_ood.inception`), which is where the CelebA
    readability evidence lives; FID/KID keep the pytorch-fid TF port in `lifelong_ood.fid`, whose
    whole purpose is comparability to published numbers. The two are NOT interchangeable, which is
    why the space each result used is recorded rather than assumed.
    """
    X = np.asarray(X, np.float64)
    img = _infer_img(X, img)
    if space == "pixel":
        return _feat(X, img=img), "pixel"
    if space in ("inception", "inception-tv"):
        from . import inception
        if not inception.available():
            return _feat(X, img=img), "pixel(fallback)"
        F = np.asarray(inception.features(np.asarray(X, np.float32), img=img, device=device),
                       np.float64)
        return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9), "inception-tv"
    raise ValueError(f"unknown feature space {space!r}")


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
        # Real ColorMNIST has almost no isolated pixels (~2e-4), so a plain ratio divides by nearly
        # zero and swings between 0.1 and 20345 across runs that look identical.  Floor the
        # denominator at one pixel in ten thousand -- below that the reference has nothing to
        # measure against and the ABSOLUTE count is the honest number.
        out["isolated_ratio"] = float(iso / max(riso, 1e-4))
    return out


def vendi_score(samples, q=1.0, space="pixel", img=None, device=None):
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
    F, _used = metric_features(samples, space, img, device)
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


def memorization(samples, train, space="pixel", img=None, device=None):
    """MOSAIC's copying test: a sample is memorized iff d_NN1 / d_NN2 < 1/3.

    Reported because "generated the held-out combination" and "retrieved a training image that
    happens to look like it" are indistinguishable in every aggregate metric here -- a method that
    copies scores WELL on precision and Frechet.  Comparing each sample's nearest training neighbour
    to its SECOND nearest makes copying visible: a genuine sample sits at comparable distance from
    several training images, while a copy is dramatically closer to the one it came from.

    -> dict(frac_memorized, median_ratio).  `train` should be images the model actually saw.
    """
    S, _used = metric_features(samples, space, img, device)
    T, _ = metric_features(train, space, img, device)
    if len(T) < 2 or not len(S):
        return dict(frac_memorized=float("nan"), median_ratio=float("nan"))
    D = np.sqrt(np.maximum(((S[:, None, :] - T[None, :, :]) ** 2).sum(-1), 0.0))
    part = np.sort(D, axis=1)[:, :2]
    ratio = part[:, 0] / np.maximum(part[:, 1], 1e-12)
    return dict(space=_used, frac_memorized=float((ratio < 1.0 / 3.0).mean()),
                median_ratio=float(np.median(ratio)))


def _knn_radius(F, k):
    D = np.sqrt(((F[:, None] - F[None]) ** 2).sum(2))
    np.fill_diagonal(D, np.inf)
    return np.sort(D, 1)[:, min(k, D.shape[1] - 1) - 1]


def precision_recall(samples, reference, k=3, space="pixel", img=None, device=None):
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
    FS, _used = metric_features(S, space, img, device)
    FR, _ = metric_features(R, space, img, device)
    rR, rS = _knn_radius(FR, k), _knn_radius(FS, k)
    D = np.sqrt(((FS[:, None] - FR[None]) ** 2).sum(2))          # [n_samples, n_ref]
    precision = float((D <= rR[None, :]).any(1).mean())          # sample inside SOME ref ball
    recall = float((D.T <= rS[None, :]).any(1).mean())           # ref inside SOME sample ball
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=float(f1), space=_used)


def density_coverage(samples, reference, k=3, space="pixel", img=None, device=None):
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
    FS, _used = metric_features(S, space, img, device)
    FR, _ = metric_features(R, space, img, device)
    rR = _knn_radius(FR, k)
    D = np.sqrt(((FS[:, None] - FR[None]) ** 2).sum(2))          # [n_samples, n_ref]
    inside = D <= rR[None, :]
    density = float(inside.sum() / (k * len(FS)))
    coverage = float(inside.any(0).mean())
    return dict(density=density, coverage=coverage)


def frechet_distance(samples, reference, space="pixel", img=None, device=None):
    """Frechet distance in the same feature space the other coverage metrics use.

    NOT "FID": FID is defined on Inception-V3 pool3 features, and Inception is the wrong detector for
    32x32 recoloured MNIST -- it was trained on natural images at 299x299 and this repository's whole
    benchmark is out of its domain.  Using it anyway would produce a number that LOOKS comparable to
    the literature and is not.  This reports the same Frechet statistic on the low-pass pixel
    features `_feat` already defines, so it is internally comparable across arms and honestly
    labelled as not comparable to published FID values.
    """
    from scipy import linalg
    FS, _used = metric_features(samples, space, img, device)
    FR, _ = metric_features(reference, space, img, device)
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


def _prf_from_features(FS, FR, k=3):
    """Kynkaanniemi precision/recall given features that are ALREADY in the target space."""
    if len(FS) < 2 or len(FR) < 2:
        return dict(precision=float("nan"), recall=float("nan"), f1=float("nan"))
    rR, rS = _knn_radius(FR, k), _knn_radius(FS, k)
    D = np.sqrt(((FS[:, None] - FR[None]) ** 2).sum(2))
    precision = float((D <= rR[None, :]).any(1).mean())
    recall = float((D.T <= rS[None, :]).any(1).mean())
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=float(f1))


def _frechet_from_features(FS, FR):
    """Frechet distance given features already in the target space (see `frechet_distance`)."""
    from scipy import linalg
    if len(FS) < 4 or len(FR) < 4:
        return float("nan")
    d = int(min(64, len(FS) - 1, len(FR) - 1))
    _, _, Vt = np.linalg.svd(FR - FR.mean(0), full_matrices=False)
    A, B = FS @ Vt[:d].T, FR @ Vt[:d].T
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    eps = 1e-8 * np.eye(d)
    covmean, _ = linalg.sqrtm((s1 + eps).dot(s2 + eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def paper_metrics(samples, reference, img=None, k=3, device=None, space="inception"):
    """The companion paper's Table 1-2 metric row, in the feature space it defines them in.

    Separate from `precision_recall` on purpose.  That function's space is `_feat` (a 2x pixel
    downsample) and every number the repository has already published rests on it; changing it in
    place would silently rewrite results whose provenance is recorded elsewhere.  This is the
    reproduction path, and it says which space it used so the two can never be confused.

    `space="inception"` matches the paper (2048-d pool3); it falls back to `_feat` with `space`
    reported as "pixel" when torchvision or the cached weights are absent, so a run on a machine
    without them degrades loudly rather than pretending.

    `cos` stands in for the paper's CLIP image-image similarity -- mean cosine between each generated
    sample and the reference set, in whichever space is in use.  It is NOT CLIP: no CLIP checkpoint
    is on this machine, and an image-image cosine in Inception space measures a related but different
    thing.  Reported because its ORDERING across arms is what their table uses it for, and labelled
    so it is never quoted as a CLIP score.
    """
    S, R = np.atleast_2d(np.asarray(samples, np.float64)), \
        np.atleast_2d(np.asarray(reference, np.float64))
    used = space
    if space == "inception":
        from lifelong_ood import inception
        if inception.available():
            FS, FR = (inception.features(S, img=img, device=device),
                      inception.features(R, img=img, device=device))
        else:
            used, FS, FR = "pixel(fallback)", _feat(S), _feat(R)
    else:
        used, FS, FR = "pixel", _feat(S), _feat(R)
    nS = FS / (np.linalg.norm(FS, axis=1, keepdims=True) + 1e-9)
    nR = FR / (np.linalg.norm(FR, axis=1, keepdims=True) + 1e-9)
    out = dict(_prf_from_features(FS, FR, k=k))
    out.update(fid=_frechet_from_features(FS, FR), cos=float((nS @ nR.T).mean()), space=used,
               n_samples=int(len(S)), n_reference=int(len(R)))
    return out


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
def _probe_features(X, mode, img=None, device=None):
    """Feature space the attribute probes live in.  -> [N, D] float32."""
    if mode != "inception":
        return np.asarray(X, np.float32)
    from . import inception
    if not inception.available():
        return np.asarray(X, np.float32)
    return inception.features(np.asarray(X, np.float32), img=img, device=device)


def fit_attr_probes(data, n_train=None, seed=0, epochs=30, attrs=None, features="pixel",
                    device=None):
    """Train one binary probe per CelebA attribute on the SEEN classes only.

    ColorMNIST has three categorical factors and a natural "joint = all three correct".  CelebA has
    14 BINARY attributes that overlap spatially, so the analogue is one probe per attribute and a
    joint score over them.  Reporting the per-attribute validation accuracy alongside matters more
    here than it did there: several CelebA attributes are close to unlearnable at 64x64 from a few
    hundred examples, and an attribute the probe cannot read is not evidence about the generator.

    `features="inception"` fits the probes on Inception-V3 pool3 activations instead of raw pixels,
    and on CelebA it roughly doubles what the benchmark can see -- measured on real images with
    `tests/diag_probe_readability.py`:

        pixel       5/14 attributes readable
        inception   9/14                       (+Black_Hair, Blond_Hair, Brown_Hair, Big_Nose)

    A linear probe on raw pixels is the right instrument for ColorMNIST, whose factors are colour and
    digit shape, and the wrong one for `Big_Lips` or `Wavy_Hair`: nine unreadable probes was a
    statement about the probe, not about the generator. The hair colours matter most, because CelebA's
    class definitions are built from them. The mode is recorded in the returned dict so
    `attr_accuracy` featurizes generated samples the same way -- a probe fitted in one space and
    applied in another would be silently meaningless.
    """
    # `.data.celeba`, not `.celeba`: the loaders moved into the `data` subpackage and this deferred
    # import was not updated with them, so `fit_attr_probes` raised ModuleNotFoundError for every
    # caller that did not pass `attrs=` explicitly.  A function-local import fails only when the
    # function runs, which is why nothing caught it at import time.
    from .data.celeba import ATTRS
    attrs = attrs or ATTRS
    seen = np.where(data["seen"])[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(seen)
    if n_train:
        seen = seen[:n_train]
    cut = int(0.9 * len(seen))
    tr, va = seen[:cut], seen[cut:]
    A = data["attrs"]
    img = data.get("img")
    X = _probe_features(data["X"], features, img=img, device=device)
    out = {"_features": features, "_img": img}
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
    # A DEGENERATE attribute is never usable, and the nan fallback below used to admit one.
    #
    # `Bald` has a base rate of **0.000** in the loaded CelebA split -- the attribute never occurs.
    # With no positives, `tpr` is nan, which sent it down the `val_acc` branch, where a probe that
    # answers "no" every time scores 1.000 and passes. It was then one of the five "readable" probes
    # that `joint` requires to be simultaneously correct: a probe that cannot be wrong, contributing
    # nothing but a free pass. The same trap catches any attribute that is always ON.
    base = probes.get(f"{a}_base")
    if base is not None and base == base and (base <= 0.0 or base >= 1.0):
        return False
    bal, tpr = probes.get(f"{a}_bal"), probes.get(f"{a}_tpr")
    if bal is None or tpr is None or bal != bal or tpr != tpr:
        # nan bal/tpr means one class was absent from validation, which is the degenerate case
        # again; fall back to plain accuracy ONLY when the statistics are simply missing (probe
        # dicts pickled before they were recorded), never when they are nan.
        if (bal is not None and bal != bal) or (tpr is not None and tpr != tpr):
            return False
        return probes.get(f"{a}_val_acc", 0.0) >= min_val_acc
    return bal >= min_bal and tpr >= min_tpr


def attr_accuracy(probes, samples, combo, attrs=None, min_val_acc=0.65):
    """Per-attribute and joint accuracy of samples against a target attribute combination.

    Attributes whose own probe cannot clear `min_val_acc` on real held-out data are EXCLUDED from
    the joint score and reported separately -- scoring a generator on an attribute no probe can read
    measures the probe, not the generator.
    """
    # `combo` is ordered by the CANONICAL attribute list, so an attribute's target must be looked up
    # by its position THERE -- not by its position in whatever `attrs` the caller passed.
    #
    # This was `combo[k]` with `k` enumerating `attrs`, which is correct only when `attrs` is the full
    # list. Two callers pass a PRE-FILTERED list (`attrs=usable`), and for them every probe was
    # scored against a different attribute's target: with the pixel probes `usable` is
    # [Gray_Hair, Wavy_Hair, No_Beard, Smiling, Male] at canonical positions [3,4,5,6,10], while
    # combo[0..4] reads [Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Wavy_Hair]. That silently
    # produced the "conditioning on a class's own embedding scores below unconditional" result, which
    # was then used to withdraw a whole column and to refute a hypothesis. Both have to be re-run.
    canon = list(probes["_attrs"])
    attrs = attrs or canon
    # featurize the SAMPLES the same way the probes were fitted, or the probe is being applied in a
    # space it never saw.  Defaults to pixels, so every existing caller is unaffected.
    S = _probe_features(samples, probes.get("_features", "pixel"), img=probes.get("_img"))
    per, usable = {}, []
    for a in attrs:
        if not probe_usable(probes, a, min_val_acc=min_val_acc):
            continue
        if a not in canon:
            continue
        hit = probes[a].predict(S) == int(combo[canon.index(a)])
        per[a] = float(hit.mean())
        usable.append(hit)
    if not usable:
        return dict(joint=float("nan"), per_attr=per, n_scored=0)
    U = np.stack(usable)
    return dict(joint=float(U.all(0).mean()), mean_attr=float(U.mean()),
                per_attr=per, n_scored=len(usable))


def fit_factor_probes(data, n_train=None, seed=0, epochs=30, min_val_acc=0.55,
                      features="pixel", device=None):
    """Generic per-FACTOR probes for any dataset satisfying `factored.py`'s contract.

    `fit_probes` is ColorMNIST's (digit/fg/bg) and `fit_attr_probes` is CelebA's (14 binary
    attributes).  Neither generalises, and a third dataset needing a third probe fitter is exactly
    the duplication `factored.py` exists to stop.  This one reads `data["factors"]`, `sizes` and
    `names`, so it works on ColorMNIST, CelebA and Shapes3D unchanged.

    -> dict with one `SoftmaxProbe` per factor name, plus `_names`, `_sizes`, and `<name>_val` /
    `<name>_base` diagnostics.

    THE VALIDATION ACCURACY IS NOT DECORATION.  On CelebA, scoring compositions through probes that
    could not read their attribute produced a headline negative that was purely a measurement
    artifact (5e-sexies), and a probe with a degenerate base rate scored 1.000 by always answering
    "no".  `probe_readable` below applies both guards, and any caller computing a joint accuracy
    must filter with it or it is measuring its own probes.
    """
    F = np.asarray(data["factors"], int)
    sizes = [int(s) for s in data["sizes"]]
    names = list(data["names"])
    seen = np.where(np.asarray(data["seen"], bool))[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(seen)
    if n_train:
        seen = seen[:n_train]
    cut = max(1, int(0.9 * len(seen)))
    tr, va = seen[:cut], seen[cut:]
    # FEATURE SPACE AS A RECORDED PARAMETER, mirroring `fit_attr_probes`. This fitter is the one
    # every non-ColorMNIST dataset routes through and it was pixel-locked, while the CelebA-only
    # fitter already supported Inception -- so the generic path could not use the stronger probe
    # even where it was measured to read more attributes (5/14 -> 9/14 at 128px). `_features` and
    # `_img` are stored so `factor_accuracy` re-featurises SAMPLES the same way; fitting in one
    # space and applying in another is its own fault class and has bitten this repo before.
    X = _probe_features(np.asarray(data["X"], np.float32), features,
                        img=data.get("img"), device=device)
    out = {"_names": names, "_sizes": sizes, "_features": features, "_img": data.get("img")}
    for k, nm in enumerate(names):
        y = F[:, k]
        p = SoftmaxProbe(sizes[k], seed=seed, epochs=epochs).fit(X[tr], y[tr])
        out[nm] = p
        if len(va):
            pred = p.predict(X[va])
            out[f"{nm}_val"] = float((pred == y[va]).mean())
            # the majority-class rate: a probe that cannot beat it has learned nothing
            cnt = np.bincount(y[va], minlength=sizes[k])
            out[f"{nm}_base"] = float(cnt.max() / max(cnt.sum(), 1))
        else:
            out[f"{nm}_val"] = float("nan")
            out[f"{nm}_base"] = float("nan")
    out["_min_val_acc"] = float(min_val_acc)
    return out


def probe_readable(probes, name, min_val_acc=None):
    """Is this factor's probe good enough to score a composition with?  -> bool."""
    v = probes.get(f"{name}_val")
    b = probes.get(f"{name}_base")
    thr = probes.get("_min_val_acc", 0.55) if min_val_acc is None else min_val_acc
    if v is None or v != v:
        return False
    if b is not None and b == b and b >= 0.999:      # degenerate: one value is everything
        return False
    return bool(v >= thr and (b is None or b != b or v > b + 0.02))


def factor_accuracy(probes, S, target, names=None):
    """Per-factor and joint accuracy of samples `S` against a target factor row.  -> dict.

    `target` is one value per factor, indexed by CANONICAL position in `probes["_names"]` -- the
    same indexing bug that produced 5e-sexies' retracted negative is avoided by looking the name up
    rather than enumerating a filtered list.
    """
    canon = list(probes["_names"])
    names = names or canon
    # Re-featurise in the SAME space the probes were fitted in. Without this a probe fitted on
    # Inception features would be handed raw pixels -- the fault `attr_accuracy` already guards.
    S = _probe_features(np.asarray(S, np.float32), probes.get("_features", "pixel"),
                        img=probes.get("_img"))
    per, usable = {}, []
    for nm in names:
        if nm not in canon or not probe_readable(probes, nm):
            continue
        hit = probes[nm].predict(S) == int(target[canon.index(nm)])
        per[nm] = float(np.mean(hit))
        usable.append(hit)
    per["joint"] = float(np.mean(np.all(np.stack(usable), 0))) if usable else float("nan")
    per["n_usable"] = len(usable)
    return per
