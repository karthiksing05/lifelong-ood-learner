"""Inception-based FID and KID, for the datasets where Inception is actually in domain.

`evals.frechet_distance` computes the Frechet statistic on low-pass PIXEL features and is deliberately
not called FID: Inception-V3 was trained on natural images at 299x299, and 32x32 recoloured MNIST is
outside its domain, so a number computed there would look comparable to the literature without being
comparable to anything.  That reasoning still holds.  This module is for the other half of the
benchmark suite -- CelebA, CIFAR-10, CLEVR -- where Inception *is* the right detector and a reviewer
will rightly expect the standard number.

Two estimators, and the choice between them is not stylistic:

  `fid`   the standard Frechet Inception Distance.  It is a BIASED estimator and the bias is O(1/n)
          with a large constant: at the sample counts this repository uses per composition (16-100)
          the bias dominates the signal, and two identical distributions can differ by more than two
          genuinely different ones.  Use it POOLED, over >= 10k samples, for a dataset-level claim.

  `kid`   Kernel Inception Distance (Binkowski et al., ICLR 2018, arXiv:1801.01401): the UNBIASED
          MMD^2 estimator with a degree-3 polynomial kernel.  Its expectation does not depend on the
          sample count, so it is the honest choice per-composition, which is where every OOD claim in
          this repository lives.  `tests/test_fid.py` demonstrates the difference empirically rather
          than asserting it on authority.

**Layout.** This repository's flat vectors are HWC as written by the loaders, and the model pipeline
*reinterprets* them as CHW -- a fixed permutation that is harmless there because training and
sampling apply it identically.  It is NOT harmless here: Inception must see an upright natural image
or its features are meaningless, and a permuted image still produces a confident-looking number.  So
this module reshapes the DISPLAY way, `(n, img, img, 3)`, matching `datasets.rgb_img` and
`evals._feat`.  This repository has already lost three hours to that exact permutation once
(`tests/test_layout.py`); `tests/test_fid.py` pins it here too.

Inputs are flat `[n, D]` in **[0, 1]** -- the output of `neural.to_data`, not model space.
"""
import numpy as np

_MODEL = {}


def _inception(device="cuda", dims=2048):
    """The TF-ported Inception weights, cached per device.

    `pytorch_fid`'s port, not `torchvision`'s: the two give different absolute FID values, and every
    published number this would be compared against uses the ported one.
    """
    key = (device, dims)
    if key not in _MODEL:
        import torch
        from pytorch_fid.inception import InceptionV3
        blk = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        m = InceptionV3([blk], resize_input=True, normalize_input=True, use_fid_inception=True)
        _MODEL[key] = m.to(device).eval()
        for p in _MODEL[key].parameters():
            p.requires_grad_(False)
    return _MODEL[key]


def _to_nchw(X, img=None):
    """Flat [n, D] in [0,1] -> float32 NCHW, reading the flat row the DISPLAY way (HWC)."""
    X = np.asarray(X, np.float32)
    if X.ndim == 1:
        X = X[None]
    n, D = X.shape
    if img is None:
        for ch in (3, 1):
            side = int(round((D / ch) ** 0.5))
            if side * side * ch == D:
                img = side
                break
        if img is None:
            raise ValueError(f"cannot infer image size from D={D}")
    ch = D // (img * img)
    if ch not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {ch} (D={D}, img={img})")
    A = X.reshape(n, img, img, ch)                   # HWC -- the display layout, see module docstring
    if ch == 1:
        A = np.repeat(A, 3, axis=3)
    return np.ascontiguousarray(A.transpose(0, 3, 1, 2)), img


def inception_features(X, img=None, device=None, batch=64, dims=2048):
    """Inception-V3 pool3 features for flat images in [0,1].  -> [n, dims]"""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    A, _ = _to_nchw(X, img)
    model = _inception(device, dims)
    out = []
    with torch.no_grad():
        for i in range(0, len(A), batch):
            t = torch.from_numpy(A[i:i + batch]).to(device)
            f = model(t)[0]                          # [b, dims, 1, 1]
            out.append(f.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _frechet(FS, FR):
    from scipy import linalg
    mu1, mu2 = FS.mean(0), FR.mean(0)
    s1 = np.cov(FS, rowvar=False)
    s2 = np.cov(FR, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(samples, reference, img=None, device=None, features=None, min_n=2048, strict=True):
    """Frechet Inception Distance.  Biased at small n -- see the module docstring.

    `strict=True` REFUSES to return a number below `min_n` samples rather than returning a
    meaningless one, because the failure mode this guards against is a small-n FID being quoted as if
    it meant something.  Pass `strict=False` to get it anyway (the reported value then carries an
    `n_warning`), or use `kid` instead, which is what small n calls for.
    """
    FS, FR = features if features else (inception_features(samples, img, device),
                                        inception_features(reference, img, device))
    n = min(len(FS), len(FR))
    if n < min_n:
        msg = (f"FID with n={n} < {min_n} is dominated by its own bias; use kid() per-composition, "
               f"or pool samples for a dataset-level FID")
        if strict:
            raise ValueError(msg)
        return dict(fid=_frechet(FS, FR), n=n, n_warning=msg)
    return dict(fid=_frechet(FS, FR), n=n)


def kid(samples, reference, img=None, device=None, features=None, n_subsets=100,
        subset_size=None, seed=0, degree=3, gamma=None, coef0=1.0):
    """Kernel Inception Distance: unbiased MMD^2, polynomial kernel (Binkowski et al., ICLR 2018).

    k(x, y) = (gamma * <x, y> + coef0)^degree, with gamma = 1/d by their default.  The unbiased
    estimator drops the diagonal terms, which is exactly what removes the sample-count dependence:

        MMD^2_u = 1/(m(m-1)) sum_{i != j} k(x_i, x_j)
                + 1/(n(n-1)) sum_{i != j} k(y_i, y_j)
                - 2/(mn)     sum_{i, j}   k(x_i, y_j)

    Averaged over `n_subsets` random subsets so the spread is reportable; the returned `std` is the
    spread across subsets, not a confidence interval on the mean.
    """
    FS, FR = features if features else (inception_features(samples, img, device),
                                        inception_features(reference, img, device))
    d = FS.shape[1]
    gamma = 1.0 / d if gamma is None else gamma
    m = min(len(FS), len(FR))
    subset_size = min(subset_size or m, m)
    if subset_size < 2:
        return dict(kid=float("nan"), std=float("nan"), n=m, subset_size=subset_size)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_subsets):
        x = FS[rng.choice(len(FS), subset_size, replace=False)]
        y = FR[rng.choice(len(FR), subset_size, replace=False)]
        kxx = (gamma * (x @ x.T) + coef0) ** degree
        kyy = (gamma * (y @ y.T) + coef0) ** degree
        kxy = (gamma * (x @ y.T) + coef0) ** degree
        s = subset_size
        # the unbiased estimator: sum off-diagonal only
        t = (kxx.sum() - np.trace(kxx)) / (s * (s - 1))
        t += (kyy.sum() - np.trace(kyy)) / (s * (s - 1))
        t -= 2 * kxy.mean()
        vals.append(t)
    return dict(kid=float(np.mean(vals)), std=float(np.std(vals)), n=m, subset_size=subset_size,
                n_subsets=n_subsets)
