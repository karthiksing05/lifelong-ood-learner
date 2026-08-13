"""Shapes3D as a third dataset, satisfying `factored.py`'s contract with no surgery anywhere else.

Shapes3D (Burgess & Kim 2018) is the canonical disentanglement benchmark and it is an **exact
Cartesian product**: 480,000 images at 64x64x3, labelled by six factors of sizes

    floor_hue 10 · wall_hue 10 · object_hue 10 · scale 8 · shape 4 · orientation 15

with every combination present exactly once.  That last property is why it is worth adding here and
also the one design problem it poses.

THE DESIGN PROBLEM, AND THE FIX.  Every experiment in this repository needs a **composition** to have
MANY images -- held-out instances to measure coverage against, a query drawn from the composition and
reference images that are not the query, N instances per concept for the batch-size sweep.  On
ColorMNIST a slot is (digit, fg, bg) and handwriting supplies the within-slot variation.  On Shapes3D
a full 6-tuple has exactly one image, so used naively every "composition" would be a single image and
every coverage metric would be undefined.

So the factors are split.  `comp_factors` are the ones a composition is made OF; the rest are
NUISANCE and supply the within-composition variation, exactly as handwriting does:

    default comp = (floor_hue, object_hue, shape)   -> 10 x 10 x 4 =  400 compositions
    nuisance     = (wall_hue, scale, orientation)   -> 10 x  8 x 15 = 1200 images each

That default is chosen to mirror ColorMNIST's three named factors, so the two datasets are directly
comparable rather than merely both present.  Pass `comp_factors` to use more of them -- with four the
contract is exercised beyond the 3-factor case, which is the point of having it.

The split is not cosmetic: it is what makes "hold out a COMBINATION without starving a VALUE"
meaningful here, which is the distinction the three-tier router exists to make (see
`factored.holdout_mask`).

    from lifelong_ood.shapes3d import load_shapes3d
    d = load_shapes3d(n_per_slot=40)
    d["X"].shape, d["sizes"], d["names"]
"""
import os

import numpy as np

FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
FACTOR_SIZES = [10, 10, 10, 8, 4, 15]
DEFAULT_COMP = ("floor_hue", "object_hue", "shape")

# Looked for in this order; override with `path=` or $SHAPES3D_H5.
_CANDIDATES = [
    os.path.expanduser("~/flash/data/shapes3d/3dshapes.h5"),
    os.path.expanduser("~/data/shapes3d/3dshapes.h5"),
    "/coc/flash10/ksingara3/data/shapes3d/3dshapes.h5",
]


def _find(path=None):
    for p in ([path] if path else []) + [os.environ.get("SHAPES3D_H5")] + _CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "3dshapes.h5 not found. Fetch it once (255 MB):\n"
        "  mkdir -p ~/flash/data/shapes3d && curl -L -o ~/flash/data/shapes3d/3dshapes.h5 \\\n"
        "    https://storage.googleapis.com/3d-shapes/3dshapes.h5")


def load_shapes3d(n_per_slot=40, seed=0, comp_factors=DEFAULT_COMP, path=None, img=64,
                  max_images=None, n_holdout=100, value_stride=1):
    """-> dict(X, factors, names, sizes, slot, seen, img) plus `nuisance` for reference.

    `X` is [N, img*img*3] float32 in [0, 1], flat rows, matching every other loader here.
    `factors` carries ONLY the composition factors, so `sizes` is what the basis is built over and
    `slot` is the mixed-radix id of a composition.  `seen` is all-True; call
    `factored.holdout_mask` to split, which is where the "never starve a value" guarantee lives.

    Images are read by SORTED index in one pass.  h5py fancy-indexing with unsorted indices is
    quadratic-ish on a 480k dataset and turns a 10-second read into minutes.
    """
    import h5py

    p = _find(path)
    # Cache: the h5 read is ~100 s because it pulls scattered rows out of a 480k dataset, and every
    # experiment calling this would pay it again. The key covers everything that changes the result.
    ck = os.path.join(os.path.dirname(p), f"cache_n{n_per_slot}_s{seed}_"
                      f"{'-'.join(comp_factors)}_{max_images}_h{n_holdout}"
                      f"_v{value_stride}.npz")
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        return dict(X=z["X"], factors=z["factors"], names=list(z["names"]),
                    sizes=[int(v) for v in z["sizes"]], slot=z["slot"], seen=z["seen"],
                    img=int(z["img"]), nuisance=z["nuisance"],
                    nuisance_names=list(z["nuisance_names"]))
    ci = [FACTOR_NAMES.index(c) for c in comp_factors]
    ni = [k for k in range(len(FACTOR_NAMES)) if k not in ci]

    with h5py.File(p, "r") as f:
        L = f["labels"][:]                                     # [480000, 6] float64, 23 MB
        # values are the physical quantities, not indices; rank them per factor to get indices
        idx = np.empty(L.shape, dtype=np.int64)
        for k in range(L.shape[1]):
            u = np.unique(L[:, k])
            idx[:, k] = np.searchsorted(u, L[:, k])
        comp = idx[:, ci]
        sizes = [FACTOR_SIZES[k] for k in ci]
        # `value_stride` keeps every k-th VALUE of each composition factor.  This is how the number
        # of compositions is reduced WITHOUT dropping a factor: 400 classes at 40 images each left
        # every class too rare for conditioning to take hold against an easy marginal (the model
        # generated excellent Shapes3D and ignored its label -- embedding pairwise cosine 0.007).
        # Fewer, denser classes is the fix; dropping to two factors would have been the wrong one,
        # because three is what makes this a composition benchmark.
        if value_stride > 1:
            # Only thin factors that HAVE values to spare.  `shape` has 4; halving it to 2 would
            # damage the benchmark to save nothing, since the class count is driven by the two
            # 10-valued hues.
            keepv = [np.arange(0, sz, value_stride) if sz >= 8 else np.arange(sz) for sz in sizes]
            ok = np.ones(len(comp), bool)
            for j, kv in enumerate(keepv):
                ok &= np.isin(comp[:, j], kv)
            # vectorised: keepv[j] is sorted and every surviving value is in it, so searchsorted
            # IS the relabelling.  The list-comprehension version built ~10^5 Python lists and was
            # killed by the OOM reaper before it finished.
            comp = comp[ok]
            for j, kv in enumerate(keepv):
                comp[:, j] = np.searchsorted(kv, comp[:, j])
            idx = idx[ok]
            sizes = [len(kv) for kv in keepv]
        slot = np.zeros(len(comp), np.int64)
        for j, s in enumerate(sizes):
            slot = slot * s + comp[:, j]

        rng = np.random.default_rng(seed)
        keep = []
        for s in np.unique(slot):
            w = np.where(slot == s)[0]
            keep.append(rng.permutation(w)[:n_per_slot])
        keep = np.sort(np.concatenate(keep))
        if max_images is not None and len(keep) > max_images:
            keep = np.sort(rng.permutation(keep)[:max_images])
        X = f["images"][keep]                                  # sorted -> one efficient read

    X = np.asarray(X, np.float32).reshape(len(keep), -1) / 255.0
    # `seen` marks the TRAINING split.  `holdout_mask` picks whole COMBINATIONS and guarantees that
    # no factor VALUE is starved -- a split that removed a value would silently convert this from
    # "compose known parts in a new arrangement" into "invent an unseen part", which is a different
    # experiment (the EXTEND BASIS tier, not ADDRESS).
    from .factored import holdout_mask
    F_keep = comp[keep].astype(int)
    if n_holdout:
        seen_mask, held = holdout_mask(F_keep, [int(v) for v in sizes],
                                       n_holdout=int(n_holdout), seed=seed)
    else:
        seen_mask = np.ones(len(keep), bool)
    out = dict(X=X, factors=F_keep, names=list(comp_factors),
               sizes=[int(s) for s in sizes], slot=slot[keep].astype(int),
               seen=np.asarray(seen_mask, bool), img=img,
               nuisance=idx[keep][:, ni].astype(int),
               nuisance_names=[FACTOR_NAMES[k] for k in ni])
    try:
        np.savez_compressed(ck, **out)
    except Exception:                                          # noqa: BLE001  cache is optional
        pass
    return out
