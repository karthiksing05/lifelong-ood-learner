"""One contract for every factored dataset, so a new one is a loader rather than a refactor.

The repository currently has two incompatible dataset shapes and every experiment hardcodes which it
is using:

    ColorMNIST   dict(X, digit, fg, bg, slot, seen)          -- three NAMED factors
    CelebA       dict(X, cls, attrs, combos, split, seen)    -- fourteen BINARY attributes

`data["digit"]` appears 14 times across the experiments, `data["slot"]` 39 times, and
`idealwords.slot_factors` decodes a slot id by hardcoded arithmetic (`s // 16`, `(s % 16) // 4`,
`s % 4`) that is true of ColorMNIST and nothing else.  Adding Shapes3D (6 factors) or CLEVR would
need a third shape and a third decode, which is how a benchmark suite becomes unmaintainable.

The unifying observation is that all of them are the same object: **a set of images, each labelled by
an integer value for each of F factors, with some combinations held out.** That is all any part of
this repository actually needs -- the factor basis, the router, the three-tier cost model and the
composition metrics are all defined over factor VALUES, never over "digit" or "Blond_Hair".

    X        [N, D]  float32 in [0, 1], flat rows
    factors  [N, F]  int, the value index of each factor
    names    [F]     str, for reporting only
    sizes    [F]     int, number of values per factor
    slot     [N]     int, a composition id (mixed-radix encoding of `factors`)
    seen     [N]     bool, False for held-out combinations
    img      int     side length

This module ADDS that view; it does not remove the old one.  `as_factored` wraps either existing
loader, so current experiments keep working untouched and new ones can be written against a contract
that Shapes3D and CLEVR satisfy without further surgery.

    from lifelong_ood.data.factored import as_factored, load_factored
    d = load_factored("colormnist", n_per_slot=50)
    d["factors"], d["sizes"], d["names"]
"""
import numpy as np


def encode_slot(factors, sizes):
    """Mixed-radix composition id.  -> [N] int.

    Replaces the per-dataset arithmetic: with sizes [10, 4, 4] this reproduces ColorMNIST's own
    `d*16 + f*4 + b`, and it is equally correct for six factors of unequal size.
    """
    F = np.asarray(factors, int)
    out = np.zeros(len(F), dtype=np.int64)
    for f, s in enumerate(sizes):
        out = out * int(s) + F[:, f]
    return out


def decode_slot(slots, sizes):
    """Inverse of `encode_slot`.  -> [N, F] int."""
    s = np.asarray(slots, np.int64).copy()
    cols = []
    for size in reversed([int(x) for x in sizes]):
        cols.append(s % size)
        s //= size
    return np.stack(cols[::-1], 1)


def as_factored(data, names=None):
    """Add the unified view to a dict from `load_colormnist` or `load_celeba`.  -> the same dict.

    Detects which shape it was handed rather than being told, because the callers that would have to
    tell it are the ones this exists to stop editing.
    """
    if "factors" in data:
        return data
    if all(k in data for k in ("digit", "fg", "bg")):
        cols = [np.asarray(data["digit"], int), np.asarray(data["fg"], int),
                np.asarray(data["bg"], int)]
        nm = names or ["digit", "fg", "bg"]
    elif "attrs" in data:
        # binary attributes ARE factors of size two; nothing else about them is special
        A = np.asarray(data["attrs"])
        cols = [A[:, k].astype(int) for k in range(A.shape[1])]
        nm = names or [f"attr{k}" for k in range(A.shape[1])]
    else:
        raise ValueError("unrecognised dataset shape: need digit/fg/bg or attrs")
    F = np.stack(cols, 1)
    sizes = [int(c.max()) + 1 for c in cols]
    data["factors"], data["names"], data["sizes"] = F, list(nm), sizes
    if "slot" not in data:
        data["slot"] = encode_slot(F, sizes)
    if "img" not in data:
        d = np.asarray(data["X"]).shape[1]
        data["img"] = int(round((d / 3.0) ** 0.5))
    return data


def holdout_mask(factors, sizes, n_holdout, seed=0, min_per_value=1):
    """Choose held-out COMBINATIONS such that every factor VALUE still appears.  -> [N] bool `seen`.

    The distinction the whole project rests on: a held-out *combination* is a new arrangement of
    known parts, while a value that never appears at all is new vocabulary.  A random split does not
    respect it -- drop every combination containing `bg=3` by chance and `bg=3` silently becomes
    vocabulary, which is a different experiment. This guarantees each value survives at least
    `min_per_value` times among the seen combinations, so held-out really means "new arrangement".
    """
    F = np.asarray(factors, int)
    slots = encode_slot(F, sizes)
    uniq = np.unique(slots)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    held, kept = [], {f: {} for f in range(F.shape[1])}
    dec = decode_slot(uniq, sizes)
    counts = {f: {} for f in range(F.shape[1])}
    for row, s in zip(dec, uniq):
        for f, v in enumerate(row):
            counts[f][int(v)] = counts[f].get(int(v), 0) + 1
    for s in order:
        if len(held) >= n_holdout:
            break
        row = decode_slot(np.array([s]), sizes)[0]
        # would holding this out starve any value?
        if all(counts[f][int(v)] - 1 >= min_per_value for f, v in enumerate(row)):
            held.append(int(s))
            for f, v in enumerate(row):
                counts[f][int(v)] -= 1
    held = set(held)
    return ~np.isin(slots, list(held)), sorted(held)


def load_factored(name, **kw):
    """-> a dict satisfying the contract above, for any registered dataset.

    Registering a new one is a loader that returns `X` plus per-image factor values; `as_factored`
    and `holdout_mask` supply everything else. Shapes3D and CLEVR are the intended next entries:
    Shapes3D ships exactly this labelling (floor hue, wall hue, object hue, scale, shape,
    orientation -- sizes [10, 10, 10, 8, 4, 15]), and CLEVR scenes reduce to it once objects are
    described by (shape, colour, size, material).
    """
    name = name.lower()
    if name in ("colormnist", "color-mnist", "cmnist"):
        from .datasets import load_colormnist
        return as_factored(load_colormnist(**kw))
    if name == "celeba":
        from .celeba import ATTRS, load_celeba
        return as_factored(load_celeba(**kw), names=list(ATTRS))
    if name in ("shapes3d", "3dshapes", "shapes-3d"):
        # Already emits the contract directly -- it has to, because its six factors are split into
        # composition and nuisance and `as_factored` cannot know which is which.
        from .shapes3d import load_shapes3d
        return load_shapes3d(**kw)
    raise ValueError(f"unknown dataset {name!r}; register a loader in factored.load_factored")
