"""Multi-object scenes: the formulation `docs/CLEVR_VERDICT.md` says is the next one, built where
it can be built exactly.

CLEVR is the natural multi-object benchmark and every distributed set labels ONE object in a scene
containing several, so "does this sample have the right composition" has no well-defined answer.
Rather than filter a 50-72 %-reliable detector over someone else's renders, the same question is
asked on a substrate where the scene label is exact BY CONSTRUCTION: two MNIST digits composited
onto one canvas.  The label is the unordered SET of the digits present.

**The question this exists to answer**, quoted from CLEVR_VERDICT:

    does a scene of *k* known objects cost *k* addresses, or does the scene itself become a
    composition over addresses?

That is the storage question a VLA actually faces -- a manipulation scene holds several objects and
a policy must condition on the relevant one.

## Why this is not the factored contract

`data/factored.py` encodes a slot as a PRODUCT of factor values, one value per factor, and
`holdout_mask` guarantees every VALUE survives.  A scene is an unordered SET drawn from one
vocabulary: there is no "factor 0" and "factor 1", because {3, 7} and {7, 3} are the same scene.
Forcing it into the product contract would make position a factor and quietly turn a set problem
back into a tuple problem -- the exact confusion CLEVR_VERDICT warns about.  So this module keeps
its own encoding and its own holdout.

## Deliberate design choices

* **Identity is SHAPE, colour is nuisance.**  Each digit is drawn in a colour sampled per object per
  image.  If colour tracked identity, a scene could be read off a colour histogram and "composition"
  would be measurable without any object ever being recognised.
* **Side is randomised.**  The lower digit is not always on the left, so the class really is the
  unordered set and a probe must be a DETECTOR ("is a 3 present?") rather than a whole-image
  classifier -- again as CLEVR_VERDICT requires.
* **Distinct digits only.**  {3,3} is excluded, so a scene is a 2-subset and the vocabulary is
  C(10,2) = 45 scenes over 10 objects.
"""
import numpy as np

from .datasets import BG_COLORS, BG_NAMES, FG_COLORS, FG_NAMES, load_mnist

IMG = 32
HALF = 16
N_OBJ = 10                      # digits 0..9 are the object vocabulary
MNIST_PER_CLASS = 5400          # the rarest MNIST digit (5) has 5421 training images


def scene_pairs():
    """The 45 unordered pairs, in a fixed canonical order.  Index into this IS the scene id."""
    return [(a, b) for a in range(N_OBJ) for b in range(a + 1, N_OBJ)]


def scene_id(a, b):
    a, b = (int(a), int(b)) if a < b else (int(b), int(a))
    return scene_pairs().index((a, b))


def _paste(canvas, tile, r0, c0):
    h, w = tile.shape
    canvas[r0:r0 + h, c0:c0 + w] = np.maximum(canvas[r0:r0 + h, c0:c0 + w], tile)


def _downscale(g):
    """32x32 -> 16x16 by 2x2 max-pool.  Max, not mean: mean thins the strokes until a 16 px digit is
    a grey smudge, and the detector probe then measures blur rather than identity."""
    return g.reshape(HALF, 2, HALF, 2).max(axis=(1, 3))


def holdout_scenes(n_holdout=9, seed=0, min_per_obj=3):
    """Held-out SCENES such that every OBJECT still appears in at least `min_per_obj` seen scenes.

    The set analogue of `factored.holdout_mask`, and it carries the same load: a held-out scene must
    be a new ARRANGEMENT of known objects.  If holding scenes out starved an object down to zero, it
    would become new VOCABULARY and the experiment would silently change from an ADDRESS test into
    an EXTEND test -- the distinction the whole project rests on.
    """
    pairs = scene_pairs()
    rng = np.random.default_rng(seed)
    counts = np.full(N_OBJ, N_OBJ - 1)          # each object starts in 9 pairs
    held = []
    for s in rng.permutation(len(pairs)):
        if len(held) >= n_holdout:
            break
        a, b = pairs[int(s)]
        if counts[a] - 1 >= min_per_obj and counts[b] - 1 >= min_per_obj:
            held.append(int(s))
            counts[a] -= 1
            counts[b] -= 1
    return sorted(held)


def load_multiobject(n_per_scene=100, seed=0, n_holdout=9, bg="black", jitter=2):
    """-> dict(X, objs, scene, seen, ...).

    `X` is [N, 3072] float32 in [0,1], HWC-flat exactly like `load_colormnist`.
    `objs` is [N, 2] int, the two digits present, ALWAYS sorted ascending -- the set, not the layout.
    `scene` is [N] int in 0..44 indexing `scene_pairs()`.
    `seen` is False for images whose scene is held out.

    Every image draws two DISTINCT MNIST source digits, so no source image appears twice in one
    scene and nothing can be solved by spotting a duplicated bitmap.
    """
    pairs = scene_pairs()
    held = set(holdout_scenes(n_holdout, seed))
    rng = np.random.default_rng(seed + 991)
    # Enough sources that no digit is reused more often than necessary -- but MNIST holds only
    # ~5.4-6.7k images of each digit, and `load_mnist` raises rather than truncating when asked for
    # more (`Cannot take a larger sample than population`).  Above the cap, sources are reused: the
    # per-digit pool reshuffles when exhausted, so reuse is spread evenly instead of the first
    # scenes taking every distinct image and the last ones repeating a handful.
    need = min(MNIST_PER_CLASS, int(np.ceil(n_per_scene * len(pairs) * 2 / N_OBJ)) + 8)
    G, L = load_mnist(n_per_class=need, seed=seed)
    pool = {d: rng.permutation(np.where(L == d)[0]) for d in range(N_OBJ)}
    cur = {d: 0 for d in range(N_OBJ)}
    bgc = np.asarray(BG_COLORS[bg], np.float32) if bg in BG_COLORS else np.zeros(3, np.float32)

    X = np.zeros((len(pairs) * n_per_scene, IMG, IMG, 3), np.float32)
    objs = np.zeros((len(X), 2), int)
    scene = np.zeros(len(X), int)
    k = 0
    for s, (a, b) in enumerate(pairs):
        for _ in range(n_per_scene):
            canvas = np.zeros((IMG, IMG), np.float32)
            col = np.zeros((IMG, IMG, 3), np.float32)
            order = [a, b] if rng.random() < 0.5 else [b, a]        # side is randomised
            for side, d in enumerate(order):
                if cur[d] >= len(pool[d]):                          # reshuffle rather than run out
                    pool[d] = rng.permutation(pool[d])
                    cur[d] = 0
                tile = _downscale(G[pool[d][cur[d]]])
                cur[d] += 1
                r0 = int(np.clip(HALF // 2 + rng.integers(-jitter, jitter + 1), 0, IMG - HALF))
                c0 = int(np.clip(side * HALF + rng.integers(-jitter, jitter + 1), 0, IMG - HALF))
                m = np.zeros((IMG, IMG), np.float32)
                _paste(m, tile, r0, c0)
                fg = np.asarray(FG_COLORS[FG_NAMES[rng.integers(len(FG_NAMES))]], np.float32)
                col += m[..., None] * fg[None, None, :]             # colour is NUISANCE
                canvas = np.maximum(canvas, m)
            img = bgc[None, None, :] * (1 - canvas[..., None]) + np.clip(col, 0, 1)
            X[k] = np.clip(img, 0, 1)
            objs[k] = (a, b)
            scene[k] = s
            k += 1
    seen = np.array([s not in held for s in scene])
    return dict(X=np.ascontiguousarray(X.reshape(len(X), -1)), objs=objs, scene=scene,
                seen=seen, held=sorted(held), pairs=pairs, n_obj=N_OBJ, img=IMG,
                names=["obj_a", "obj_b"])


def object_membership(scene_ids):
    """[n_scenes, N_OBJ] 0/1 -- the design matrix a per-object basis is fitted against.

    This is what makes the ADDRESS question askable: if the scene embedding table is (approximately)
    a sum of per-object directions, then a scene is a COMPOSITION over addresses and an unseen pair
    of known objects should cost nothing to store.
    """
    pairs = scene_pairs()
    M = np.zeros((len(scene_ids), N_OBJ), np.float32)
    for i, s in enumerate(np.asarray(scene_ids, int)):
        a, b = pairs[int(s)]
        M[i, a] = 1.0
        M[i, b] = 1.0
    return M
