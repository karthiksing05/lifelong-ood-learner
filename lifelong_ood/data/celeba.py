"""CelebA as a compositional benchmark, to the poe-diffusion Appendix B / Table 4 specification.

  "A compositional class is a unique combination of 14 binary CelebA attributes (Black/Blond/Brown/
   Gray hair, Wavy hair, No_Beard, Smiling, Big_Nose, Big_Lips, Oval_Face, Male, Sideburns,
   Wearing_Necklace, Bald).  The 74 realized combinations used in this experiment are partitioned
   into a seen tier used for backbone training and a holdout tier accessible only at test time."
   -- 51 seen classes / 23 held out, images at 128x128.

WHY THIS BENCHMARK MATTERS HERE, beyond being the paper's second dataset.  Every conclusion in this
repository so far rests on ColorMNIST, whose factors are *spatially separable* -- the stroke carries
the digit and its colour, the background carries nothing else.  That separability is load-bearing in
places: it is why the per-dimension PoE can assemble an unseen pairing from base concepts at all, and
why E4's closure hypothesis could not be tested (PLAN 3.2).  CelebA attributes are NOT separable --
"smiling" and "big lips" occupy the same pixels -- so it tests the parts of the method that
ColorMNIST structurally cannot.

TWO DEVIATIONS, both forced and both flagged:
  * Resolution.  The paper trains at 128x128 on 4 GPUs for ~50 hours.  `IMG_CELEBA` defaults to 64,
    which keeps the backbone trainable in hours on one GPU.  Absolute numbers are therefore not
    comparable to the paper's; within-table comparisons are, which is all this repo ever claims.
  * Class construction.  The paper does not publish which 74 combinations it realized or how it split
    them, so `build_classes` derives them by the stated rule -- take the combinations that actually
    occur with enough support, then hold out those differing from a seen class on a single attribute,
    which is the paper's own stated criterion ("chosen so that their attribute combination differs
    from at least one seen class on a single attribute").
"""
import hashlib
import io
import os
import zipfile

import numpy as np

IMG_CELEBA = 64
ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Wavy_Hair", "No_Beard",
         "Smiling", "Big_Nose", "Big_Lips", "Oval_Face", "Male", "Sideburns",
         "Wearing_Necklace", "Bald"]


def set_attrs(names):
    """Replace the conditioning attribute set, IN PLACE.

    Every module does `from .celeba import ATTRS` and therefore holds a reference to this exact list
    object, so mutating it in place is what makes a change visible everywhere rather than only in
    this module.

    The default 14 were inherited, never measured, and `tests/diag_celeba_attrs.py` shows only two of
    them survive scrutiny: `Bald` has base rate 0.015 and probe TPR **0.00**, `Sideburns` 0.00,
    `Wavy_Hair` 0.10, `Oval_Face` 0.16, and `Big_Lips` answers "yes" to 98% of images.  An attribute
    the backbone cannot learn still multiplies the class count -- diluting every class's training
    data -- and an attribute the probe cannot read caps every score that includes it, which is why
    real held-out images score mean-attribute 0.827 rather than ~1.0.
    """
    ATTRS[:] = list(names)
    return ATTRS


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.environ.get("LOL_CELEBA_DIR", ""),
               os.path.join(_HERE, "data", "celeba"),
               os.path.join(os.path.dirname(_HERE), "data", "celeba"),
               "/coc/flash10/ksingara3/data/celeba",
               os.path.expanduser("~/flash/data/celeba")]


def find_celeba():
    """Locate a CelebA directory: either the official layout or HuggingFace parquet shards."""
    for d in _CANDIDATES:
        if not d:
            continue
        if os.path.exists(os.path.join(d, "list_attr_celeba.txt")):
            return d
        import glob
        if glob.glob(os.path.join(d, "*.parquet")):
            return d
    raise FileNotFoundError(
        "CelebA not found.  Either set $LOL_CELEBA_DIR to a directory with list_attr_celeba.txt and "
        "the aligned images, or drop HuggingFace parquet shards there:\n"
        "  curl -fsSLO https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/"
        "img_align%2Bidentity%2Battr/train-00000-of-00019.parquet")


def _parquet_shards(root):
    import glob
    return sorted(glob.glob(os.path.join(root, "*.parquet")))


def load_from_parquet(root=None, img=IMG_CELEBA, max_rows=None):
    """Read HuggingFace CelebA parquet shards -> (images [N, img*img*3], attrs [N, 14]).

    The shards carry the aligned image bytes alongside all 40 boolean attributes, so this needs no
    separate attribute file and no 1.4GB zip extraction.  Images are centre-cropped before resizing,
    the standard CelebA preprocessing.
    """
    import pyarrow.parquet as pq
    from PIL import Image
    root = root or find_celeba()
    shards = _parquet_shards(root)
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {root}")
    Xs, As, n = [], [], 0
    for sp in shards:
        f = pq.ParquetFile(sp)
        for batch in f.iter_batches(batch_size=512, columns=["image"] + ATTRS):
            d = batch.to_pydict()
            for k in range(len(d["image"])):
                im = Image.open(io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
                w, h = im.size
                top = max(0, (h - w) // 2)
                im = im.crop((0, top, w, top + w)).resize((img, img), Image.BILINEAR)
                Xs.append((np.asarray(im, np.float32) / 255.0).ravel())
                As.append([1 if d[a][k] else 0 for a in ATTRS])
                n += 1
                if max_rows and n >= max_rows:
                    return np.stack(Xs), np.array(As, np.int8)
    return np.stack(Xs), np.array(As, np.int8)


def load_attributes(root=None):
    """-> (filenames [N], attribute matrix [N, 14] in {0,1}) for the 14 attributes above."""
    root = root or find_celeba()
    path = os.path.join(root, "list_attr_celeba.txt")
    with open(path) as fh:
        lines = fh.read().strip().split("\n")
    header = lines[1].split()
    idx = [header.index(a) for a in ATTRS]
    names, rows = [], []
    for ln in lines[2:]:
        parts = ln.split()
        names.append(parts[0])
        v = np.array(parts[1:], dtype=np.int8)
        rows.append((v[idx] > 0).astype(np.int8))
    return np.array(names), np.stack(rows)


def build_classes(A, min_count=120, n_seen=51, n_holdout=23, seed=0):
    """Attribute rows -> compositional classes, split into seen and held-out tiers.

    Follows the paper's stated construction: keep combinations with enough support to train on, then
    choose the holdout tier from combinations that differ from a seen one on a SINGLE attribute, so
    each held-out class is reachable from a near neighbour and the test is compositional rather than
    a novel-attribute test.
    """
    keys, inv, counts = np.unique(A, axis=0, return_inverse=True, return_counts=True)
    keep = np.where(counts >= min_count)[0]
    keep = keep[np.argsort(-counts[keep])][:n_seen + n_holdout]
    combos = keys[keep]
    rng = np.random.default_rng(seed)

    order = rng.permutation(len(combos))
    seen_ids, hold_ids = list(order[:n_seen]), []
    for i in order[n_seen:]:
        d = (combos[i][None] != combos[seen_ids]).sum(1)
        (hold_ids if d.min() == 1 else seen_ids).append(i)
    hold_ids = hold_ids[:n_holdout]

    lookup = {tuple(c): j for j, c in enumerate(combos)}
    cls = np.full(len(A), -1)
    for i, row in enumerate(A):
        j = lookup.get(tuple(row))
        if j is not None:
            cls[i] = j
    seen_mask = np.isin(cls, seen_ids)
    hold_mask = np.isin(cls, hold_ids)
    return dict(combos=combos, cls=cls, seen_ids=np.array(seen_ids), hold_ids=np.array(hold_ids),
                seen_mask=seen_mask, hold_mask=hold_mask,
                counts=counts[keep])


def load_images(names, root=None, img=IMG_CELEBA, cache=True):
    """Load and centre-crop the named aligned images -> [N, img*img*3] float32 in [0,1].

    Reads from `img_align_celeba/` if extracted, otherwise straight out of the zip so the 1.4GB
    archive never has to be expanded.  The 178x218 aligned images are centre-cropped to 178x178
    before resizing, which is the standard CelebA preprocessing.
    """
    from PIL import Image
    root = root or find_celeba()
    d = os.path.join(root, "img_align_celeba")
    zf = None
    if not os.path.isdir(d):
        zp = os.path.join(root, "img_align_celeba.zip")
        if not os.path.exists(zp):
            raise FileNotFoundError(f"neither {d} nor {zp}")
        zf = zipfile.ZipFile(zp)
        inner = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".jpg")}
    out = np.empty((len(names), img * img * 3), np.float32)
    for i, n in enumerate(names):
        if zf is None:
            im = Image.open(os.path.join(d, n))
        else:
            im = Image.open(zf.open(inner[n]))
        im = im.convert("RGB").crop((0, 20, 178, 198)).resize((img, img), Image.BILINEAR)
        out[i] = (np.asarray(im, np.float32) / 255.0).ravel()
    return out


def load_celeba(n_per_class=200, img=IMG_CELEBA, seed=0, root=None, min_count=120,
                max_rows=25000):
    """-> dict(X, cls, attrs, seen, combos, seen_ids, hold_ids) mirroring `load_colormnist`.

    Downstream code treats a compositional class exactly as it treats a ColorMNIST slot, so the
    experiments run unchanged; only the factor structure differs, which is the point -- CelebA's
    attributes overlap spatially where ColorMNIST's do not.
    """
    root = root or find_celeba()
    if _parquet_shards(root):
        X_all, A = load_from_parquet(root, img=img, max_rows=max_rows)
        names = None
    else:
        names, A = load_attributes(root)
        X_all = None
    info = build_classes(A, min_count=min_count, seed=seed)
    rng = np.random.default_rng(seed)
    pick = []
    for j in np.concatenate([info["seen_ids"], info["hold_ids"]]):
        idx = np.where(info["cls"] == j)[0]
        rng.shuffle(idx)
        pick.append(idx[:n_per_class])
    pick = np.concatenate(pick)
    X = X_all[pick] if X_all is not None else load_images(names[pick], root=root, img=img)
    return dict(X=X, cls=info["cls"][pick], attrs=A[pick],
                seen=np.isin(info["cls"][pick], info["seen_ids"]),
                combos=info["combos"], seen_ids=info["seen_ids"], hold_ids=info["hold_ids"],
                img=img)


def build_comp_split(A, min_count=60, n_holdout=24, holdout_attr=None, holdout_pair=None, seed=0):
    """Compositional split for the FACTORED backbone: hold out whole attribute COMBINATIONS.

    Differs from `build_classes` in what it is allowed to train on.  `build_classes` exists for a
    per-class embedding table, so it keeps only combinations with enough support to estimate a row
    (>=120 images) and discards the other ~85% of the data.  A factored conditioning head has no
    per-combination parameters -- it learns one vector per (attribute, value) -- so it can train on
    EVERY image whose combination is not held out, which is what makes CelebA trainable at all
    (RESULTS 5o: per-class conditioning reached 34% of ceiling and 2.3x data did not move it).

    Held-out combinations are chosen to differ from a retained one on a SINGLE attribute, which is
    the paper's own criterion (App. B) and makes the test compositional rather than a novel-attribute
    test.  `holdout_attr` runs the OTHER regime -- an attribute whose ON value is withheld
    ENTIRELY, so the backbone has never seen that vocabulary item in any combination.  That is the
    CelebA analogue of `n1_train --holdout-bg` and is what separates "new combination" (free) from
    "new vocabulary" (needs real data) on spatially overlapping attributes.
    """
    rng = np.random.default_rng(seed)
    keys, inv, counts = np.unique(A, axis=0, return_inverse=True, return_counts=True)
    cand = np.where(counts >= min_count)[0]
    if len(cand) < n_holdout + 2:
        raise ValueError(f"only {len(cand)} combos with >={min_count} support")

    if holdout_pair is not None:
        # THE STRONG COMPOSITIONAL SPLIT, and the direct analogue of ColorMNIST's held-out (fg, bg)
        # pairs: withhold every combination in which two attributes are ON TOGETHER.  Each attribute
        # is still seen thousands of times on its own, so only their CO-OCCURRENCE is novel -- which
        # is what "compositional generalisation" is supposed to mean, and is a far stronger demand
        # than holding out 24 combinations out of ~1200 individually.
        #
        # Choose a pair that occupies the SAME PIXELS (Wavy_Hair x Blond_Hair is shape and colour of
        # the same hair) and the split also stops being separable, which is the whole reason CelebA
        # is in this repository.
        ka, kb = [ATTRS.index(a) if isinstance(a, str) else int(a) for a in holdout_pair]
        both = (keys[:, ka] == 1) & (keys[:, kb] == 1)
        hold_ids = np.where(both & (counts >= 25))[0]
        train_mask = ~((A[:, ka] == 1) & (A[:, kb] == 1))
        return dict(all_combos=keys, counts=counts, combo_of=inv,
                    hold_ids=np.array(sorted(hold_ids), dtype=int),
                    hold_combos=keys[sorted(hold_ids)], train_mask=train_mask,
                    holdout_attr=None, holdout_attr_idx=None,
                    holdout_pair=(ATTRS[ka], ATTRS[kb]), holdout_pair_idx=(ka, kb))

    hold_ids = []
    for i in rng.permutation(cand):
        if len(hold_ids) >= n_holdout:
            break
        rest = np.setdiff1d(cand, np.array(hold_ids + [i], dtype=int))
        if len(rest) and (keys[i][None] != keys[rest]).sum(1).min() == 1:
            hold_ids.append(int(i))
    hold_ids = np.array(sorted(hold_ids), dtype=int)

    train_mask = ~np.isin(inv, hold_ids)
    attr_idx = None
    if holdout_attr is not None:
        attr_idx = ATTRS.index(holdout_attr) if isinstance(holdout_attr, str) else int(holdout_attr)
        train_mask = train_mask & (A[:, attr_idx] == 0)

    return dict(all_combos=keys, counts=counts, combo_of=inv, hold_ids=hold_ids,
                hold_combos=keys[hold_ids], train_mask=train_mask,
                holdout_attr=(ATTRS[attr_idx] if attr_idx is not None else None),
                holdout_attr_idx=attr_idx)


def nearest_seen(combo, split):
    """-> (index into all_combos, differing attribute indices) of the closest TRAINED combination.

    Held-out combinations differ from this neighbour on exactly one attribute by construction, so
    the differing index is the attribute a compositional generator has to get right *while holding
    the other thirteen* -- a far sharper test than a 14-way conjunction whose ceiling is 0.25.
    """
    keys, counts = split["all_combos"], split["counts"]
    ok = np.ones(len(keys), bool)
    ok[split["hold_ids"]] = False
    if split.get("holdout_attr_idx") is not None:
        ok &= keys[:, split["holdout_attr_idx"]] == 0
    if split.get("holdout_pair_idx") is not None:
        ka, kb = split["holdout_pair_idx"]
        ok &= ~((keys[:, ka] == 1) & (keys[:, kb] == 1))
    ok &= counts >= 25
    d = (np.asarray(combo)[None] != keys).sum(1)
    d = np.where(ok, d, 10 ** 6)
    j = int(np.argmin(d))
    return j, np.where(np.asarray(combo) != keys[j])[0]


def load_celeba_comp(img=IMG_CELEBA, root=None, max_rows=None, min_count=60, n_holdout=24,
                     holdout_attr=None, holdout_pair=None, seed=0, cache_dir=None,
                     n_test_per_combo=64, verify_hold_ids=None):
    """Full compositional CelebA: every usable image, split by held-out COMBINATION.

    Decoding ~80k JPEGs takes minutes, so the decoded array is cached to `cache_dir` keyed by
    (img, max_rows) and reused by every experiment.  Images are kept as uint8 -- at 64x64x3 the
    float32 copy of 80k images is 3.8 GB and the uint8 one is 0.95 GB.
    """
    root = root or find_celeba()
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # The attribute SET must be in the key.  `A`'s columns are exactly ATTRS, so a run with a
        # different attribute list that reads a cache written under the old one gets a matrix whose
        # columns mean something else entirely -- and it fails silently, because the shape is only
        # wrong in a dimension nothing checks.  Measured: a 4-attribute run reported "1226 distinct
        # combos", which is impossible for 4 binary attributes and is exactly the 14-attribute count.
        key = hashlib.sha1(",".join(ATTRS).encode()).hexdigest()[:8]
        cache = os.path.join(cache_dir, f"celeba_raw_img{img}_n{max_rows or 0}_a{key}.npz")
    if cache and os.path.exists(cache):
        z = np.load(cache)
        X8, A = z["X"], z["A"]
    else:
        Xf, A = load_from_parquet(root, img=img, max_rows=max_rows)
        X8 = np.clip(Xf * 255.0, 0, 255).astype(np.uint8)
        del Xf
        if cache:
            # write-then-rename: several experiments start at once and a reader must never see a
            # half-written array.  Uncompressed on purpose -- deflating ~1.2 GB of uint8 costs
            # minutes per run and buys nothing on a filesystem with terabytes free.
            tmp = f"{cache}.tmp{os.getpid()}.npz"
            np.savez(tmp, X=X8, A=A)
            os.replace(tmp, cache)
    split = build_comp_split(A, min_count=min_count, n_holdout=n_holdout,
                             holdout_attr=holdout_attr, holdout_pair=holdout_pair, seed=seed)
    if verify_hold_ids is not None:
        # An evaluation that recomputes the split with any argument different from training would
        # silently score against a DIFFERENT set of held-out combinations -- images the backbone had
        # actually trained on -- and every CelebA number would be quietly wrong with nothing raised.
        got, want = np.asarray(split["hold_ids"]), np.asarray(verify_hold_ids)
        if got.shape != want.shape or not np.array_equal(got, want):
            raise ValueError(
                f"split mismatch: recomputed hold_ids {got[:6]}... do not match the checkpoint's "
                f"{want[:6]}...  min_count/n_holdout/holdout_attr/seed/max_rows must match training")
    rng = np.random.default_rng(seed)
    tests = {}
    if holdout_attr is None:
        targets = list(split["hold_ids"])
        for j in targets:
            idx = np.where(split["combo_of"] == j)[0]
            rng.shuffle(idx)
            tests[int(j)] = idx[:n_test_per_combo]
    else:
        k = split["holdout_attr_idx"]
        idx = np.where(A[:, k] == 1)[0]
        rng.shuffle(idx)
        tests["attr"] = idx[:n_test_per_combo * 8]
    return dict(X8=X8, attrs=A, img=img, split=split, tests=tests,
                train_idx=np.where(split["train_mask"])[0])


def as_float(X8):
    """uint8 image store -> [0,1] float32, the space every probe and metric in this repo uses."""
    return np.asarray(X8, np.float32) / 255.0


def class_name(combo):
    on = [a for a, v in zip(ATTRS, combo) if v]
    return ", ".join(on) if on else "(none)"


def rgb_img(vec, img=IMG_CELEBA):
    return np.clip(np.asarray(vec, np.float32).reshape(img, img, 3), 0, 1)
