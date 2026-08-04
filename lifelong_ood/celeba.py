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
import io
import os
import zipfile

import numpy as np

IMG_CELEBA = 64
ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Wavy_Hair", "No_Beard",
         "Smiling", "Big_Nose", "Big_Lips", "Oval_Face", "Male", "Sideburns",
         "Wearing_Necklace", "Bald"]

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


def class_name(combo):
    on = [a for a, v in zip(ATTRS, combo) if v]
    return ", ".join(on) if on else "(none)"


def rgb_img(vec, img=IMG_CELEBA):
    return np.clip(np.asarray(vec, np.float32).reshape(img, img, 3), 0, 1)
