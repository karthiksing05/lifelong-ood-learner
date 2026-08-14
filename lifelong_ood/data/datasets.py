"""ColorMNIST: the compositional benchmark (digit x fg-colour x bg-colour).

Built to the specification in Appendix B of the poe-diffusion paper (`paper/poe-diffusion.pdf`):
32x32 RGB, three primitive factors, 10 x 4 x 4 = 160 compositional slots, of which 120 are seen during
training and 40 are held out as OOD compositions.  The held-out 40 are FOUR (fg, bg) PAIRS crossed with
all ten digits, so every primitive VALUE is seen during training and only the COMBINATION is novel.

The palette and split are IDENTICAL to `cobweb-poincare/cobweb_poincare/datasets.py`, deliberately:
numbers produced by this repository and that one are comparable slot-for-slot.  The palette is that
repository's reconstruction of the paper's named colours ("yellow, green, cyan, pink" on "deep red,
navy, dark purple, dark brown"), chosen so no digit colour collapses against any background.

MNIST is read directly from idx files with numpy -- no torch/torchvision anywhere in this repository
(the login nodes of the cluster this was developed on kill torch processes).  Search order for the raw
files: $LOL_MNIST_DIR, ./data/MNIST/raw, ../cobweb-poincare/data/MNIST/raw, then download.
"""
import gzip
import os
import struct
import urllib.request

import numpy as np

IMG = 32
D = IMG * IMG * 3

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MNIST_CANDIDATES = [
    os.environ.get("LOL_MNIST_DIR", ""),
    os.path.join(_HERE, "data", "MNIST", "raw"),
    os.path.join(os.path.dirname(_HERE), "cobweb-poincare", "data", "MNIST", "raw"),
]
_MNIST_URLS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
               "https://storage.googleapis.com/cvdf-datasets/mnist/"]


def _read_idx(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), np.uint8).reshape(shape)


def _find_mnist():
    fname = "train-images-idx3-ubyte"
    for d in _MNIST_CANDIDATES:
        if d and (os.path.exists(os.path.join(d, fname)) or os.path.exists(os.path.join(d, fname + ".gz"))):
            return d
    dest = os.path.join(_HERE, "data", "MNIST", "raw")
    os.makedirs(dest, exist_ok=True)
    for base in _MNIST_URLS:
        try:
            for f in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
                urllib.request.urlretrieve(base + f, os.path.join(dest, f))
            return dest
        except Exception:
            continue
    raise FileNotFoundError("MNIST raw idx files not found and download failed; set $LOL_MNIST_DIR")


def load_mnist(n_per_class=1600, seed=0):
    """Balanced MNIST sample, padded 28->32, as ([N, 32, 32] float32 in [0,1], labels)."""
    d = _find_mnist()

    def pick(stem):
        for suf in ("", ".gz"):
            p = os.path.join(d, stem + suf)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(p)

    G = _read_idx(pick("train-images-idx3-ubyte")).astype(np.float32) / 255.0
    L = _read_idx(pick("train-labels-idx1-ubyte")).astype(int)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(L == c)[0], n_per_class, replace=False) for c in range(10)])
    rng.shuffle(idx)
    X = np.zeros((len(idx), IMG, IMG), np.float32)
    X[:, 2:30, 2:30] = G[idx]
    return X, L[idx]


# ── palette + split, byte-identical to cobweb-poincare ─────────────────────────
FG_COLORS = {"yellow": (0.90, 0.85, 0.20), "green":       (0.30, 0.80, 0.35),
             "cyan":   (0.25, 0.80, 0.85), "pink":        (0.95, 0.45, 0.70)}
BG_COLORS = {"deep_red":    (0.35, 0.06, 0.08), "navy":       (0.05, 0.09, 0.30),
             "dark_purple": (0.20, 0.06, 0.28), "dark_brown": (0.25, 0.15, 0.06)}
FG_NAMES, BG_NAMES = list(FG_COLORS), list(BG_COLORS)
OOD_PAIRS = (("cyan", "deep_red"), ("green", "navy"),
             ("pink", "dark_purple"), ("yellow", "dark_brown"))


def colorize(G, fg, bg):
    """Grayscale masks [n, 32, 32] in [0,1] -> flat RGB [n, 3072] by alpha-compositing.

    Soft edges matter: a hard mask would make every image in a slot share an identical background
    region, collapsing the diagonal-Gaussian variance of a colour concept onto the prior floor.
    """
    m = np.asarray(G, np.float32)[..., None]
    return np.ascontiguousarray(
        (np.asarray(bg, np.float32) * (1 - m) + np.asarray(fg, np.float32) * m
         ).reshape(len(m), -1).astype(np.float32))


def slot_id(d, f, b):
    return int(d) * 16 + int(f) * 4 + int(b)


def slot_name(d, f, b):
    return f"digit {d}, {FG_NAMES[f]} on {BG_NAMES[b]}"


def load_colormnist(n_per_slot=100, seed=0):
    """-> dict(X, digit, fg, bg, slot, seen).

    `X` is [N, 3072] float32 in [0,1]; `seen` is True for the 120 training slots.  Every slot gets
    `n_per_slot` DISTINCT MNIST source digits -- the same source image is never rendered under two
    colourings, so nothing can be solved by recognising a memorised shape in a new colour.
    """
    G, L = load_mnist(n_per_class=n_per_slot * 16, seed=seed)
    rng = np.random.default_rng(seed)
    ood = set(OOD_PAIRS)
    Xs, dig, fgi, bgi, slot, seen = [], [], [], [], [], []
    for d in range(10):
        pool = np.where(L == d)[0]
        rng.shuffle(pool)
        cur = 0
        for fi, fn in enumerate(FG_NAMES):
            for bi, bn in enumerate(BG_NAMES):
                take = pool[cur:cur + n_per_slot]
                cur += n_per_slot
                Xs.append(colorize(G[take], FG_COLORS[fn], BG_COLORS[bn]))
                n = len(take)
                dig.append(np.full(n, d)); fgi.append(np.full(n, fi)); bgi.append(np.full(n, bi))
                slot.append(np.full(n, slot_id(d, fi, bi)))
                seen.append(np.full(n, (fn, bn) not in ood))
    return dict(X=np.ascontiguousarray(np.concatenate(Xs)),
                digit=np.concatenate(dig).astype(int), fg=np.concatenate(fgi).astype(int),
                bg=np.concatenate(bgi).astype(int), slot=np.concatenate(slot).astype(int),
                seen=np.concatenate(seen))


def rgb_img(vec):
    return np.clip(np.asarray(vec, np.float32).reshape(IMG, IMG, 3), 0, 1)
