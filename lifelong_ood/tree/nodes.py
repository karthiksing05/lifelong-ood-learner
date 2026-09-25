"""The prototype tree: nodes are modes of the noisy marginals p_u, one level per noise level, each
node a diagonal Gaussian with Cobweb statistics (count, mode, extent), and every byte of what a node
STORES is what it USES.

A node's working mode is regenerated from its stored form (uint8 at the level's resolution), so the
bytes charged are the bytes the responsibility actually reads.  The resolution rule (`mode_res`) is
part of the method and is reported: coarse levels (u >= 0.55) keep 8x8, mid levels 16x16, the finest
level the full image.  Extents are one variance per channel plus, optionally, a support mask.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..backbones.schedule import cosine_abar


def mode_res(u, img):
    """Stored resolution of a mode found at noise level u (px)."""
    if u >= 0.55:
        return max(min(8, img), 1)
    if u >= 0.2:
        return max(min(16, img), 1)
    return int(img)


def _down(x_img, r):
    """[H, W, C] float -> [r, r, C] by block averaging (H, W multiples of r)."""
    H, W, C = x_img.shape
    if r == H:
        return x_img
    f = H // r
    return x_img.reshape(r, f, r, f, C).mean(axis=(1, 3))


def _up(x_small, H):
    r = x_small.shape[0]
    if r == H:
        return x_small
    return np.repeat(np.repeat(x_small, H // r, axis=0), H // r, axis=1)


def quantize_mode(rows, img, res, lo, hi):
    """flat rows (HWC, in [lo, hi]) -> uint8 [res, res, 3]."""
    x = np.asarray(rows, np.float32).reshape(img, img, 3)
    x = _down(x, res)
    q = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.round(q * 255.0).astype(np.uint8)


def dequantize_mode(q, img, lo, hi):
    """uint8 [res, res, 3] -> flat rows [img*img*3] in [lo, hi]."""
    x = q.astype(np.float32) / 255.0 * (hi - lo) + lo
    return _up(x, img).reshape(-1)


@dataclasses.dataclass
class Node:
    id: int
    level: int
    u: float
    q: np.ndarray                  # stored mode, uint8 [res, res, 3]
    extent: np.ndarray             # stored per-channel variance, float32 [3] (x0 space)
    count: float = 0.0
    parent: int = -1               # -1 = the root
    expert: str | None = None
    gain_accrued: float = 0.0
    mode: np.ndarray = dataclasses.field(default=None, repr=False)   # working copy, flat rows
    var: np.ndarray = dataclasses.field(default=None, repr=False)    # working per-dim variance, flat rows

    def bytes(self):
        """level u8 + count f32 + parent u16 + expert u16 + gain f32 = 13 B, rounded to 16, plus the
        stored mode and the three channel variances."""
        return 16 + int(self.q.size) + 4 * int(self.extent.size)


class Tree:
    """Nodes at a grid of noise levels.  `levels[l]` lists node ids at level l (l = 0 is the finest)."""

    def __init__(self, u_grid, img, lo, hi):
        self.u_grid = [float(u) for u in u_grid]
        self.img, self.lo, self.hi = int(img), float(lo), float(hi)
        self.nodes: dict[int, Node] = {}
        self.levels: list[list[int]] = [[] for _ in self.u_grid]
        self._next = 0

    # ── construction ────────────────────────────────────────────────────────
    def add(self, level, mode_rows, extent_perdim=None, count=1.0, parent=-1, extent_channels=None):
        """A node from a working mode (flat rows) and a per-dim variance (or per-channel [3])."""
        u = self.u_grid[level]
        res = mode_res(u, self.img)
        q = quantize_mode(mode_rows, self.img, res, self.lo, self.hi)
        if extent_channels is None:
            v = np.asarray(extent_perdim, np.float32).reshape(self.img, self.img, 3)
            extent_channels = v.mean(axis=(0, 1))
        n = Node(id=self._next, level=int(level), u=u, q=q, extent=np.asarray(extent_channels, np.float32),
                 count=float(count), parent=int(parent))
        self._next += 1
        self._materialise(n)
        self.nodes[n.id] = n
        self.levels[level].append(n.id)
        return n.id

    def _materialise(self, n):
        n.mode = dequantize_mode(n.q, self.img, self.lo, self.hi)
        n.var = np.repeat(n.extent.reshape(1, 3), self.img * self.img, axis=0).reshape(-1)

    def remove(self, nid):
        n = self.nodes.pop(nid)
        self.levels[n.level].remove(nid)
        for m in self.nodes.values():
            if m.parent == nid:
                m.parent = n.parent

    # ── queries ─────────────────────────────────────────────────────────────
    def level_of(self, u):
        return int(np.argmin(np.abs(np.asarray(self.u_grid) - float(u))))

    def children(self, nid):
        return [m.id for m in self.nodes.values() if m.parent == nid]

    def stats(self, level):
        """-> (ids, modes [K, D], vars [K, D], counts [K]) for a level."""
        ids = list(self.levels[level])
        if not ids:
            return ids, np.zeros((0, 3 * self.img * self.img), np.float32), np.zeros((0, 3 * self.img * self.img), np.float32), np.zeros(0)
        M = np.stack([self.nodes[i].mode for i in ids]).astype(np.float32)
        V = np.stack([self.nodes[i].var for i in ids]).astype(np.float32)
        c = np.asarray([self.nodes[i].count for i in ids], np.float64)
        return ids, M, V, c

    def bytes(self):
        return int(sum(n.bytes() for n in self.nodes.values()))

    def summary(self):
        return {"n_nodes": len(self.nodes), "per_level": [len(l) for l in self.levels], "bytes": self.bytes(),
                "u_grid": list(self.u_grid), "img": self.img}

    # ── (de)serialisation: exactly the stored fields ─────────────────────────
    def to_dict(self):
        return dict(u_grid=self.u_grid, img=self.img, lo=self.lo, hi=self.hi, next=self._next,
                    nodes=[dict(id=n.id, level=n.level, q=n.q, extent=n.extent, count=n.count, parent=n.parent,
                                expert=n.expert, gain=n.gain_accrued) for n in self.nodes.values()])

    @classmethod
    def from_dict(cls, d):
        t = cls(d["u_grid"], d["img"], d["lo"], d["hi"])
        t._next = int(d["next"])
        for r in d["nodes"]:
            n = Node(id=int(r["id"]), level=int(r["level"]), u=t.u_grid[int(r["level"])], q=np.asarray(r["q"], np.uint8),
                     extent=np.asarray(r["extent"], np.float32), count=float(r["count"]), parent=int(r["parent"]),
                     expert=r.get("expert"), gain_accrued=float(r.get("gain", 0.0)))
            t._materialise(n)
            t.nodes[n.id] = n
            t.levels[n.level].append(n.id)
        return t

    def save(self, path):
        d = self.to_dict()
        np.savez_compressed(path, meta=np.array([d["u_grid"], d["img"], d["lo"], d["hi"], d["next"]], dtype=object),
                            nodes=np.array(d["nodes"], dtype=object), allow_pickle=True)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        u_grid, img, lo, hi, nxt = z["meta"]
        return cls.from_dict(dict(u_grid=list(u_grid), img=int(img), lo=float(lo), hi=float(hi), next=int(nxt),
                                  nodes=list(z["nodes"])))


def level_gaussian(tree, level):
    """The noisy-marginal parameters of a level's nodes: (ids, centres sqrt(a) m, variances a V + (1-a), log pi)."""
    ids, M, V, c = tree.stats(level)
    a = float(cosine_abar(tree.u_grid[level]))
    if len(ids) == 0:
        return ids, M, V, np.zeros(0)
    pi = c / max(c.sum(), 1e-12)
    return ids, np.sqrt(a) * M, a * V + (1.0 - a), np.log(np.clip(pi, 1e-12, None))
