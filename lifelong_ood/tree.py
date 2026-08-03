"""GaussianCobweb: the PORTABILITY FALLBACK tree, in pure numpy.

The real Cobweb used by this repository is cobweb-private's C++ `CobwebContinuousTree`, wrapped in
`backend.CobwebPrivateTree` and selected automatically whenever it is importable (see backend.py).
This module exists so the repository still runs -- tests, smoke experiments, other machines -- where
cobweb-private is not installed.  It is a faithful small continuous Cobweb in the COBWEB/3 lineage:
instances arrive one at a time, each node stores diagonal-Gaussian sufficient statistics (count, sum,
sum-of-squares), and `ifit` descends choosing among the four classic operators -- host into the best
child, create a new child, merge the two best children, split the best child -- scored by a
continuous category utility.  No gradient descent, no batch step; the stream IS the training.

Both backends expose the same `snapshot(min_count)` -> arrays (means, variances, mixture weights,
depths) interface, and everything downstream reads only that dict.  Lifelong adapters are NOT grown
inside the tree: they are separate trees whose snapshots `backend.merge_snapshots` concatenates under
the frozen base -- that is what makes the frozen-base property exact under either backend.
"""
import numpy as np

_SQ4PI = 1.0 / (2.0 * np.sqrt(np.pi))


class Node:
    __slots__ = ("n", "s", "ss", "children", "branch")

    def __init__(self, n=0.0, s=None, ss=None, branch="base"):
        self.n, self.s, self.ss = float(n), s, ss
        self.children = []
        self.branch = branch

    @classmethod
    def leaf(cls, x, branch="base"):
        x = np.asarray(x, np.float64)
        return cls(1.0, x.copy(), x * x, branch)

    def update(self, x):
        self.n += 1.0
        self.s += x
        self.ss += x * x

    def mean(self):
        return self.s / self.n

    def var(self, prior_var):
        m = self.s / self.n
        return np.maximum(self.ss / self.n - m * m, 0.0) + prior_var


class GaussianCobweb:
    """See module docstring.  `prior_var` is the variance floor added to every concept, exactly the
    role it plays in cobweb-poincare; it is the only structural hyperparameter."""
    backend = "numpy"

    def __init__(self, prior_var=0.05):
        self.prior_var = float(prior_var)
        self.root = None

    # ── category utility ─────────────────────────────────────────────────────
    def _score_stats(self, n, s, ss):
        """Per-child expected-correct-guess score for continuous attributes: mean_d 1/(2*sqrt(pi)*sigma_d)."""
        m = s / n
        var = np.maximum(ss / n - m * m, 0.0) + self.prior_var
        return _SQ4PI * float(np.mean(1.0 / np.sqrt(var)))

    def _cu(self, stats):
        """Category utility of a partition given [(n, s, ss), ...] hypothetical child stats."""
        if not stats:
            return -np.inf
        n_all = sum(st[0] for st in stats)
        if n_all <= 0:
            return -np.inf
        parent = (n_all, sum(st[1] for st in stats), sum(st[2] for st in stats))
        base = self._score_stats(*parent)
        cu = 0.0
        for (n, s, ss) in stats:
            cu += (n / n_all) * (self._score_stats(n, s, ss) - base)
        return cu / len(stats)

    def _loglik(self, node, x):
        m = node.mean()
        v = node.var(self.prior_var)
        return -0.5 * float(np.sum((x - m) ** 2 / v + np.log(2 * np.pi * v)))

    # ── the four operators, chosen by CU ─────────────────────────────────────
    def ifit(self, x):
        """Absorb ONE instance."""
        x = np.asarray(x, np.float64).ravel()
        if self.root is None:
            self.root = Node.leaf(x)
            return self
        node = self.root
        node.update(x)
        while True:
            if not node.children:
                # a leaf that has just absorbed x: fork into (old prototype, new instance)
                if node.n >= 2.0:
                    old = Node(node.n - 1.0, node.s - x, node.ss - x * x, node.branch)
                    if float(np.max(old.var(0.0))) > 1e-12 or float(np.sum((old.mean() - x) ** 2)) > 1e-12:
                        node.children = [old, Node.leaf(x, node.branch)]
                return self

            ch = node.children
            stats = [(c.n, c.s, c.ss) for c in ch]
            ll = np.array([self._loglik(c, x) for c in ch])
            order = np.argsort(-ll)
            i1 = int(order[0])
            i2 = int(order[1]) if len(ch) > 1 else None
            xx = x * x

            def with_added(st):
                return (st[0] + 1.0, st[1] + x, st[2] + xx)

            cand = {}
            cand["best"] = self._cu([with_added(st) if i == i1 else st for i, st in enumerate(stats)])
            cand["new"] = self._cu(stats + [(1.0, x, xx)])
            if i2 is not None:
                merged = (stats[i1][0] + stats[i2][0], stats[i1][1] + stats[i2][1], stats[i1][2] + stats[i2][2])
                cand["merge"] = self._cu([with_added(merged)] +
                                         [st for i, st in enumerate(stats) if i not in (i1, i2)])
            if ch[i1].children:
                cand["split"] = self._cu([st for i, st in enumerate(stats) if i != i1] +
                                         [(c.n, c.s, c.ss) for c in ch[i1].children])
            act = max(cand, key=cand.get)

            if act == "new":
                node.children.append(Node.leaf(x, node.branch))
                return self
            if act == "best":
                ch[i1].update(x)
                node = ch[i1]
                continue
            if act == "merge":
                m = Node(stats[i1][0] + stats[i2][0], stats[i1][1] + stats[i2][1],
                         stats[i1][2] + stats[i2][2], node.branch)
                m.children = [ch[i1], ch[i2]]
                node.children = [c for i, c in enumerate(ch) if i not in (i1, i2)] + [m]
                m.update(x)
                node = m
                continue
            # split: promote the best child's children and re-decide at this node
            node.children = [c for i, c in enumerate(ch) if i != i1] + list(ch[i1].children)
            continue

    def fit(self, X, order=None, seed=0, verbose=False):
        """Absorb many instances one at a time.  Order matters to Cobweb; default is a seeded shuffle."""
        X = np.asarray(X, np.float64)
        if order is None:
            order = np.random.default_rng(seed).permutation(len(X))
        for k, i in enumerate(np.asarray(order)):
            self.ifit(X[i])
            if verbose and (k + 1) % 500 == 0:
                print(f"    ifit {k + 1}/{len(order)}  nodes={self.n_nodes}", flush=True)
        return self

    # ── reading the tree off as arrays ───────────────────────────────────────
    def iter_nodes(self):
        stack = [self.root] if self.root is not None else []
        while stack:
            nd = stack.pop()
            yield nd
            stack.extend(nd.children)

    @property
    def n_nodes(self):
        return sum(1 for _ in self.iter_nodes())

    def snapshot(self, min_count=3):
        """-> dict(M, V, W, depth, is_leaf, parent, branch) over nodes with count >= min_count.

        `is_leaf` marks EFFECTIVE leaves: kept nodes none of whose kept children exist.  The mixture
        over effective leaves, weighted by count, is the tree's estimate of p_data; shallower kept
        nodes are its coarser abstractions.  Everything downstream reads only this dict.
        """
        keep, depth, parent, order = [], [], [], {}
        def walk(nd, d, par):
            if nd.n < min_count:
                return
            order[id(nd)] = len(keep)
            keep.append(nd); depth.append(d); parent.append(par)
            for c in nd.children:
                walk(c, d + 1, order[id(nd)])
        if self.root is not None:
            walk(self.root, 0, -1)
        if not keep:
            raise ValueError("empty snapshot: lower min_count or fit more data")
        N = len(keep)
        has_kept_child = np.zeros(N, bool)
        for i, p in enumerate(parent):
            if p >= 0:
                has_kept_child[p] = True
        M = np.stack([nd.mean() for nd in keep]).astype(np.float64)
        V = np.stack([nd.var(self.prior_var) for nd in keep]).astype(np.float64)
        n = np.array([nd.n for nd in keep])
        is_leaf = ~has_kept_child
        W = np.where(is_leaf, n, 0.0)
        W = W / W.sum()
        return dict(M=M, V=V, W=W, count=n, depth=np.array(depth), parent=np.array(parent),
                    is_leaf=is_leaf, branch=np.array([nd.branch for nd in keep]))


def frontier_mask(snap, level):
    """Nodes forming the depth-`level` cut of the snapshot tree: at depth `level`, or effective leaves
    shallower than it.  These are the mixture components of the coarse marginal at that level."""
    d, leaf = snap["depth"], snap["is_leaf"]
    return (d == level) | (leaf & (d < level))


def frontier_weights(snap, mask):
    w = np.where(mask, snap["count"], 0.0).astype(np.float64)
    return w / w.sum()
