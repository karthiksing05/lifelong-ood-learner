"""Tree backends.  The REAL Cobweb is cobweb-private's C++ `CobwebContinuousTree` (nanobind), used
whenever it is importable; the pure-numpy `GaussianCobweb` in tree.py is only a portability fallback
so the repository still runs where cobweb-private isn't installed.  Both expose the same three
methods -- ifit / fit / snapshot -- and everything downstream (ScoreField, compose, LifelongLearner)
reads only the snapshot dict, so the two are interchangeable.

The cobweb-private configuration matches cobweb-poincare exactly (covar_from=1: a node's variance is
its OWN sum_sq/count + prior_var, which is what CobwebContinuousNode::log_prob uses), so trees built
here agree with trees built there.
"""
import numpy as np

COBWEB_DEFAULTS = dict(covar_type=1, covar_from=1, alpha=1e-6, insert_only=False,
                       depth_max=999999, branch_max=999999, num_labels=0)


def _assemble(parent, depth, count, M, V, min_count):
    """Rows (already restricted to kept nodes, parents reindexed) -> the snapshot dict."""
    N = len(count)
    has_kept_child = np.zeros(N, bool)
    for i, p in enumerate(parent):
        if p >= 0:
            has_kept_child[p] = True
    is_leaf = ~has_kept_child
    W = np.where(is_leaf, count, 0.0).astype(np.float64)
    return dict(M=np.asarray(M, np.float64), V=np.asarray(V, np.float64), W=W / W.sum(),
                count=np.asarray(count, np.float64), depth=np.asarray(depth, np.int64),
                parent=np.asarray(parent, np.int64), is_leaf=is_leaf,
                branch=np.array(["base"] * N))


class CobwebPrivateTree:
    """cobweb-private's CobwebContinuousTree behind the common ifit/fit/snapshot interface."""
    backend = "cobweb-private"

    def __init__(self, prior_var=0.05, **kw):
        self.prior_var = float(prior_var)
        self.cfg = dict(COBWEB_DEFAULTS, **kw)
        self.tree = None
        self._empty = np.zeros(0, np.float32)

    def _new(self, size):
        from cobweb.cobweb_continuous import CobwebContinuousTree
        c = self.cfg
        self.tree = CobwebContinuousTree(
            size=int(size), num_labels=c["num_labels"], covar_type=c["covar_type"],
            covar_from=c["covar_from"], alpha=c["alpha"], prior_var=self.prior_var,
            insert_only=c["insert_only"], depth_max=c["depth_max"], branch_max=c["branch_max"])

    def ifit(self, x):
        x = np.ascontiguousarray(np.asarray(x, np.float32).ravel())
        if self.tree is None:
            self._new(len(x))
        self.tree.ifit(x, self._empty)
        return self

    def fit(self, X, order=None, seed=0, verbose=False):
        X = np.asarray(X, np.float32)
        if order is None:
            order = np.random.default_rng(seed).permutation(len(X))
        for k, i in enumerate(np.asarray(order)):
            self.ifit(X[i])
            if verbose and (k + 1) % 1000 == 0:
                print(f"    ifit {k + 1}/{len(order)}", flush=True)
        return self

    @property
    def n_nodes(self):
        n, q = 0, [self.tree.root] if self.tree is not None else []
        while q:
            nd = q.pop()
            n += 1
            q.extend(nd.children)
        return n

    def snapshot(self, min_count=3):
        """BFS the live C++ tree into arrays; `var = sum_sq/count + prior_var` reproduces
        CobwebContinuousNode::log_prob exactly for covar_from=1 (same convention as cobweb-poincare).
        Nodes with count < min_count are pruned with their subtrees (counts only fall with depth)."""
        if self.tree is None:
            raise ValueError("empty tree")
        parent, depth, count, M, V = [], [], [], [], []
        q = [(self.tree.root, -1, 0)]
        while q:
            nd, par, d = q.pop(0)
            c = float(nd.count)
            if c < min_count and par >= 0:
                continue
            me = len(count)
            parent.append(par); depth.append(d); count.append(c)
            m = np.asarray(nd.mean, np.float64)
            M.append(m)
            V.append(np.asarray(nd.sum_sq, np.float64) / max(c, 1.0) + self.prior_var)
            for ch in nd.children:
                q.append((ch, me, d + 1))
        return _assemble(parent, depth, count, np.stack(M), np.stack(V), min_count)


def make_tree(prior_var=0.05, backend="auto", **kw):
    """backend='auto': cobweb-private if importable, else the numpy fallback.  'cobweb'/'numpy' force."""
    if backend in ("auto", "cobweb"):
        try:
            from cobweb.cobweb_continuous import CobwebContinuousTree  # noqa: F401
            return CobwebPrivateTree(prior_var, **kw)
        except ImportError:
            if backend == "cobweb":
                raise
    from .tree import GaussianCobweb
    return GaussianCobweb(prior_var)


def merge_snapshots(base, adapters):
    """One field over a frozen base plus additive adapters: [(name, snapshot), ...] concatenated,
    each adapter hanging under the base root (depth shifted by 1), weights renormalized by count.

    This is the LoRA composition rule made literal: the base arrays are UNTOUCHED -- every base mean,
    variance and subtree is bit-identical after any number of merges -- and an adapter contributes
    only new mixture components.  The only global effect is the weight renormalization, which bounds
    the log-likelihood drift on old data by log(1 + n_adapters / n_base).
    """
    if not adapters:
        return base
    Ms, Vs, cs, ds, ps, ls, bs = [base["M"]], [base["V"]], [base["count"]], [base["depth"]], \
        [base["parent"]], [base["is_leaf"]], [base["branch"]]
    off = len(base["count"])
    for name, s in adapters:
        p = s["parent"].copy() + off
        p[s["parent"] < 0] = 0                       # adapter root hangs under the base root
        Ms.append(s["M"]); Vs.append(s["V"]); cs.append(s["count"])
        ds.append(s["depth"] + 1); ps.append(p); ls.append(s["is_leaf"])
        bs.append(np.full(len(s["count"]), name, dtype=object))
        off += len(s["count"])
    count = np.concatenate(cs)
    is_leaf = np.concatenate(ls)
    W = np.where(is_leaf, count, 0.0).astype(np.float64)
    return dict(M=np.concatenate(Ms), V=np.concatenate(Vs), W=W / W.sum(), count=count,
                depth=np.concatenate(ds), parent=np.concatenate(ps), is_leaf=is_leaf,
                branch=np.concatenate([np.asarray(b, dtype=object) for b in bs]).astype(str))
