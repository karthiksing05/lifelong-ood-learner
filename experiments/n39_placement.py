"""N39 -- why is the tree a caterpillar, and what placement rule would not be?

N38 drew the hierarchy and found a degenerate one: 31 of 40 concepts share the identical path prefix,
3 shared nodes, branching 10.0, depth 4. §24f blamed the descent rule from reading the code. That is
a hypothesis, and it has an obvious competitor that must be ruled out first.

**The control that could exonerate the rule entirely.** If the composed teacher means carry no
cluster structure -- if the 40 mu's are roughly equidistant in a 3072-dimensional space -- then no
placement rule can group them and the caterpillar is the correct answer to a question with no
structure in it. So this measures the structure available BEFORE blaming the consumer of it:
silhouette of the true factor labels in mu-space, the pairwise distance distribution, and what an
offline agglomerative clustering recovers with the whole set visible at once.

**Then, why the current rule fails.** `_descend` is instrumented to record, for every step, whether
it descended because the child genuinely fit better (`quality`) or because the current node had
already hit `branch_max` (`forced`). §24f predicts the forced fraction is high; that is now measured
rather than argued.

Three defects are visible in the code by inspection, and each gets an arm:

    ConceptTree._ll:  -0.5 * mean((mu - m)^2) / self.var        <- ONE global scalar variance
    GaussianCobweb._loglik: -0.5 * sum((x-m)^2/v + log(2*pi*v)) <- PER-NODE variance, with log-det

  1. a fixed scalar `var` means a node that has absorbed forty diverse concepts scores as tight as a
     singleton -- there is no notion of a node's own spread;
  2. `mean` over dimensions rather than `sum` shrinks every likelihood difference by 1/D;
  3. there is no partition-level criterion at all.  Cobweb chooses between "host into the best child"
     and "create a new child" by CATEGORY UTILITY, which is what produces balance; this rule only
     ever goes to the nearest child, with a saturation override.

Arms:

  `current`     ConceptTree exactly as shipped (branch_max 4, max_depth 3, scalar var)
  `caps`        same rule, branch_max 16 / max_depth 6 -- isolates the caps from the criterion
  `nodevar`     per-node variance from its members, summed not averaged, with the log-det term
  `cobweb`      the real `GaussianCobweb` on the same means -- the reference this is measured against,
                since "well-distributed like Cobweb" is the actual target
  `agglom`      offline average-linkage over all 40 means at once: an ORACLE, not a method, since it
                is neither incremental nor insert-only.  It bounds what any online rule could reach.

Scored on shape (nodes, depth, branching, balance) and on whether the structure is MEANINGFUL:
sibling purity, the fraction of concepts under an internal node sharing that node's modal value of
each true factor, against the 1/n_values a random tree gets.

    python experiments/n39_placement.py --n-concepts 40
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                                # noqa: E402

from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import save_json                                      # noqa: E402
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_model   # noqa: E402
from lifelong_ood.neural.adapters import ConceptTree                          # noqa: E402
from lifelong_ood.tree import GaussianCobweb                                  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
FACTORS = ("d", "f", "b")
NVALS = dict(d=10, f=4, b=4)


class InstrumentedTree(ConceptTree):
    """ConceptTree that records WHY each descent happened, and can use a per-node variance."""

    def __init__(self, *a, node_var=False, **kw):
        super().__init__(*a, **kw)
        self.node_var = bool(node_var)
        self.sq = {0: None}                 # running sum of squares, for a per-node variance
        self.log = []                       # ("quality"|"forced"|"stop", depth)

    def _nvar(self, nid):
        if not self.node_var or self.sq.get(nid) is None or self.count.get(nid, 0) < 2:
            return self.var
        m = self.mean[nid]
        v = self.sq[nid] / self.count[nid] - m * m
        return float(np.mean(np.maximum(v, 0.0))) + self.var

    def _ll(self, nid, mu):
        m = self.mean[nid]
        if m is None:
            return 0.0
        v = self._nvar(nid)
        if self.node_var:                   # sum, not mean, and pay for the node's spread
            return -0.5 * float(np.sum((mu - m) ** 2)) / v - 0.5 * len(mu) * np.log(v)
        return -0.5 * float(np.mean((mu - m) ** 2)) / self.var

    def _descend(self, mu):
        if self.flat:
            return [0]
        if self.rule == "cu":          # instrumentation is about the `ll` caps; cu has no forcing
            return self._descend_cu(mu)
        path, cur = [0], 0
        for _ in range(self.max_depth):
            kids = list(self.children[cur])
            inner = [k for k in kids if self.children[k]] or kids
            if not inner:
                self.log.append(("stop-noinner", len(path)))
                break
            best = max(inner, key=lambda k: self._ll(k, mu))
            better = self._ll(best, mu) >= self._ll(cur, mu) + self.margin
            saturated = len(kids) >= self.branch_max
            if not better and not saturated:
                self.log.append(("stop-fits-here", len(path)))
                break
            self.log.append(("quality" if better else "forced", len(path)))
            path.append(best)
            cur = best
        return path

    def _touch(self, path, mu):
        super()._touch(path, mu)
        for nid in path:
            self.sq[nid] = mu * mu if self.sq.get(nid) is None else self.sq[nid] + mu * mu

    def _new(self, mu, parent):
        nid = super()._new(mu, parent)
        self.sq[nid] = np.asarray(mu, np.float64) ** 2
        return nid


def shape(parent_of, leaves_of, labels):
    """Structural + semantic summary given parent links and each node's descendant concept ids."""
    kids = {}
    for c, p in parent_of.items():
        kids.setdefault(p, []).append(c)
    internal = [n for n in kids if len(leaves_of.get(n, [])) >= 2 and n in leaves_of]
    depth = {}

    def dep(n):
        if n not in parent_of:
            return 0
        if n not in depth:
            depth[n] = dep(parent_of[n]) + 1
        return depth[n]

    all_nodes = set(kids) | {c for c in parent_of}
    dd = [dep(n) for n in all_nodes]
    # balance: normalized entropy of how the concepts split among a node's children, averaged
    bal = []
    for n, cs in kids.items():
        sz = np.array([len(leaves_of.get(c, [])) for c in cs], float)
        sz = sz[sz > 0]
        if len(sz) > 1:
            p = sz / sz.sum()
            bal.append(float(-(p * np.log(p)).sum() / np.log(len(sz))))
    # purity: of the concepts under an internal node, what fraction share its modal factor value
    pur = {}
    for f in FACTORS:
        num = den = 0.0
        for n in internal:
            vals = [labels[i][f] for i in leaves_of[n]]
            if len(vals) < 2:
                continue
            _, cnt = np.unique(vals, return_counts=True)
            num += cnt.max(); den += len(vals)
        pur[f] = float(num / den) if den else float("nan")
    return dict(n_nodes=len(all_nodes), depth=int(max(dd)) if dd else 0,
                branching=float(np.mean([len(c) for c in kids.values()])) if kids else 0.0,
                n_internal=len(internal),
                balance=float(np.mean(bal)) if bal else float("nan"),
                **{f"purity_{f}": pur[f] for f in FACTORS})


def from_concept_tree(tree, paths):
    parent_of, leaves_of = {}, {}
    for i, p in enumerate(paths):
        for a, b in zip(p, p[1:]):
            parent_of[b] = a
        for n in p:
            leaves_of.setdefault(n, []).append(i)
    return parent_of, leaves_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=40)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_per_slot, args.steps = 10, 20, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    # Mode ascent over 40 queries is ~19 min and is the ONLY GPU cost here; the placement study
    # varies nothing that affects it, so cache the means and let rule iteration be seconds.
    cache = os.path.join(CACHE, f"n39_mus_{args.n_concepts}_{args.seed}.npz")
    labels = [dict(d=int(q["d"]), f=int(q["f"]), b=int(q["b"])) for q in qs]
    if os.path.exists(cache):
        M = np.load(cache)["M"]
        print(f"[n39] loaded {len(M)} cached teacher means from {os.path.basename(cache)}",
              flush=True)
    else:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        mus = []
        print(f"[n39] composing {len(qs)} teachers", flush=True)
        for q in qs:
            mus.append(np.ravel(L.compose(to_model(data["X"][q["query"]]).ravel())["mu"]))
        del field, L
        torch.cuda.empty_cache()
        M = np.stack(mus)
        np.savez_compressed(cache, M=M)

    # ── PART 1: is there structure to find at all? ───────────────────────────
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    D = squareform(pdist(M))
    off = D[~np.eye(len(M), dtype=bool)]
    print(f"\n[n39] PART 1 -- structure available in the teacher means")
    print(f"    pairwise distance: mean {off.mean():.3f}  sd {off.std():.3f}  "
          f"min {off.min():.3f}  max {off.max():.3f}  (sd/mean {off.std()/off.mean():.3f})")
    sil = {}
    for f in FACTORS:
        lab = np.array([l[f] for l in labels])
        s = []
        for i in range(len(M)):
            same = (lab == lab[i]); same[i] = False
            if not same.any() or (~same).sum() == 0:
                continue
            a = D[i][same].mean()
            b = min(D[i][lab == v].mean() for v in np.unique(lab[lab != lab[i]]))
            s.append((b - a) / max(a, b))
        sil[f] = float(np.mean(s)) if s else float("nan")
    print(f"    silhouette of TRUE labels in mu-space: " +
          "  ".join(f"{f}={sil[f]:+.3f}" for f in FACTORS)
          + "   (>0 = that factor forms clusters a rule could find)")

    # ── PART 2 & 3: the placement arms ──────────────────────────────────────
    rows = []
    arms = [("current", dict(branch_max=4, max_depth=3, node_var=False)),
            ("caps", dict(branch_max=16, max_depth=6, node_var=False)),
            ("nodevar", dict(branch_max=16, max_depth=6, node_var=True)),
            ("cu-insertonly", dict(branch_max=16, max_depth=6, node_var=False, rule="cu")),
            ("cu-shippedcaps", dict(branch_max=4, max_depth=3, node_var=False, rule="cu"))]
    for name, kw in arms:
        t = InstrumentedTree(var=0.1, **kw)
        paths = [t.insert(m)[0] for m in M]
        parent_of, leaves_of = from_concept_tree(t, paths)
        s = shape(parent_of, leaves_of, labels)
        n_forced = sum(1 for k, _ in t.log if k == "forced")
        n_qual = sum(1 for k, _ in t.log if k == "quality")
        s.update(arm=name, forced=n_forced, quality=n_qual,
                 forced_frac=n_forced / max(n_forced + n_qual, 1),
                 max_prefix=int(max(collections.Counter(
                     tuple(p[:-1]) for p in paths).values())))
        rows.append(s)
        print(f"[n39] {name:9s} nodes {s['n_nodes']:3d} depth {s['depth']} "
              f"branch {s['branching']:5.2f} internal {s['n_internal']:2d} "
              f"balance {s['balance']:.3f} purity f={s['purity_f']:.2f} b={s['purity_b']:.2f} "
              f"| descents {n_qual} quality / {n_forced} FORCED "
              f"| biggest shared prefix {s['max_prefix']}/{len(M)}", flush=True)

    # the real Cobweb, on the same means
    cw = GaussianCobweb(prior_var=0.05)
    for m in M:
        cw.ifit(m)
    node_id, parent_of, leaves_of = {}, {}, {}
    nxt = [0]
    assign = {}

    def walk(nd, par):
        i = nxt[0]; nxt[0] += 1
        node_id[id(nd)] = i
        if par is not None:
            parent_of[i] = par
        for c in nd.children:
            walk(c, i)
        return i
    walk(cw.root, None)
    # assign each concept to its best leaf, then propagate membership upward
    def descend(nd, x):
        while nd.children:
            nd = max(nd.children, key=lambda c: cw._loglik(c, x))
        return nd
    for i, m in enumerate(M):
        nd = descend(cw.root, m)
        n = node_id[id(nd)]
        while True:
            leaves_of.setdefault(n, []).append(i)
            if n not in parent_of:
                break
            n = parent_of[n]
    s = shape(parent_of, leaves_of, labels)
    s.update(arm="cobweb", forced=0, quality=0, forced_frac=float("nan"),
             max_prefix=int(max(len(v) for v in leaves_of.values())))
    rows.append(s)
    print(f"[n39] {'cobweb':9s} nodes {s['n_nodes']:3d} depth {s['depth']} "
          f"branch {s['branching']:5.2f} internal {s['n_internal']:2d} "
          f"balance {s['balance']:.3f} purity f={s['purity_f']:.2f} b={s['purity_b']:.2f}",
          flush=True)

    # offline oracle: average-linkage over everything at once
    Z = linkage(M, method="average")
    parent_of, leaves_of = {}, {}
    n = len(M)
    for i, (a, b, _, _) in enumerate(Z):
        parent_of[int(a)] = n + i
        parent_of[int(b)] = n + i
    for i in range(len(M)):
        cur = i
        while True:
            leaves_of.setdefault(cur, []).append(i)
            if cur not in parent_of:
                break
            cur = parent_of[cur]
    s = shape(parent_of, leaves_of, labels)
    s.update(arm="agglom (oracle)", forced=0, quality=0, forced_frac=float("nan"),
             max_prefix=len(M))
    rows.append(s)
    print(f"[n39] {'agglom':9s} nodes {s['n_nodes']:3d} depth {s['depth']} "
          f"branch {s['branching']:5.2f} internal {s['n_internal']:2d} "
          f"balance {s['balance']:.3f} purity f={s['purity_f']:.2f} b={s['purity_b']:.2f}",
          flush=True)

    save_json(dict(rows=rows, silhouette=sil, labels=labels,
                   dist=dict(mean=float(off.mean()), sd=float(off.std()),
                             lo=float(off.min()), hi=float(off.max())),
                   config=vars(args)),
              os.path.join(RESULTS, f"n39_placement{args.tag}.json"), seed=args.seed)

    print(f"\n[n39] {'arm':16s}{'nodes':>6s}{'depth':>6s}{'branch':>8s}{'balance':>9s}"
          f"{'pur_f':>7s}{'pur_b':>7s}{'pur_d':>7s}{'forced%':>9s}")
    for r in rows:
        ff = "" if r["forced_frac"] != r["forced_frac"] else f"{100*r['forced_frac']:8.0f}%"
        print(f"    {r['arm']:16s}{r['n_nodes']:6d}{r['depth']:6d}{r['branching']:8.2f}"
              f"{r['balance']:9.3f}{r['purity_f']:7.2f}{r['purity_b']:7.2f}{r['purity_d']:7.2f}"
              f"{ff:>9s}")
    print(f"    {'random baseline':16s}{'':6s}{'':6s}{'':8s}{'':9s}"
          f"{1/NVALS['f']:7.2f}{1/NVALS['b']:7.2f}{1/NVALS['d']:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
