"""N38 -- the hierarchy itself: what it looks like, and how its shape grows with the stream.

Everything this repository reports about the tree is a scalar buried in a table -- node counts in the
growth accounting, a depth in `summary()`, "10 shared nodes" in the unbounded test.  Nobody has drawn
it.  This does, on ColorMNIST with the ordinary well-trained backbone, and adds the structural
learning curves: size, depth, branching factor and sharing as a function of how many OOD concepts
have arrived.

**This needs no training, and that is exact rather than an approximation.**  Placement reads only the
teacher mean: `add_concept(name, mu)` calls `tree.insert(mu)`, and consolidation never touches
`tree.mean` / `tree.count` / `tree.children`.  So the structure produced by absorbing a stream is
fully determined by the sequence of composed means, and this computes those (mode ascent, the only
GPU cost) and inserts them -- skipping the epochs entirely.  The tree drawn here is bit-for-bit the
tree a full run would build from the same stream.

**Order is swept, not fixed.**  §23 established that the tree is order-dependent -- a different
arrival order gives concepts different parents, and only ~5.6% keep their path.  Reporting one
tree's statistics as "the" statistics would hide exactly that, so the curves are mean ± spread over
`--n-orders` permutations of the same concept set, and the drawn tree is labelled as one draw.

Statistics recorded after every insertion:

  `n_nodes`        total nodes allocated (each costs the same fixed adapter block)
  `depth`          deepest path, capped by `max_depth`
  `branching`      mean children per node that has any
  `n_shared`       internal nodes with >= 2 concepts beneath them -- the ones sharing is supposed to
                   happen at
  `frac_sharing`   fraction of concepts that share a non-root ancestor with at least one other
  `mean_path`      mean root-to-leaf length, i.e. how many blocks a concept's forward pass sums

Writes `figures/f14_hierarchy.png`.

    python experiments/n38_hierarchy_growth.py --n-concepts 40 --n-orders 5
"""
import argparse
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

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def stats(tree, paths):
    """Structural summary of a tree holding `paths` (one root-to-leaf list per concept)."""
    live = {n for p in paths for n in p}
    kids = {n: [c for c in tree.children.get(n, []) if c in live] for n in live}
    inner = [n for n, c in kids.items() if c]
    # how many concepts sit beneath each node
    beneath = {}
    for p in paths:
        for n in p:
            beneath[n] = beneath.get(n, 0) + 1
    shared = [n for n in inner if beneath.get(n, 0) >= 2 and n != 0]
    sharing = sum(1 for p in paths if any(beneath.get(n, 0) >= 2 for n in p[1:-1]))
    return dict(n_concepts=len(paths), n_nodes=len(live),
                depth=max((len(p) - 1 for p in paths), default=0),
                branching=float(np.mean([len(c) for c in kids.values() if c])) if inner else 0.0,
                n_shared=len(shared),
                frac_sharing=sharing / max(len(paths), 1),
                mean_path=float(np.mean([len(p) for p in paths])) if paths else 0.0)


def layout(tree, paths):
    """(x, y) per node for a tidy top-down drawing; leaves evenly spaced, parents centred."""
    live = {n for p in paths for n in p}
    kids = {n: sorted(c for c in tree.children.get(n, []) if c in live) for n in live}
    depth = {}
    for p in paths:
        for d, n in enumerate(p):
            depth[n] = min(depth.get(n, d), d)
    leaves = [n for n in live if not kids.get(n)]
    order, seen = [], set()

    def walk(n):
        if n in seen:
            return
        seen.add(n)
        if not kids.get(n):
            order.append(n)
            return
        for c in kids[n]:
            walk(c)
    walk(0)
    for n in leaves:                      # any leaf the walk missed (defensive)
        if n not in order:
            order.append(n)
    xs = {n: i for i, n in enumerate(order)}
    for n in sorted(live, key=lambda m: -depth.get(m, 0)):
        if kids.get(n):
            xs[n] = float(np.mean([xs[c] for c in kids[n] if c in xs]))
    return {n: (xs[n], depth.get(n, 0)) for n in live if n in xs}, kids, depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=40)
    p.add_argument("--n-orders", type=int, default=5)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--branch-max", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--var", type=float, default=0.1)
    p.add_argument("--params-per-node", type=int, default=551408,
                   help="adapter parameters per node at rank 8 on this backbone (measured)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_orders, args.n_per_slot, args.steps = 8, 2, 20, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    print(f"[n38] composing {len(qs)} OOD teachers (the only GPU cost; placement needs no "
          f"training)\n", flush=True)
    mus, labels = [], []
    for i, q in enumerate(qs):
        t = L.compose(to_model(data["X"][q["query"]]).ravel())
        mus.append(np.ravel(t["mu"]))
        labels.append(dict(slot=int(q["slot"]), d=int(q["d"]), f=int(q["f"]), b=int(q["b"])))
        if (i + 1) % 10 == 0:
            print(f"[n38]   {i + 1}/{len(qs)}", flush=True)
    del field, L
    torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    orders = [list(range(len(mus)))] + [list(rng.permutation(len(mus)))
                                        for _ in range(args.n_orders - 1)]
    curves, finals = [], None
    for oi, order in enumerate(orders):
        tree = ConceptTree(var=args.var, branch_max=args.branch_max, max_depth=args.max_depth)
        paths, traj = [], []
        for j in order:
            path, _ = tree.insert(mus[j])
            paths.append(path)
            traj.append(stats(tree, paths))
        curves.append(traj)
        if oi == 0:
            finals = dict(tree=tree, paths=list(paths), order=list(order))
        s = traj[-1]
        print(f"[n38] order {oi}: {s['n_nodes']} nodes, depth {s['depth']}, branching "
              f"{s['branching']:.2f}, {s['n_shared']} shared, {s['frac_sharing']:.2f} of concepts "
              f"share an ancestor, mean path {s['mean_path']:.2f}", flush=True)

    keys = ("n_nodes", "depth", "branching", "n_shared", "frac_sharing", "mean_path")
    agg = {k: (np.array([[t[k] for t in c] for c in curves], float)) for k in keys}
    save_json(dict(final=stats(finals["tree"], finals["paths"]),
                   curves={k: v.tolist() for k, v in agg.items()},
                   paths=[list(map(int, p)) for p in finals["paths"]],
                   order=[int(i) for i in finals["order"]],
                   labels=labels, config=vars(args)),
              os.path.join(RESULTS, f"n38_hierarchy_growth{args.tag}.json"), seed=args.seed)

    # ── figure ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig = plt.figure(figsize=(17.5, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.45, 1, 1], hspace=0.42, wspace=0.26)

    # the tree
    ax = fig.add_subplot(gs[0, :])
    pos, kids, depth = layout(finals["tree"], finals["paths"])
    beneath = {}
    for pth in finals["paths"]:
        for n in pth:
            beneath[n] = beneath.get(n, 0) + 1
    maxd = max(depth.values()) if depth else 0
    for n, (x, d) in pos.items():
        for c in kids.get(n, []):
            if c in pos:
                ax.plot([x, pos[c][0]], [-d, -pos[c][1]], color="#BFBFBF", lw=1.0, zorder=1)
    for n, (x, d) in pos.items():
        leaf = not kids.get(n)
        r = 0.16 + 0.30 * (beneath.get(n, 1) / max(beneath.values())) ** 0.5
        ax.add_patch(Circle((x, -d), r, color="#8A8A8A" if leaf else "#1B7F4B",
                            alpha=0.55 if leaf else 0.95, zorder=3))
        if not leaf:
            ax.text(x, -d, str(beneath.get(n, 0)), ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=4)
    f = stats(finals["tree"], finals["paths"])
    ax.set_xlim(-1.5, len(finals["paths"]) + 0.5)
    ax.set_ylim(-maxd - 0.8, 0.9)
    ax.axis("off")
    ax.set_title(
        f"The hierarchy after {f['n_concepts']} OOD concepts (ColorMNIST, standard backbone; "
        f"one arrival order of {len(orders)})\n"
        f"{f['n_nodes']} nodes  ·  depth {f['depth']}  ·  branching {f['branching']:.2f}  ·  "
        f"{f['n_shared']} shared nodes  ·  {f['frac_sharing']:.0%} of concepts share a non-root "
        f"ancestor  ·  mean path {f['mean_path']:.2f} blocks",
        fontsize=11)
    for d in range(maxd + 1):
        ax.text(-1.2, -d, f"L{d}", fontsize=8, color="#6A6A6A", va="center")
    ax.text(0.995, 0.02, "green = internal (number = concepts beneath)   grey = leaf (one concept)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#555")

    # the curves
    panels = [("n_nodes", "tree size (nodes)"), ("depth", "depth"),
              ("branching", "mean branching factor"), ("n_shared", "shared internal nodes"),
              ("frac_sharing", "fraction of concepts sharing"), ("mean_path", "mean path length")]
    x = np.arange(1, len(mus) + 1)
    for i, (k, title) in enumerate(panels):
        axc = fig.add_subplot(gs[1 + i // 3, i % 3])
        v = agg[k]
        m, lo, hi = v.mean(0), v.min(0), v.max(0)
        axc.fill_between(x, lo, hi, color="#1B7F4B", alpha=0.16, lw=0)
        axc.plot(x, m, color="#1B7F4B", lw=2)
        if k == "n_nodes":                      # what the growth actually costs
            ax2 = axc.twinx()
            ax2.plot(x, m * args.params_per_node / 1e6, color="#B03A2E", lw=1.2, ls=(0, (4, 3)))
            ax2.set_ylabel("adapter params (M)", fontsize=8, color="#B03A2E")
            ax2.tick_params(labelsize=7.5, colors="#B03A2E")
        axc.set_title(title, fontsize=10)
        axc.set_xlabel("OOD concepts absorbed", fontsize=8.5)
        axc.grid(alpha=0.25, lw=0.6)
        axc.tick_params(labelsize=8)
    fig.suptitle("N38 — the hierarchy on ColorMNIST: final shape, and how it grows with the stream. "
                 f"Band = min–max over {len(orders)} arrival orders (§23: the tree is "
                 "order-dependent).", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, f"f14_hierarchy{args.tag}.png")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"\n[n38] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
