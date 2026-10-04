"""M5 (offline): does the library GROW by motions, not by tasks?  Stream LIBERO-90 demos through ADDRESS + the node
half of FOLD, no rollouts.

    python experiments/r5_growth.py --tree results/robot/r2b_libero_40_tree.npz --library results/robot/r2b_libero_40.npz \
        --order 42+40+41+18+35+33+39+20

For each task in order: the novelty read of its ONE demo against the CURRENT tree (thresholds fixed once, on held-out
frames of the base library), then the unexplained chunks' mixture modes are added as nodes at the band levels (what
r4_fold adds).  Reported per step: unexplained fraction before folding, nodes and bytes added, the tree's size.  A
second pass reads every task against the FINAL tree (all folds done): a task whose motion another task folded should
now be explained.  Also run on a shuffled order: if cost tracks motions, related tasks after the first are cheap
whatever the order.  CPU; ledger row r5_growth.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy import novelty                                # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fold_nodes(tree, U, band, r2b, device="cpu"):
    """Add the mixture modes of the unexplained chunks U at every band level.  -> new node ids."""
    new = []
    for l in band:
        u = tree.u_grid[l]
        al, sg = 1.0 - u, u
        modes = r2b.gmm_modes(al * U, np.full_like(U, sg * sg) + al * al * 1e-3, device=device) / al
        kept, owner = r2b.basins(modes, 0.05 * max(sg / al, 1e-3))
        for k in kept:
            mm = owner == k
            v = (1e-3 + (U[mm] - modes[k][None]) ** 2).mean(0)
            new.append(tree.add(l, modes[k], extent_perdim=v, count=float(mm.sum()), var_floor=1e-4, var_cap=4.0))
    return new


def unexplained(tree, rows, band, thr):
    return float(novelty.read(tree, rows, band, thr)[1].mean())


def stream(tree_path, order, band, thr, texts, r2b, rows_of, held_of):
    """-> per step: the task's demo read against the base library and against the library grown by the tasks before
    it (TRANSFER), the nodes / bytes its own fold adds; and per task a HELD-OUT demo read against the base library and
    against the final library (does the fold GENERALISE to a new instance of the same task?)."""
    base = Tree.load(tree_path)
    tree = Tree.load(tree_path)
    steps = []
    for tid in order:
        rows = rows_of[tid]
        nll, unexp, depth = novelty.read(tree, rows, band, thr)
        b0 = tree.bytes()
        new = fold_nodes(tree, rows[unexp], band, r2b) if unexp.any() else []
        steps.append(dict(task=tid, text=texts[tid], frames=int(len(rows)), unexplained_base=unexplained(base, rows, band, thr),
                          unexplained_stream=float(unexp.mean()), nodes_added=len(new), bytes_added=int(tree.bytes() - b0),
                          tree_nodes=len(tree.nodes)))
    held = {tid: dict(base=unexplained(base, held_of[tid], band, thr), final=unexplained(tree, held_of[tid], band, thr))
            for tid in order if held_of.get(tid) is not None}
    return steps, held, tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--order", required=True, help="LIBERO-90 ids joined by +")
    ap.add_argument("--band", default="2,3,4")
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    band = [int(x) for x in a.band.split(",")]
    order = [int(x) for x in a.order.split("+")]
    r2b, r3p = _mod("r2b_marginal"), _mod("r3_prepare")
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    lib = np.load(a.library)
    thr = novelty.calibrate(Tree.load(a.tree), novelty.heldout_rows(D.Demos(), lib), band)
    rows_of, held_of = {}, {}
    for tid in order:
        eps = d90.episodes_of_task(tid)
        ep = d90.load(eps[0])
        rows_of[tid] = D.to_rows(ep["actions"][D.chunk_indices(len(ep["actions"]), 10)[0]], norm)
        if len(eps) > 1:
            h = d90.load(eps[1])
            held_of[tid] = D.to_rows(h["actions"][D.chunk_indices(len(h["actions"]), 10)[0]], norm)
    steps, held, tree = stream(a.tree, order, band, thr, texts, r2b, rows_of, held_of)
    print(f"[r5g] thresholds {[round(v, 2) for v in thr.values()]} (q95 of held-out library frames)", flush=True)
    for s in steps:
        h = held.get(s["task"], {})
        print(f"[r5g] {s['task']:>3} {s['text'][:44]:44} unexplained: base {s['unexplained_base']:.2f} -> after earlier folds "
              f"{s['unexplained_stream']:.2f}; +{s['nodes_added']} nodes (+{s['bytes_added']} B) | held-out demo: base "
              f"{h.get('base', float('nan')):.2f} -> final {h.get('final', float('nan')):.2f}", flush=True)
    rng = np.random.default_rng(a.seed)
    shuf = []
    for k in range(a.shuffles):
        o2 = [int(x) for x in rng.permutation(order)]
        st2, _, _ = stream(a.tree, o2, band, thr, texts, r2b, rows_of, {})
        shuf.append(dict(order=o2, steps=st2))
        print(f"[r5g] shuffle {k} {o2}: bytes per step {[s['bytes_added'] for s in st2]} total {sum(s['bytes_added'] for s in st2)}", flush=True)
    m = dict(order=order, band=band, thresholds={str(k): v for k, v in thr.items()}, steps=steps, heldout=held,
             total_bytes_added=int(sum(s["bytes_added"] for s in steps)), shuffles=shuf)
    ledger.record("r5_growth", bench="libero_90", tag=a.tag, seed=a.seed, keys=dict(order="+".join(map(str, order))), metrics=m,
                  bytes=dict(experts=0, tree=m["total_bytes_added"], codes=0, samples=0, teacher_copies=0, total=m["total_bytes_added"]),
                  config=vars(a), script="experiments/r5_growth.py", command=" ".join(sys.argv))


if __name__ == "__main__":
    main()
