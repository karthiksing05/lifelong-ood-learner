"""GPU probe p1 (Phase 0): build the prototype tree label-free on a real backbone and score its nodes
against held-out LABELS (evaluation only; the build never sees them).

    sbatch --export=ALL,SCRIPT="experiments/p1_tree_build.py --bench colormnist --seed 0" slurm/jobs/c_gpu.sbatch

Kill criterion (docs/PLAN_2026-09.md, Section 5, phase 0): no level gives background purity >= 0.9 and
digit purity >= 0.7 with <= 300 nodes -> the pixel-space tree is dead.  Writes the tree to
results/trees/, a mode-grid figure to figures/p1/, and a ledger row (bytes = the tree's).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.accounting import charge                             # noqa: E402
from lifelong_ood.backbones.loader import load_backbone                # noqa: E402
from lifelong_ood.tree.build import DEFAULT_GRID, build                # noqa: E402
from lifelong_ood.operators.address import best_nll                   # noqa: E402
from lifelong_ood.tree.responsibility import basin                     # noqa: E402

CKPT = {"colormnist": "results/cache/ddpm_colormnist_nobg3.pt", "shapes3d": "results/cache/ddpm_s3d_full.pt"}


def purity(assign, labels):
    """Weighted majority purity of a partition w.r.t. integer labels, plus NMI."""
    assign, labels = np.asarray(assign), np.asarray(labels)
    tot, n = 0, len(labels)
    for a in np.unique(assign):
        m = assign == a
        tot += np.bincount(labels[m]).max()
    try:
        from sklearn.metrics import normalized_mutual_info_score as nmi
        s = float(nmi(labels, assign))
    except Exception:
        s = float("nan")
    return float(tot / n), s


def colormnist_eval(bk, n_eval, seed):
    """Real images of the TRAINED slots (the withheld background is reported separately)."""
    from lifelong_ood.data.datasets import load_colormnist
    d = load_colormnist(n_per_slot=n_eval, seed=seed + 1)
    ck = torch.load(bk.path, map_location="cpu", weights_only=False)
    trained = {int(k) for k in ck["remap"]}
    slot = np.asarray(d["slot"], int)
    m = np.array([s in trained for s in slot])
    X = bk.rows_from_data(d["X"])
    labels = {"digit": np.asarray(d["digit"], int), "fg": np.asarray(d["fg"], int), "bg": np.asarray(d["bg"], int)}
    return X[m], {k: v[m] for k, v in labels.items()}, X[~m], {k: v[~m] for k, v in labels.items()}


def mode_grid(tree, bk, path, per_level=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = len(tree.u_grid)
    fig, axes = plt.subplots(L, per_level, figsize=(per_level * 0.9, L * 1.0))
    for l in range(L):
        ids = sorted(tree.levels[l], key=lambda i: -tree.nodes[i].count)[:per_level]
        for c in range(per_level):
            ax = axes[l, c] if L > 1 else axes[c]
            ax.axis("off")
            if c < len(ids):
                n = tree.nodes[ids[c]]
                img = bk.rows_to_data(n.mode[None])[0].reshape(bk.img, bk.img, 3)
                ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
                ax.set_title(f"u{tree.u_grid[l]:.2f} n{int(n.count)}", fontsize=5)
    fig.suptitle("prototype tree: modes per level (coarse at the bottom), sorted by basin count", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="colormnist", choices=sorted(CKPT))
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--grid", default=":".join(f"{u:.2f}" for u in DEFAULT_GRID))
    ap.add_argument("--caps", default="32:24:16:16:8:8")
    ap.add_argument("--n-eval", type=int, default=40, help="real images per slot for the purity read")
    ap.add_argument("--select", default="dedup", choices=("dedup", "kmeans"))
    ap.add_argument("--channel-extent", action="store_true", help="collapse extents to one variance per channel (the 2026-09-25 default, killed)")
    ap.add_argument("--tree", default=None, help="reuse this tree .npz instead of building")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = tuple(float(x) for x in args.grid.split(":"))
    caps = {u: int(c) for u, c in zip(grid, args.caps.split(":"))}
    bk = load_backbone(os.path.join(ROOT, CKPT[args.bench]), device=dev)
    print(f"[p1] {bk.describe()}", flush=True)
    t0 = time.time()
    if args.tree:
        from lifelong_ood.tree.nodes import Tree
        tree = Tree.load(args.tree)
        tree.build_info = dict(reused=args.tree, per_level=[dict(secs=0.0)] * len(tree.u_grid))
        grid = tuple(tree.u_grid)
    else:
        tree = build(bk, u_grid=grid, n_samples=args.n_samples, seed=args.seed, iters=args.iters, lr=args.lr,
                     tol=args.tol, caps=caps, steps=args.steps, select=args.select, per_dim=not args.channel_extent)
    secs = time.time() - t0
    os.makedirs(os.path.join(ROOT, "results", "trees"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures", "p1"), exist_ok=True)
    tpath = os.path.join(ROOT, "results", "trees", f"tree_{args.bench}_s{args.seed}{args.tag}.npz")
    tree.save(tpath)
    fpath = os.path.join(ROOT, "figures", "p1", f"tree_{args.bench}_s{args.seed}{args.tag}.png")
    if args.tree:
        tpath = args.tree
    mode_grid(tree, bk, fpath)
    metrics = dict(n_nodes=len(tree.nodes), per_level=tree.summary()["per_level"], build_secs=secs,
                   per_level_secs=[p["secs"] for p in tree.build_info["per_level"]], levels={})
    if args.bench == "colormnist":
        Xin, lab_in, Xout, lab_out = colormnist_eval(bk, args.n_eval, args.seed)
        best = None
        for l, u in enumerate(tree.u_grid):
            assign, r = basin(tree, Xin, l, seed=args.seed)
            row = dict(u=u, n_nodes=len(tree.levels[l]), max_r=float(r.max(1).mean()))
            for f in ("digit", "fg", "bg"):
                p, s = purity(assign, lab_in[f])
                row[f"purity_{f}"], row[f"nmi_{f}"] = p, s
            nin, _ = best_nll(tree, Xin, l, seed=args.seed)
            row["nll_in"] = float(nin.mean())
            if len(Xout):
                a2, r2 = basin(tree, Xout, l, seed=args.seed)
                row["withheld_bg_max_r"] = float(r2.max(1).mean())
                row["withheld_bg_purity_digit"] = purity(a2, lab_out["digit"])[0]
                nout, _ = best_nll(tree, Xout, l, seed=args.seed)
                row["nll_withheld"] = float(nout.mean())
                row["nll_auroc"] = float((nout[:, None] > nin[None, :]).mean() + 0.5 * (nout[:, None] == nin[None, :]).mean())
            metrics["levels"][str(l)] = row
            print(f"[p1] level {l} u={u:.2f} nodes {row['n_nodes']:3d}: purity digit {row['purity_digit']:.2f} "
                  f"fg {row['purity_fg']:.2f} bg {row['purity_bg']:.2f} | nmi {row['nmi_digit']:.2f}/{row['nmi_fg']:.2f}/"
                  f"{row['nmi_bg']:.2f} | mean max r {row['max_r']:.2f}"
                  + (f" | NLL in {row['nll_in']:.3f} withheld {row['nll_withheld']:.3f} AUROC {row['nll_auroc']:.2f}" if len(Xout) else ""), flush=True)
            ok = row["purity_bg"] >= 0.9 and row["purity_digit"] >= 0.7 and len(tree.nodes) <= 300
            if ok and best is None:
                best = l
        metrics["phase0_pass"] = best is not None
        metrics["phase0_level"] = best
        # the per-factor reading (each factor at its own level), pre-registered 2026-09-25 after the strict
        # criterion failed: bg >= 0.9, fg >= 0.8, digit >= 0.5 at SOME level each, <= 300 nodes
        best_of = {f: max(r[f"purity_{f}"] for r in metrics["levels"].values()) for f in ("digit", "fg", "bg")}
        metrics["best_purity"] = best_of
        metrics["phase0b_pass"] = bool(best_of["bg"] >= 0.9 and best_of["fg"] >= 0.8 and best_of["digit"] >= 0.5 and len(tree.nodes) <= 300)
        print(f"[p1] phase 0 (strict, one level) {'PASS at level ' + str(best) if best is not None else 'KILL: no level reaches bg >= 0.9 and digit >= 0.7'}; "
              f"per-factor best {best_of} -> {'PASS' if metrics['phase0b_pass'] else 'KILL'}", flush=True)
    bill = charge(tree=tree)
    ledger.record("p1_tree_build", bench=args.bench, tag=args.tag, seed=args.seed, metrics=metrics, bytes=bill,
                  config=vars(args) | dict(build=tree.build_info), script="experiments/p1_tree_build.py",
                  command=" ".join(sys.argv), seconds=secs,
                  backbone=dict(path=CKPT[args.bench], sha256=bk.sha256, kind=bk.kind, layout=bk.layout, img=bk.img,
                                n_params=bk.n_params),
                  artifacts=dict(tree=os.path.relpath(tpath, ROOT), figure=os.path.relpath(fpath, ROOT)))
    print(f"[p1] tree {tree.summary()} bytes {bill['total']:,} in {secs:.0f}s -> {tpath}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
