"""GPU probe p2 (Phase 1): DECOMPOSE over the tree vs Zekun's per-query ascent, on the backbone's
held-out combinations of KNOWN values (the nobg3 bench's `held_comb`: the OOD (fg, bg) pairs with the
withheld background excluded), from ONE real query image each.

For each held-out cell: (a) tree PoE: decompose the query over the tree's nodes (K=3) and sample with
the analytic teacher as the conditional branch (Eq. 14-16, null-conditioned model); (b) ascent PoE:
Zekun's candidates_ascent on the legacy field with the paper's ColorMNIST settings (the `poe-guided`
arm), same sampler; (c) the same, tree + local refinement (a few ascent chains from the query at two
levels added to the pool).  Scores: joint accuracy (the bench's probes), Inception coverage, vendi,
seconds per query.  Kill (plan Section 5, phase 1): tree < 0.8 x ascent even with refinement.

    sbatch --export=ALL,SCRIPT="experiments/p2_decompose.py --seed 0" slurm/jobs/c_gpu.sbatch
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
from lifelong_ood.backbones.loader import Backbone                     # noqa: E402
from lifelong_ood.backbones.sampler import sample                      # noqa: E402
from lifelong_ood.backbones.score_ops import laplace_diag, mode_ascent  # noqa: E402
from lifelong_ood.evals import density_coverage, vendi_score           # noqa: E402
from lifelong_ood.operators.decompose import Code, decompose, guide, loglik_perdim, submodular_select  # noqa: E402
from lifelong_ood.streams.bench import load_bench                      # noqa: E402
from lifelong_ood.tree.build import DEFAULT_GRID, build                # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402


def poe_from_pool(xq, M, V, K=3, tau=0.05):
    """Zekun's selection + adaptive-temperature product over an arbitrary pool (mixture extent)."""
    L = loglik_perdim(xq, M, V)
    sel = submodular_select(L, K)
    Ls = L[sel]
    T = tau * max(float(Ls.std(0).mean()), 1e-6)
    Z = Ls / T
    Z -= Z.max(0, keepdims=True)
    w = np.exp(Z)
    w /= w.sum(0, keepdims=True)
    prec = (w / V[sel]).sum(0)
    mu = (w * M[sel] / V[sel]).sum(0) / prec
    var = (w * V[sel]).sum(0) + ((M[sel] - mu[None]) ** 2 * w).sum(0)
    return mu.astype(np.float32), var.astype(np.float32), sel


def ascent_pool(bk, xq, u_grid, n_starts, iters, lr, seed, probes=4):
    """Per-query discovery: chains from the noised query at each level; Tweedie modes + Laplace extents."""
    Ms, Vs = [], []
    Q = np.repeat(xq[None], n_starts, 0)
    for i, u in enumerate(u_grid):
        xt, x0 = mode_ascent(bk, Q, float(u), iters=iters, lr=lr, seed=seed + i, init="query", adam=True, tweedie=True)
        ext = laplace_diag(bk, xt, float(u), probes=probes, seed=seed + 100 + i)
        Ms.append(np.clip(x0, *bk.clip))
        Vs.append(ext)
    return np.concatenate(Ms), np.concatenate(Vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cells", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--w0", type=float, default=1.2)
    ap.add_argument("--wmax", type=float, default=2.0)
    ap.add_argument("--ascent-starts", type=int, default=16, help="chains per level for the per-query pool (paper: 128)")
    ap.add_argument("--ascent-iters", type=int, default=150)
    ap.add_argument("--ascent-lr", type=float, default=0.1)
    ap.add_argument("--refine-starts", type=int, default=8)
    ap.add_argument("--refine-iters", type=int, default=50)
    ap.add_argument("--tree", default=None)
    ap.add_argument("--n-tree-samples", type=int, default=512)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_cells, args.n_samples, args.ascent_starts, args.ascent_iters, args.n_tree_samples = 4, 12, 4, 40, 128
        args.tag = args.tag or "_quick"
    if not torch.cuda.is_available():
        raise SystemExit("[p2] refusing to run on CPU")
    dev = "cuda"
    torch.manual_seed(args.seed)
    B = load_bench("colormnist", seed=args.seed, device=dev, probe_epochs=10 if args.quick else 30)
    base = B.model.eval()
    bk = Backbone(model=base, kind="unet", layout=B.layout, img=B.img, clip=B.clip, space="model",
                  null_id=int(base.null_id), emb_dim=int(base.emb_dim),
                  n_params=int(sum(q.numel() for q in base.parameters())), device=dev)
    u_grid = tuple(round(t / 1000, 4) for t in range(50, 401, 25))      # the paper's ColorMNIST grid, in u
    tpath = args.tree or os.path.join(ROOT, "results", "trees", f"tree_colormnist_s{args.seed}.npz")
    if os.path.exists(tpath):
        tree = Tree.load(tpath)
    else:
        own = sample(bk, args.n_tree_samples, steps=args.steps, seed=args.seed)
        tree = build(bk, u_grid=DEFAULT_GRID, samples=own, seed=args.seed, iters=150, lr=0.1, steps=args.steps)
        os.makedirs(os.path.dirname(tpath), exist_ok=True)
        tree.save(tpath)
    print(f"[p2] tree {tree.summary()}", flush=True)
    rng = np.random.default_rng(args.seed)
    cells = [B.held_comb[i] for i in rng.permutation(len(B.held_comb))[:args.n_cells]]
    rows = []
    for ci, cell in enumerate(cells):
        X = B.images(cell)
        q_idx = int(rng.integers(len(X)))
        xq = B.to_m(X[q_idx:q_idx + 1])[0]
        ref = np.delete(X, q_idx, 0)[:100]
        arms = {}
        # (a) tree
        t0 = time.time()
        code = decompose(tree, xq, K=args.K)
        arms["tree"] = dict(mu=code.mu, var=code.var, secs=time.time() - t0, nodes=code.nodes, bytes=code.bytes())
        # (b) per-query ascent pool
        t0 = time.time()
        M, V = ascent_pool(bk, xq, u_grid, args.ascent_starts, args.ascent_iters, args.ascent_lr, seed=args.seed + 17 * ci)
        mu, var, sel = poe_from_pool(xq, M, V, K=args.K)
        arms["ascent"] = dict(mu=mu, var=var, secs=time.time() - t0, nodes=[int(s) for s in sel], bytes=int(4 * xq.size * 2))
        # (c) tree + local refinement: a few chains from the query at two levels join the tree's pool
        t0 = time.time()
        Mr, Vr = ascent_pool(bk, xq, (0.1, 0.3), args.refine_starts, args.refine_iters, args.ascent_lr, seed=args.seed + 31 * ci)
        from lifelong_ood.operators.decompose import candidates
        ids, lv, Mt, Vt = candidates(tree)
        mu, var, sel = poe_from_pool(xq, np.concatenate([Mt, Mr]), np.concatenate([Vt, Vr]), K=args.K)
        arms["tree+refine"] = dict(mu=mu, var=var, secs=time.time() - t0, nodes=[int(s) for s in sel],
                                   bytes=int(4 * args.K + sum(4 * xq.size * 2 for s in sel if s >= len(ids))))
        rec = dict(cell=list(cell), query=q_idx)
        for name, a in arms.items():
            eps_fn, w_fn = guide(a["mu"], a["var"], w0=args.w0, wmax=args.wmax)
            t0 = time.time()
            S = B.to_d(sample(bk, args.n_samples, steps=args.steps, seed=3000 + ci, guide=(eps_fn, w_fn)))
            acc = B.acc(S, cell)
            dc = density_coverage(S, ref, space="inception", img=B.img, device=dev)
            rec[name] = dict(joint=acc["joint"], acc={nm: acc[nm] for nm in B.names}, coverage=dc["coverage"],
                             density=dc["density"], vendi=float(vendi_score(S)), secs_discover=a["secs"],
                             secs_sample=time.time() - t0, code_bytes=a["bytes"], nodes=a["nodes"])
        rows.append(rec)
        print(f"[p2] cell {cell}: " + " | ".join(f"{n} joint {rec[n]['joint']:.2f} cov {rec[n]['coverage']:.2f} "
                                                   f"disc {rec[n]['secs_discover']:.1f}s" for n in arms), flush=True)
    summary = {n: dict(joint=float(np.mean([r[n]["joint"] for r in rows])), coverage=float(np.mean([r[n]["coverage"] for r in rows])),
                       vendi=float(np.mean([r[n]["vendi"] for r in rows])), secs_discover=float(np.mean([r[n]["secs_discover"] for r in rows])),
                       code_bytes=float(np.mean([r[n]["code_bytes"] for r in rows]))) for n in ("tree", "ascent", "tree+refine")}
    ratio = summary["tree"]["joint"] / max(summary["ascent"]["joint"], 1e-6)
    ratio_r = summary["tree+refine"]["joint"] / max(summary["ascent"]["joint"], 1e-6)
    summary["phase1_pass"] = bool(max(ratio, ratio_r) >= 0.8)
    print(f"[p2] summary {summary}\n[p2] phase 1 {'PASS' if summary['phase1_pass'] else 'KILL'} "
          f"(tree/ascent {ratio:.2f}, refined {ratio_r:.2f})", flush=True)
    for n in ("tree", "ascent", "tree+refine"):
        ledger.record("p2_decompose", bench="colormnist", tag=args.tag, seed=args.seed, keys={"arm": n},
                      metrics=dict(summary[n], n_cells=len(rows), phase1_pass=summary["phase1_pass"], per_cell=[r[n] | {"cell": r["cell"]} for r in rows]),
                      bytes=charge(codes=[int(summary[n]["code_bytes"])], tree=tree.bytes() if n != "ascent" else 0),
                      config=vars(args), script="experiments/p2_decompose.py", command=" ".join(sys.argv),
                      backbone=dict(bench="colormnist", n_params=bk.n_params, layout=bk.layout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
