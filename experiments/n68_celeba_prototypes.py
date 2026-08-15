"""N68 -- fix the CelebA prototype pool, which has collapsed in DISCOVERY.

`tests/diag_prototype_diversity.py` measures the CelebA pool at **0.30x** the spread of real images
of the held-out class, against ColorMNIST's 1.51x, with all three selected prototypes coming from
query-seeded chains.  Two causes compound, and both are in the paper's own CelebA column:

    PAPER_CELEBA = u_grid 10 levels, n_starts=8, iters=100, lr=0.02, init="query"
    PAPER_COLORMNIST = u_grid 15 levels, n_starts=128, iters=150, lr=0.1, init="hybrid"

`init="query"` means EVERY chain starts at the query, so there are no query-agnostic candidates for
submodular selection to pick -- Eq. 8-9 exists to find experts that DISAGREE and is being handed
only experts that agree.  And 8 starts per level is 16x thinner than the ColorMNIST column, so even
among query-seeded chains the pool barely spreads.

The consequence is not cosmetic.  Eq. 7's per-dimension product over three near-copies of one mode
IS that mode, so the teacher cannot compose, and any arm reading the teacher inherits the failure --
including the N67 hybrid, which reads `mu_T` for the factor values.

So this sweeps the discovery settings and scores each on what actually matters downstream:

  pool_spread    pairwise RMSE across the pool, over the spread of REAL images of the class
  sel_spread     the same for the K selected -- what the product actually sees
  n_random       how many query-agnostic candidates exist at all (0 makes selection a formality)
  read_correct   whether mu_T probe-reads the held-out ATTRIBUTES -- the number the hybrid needs

`read_correct` is the target, not diversity for its own sake: a diverse pool that identifies the
wrong composition is no better than a collapsed one.

    python experiments/n68_celeba_prototypes.py
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.compose import (PAPER_CELEBA, candidates_ascent,     # noqa: E402
                                  poe_teacher)
from lifelong_ood.evals import save_json                               # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data, to_model     # noqa: E402


def rmse_pairs(A, cap=24):
    A = A[:cap]
    if len(A) < 2:
        return float("nan")
    return float(np.mean([float(np.sqrt(np.mean((A[i] - A[j]) ** 2)))
                          for i in range(len(A)) for j in range(i + 1, len(A))]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba128.pt"))
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from lifelong_ood.data.celeba import load_celeba
    from lifelong_ood.evals import fit_attr_probes
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    field = load_field(args.ckpt, device=dev)
    d = load_celeba(img=int(ck.get("img", 128)), min_count=int(ck.get("min_count", 120)),
                    seed=int(ck.get("seed", 0)))
    cls = np.asarray(d["cls"] if "cls" in d else d["slot"])
    X = np.asarray(d["X"], np.float32)
    A = np.asarray(d["attrs"])
    probes = fit_attr_probes(d, seed=args.seed) if "attrs" in d else None
    hold = [int(h) for h in np.asarray(ck.get("hold_ids", [])).ravel()]
    cases = [h for h in hold if (cls == h).any()][:args.n_cases]
    print(f"[n68] {len(cases)} held-out classes: {cases}", flush=True)

    # the paper's CelebA column, then the two knobs it differs from their ColorMNIST column by
    grid = []
    for init, ns in itertools.product(("query", "hybrid"), (8, 32, 64)):
        grid.append(dict(PAPER_CELEBA, init=init, n_starts=ns))
    rows = []
    for cfg in grid:
        tag = f"init={cfg['init']:6s} n_starts={cfg['n_starts']:<3d}"
        ps, ss, nr, rc = [], [], [], []
        for h in cases:
            idx = np.where(cls == h)[0]
            xq = to_model(X[int(idx[0])]).ravel()
            M, V, meta = candidates_ascent(field, xq, seed=args.seed, **cfg)
            t = poe_teacher(xq, M, V, K=3, tau=0.05)
            real = np.stack([to_model(X[int(i)]).ravel() for i in idx[:6]])
            scale = max(rmse_pairs(real), 1e-9)
            ps.append(rmse_pairs(M) / scale)
            ss.append(rmse_pairs(M[t["sel"]]) / scale)
            nr.append(int(np.sum(np.asarray(meta["src"]) == "random")))
            if probes is not None:
                mu = to_data(t["mu"]).reshape(1, -1)
                tgt = A[int(idx[0])]
                hit = np.mean([int(probes[k].predict(mu)[0]) == int(tgt[k])
                               for k in range(len(probes))]) if len(probes) else float("nan")
                rc.append(float(hit))
        row = dict(init=cfg["init"], n_starts=cfg["n_starts"],
                   pool_spread=float(np.mean(ps)), sel_spread=float(np.mean(ss)),
                   n_random=float(np.mean(nr)),
                   read_correct=float(np.mean(rc)) if rc else float("nan"))
        rows.append(row)
        print(f"[n68] {tag} | pool {row['pool_spread']:.2f}x  selected {row['sel_spread']:.2f}x  "
              f"query-agnostic candidates {row['n_random']:.0f}  "
              f"mu_T attr-read {row['read_correct']:.3f}", flush=True)

    print(f"\n[n68] {'=' * 70}")
    base = next(r for r in rows if r["init"] == "query" and r["n_starts"] == 8)
    best = max(rows, key=lambda r: (r["read_correct"] if r["read_correct"] == r["read_correct"]
                                    else 0, r["sel_spread"]))
    print(f"[n68] paper's own CelebA column: pool {base['pool_spread']:.2f}x, "
          f"selected {base['sel_spread']:.2f}x, mu_T attr-read {base['read_correct']:.3f}")
    print(f"[n68] best swept:  init={best['init']} n_starts={best['n_starts']} -> "
          f"pool {best['pool_spread']:.2f}x, selected {best['sel_spread']:.2f}x, "
          f"mu_T attr-read {best['read_correct']:.3f}")
    if best["sel_spread"] > 2 * base["sel_spread"] or best["read_correct"] > base["read_correct"]:
        print("[n68] VERDICT: the collapse is a SETTING, not the backbone.  CelebA discovery should")
        print("[n68]   use hybrid initialisation and a ColorMNIST-scale start count -- which is what")
        print("[n68]   the paper itself uses on ColorMNIST.  Deviating from their CelebA column is")
        print("[n68]   justified by this measurement and must be recorded as a deviation.")
    else:
        print("[n68] VERDICT: no setting in this sweep decollapses the pool; the CelebA backbone's")
        print("[n68]   density has one dominant basin and prototype discovery cannot work on it.")
    save_json(dict(rows=rows, cases=cases, config=vars(args)),
              os.path.join(RESULTS, "n68_celeba_prototypes.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
