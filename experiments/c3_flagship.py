"""C3 -- the Shapes3D flagship, measured the way the 2026-09-24 audit said it has to be.

n78 (RESULTS 7jj) reported ADDRESS 0.983 against retrieval 0.010.  The audit found that it composed
from the held-out cell's TRUE labels (no query was ever read), scored the first 12 of 40 held-out
cells in slot order (floor_hue 0-3 only), used a nearest-class baseline that differs from every target
in exactly one factor (so ~0 by construction), checked memorisation against 15 of 360 classes, and
certified a "resolution" of 0.0004 from a Wald formula evaluated at p = 1.  Numbers stand; what they
show does not.

This measures the same thing honestly, on the same backbone and split:
  * ALL 40 held-out cells;
  * ADDRESS with the factors READ -- from ONE query image and from 16 -- beside the oracle;
  * a baseline that CAN win: a 1 KB embedding learned from the same one image (textual inversion),
    against the 72 B address;
  * memorisation against a random sample spanning EVERY trained class;
  * joint accuracy with Wilson 95% intervals, Naeem coverage (reference-anchored -- pixel recall
    measures diversity, not class coverage), vendi, over `--seeds` sampling seeds.

NOTE on semantics: in this split the factor the repository calls `floor_hue` is the image's WALL hue
(loader bug, d8298e1); `c_bench` names it `wall`.  The backbone's pretraining (eurecom) contains the
held-out cells' images, so this is a test of ADDRESSING a combination the generator can already draw,
not of generating one it never saw.

    python experiments/c3_flagship.py --seeds 3
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c_bench import load_bench                                        # noqa: E402
from common import RESULTS                                            # noqa: E402

from lifelong_ood.evals import density_coverage, memorization, save_json, vendi_score  # noqa: E402
from lifelong_ood.neural.cobweb_update import (candidate_losses, ddim_sample, eps_loss,  # noqa: E402
                                               to_img)

U_GRID = np.round(np.arange(0.1, 1.0, 0.1), 2)


def wilson(k, n, z=1.96):
    p = k / max(n, 1)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-read", type=int, default=16)
    p.add_argument("--n-ref", type=int, default=60)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--n-mem", type=int, default=4000, help="memorisation reference, all classes")
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("[c3] refusing to run on CPU")
    dev = "cuda"
    t0 = time.time()
    B = load_bench("shapes3d", seed=0, device=dev)
    model = B.model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    kw = dict(layout=B.layout, img=B.img, device=dev)
    fac = B.fac
    mu = np.asarray(fac["mu"], np.float32)
    V = [np.asarray(v, np.float32) for v in fac["V"]]
    nF = len(V)
    combos = list(itertools.product(*[range(len(v)) for v in V]))
    E_all = np.stack([mu + sum(V[f][c[f]] for f in range(nF)) for c in combos]).astype(np.float32)
    seen_c = np.array([c for c, _ in B.seen])
    seen_t = [t for _, t in B.seen]
    C_seen = B.model.cemb.weight.detach().cpu().numpy()[seen_c]
    cells = B.held_comb if args.max_cells is None else B.held_comb[:args.max_cells]
    rng = np.random.default_rng(0)
    # memorisation reference spanning EVERY trained class (the audit: n78 used 15 of 360)
    pool = np.concatenate([B.images(t) for t in seen_t])
    mem_ref = pool[rng.permutation(len(pool))[:args.n_mem]]
    del pool
    print(f"[c3] {len(cells)} held-out cells; basis R2 {fac['r2']:.3f}; memorisation reference "
          f"{len(mem_ref)} images over {len(seen_t)} classes ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for ci, cell in enumerate(cells):
        X = B.images(cell)
        q1, q16 = X[:1], X[:args.n_read]
        ref = X[-args.n_ref:]
        assert args.n_read + args.n_ref <= len(X), f"{len(X)} images for {cell}"
        L1 = candidate_losses(model, B.to_m(q1), U_GRID, embs=E_all, seed=11 + ci, batch=256,
                              **kw)
        L16 = candidate_losses(model, B.to_m(q16), U_GRID, embs=E_all, seed=11 + ci, n_noise=1,
                               batch=256, **kw)
        r1, r16 = combos[int(np.argmin(L1))], combos[int(np.argmin(L16))]
        e_true = mu + sum(V[f][cell[f]] for f in range(nF))
        j_near = int(np.argmin(((C_seen - e_true[None]) ** 2).sum(1)))
        # textual inversion from the SAME one image: a free 1 KB vector, initialised at the mean
        e = torch.as_tensor(mu, device=dev).clone().requires_grad_(True)
        opt = torch.optim.Adam([e], lr=0.02)
        xq = to_img(torch.as_tensor(B.to_m(q1), device=dev), B.img, B.layout).expand(8, -1, -1, -1)
        g = torch.Generator(device="cpu").manual_seed(ci)
        for _ in range(args.inv_steps):
            loss = eps_loss(model, xq, emb=e[None].expand(8, -1), gen=g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        e_inv = e.detach().cpu().numpy()
        arms = {"address (oracle labels)": dict(emb=e_true),
                "address (read, 1 image)": dict(emb=E_all[combos.index(r1)]),
                f"address (read, {args.n_read} images)": dict(emb=E_all[combos.index(r16)]),
                "inversion (1 KB, 1 image)": dict(emb=e_inv),
                "nearest trained class": dict(cls=int(seen_c[j_near])),
                "unconditional": dict(cls=int(model.null_id))}
        for seed in range(args.seeds):
            for arm, cond in arms.items():
                S = B.to_d(ddim_sample(model, args.n_samples, steps=args.steps,
                                       seed=1000 * seed + ci, clip=B.clip, **cond, **kw))
                a = B.acc(S, cell)
                dc = density_coverage(S, ref)
                rows.append(dict(cell=list(cell), seed=seed, arm=arm, joint=a["joint"],
                                 k=int(round(a["joint"] * args.n_samples)), n=args.n_samples,
                                 acc={nm: a[nm] for nm in B.names}, coverage=dc["coverage"],
                                 density=dc["density"], vendi=float(vendi_score(S)),
                                 memorized=memorization(S, mem_ref)["frac_memorized"],
                                 read1=list(r1), read16=list(r16),
                                 near=list(seen_t[j_near])))
        print(f"[c3] {ci + 1:2d}/{len(cells)} {cell}: read1 {r1} {'OK' if r1 == tuple(cell) else 'x'} "
              f"read16 {r16} {'OK' if r16 == tuple(cell) else 'x'} near {seen_t[j_near]} | "
              + " ".join(f"{a_.split(' (')[0][:9]}={np.mean([r['joint'] for r in rows if r['arm'] == a_ and r['cell'] == list(cell)]):.2f}"
                         for a_ in arms) + f" ({time.time() - t0:.0f}s)", flush=True)

    print("\n[c3] ===== pooled over cells and seeds (joint: Wilson 95%)")
    summary = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        k, n = sum(r["k"] for r in rs), sum(r["n"] for r in rs)
        per_cell = [np.mean([r["joint"] for r in rs if r["cell"] == c]) for c in map(list, cells)]
        summary[arm] = dict(joint=k / n, ci=wilson(k, n), cells_ge_half=float(np.mean(
            np.asarray(per_cell) >= 0.5)), coverage=float(np.mean([r["coverage"] for r in rs])),
            vendi=float(np.mean([r["vendi"] for r in rs])),
            memorized=float(np.mean([r["memorized"] for r in rs])))
        s = summary[arm]
        print(f"  {arm:30s} joint {s['joint']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  "
              f"cells>=0.5 {s['cells_ge_half']:.2f}  coverage {s['coverage']:.3f}  vendi "
              f"{s['vendi']:.2f}  memorised {s['memorized']:.3f}")
    reads = [(tuple(r["read1"]) == tuple(r["cell"]), tuple(r["read16"]) == tuple(r["cell"]))
             for r in rows if r["seed"] == 0 and r["arm"] == "unconditional"]
    summary["read_acc_1"] = float(np.mean([a for a, _ in reads]))
    summary[f"read_acc_{args.n_read}"] = float(np.mean([b for _, b in reads]))
    print(f"  read accuracy: 1 image {summary['read_acc_1']:.3f}, {args.n_read} images "
          f"{summary[f'read_acc_{args.n_read}']:.3f} (over {len(reads)} cells)")
    save_json(dict(rows=rows, summary=summary, config=vars(args), cells=[list(c) for c in cells],
                   basis_r2=float(fac["r2"])),
              os.path.join(RESULTS, f"c3_flagship{args.tag}.json"), seed=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
