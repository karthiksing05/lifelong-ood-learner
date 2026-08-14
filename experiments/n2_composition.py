"""N2 — test-time compositional generalization on the NEURAL backbone (poe-diffusion Section 4).

The same method as E2, run against a trained DDPM instead of the analytic tree, which lets us ask
the question the analytic track could not answer cleanly: how much of the composition gap is the
METHOD and how much is the backbone's ability to hold density at an unseen combination?

Discovery here is necessarily the paper's route -- mode ascent on the learned score at several noise
levels (their Eq. 4) -- because a UNet has no concept table to rank.  Baselines follow the paper:

  poe         discovered prototypes -> submodular selection -> per-dimension PoE -> guided sampling
  query_only  one isotropic Gaussian around the query, through the same sampler
  top1/top3   the paper's trained-class baselines: score every SEEN class by its average DDPM loss
              on the query, take the best k, and condition on them (k=3 composes the three scores)

Reported with the same metric set as E2 -- composition accuracy plus the novelty/diversity/gen
geometry -- so the two backbones sit in the same table.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.compose import cfg_weight, teacher_eps               # noqa: E402
from lifelong_ood.data.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, sample_geometry,  # noqa: E402
                                save_grid, save_json)
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model  # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402


def class_loss(field, xq_model, cid, u_grid=(0.2, 0.4, 0.6), seed=0, reps=2):
    """The paper's class score: average epsilon-prediction error of x_q under class `cid`."""
    rng = np.random.default_rng(seed)
    x = np.atleast_2d(xq_model)
    tot = 0.0
    for u in u_grid:
        a = float(cosine_abar(u))
        for _ in range(reps):
            e = rng.standard_normal(x.shape)
            xt = np.sqrt(a) * x + np.sqrt(1 - a) * e
            tot += float(np.mean((field.eps(xt, u, cid) - e) ** 2))
    return tot / (len(u_grid) * reps)


def topk_eps(field, cids, weights):
    """Composable-diffusion style combination of several trained class conditionals."""
    def _eps(x, u):
        out = None
        for c, w in zip(cids, weights):
            e = field.eps(x, u, int(c))
            out = w * e if out is None else out + w * e
        return out
    return _eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-slots", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk-classes", type=int, default=24,
                   help="candidate seen classes scored for the top-k baselines")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_samples, args.n_per_slot, args.topk_classes = 2, 8, 20, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    print(f"[n2] backbone {args.ckpt}, {len(remap)} seen classes, device {dev}", flush=True)

    qs = ood_queries(data, n_slots=args.n_slots, seed=args.seed)
    cand = np.array(sorted(remap.values()))[:args.topk_classes]
    rows = []
    for qi, q in enumerate(qs):
        xq01 = data["X"][q["query"]]
        xq = to_model(xq01).ravel()
        ref = data["X"][q["rest"][:100]]
        res = {}

        teacher = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4),
                            n_starts=8 if args.quick else 16,
                            iters=40 if args.quick else 120)
        res["poe"] = L.sample_teacher(teacher, n=args.n_samples, seed=args.seed + qi)
        res["query_only"] = L.sample_teacher(L.baseline_teacher(xq), n=args.n_samples,
                                             seed=args.seed + qi)

        losses = np.array([class_loss(field, xq, int(c), seed=args.seed) for c in cand])
        order = np.argsort(losses)
        for k in (1, 3):
            cids = cand[order[:k]]
            w = np.exp(-(losses[order[:k]] - losses[order[0]]) / 0.01)
            w = w / w.sum()
            res[f"top{k}"] = field.sample(n=args.n_samples, steps=args.steps, seed=args.seed + qi,
                                          u_start=1.0, clip=(-1, 1),
                                          guide_eps=topk_eps(field, cids, w),
                                          guide_w=lambda u: 1.0)

        row = dict(slot=q["slot"], name=slot_name(q["d"], q["f"], q["b"]))
        for name, Sm in res.items():
            S = to_data(Sm).astype(np.float32)
            row[name] = dict(composition_accuracy(probes, S, q["d"], q["f"], q["b"]),
                             **sample_geometry(S, xq01, ref))
            if qi < 2:
                save_grid(np.concatenate([[xq01], S[:15]]),
                          os.path.join(RESULTS, f"n2_slot{q['slot']}_{name}.png"),
                          title=f"{row['name']} — {name} (neural; first tile = query)")
        rows.append(row)
        print(f"[n2] slot {q['slot']:3d} ({row['name']}): "
              + "  ".join(f"{k}={row[k]['joint']:.2f}" for k in res), flush=True)

    keys = ("digit", "fg", "bg", "joint", "novelty", "diversity", "gen_rmse", "ref_spread")
    summary = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in keys if m in rows[0][k]}
               for k in ("poe", "query_only", "top1", "top3")}
    save_json(dict(rows=rows, summary=summary), os.path.join(RESULTS, "n2_composition.json"),
              seed=args.seed)
    print(f"\n[n2] {len(rows)} OOD slots on the neural backbone")
    print(f"    {'method':11s} {'joint':>6s} {'digit':>6s} {'fg':>5s} {'bg':>5s} "
          f"{'novel':>7s} {'divers':>7s} {'gen':>7s}")
    for k, v in summary.items():
        print(f"    {k:11s} {v['joint']:6.3f} {v['digit']:6.2f} {v['fg']:5.2f} {v['bg']:5.2f} "
              f"{v['novelty']:7.3f} {v['diversity']:7.3f} {v['gen_rmse']:7.3f}")
    print(f"    (reference set internal spread {summary['poe']['ref_spread']:.3f})")


if __name__ == "__main__":
    main()
