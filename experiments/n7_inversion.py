"""N7 — does conditioning the manifold beat guiding it toward a Gaussian?

The architecture under test: **the manifold follows the teacher, and sampling follows the manifold.**
The PoE teacher locates the concept; the model's own class conditional generates it.

Arms, all on the same trained backbone and the same held-out queries:

  `teacher-guided`   what we shipped: CFG toward the analytic PoE Gaussian.  RESULTS 5e showed this
                     cannot cover a class no matter how it is tuned (recall 0.00 across 7 guidance
                     schedules, 2 covariance models, 3 variance scales, ancestral sampling, and
                     pooling over queries).
  `inversion`        learn c_new in the span of the trained class embeddings from the single query,
                     then sample the model conditioned on it.
  `inversion+init`   the same, but alpha initialized from the teacher's ranking of trained classes,
                     so composition decides where the search starts.
  `inversion-simplex` alpha constrained to a convex combination -- cannot leave the hull of trained
                     embeddings, so diversity is inherited by construction but the concept cannot
                     extrapolate past what the model already knows.
  `nearest-class`    condition on the single best-matching trained class embedding.  The control
                     that says whether inversion is doing anything a lookup could not.

CEILINGS reported alongside: the base model on a SEEN class (what conditioning can achieve at best)
and the real held-out images against themselves (what the data itself scores).

Scored with the paper's metric set -- k-NN precision / recall / F1 -- plus composition accuracy and
content diversity, because a method can win recall by generating the wrong class diversely.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry, save_grid, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, learn_embedding,  # noqa: E402
                                 load_field, sample_embedding, teacher_init_alpha,
                                 to_data, to_model)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def score(probes, S01, q, R, tag):
    pr = precision_recall(S01, R)
    g = sample_geometry(S01, R[0], R)
    a = composition_accuracy(probes, S01, q["d"], q["f"], q["b"])
    return dict(arm=tag, joint=a["joint"], digit=a["digit"], fg=a["fg"], bg=a["bg"],
                precision=pr["precision"], recall=pr["recall"], f1=pr["f1"],
                content_ratio=g.get("content_ratio", float("nan")),
                diversity=g["diversity"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--inv-lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, nargs="+", default=[1.5, 2.5])
    p.add_argument("--init-temp", type=float, default=0.02)
    p.add_argument("--pool-n", type=int, default=64)
    p.add_argument("--anchor", type=float, nargs="+", default=[0.5, 2.0, 10.0])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_samples, args.n_per_slot = 2, 12, 20
        args.steps, args.inv_steps, args.pool_n = 20, 60, 16
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    qs = ood_queries(data, n_slots=args.n_slots, seed=args.seed)
    print(f"[n7] {len(qs)} held-out compositions, {args.n_samples} samples each, device {dev}",
          flush=True)

    # ── ceilings ─────────────────────────────────────────────────────────────
    seen_slot = int(sorted(remap)[0])
    idx = np.where(data["slot"] == seen_slot)[0]
    Rs = data["X"][idx[:100]]
    d0, f0, b0 = seen_slot // 16, (seen_slot % 16) // 4, seen_slot % 4
    S_base = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                  clip=(-1, 1), branch=remap[seen_slot])).astype(np.float32)
    ceil_model = score(probes, S_base, dict(d=d0, f=f0, b=b0), Rs, "CEILING base model, seen class")
    half = len(Rs) // 2
    pr = precision_recall(Rs[:half], Rs[half:])
    print(f"\n  CEILING real images vs themselves : precision {pr['precision']:.2f} "
          f"recall {pr['recall']:.2f} F1 {pr['f1']:.2f}")
    print(f"  CEILING base model on a SEEN class: precision {ceil_model['precision']:.2f} "
          f"recall {ceil_model['recall']:.2f} F1 {ceil_model['f1']:.2f} "
          f"acc {ceil_model['joint']:.2f}\n")

    rows, per_slot = [], []
    for qi, q in enumerate(qs):
        xq01 = data["X"][q["query"]]
        xq = to_model(xq01)
        R = data["X"][q["rest"][:100]]
        t = L.compose(xq.ravel(), u_grid=(0.1, 0.2, 0.3, 0.4),
                      n_starts=8 if args.quick else 16, iters=30 if args.quick else 120)
        got = {}

        got["teacher-guided"] = to_data(L.sample_teacher(t, n=args.n_samples,
                                                         seed=args.seed + qi)).astype(np.float32)

        a0 = teacher_init_alpha(field, t, xq, temperature=args.init_temp, seed=args.seed)
        # THE TEACHER'S POOL AS THE INVERSION TARGET.  Inverting on the single query leaves 120
        # parameters underdetermined by one image, and N7's first run showed the consequence:
        # coverage recovered (recall 0.89) while the composition drifted (accuracy 0.58 -> 0.24).
        # The teacher's pool is collapsed as a SAMPLE SET but it is on-composition as a CONSTRAINT
        # -- which is the division of labour intended here.  The teacher constrains the manifold;
        # the manifold, being a learned multimodal conditional, supplies the diversity the pool lacks.
        pool_m = L.sample_teacher(t, n=args.pool_n, seed=args.seed + qi)
        targets = {"inversion": xq, "inversion+init": xq, "inversion@pool": pool_m}
        specs = [("inversion", dict()), ("inversion+init", dict(init_alpha=a0)),
                 ("inversion@pool", dict())]
        # the teacher as a PRIOR on alpha, not just a starting point: `anchor` dials between what
        # the teacher composed and what this one query happens to look like
        for anc in args.anchor:
            tag = f"inversion+anchor {anc}"
            targets[tag] = xq
            specs.append((tag, dict(init_alpha=a0, anchor=anc, anchor_alpha=a0)))
        for tag, kw in specs:
            c_new, alpha, hist = learn_embedding(field, targets[tag], n_steps=args.inv_steps,
                                                 lr=args.inv_lr, seed=args.seed, **kw)
            got[tag] = to_data(sample_embedding(field, c_new, n=args.n_samples, steps=args.steps,
                                                seed=args.seed + qi)).astype(np.float32)
            if tag == f"inversion+anchor {args.anchor[0]}":
                top = np.argsort(-alpha)[:3]
                per_slot.append(dict(slot=q["slot"], top_classes=[int(sorted(remap)[i]) for i in top],
                                     weights=[float(alpha[i]) for i in top],
                                     final_loss=float(np.mean(hist[-20:]))))
                # CFG on the LEARNED EMBEDDING -- the accuracy knob that Gaussian guidance could not
                # provide.  Both branches are now the model's own conditional, so sharpening moves
                # along the learned manifold instead of off it toward an analytic mean.
                for gw in args.guidance:
                    got[f"inversion+anchor {args.anchor[0]} w={gw}"] = to_data(
                        sample_embedding(field, c_new, n=args.n_samples, steps=args.steps,
                                         seed=args.seed + qi, guidance=gw)).astype(np.float32)

        best_cid = int(np.argmax(a0))
        got["nearest-class"] = to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                    seed=args.seed + qi, clip=(-1, 1),
                                                    branch=best_cid)).astype(np.float32)

        for tag, S in got.items():
            rows.append(dict(score(probes, S, q, R, tag), slot=q["slot"]))
        if qi < 3:
            grid = [xq01]
            for tag in ("teacher-guided", f"inversion+anchor {args.anchor[0]}"):
                grid += list(got[tag][:7])
            save_grid(grid, os.path.join(FIG, f"n7_slot{q['slot']}.png"), ncol=8,
                      title=f"{slot_name(q['d'], q['f'], q['b'])}: query | 7 teacher-guided "
                            f"| 7 inversion")
        print(f"[n7] slot {q['slot']:3d}: " + "  ".join(
            f"{k}={[r for r in rows if r['slot'] == q['slot'] and r['arm'] == k][0]['recall']:.2f}"
            for k in got), flush=True)

    arms = (["teacher-guided", "inversion", "inversion+init", "inversion@pool"]
            + [f"inversion+anchor {a}" for a in args.anchor]
            + [f"inversion+anchor {args.anchor[0]} w={g}" for g in args.guidance]
            + ["nearest-class"])
    summary = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        summary[a] = {k: float(np.mean([r[k] for r in rs]))
                      for k in ("joint", "digit", "fg", "bg", "precision", "recall", "f1",
                                "content_ratio")}
    save_json(dict(rows=rows, summary=summary, ceiling_model=ceil_model,
                   ceiling_real=pr, alphas=per_slot),
              os.path.join(RESULTS, "n7_inversion.json"), seed=args.seed)

    print(f"\n[n7] {len(qs)} held-out compositions")
    print(f"    {'arm':20s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'content div/ref':>17s}")
    for a in arms:
        v = summary[a]
        print(f"    {a:20s}{v['joint']:6.2f}{v['digit']:7.2f}{v['fg']:5.2f}{v['bg']:5.2f}"
              f"{v['precision']:7.2f}{v['recall']:8.2f}{v['f1']:7.2f}{v['content_ratio']:17.2f}")
    print(f"    {'-- ceiling: base model on a seen class':46s}"
          f"{ceil_model['precision']:7.2f}{ceil_model['recall']:8.2f}{ceil_model['f1']:7.2f}"
          f"{ceil_model['content_ratio']:17.2f}")
    if per_slot:
        print("\n  what the composition chose (top trained classes by weight):")
        for s in per_slot[:4]:
            print(f"    slot {s['slot']:3d} -> classes {s['top_classes']} "
                  f"w={[round(x, 2) for x in s['weights']]}")
    print(f"[n7] wrote {RESULTS}/n7_inversion.json")


if __name__ == "__main__":
    main()
