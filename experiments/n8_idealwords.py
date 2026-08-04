"""N8 — compose in the FACTOR basis, not the class basis.

The span oracle (RESULTS 5h) fitted weights over the 120 trained class embeddings using 64 real
images of a held-out slot and still could not place the composition -- it recovered the background or
the foreground, never both.  The natural reading was that the conditioning space cannot express these
pairings.  That reading was wrong, and a one-line measurement shows why: the embedding table is
approximately ADDITIVELY FACTORED,

    c(digit, fg, bg)  ~  mu + D_digit + F_fg + B_bg      R^2 = 0.816

so a held-out combination *is* expressible -- as a sum of vectors that each appear in the training
data.  The oracle failed because a free weighting over 120 entangled class vectors can trade one
factor off against another, and with a single global conditioning vector nothing stops it.  The fix
is to search the factor basis instead ("ideal words", Trager et al. ICCV 2023), where composition is
structural rather than something the optimizer has to discover.

Arms:
  `teacher-guided`     the fidelity reference (accurate, zero coverage)
  `inversion`          weights over 120 class embeddings -- the previous best
  `factored`           weights over the ~18 FACTOR vectors, learned from the single query
  `factored+init`      the same, warm-started from the teacher's read of the composition
  `ideal-words ORACLE` compose mu + D + F + B directly from the TRUE labels: no query, no
                       optimization.  An upper bound on what the factor structure supports, and
                       labelled as such -- it uses information the method does not receive.
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
                                 load_field, sample_embedding, to_data, to_model)
from lifelong_ood.neural.idealwords import (compose, factorize, learn_factored,  # noqa: E402
                                            slot_factors)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def score(probes, S01, q, R, tag):
    pr = precision_recall(S01, R)
    g = sample_geometry(S01, R[0], R)
    a = composition_accuracy(probes, S01, q["d"], q["f"], q["b"])
    return dict(arm=tag, slot=q["slot"], joint=a["joint"], digit=a["digit"], fg=a["fg"], bg=a["bg"],
                precision=pr["precision"], recall=pr["recall"], f1=pr["f1"],
                content_ratio=g.get("content_ratio", float("nan")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--inv-lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_samples, args.n_per_slot = 2, 12, 20
        args.steps, args.inv_steps = 20, 60
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)

    # ── the factorization, fitted on the SEEN slots only ─────────────────────
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots]]
    fac = factorize(C, slot_factors(slots))
    print(f"[n8] additive factorization of the class-embedding table: R^2 = {fac['r2']:.4f}")
    print(f"[n8] factor vector norms: digit {np.linalg.norm(fac['V'][0], axis=1).mean():.3f}  "
          f"fg {np.linalg.norm(fac['V'][1], axis=1).mean():.3f}  "
          f"bg {np.linalg.norm(fac['V'][2], axis=1).mean():.3f}\n", flush=True)

    qs = ood_queries(data, n_slots=args.n_slots, seed=args.seed)
    rows = []
    for qi, q in enumerate(qs):
        xq01 = data["X"][q["query"]]
        xq = to_model(xq01)
        R = data["X"][q["rest"][:100]]
        got = {}

        t = L.compose(xq.ravel(), u_grid=(0.1, 0.2, 0.3, 0.4),
                      n_starts=8 if args.quick else 16, iters=30 if args.quick else 120)
        got["teacher-guided"] = to_data(L.sample_teacher(t, n=args.n_samples,
                                                         seed=args.seed + qi)).astype(np.float32)

        c_cls, _, _ = learn_embedding(field, xq, n_steps=args.inv_steps, lr=args.inv_lr,
                                      seed=args.seed)
        got["inversion (class basis)"] = to_data(sample_embedding(
            field, c_cls, n=args.n_samples, steps=args.steps, seed=args.seed + qi)).astype(np.float32)

        c_fac, ws, _ = learn_factored(field, xq, fac, n_steps=args.inv_steps, lr=args.inv_lr,
                                      seed=args.seed)
        got["factored (ideal words)"] = to_data(sample_embedding(
            field, c_fac, n=args.n_samples, steps=args.steps, seed=args.seed + qi)).astype(np.float32)

        # ORACLE: compose straight from the true factor values -- no query, no optimization
        c_or = compose(fac, (q["d"], q["f"], q["b"]))
        got["ideal-words ORACLE"] = to_data(sample_embedding(
            field, torch.as_tensor(c_or, dtype=torch.float32, device=dev),
            n=args.n_samples, steps=args.steps, seed=args.seed + qi)).astype(np.float32)

        for tag, S in got.items():
            rows.append(score(probes, S, q, R, tag))
        pred = [int(np.argmax(w)) for w in ws]
        print(f"[n8] slot {q['slot']:3d} ({slot_name(q['d'], q['f'], q['b'])}): "
              + "  ".join(f"{k.split()[0]}={[r for r in rows if r['slot'] == q['slot'] and r['arm'] == k][0]['joint']:.2f}"
                          for k in got)
              + f"  | factored read (d,f,b)=({pred[0]},{pred[1]},{pred[2]}) "
                f"true=({q['d']},{q['f']},{q['b']})", flush=True)
        if qi < 3:
            grid = [xq01]
            for tag in ("teacher-guided", "factored (ideal words)", "ideal-words ORACLE"):
                grid += list(got[tag][:5])
            save_grid(grid, os.path.join(FIG, f"n8_slot{q['slot']}.png"), ncol=8,
                      title=f"{slot_name(q['d'], q['f'], q['b'])}: query | teacher-guided | "
                            f"factored | oracle")

    arms = ["teacher-guided", "inversion (class basis)", "factored (ideal words)",
            "ideal-words ORACLE"]
    summary = {a: {k: float(np.mean([r[k] for r in rows if r["arm"] == a]))
                   for k in ("joint", "digit", "fg", "bg", "precision", "recall", "f1",
                             "content_ratio")} for a in arms}
    save_json(dict(rows=rows, summary=summary, factorization_r2=fac["r2"]),
              os.path.join(RESULTS, "n8_idealwords.json"), seed=args.seed)

    print(f"\n[n8] {len(qs)} held-out compositions  (factorization R^2 {fac['r2']:.3f})")
    print(f"    {'arm':26s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'content div/ref':>17s}")
    for a in arms:
        v = summary[a]
        print(f"    {a:26s}{v['joint']:6.2f}{v['digit']:7.2f}{v['fg']:5.2f}{v['bg']:5.2f}"
              f"{v['precision']:7.2f}{v['recall']:8.2f}{v['f1']:7.2f}{v['content_ratio']:17.2f}")
    print(f"[n8] wrote {RESULTS}/n8_idealwords.json")


if __name__ == "__main__":
    main()
