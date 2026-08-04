"""N6 — recover the paper's sample diversity by restoring the schedule we replaced.

WHY THIS EXPERIMENT EXISTS.  The paper's PoE reaches **recall 80.7%** on ColorMNIST while its
query-only baseline reaches **2.5%**, and it explicitly describes query-only as the arm that
"repeats near copies of x_q".  Vetting our case studies showed our PoE doing exactly that: content
diversity 0.45x the real class spread, and diversity/novelty 1.04 where independent draws would give
1.41.  We had become their failing baseline, and the cause is a schedule we substituted.

THE MECHANISM WE BROKE.  Their Eq. 14 sets `w*(t) = min(w0 * tr(Sigma_t)/d, wmax)` with
`Sigma_t = abar_t * Sigma_T + (1 - abar_t) I`.  Read it at the two ends:

    high noise  abar -> 0,  Sigma_t -> I,        so w -> w0     : the TEACHER steers, choosing which
                                                                  composition the trajectory heads to
    low noise   abar -> 1,  Sigma_t -> Sigma_T,  so w -> w0*mean(Sigma_T) ~ 0 for a sharp teacher
                                                                : guidance VANISHES and the
                                                                  unconditional model renders the
                                                                  instance detail

So their schedule is a hand-off: the teacher picks the mode, the diffusion model paints a different
member of it every time.  That is where recall 80.7% comes from.  It is also SELF-CORRECTING for
teacher sharpness -- a tighter Sigma_T hands off *earlier*, so a sharp PoE costs diversity nothing.

Our `anneal` schedule runs the other way (w from wmin at high noise up to wmax at low noise), which
clamps every trajectory onto mu_T at exactly the point where instance detail is decided.  Annealing
was a justified fix for the ANALYTIC backbone, whose Gaussian mixture has no density at an unseen
composition and therefore fights it; it was then carried to the neural backbone by default, where
the unconditional branch is a good UNet that should be trusted late.  `n0_tune` never caught this
because it only ever swept parameters *within* the annealed schedule.

It also reframes an earlier finding: the "teacher covariance is 13.6x too tight" result was a
symptom of the wrong schedule, not a defect needing the `teacher_var_scale` correction we added.
Under Eq. 14 a tight teacher is fine.  This sweep tests that prediction directly.

Scored with the paper's own metric set -- precision, recall and F1 from a k-NN density estimator --
alongside composition accuracy and the content-diversity measure that exposed the problem.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.datasets import load_colormnist, rgb_img             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry, save_grid, save_json)
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pool-queries", type=int, default=24)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_samples, args.n_per_slot, args.steps = 3, 12, 20, 20
        args.n_pool_queries = 6
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    qs = ood_queries(data, n_slots=args.n_slots, seed=args.seed)
    xqs = [to_model(data["X"][q["query"]]).ravel() for q in qs]
    teachers = L.compose_batch(xqs, u_grid=(0.1, 0.2, 0.3, 0.4),
                               n_starts=8 if args.quick else 16,
                               iters=30 if args.quick else 120)

    # the REAL class, scored against itself: the ceiling every row below is chasing
    ref_rows = []
    for q in qs:
        R = data["X"][q["rest"][:100]]
        half = len(R) // 2
        pr = precision_recall(R[:half], R[half:])
        ref_rows.append(dict(pr, **sample_geometry(R[:half], data["X"][q["query"]], R[half:])))
    m_ref = lambda k: float(np.mean([r[k] for r in ref_rows]))          # noqa: E731
    print(f"[n6] REAL held-out images scored against themselves (the ceiling): "
          f"precision {m_ref('precision'):.2f} recall {m_ref('recall'):.2f} "
          f"F1 {m_ref('f1'):.2f}\n", flush=True)

    # extent="poe" is the product covariance (what we shipped); extent="mixture" adds the
    # between-expert term.  Crossed with the guidance schedule, because the two interact: a broader
    # teacher also changes where Eq. 14 hands off.
    configs = [("PoE cov + anneal (shipped)", "poe", dict(schedule="anneal", wmin=0.8, wmax=2.0), 30.0),
               ("PoE cov + Eq.14", "poe", dict(schedule="variance", w0=1.2, wmax=2.0, wmin=0.0), 1.0),
               ("MIXTURE cov + Eq.14", "mixture", dict(schedule="variance", w0=1.2, wmax=2.0,
                                                       wmin=0.0), 1.0),
               ("MIXTURE cov + Eq.14 w0=2", "mixture", dict(schedule="variance", w0=2.0, wmax=2.0,
                                                            wmin=0.0), 1.0),
               ("MIXTURE cov + anneal", "mixture", dict(schedule="anneal", wmin=0.8, wmax=2.0), 1.0),
               ("MIXTURE cov + const w=1", "mixture", dict(schedule="const", wmax=1.0), 1.0),
               ("MIXTURE cov + const w=0.5", "mixture", dict(schedule="const", wmax=0.5), 1.0),
               # eta>0 = ancestral sampling: noise re-injected at every step, so diversity no
               # longer has to survive the whole deterministic trajectory from x_T alone
               ("MIX + Eq.14 + eta=0.5", "mixture", dict(schedule="variance", w0=1.2, wmax=2.0,
                                                         wmin=0.0, eta=0.5), 1.0),
               ("MIX + Eq.14 + eta=1.0", "mixture", dict(schedule="variance", w0=1.2, wmax=2.0,
                                                         wmin=0.0, eta=1.0), 1.0),
               ("PoE + anneal + eta=1.0", "poe", dict(schedule="anneal", wmin=0.8, wmax=2.0,
                                                      eta=1.0), 30.0)]

    rows = []
    print(f"    {'config':30s}{'acc':>6s}{'prec':>7s}{'recall':>8s}{'F1':>7s}"
          f"{'content div/ref':>17s}{'div/nov':>9s}{'nov':>7s}")
    for label, ext, kw, vs in configs:
        A, G, PR = [], [], []
        T2 = [dict(t, var=(t["var_mixture"] if ext == "mixture" else t["var_poe"]))
              for t in teachers]
        pools = L.sample_teacher_batch(T2, n=args.n_samples, seed=args.seed,
                                       var_scale=vs, u_start=1.0, **kw)
        for q, P in zip(qs, pools):
            S = to_data(P).astype(np.float32)
            R = data["X"][q["rest"][:100]]
            A.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"]))
            G.append(sample_geometry(S, data["X"][q["query"]], R))
            PR.append(precision_recall(S, R))
        m = lambda k, L_: float(np.mean([x[k] for x in L_]))            # noqa: E731
        r = dict(config=label, extent=ext, var_scale=vs, **kw,
                 joint=m("joint", A), precision=m("precision", PR), recall=m("recall", PR),
                 f1=m("f1", PR), content_ratio=m("content_ratio", G),
                 div_over_nov=m("div_over_nov", G), novelty=m("novelty", G),
                 diversity=m("diversity", G))
        rows.append(r)
        print(f"    {label:30s}{r['joint']:6.2f}{r['precision']:7.2f}{r['recall']:8.2f}"
              f"{r['f1']:7.2f}{r['content_ratio']:17.2f}{r['div_over_nov']:9.2f}"
              f"{r['novelty']:7.3f}", flush=True)
        if label.startswith("MIXTURE cov + Eq.14") and "w0=2" not in label or label.startswith("PoE cov + anneal"):
            tag = "fixed" if label.startswith("MIXTURE") else "shipped"
            grid = []
            for q, P in list(zip(qs, pools))[:4]:
                grid.append(data["X"][q["query"]])
                grid += list(to_data(P)[:7].astype(np.float32))
            save_grid(grid, os.path.join(os.path.dirname(RESULTS), "figures",
                                         f"n6_samples_{tag}.png"), ncol=8,
                      title=f"{label} — query then 7 samples, one row per composition")

    # ── the paper's PROTOCOL, which is not the one measured above ────────────
    # "For each OOD class, we run the full framework independently on each of the 100 queries",
    # with 16 saved generations per query, scored against 100 reference images.  So their generated
    # set is POOLED OVER 100 DIFFERENT QUERIES and inherits those queries' own diversity.  Every row
    # above instead measures ONE query's samples -- within-query diversity, which the paper never
    # reports.  If pooling closes the gap, our pipeline is not worse than theirs; it was being held
    # to a stricter test, and the honest conclusion changes.
    print("\n[n6] the paper's protocol: pool samples from MANY queries of the same OOD class")
    slot = qs[0]["slot"]
    idx = np.concatenate([[qs[0]["query"]], qs[0]["rest"]])
    pool_qs = idx[:args.n_pool_queries]
    R = data["X"][idx[args.n_pool_queries:args.n_pool_queries + 100]]
    xq_many = [to_model(data["X"][i]).ravel() for i in pool_qs]
    t_many = L.compose_batch(xq_many, u_grid=(0.1, 0.2, 0.3, 0.4),
                             n_starts=8 if args.quick else 16, iters=30 if args.quick else 120)
    per_q = max(1, args.n_samples // 4)
    P_many = L.sample_teacher_batch(t_many, n=per_q, seed=args.seed, var_scale=1.0, u_start=1.0,
                                    schedule="variance", w0=1.2, wmax=2.0, wmin=0.0)
    pooled = to_data(np.concatenate(P_many)).astype(np.float32)
    single = to_data(P_many[0]).astype(np.float32)
    pr_pooled = precision_recall(pooled, R)
    pr_single = precision_recall(single, R)
    q0 = qs[0]
    print(f"    slot {slot}, {len(pool_qs)} queries x {per_q} samples = {len(pooled)} pooled")
    print(f"    {'protocol':34s}{'prec':>7s}{'recall':>8s}{'F1':>7s}{'acc':>6s}")
    print(f"    {'ONE query (what we measured)':34s}{pr_single['precision']:7.2f}"
          f"{pr_single['recall']:8.2f}{pr_single['f1']:7.2f}"
          f"{composition_accuracy(probes, single, q0['d'], q0['f'], q0['b'])['joint']:6.2f}")
    print(f"    {chr(34)+'100 queries pooled'+chr(34)+' (the paper)':34s}{pr_pooled['precision']:7.2f}"
          f"{pr_pooled['recall']:8.2f}{pr_pooled['f1']:7.2f}"
          f"{composition_accuracy(probes, pooled, q0['d'], q0['f'], q0['b'])['joint']:6.2f}")
    print(f"    {'the real class vs itself':34s}{m_ref('precision'):7.2f}{m_ref('recall'):8.2f}"
          f"{m_ref('f1'):7.2f}")
    protocol = dict(slot=int(slot), n_queries=len(pool_qs), per_query=per_q,
                    single=pr_single, pooled=pr_pooled)

    save_json(dict(rows=rows, protocol=protocol,
                   reference=dict(precision=m_ref("precision"), recall=m_ref("recall"),
                                  f1=m_ref("f1"))),
              os.path.join(RESULTS, "n6_diversity.json"), seed=args.seed)
    best = max(rows, key=lambda r: r["f1"])
    print(f"\n[n6] best F1: {best['config']} -> recall {best['recall']:.2f}, "
          f"acc {best['joint']:.2f}, content div/ref {best['content_ratio']:.2f}")
    cur = rows[0]
    print(f"[n6] vs current default: recall {cur['recall']:.2f} -> {best['recall']:.2f}, "
          f"content div/ref {cur['content_ratio']:.2f} -> {best['content_ratio']:.2f}, "
          f"acc {cur['joint']:.2f} -> {best['joint']:.2f}")
    print(f"[n6] wrote {RESULTS}/n6_diversity.json")


if __name__ == "__main__":
    main()
