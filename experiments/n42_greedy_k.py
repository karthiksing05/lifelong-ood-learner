"""N42 -- why K = 3?  Choose K per instance, by explaining the query until nothing is left.

Every composition in this repository selects a FIXED K = 3 experts (`poe_teacher(..., K=3)`), which
is a number nobody has justified and a reviewer will ask about.  It is also cognitively odd: Cobweb
does not decide in advance how many concepts an instance needs, it descends until the instance is
accounted for.

**The greedy alternative.**  Mode ascent on the frozen score field already produces a dictionary of
prototypes -- in the ACS framing (Wang, Singaravadivelan & MacLellan) those modes ARE the concepts
of the hierarchy the diffusion model has learned.  So instead of taking the top-3, take them one at a
time, each time keeping the mode that explains the most of what is still UNEXPLAINED, and stop when
the query is accounted for:

    S <- {}                                     the concepts chosen so far
    repeat
        for each candidate mode m:  score m by  expl(S + {m})
        S <- S + {argmax}                       keep the best
        r <- x_q - mu_T(S)                      what the chosen set still fails to explain
    until  expl(S) >= tau   or   the marginal gain falls below eps

with `expl(S) = 1 - Var(x_q - mu_T(S)) / Var(x_q)`, the fraction of the query's own variance the
composition accounts for, and `mu_T(S)` the per-dimension product-of-experts mean over S -- so the
reconstruction is the SAME object the pipeline goes on to use, not a proxy fitted for this purpose.
This is matching pursuit over the model's own modes.

Three things this is for.

1. **K becomes a per-instance decision with a reason**, and the trajectory of `expl` against k is a
   picture of the prototype being assembled -- what each concept contributed and in what order.
2. **A new routing signal.**  A query the model can already express should be explained by few
   concepts and reach a high `expl`; a genuinely novel one should not.  So `K_chosen` and the final
   residual are candidate in-distribution/OOD detectors, and they are compared here against the
   deployed router (denoising error, AUROC 0.980 / 0.995) rather than assumed better.
3. **Robustness**, because a greedy rule with a threshold invites exactly one objection: that the
   threshold is doing the work.  `tau` is swept, the candidate pool is resized, and the seed is
   varied; the reported result is the sensitivity, not one setting.

    python experiments/n42_greedy_k.py --n-cases 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries                # noqa: E402

from lifelong_ood.compose import candidates_ascent, loglik_perdim             # noqa: E402
from lifelong_ood.data.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                       # noqa: E402
from lifelong_ood.neural.router import auroc                                  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")


def poe_mean(M, V, sel, xq, tau=0.05):
    """Per-dimension product-of-experts mean over the selected modes -- the pipeline's own mu_T."""
    if not len(sel):
        return np.zeros_like(xq)
    L = loglik_perdim(xq, M, V)[sel]
    T = tau * max(float(L.std(0).mean()), 1e-6)
    Z = L / T
    Z -= Z.max(0, keepdims=True)
    w = np.exp(Z)
    w /= w.sum(0, keepdims=True)
    prec = (w / V[sel]).sum(0)
    return ((w * M[sel] / V[sel]).sum(0)) / np.maximum(prec, 1e-12)


def explained(xq, mu):
    """Fraction of the query's own variance accounted for by the reconstruction."""
    v = float(np.var(xq))
    if v <= 0:
        return 0.0
    return float(1.0 - np.var(xq - mu) / v)


def greedy_select(xq, M, V, tau_expl=0.99, eps=1e-3, k_max=12):
    """Matching pursuit over the mode dictionary.  -> (selected idx, trajectory of expl)."""
    sel, traj, prev = [], [], 0.0
    pool = list(range(len(M)))
    for _ in range(min(k_max, len(M))):
        best, best_e = None, -np.inf
        for j in pool:
            if j in sel:
                continue
            e = explained(xq, poe_mean(M, V, np.array(sel + [j]), xq))
            if e > best_e:
                best, best_e = j, e
        if best is None:
            break
        sel.append(best)
        traj.append(best_e)
        if best_e >= tau_expl or (best_e - prev) < eps:
            break
        prev = best_e
    return np.array(sel), traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--n-seen", type=int, default=8, help="in-distribution queries, for the detector")
    p.add_argument("--taus", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    p.add_argument("--n-starts", type=int, nargs="+", default=[8, 16],
                   help="candidate pool sizes, for the robustness check")
    p.add_argument("--fixed-k", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_seen, args.n_per_slot = 2, 2, 20
        args.steps, args.n_samples = 20, 8
        args.taus, args.n_starts = [0.95], [8]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)[:args.n_cases]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, slot_factors(slots),
                           gains=backbone_gains(field, C, slots, probes, data, seed=args.seed,
                                                steps=args.steps, n=8 if args.quick else 16))

    # in-distribution queries, for the detector comparison
    rng = np.random.default_rng(args.seed)
    seen_slots = np.unique(data["slot"][data["seen"]])
    seen_pick = rng.choice(seen_slots, min(args.n_seen, len(seen_slots)), replace=False)
    seen_q = []
    for s in seen_pick:
        idx = np.where(data["slot"] == s)[0]
        i = int(idx[rng.integers(len(idx))])
        seen_q.append(dict(slot=int(s), query=i, rest=idx[idx != i],
                           d=int(data["digit"][i]), f=int(data["fg"][i]), b=int(data["bg"][i])))

    rows, trajs = [], {}
    for n_starts in args.n_starts:
        for q, kind in [(x, "ood") for x in qs] + [(x, "seen") for x in seen_q]:
            xq = to_model(data["X"][q["query"]]).ravel()
            M, V, _ = candidates_ascent(field, xq, seed=args.seed, n_starts=n_starts,
                                        iters=30 if args.quick else 120)
            R = data["X"][q["rest"][:100]].astype(np.float32)
            for tau in args.taus:
                sel, traj = greedy_select(xq, M, V, tau_expl=tau)
                mu_g = poe_mean(M, V, sel, xq)
                # the fixed-K baseline on the IDENTICAL candidate pool
                from lifelong_ood.compose import submodular_select
                selK = submodular_select(loglik_perdim(xq, M, V), args.fixed_k)
                mu_k = poe_mean(M, V, selK, xq)
                row = dict(kind=kind, slot=int(q["slot"]), n_starts=int(n_starts), tau=float(tau),
                           k_greedy=int(len(sel)), k_fixed=int(args.fixed_k),
                           expl_greedy=explained(xq, mu_g), expl_fixed=explained(xq, mu_k),
                           n_candidates=int(len(M)))
                if kind == "ood" and tau == args.taus[-1] and n_starts == args.n_starts[-1]:
                    trajs[int(q["slot"])] = traj
                    for tag, mu in (("greedy", mu_g), ("fixed", mu_k)):
                        t = dict(mu=mu, var=np.full_like(mu, 0.05), fixed_var=True)
                        S = to_data(L.sample_teacher(t, n=args.n_samples, seed=args.seed,
                                                     schedule="variance", w0=1.2, wmax=2.0,
                                                     wmin=0.0)).astype(np.float32)
                        a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                        row[f"joint_{tag}"] = a["joint"]
                        row[f"f1_{tag}"] = precision_recall(S, R)["f1"]
                rows.append(row)
            print(f"[n42] {kind:4s} slot {q['slot']:3d} starts {n_starts:2d}: "
                  + "  ".join(f"tau={r['tau']:.2f}->K={r['k_greedy']} "
                              f"(expl {r['expl_greedy']:.3f})"
                              for r in rows[-len(args.taus):]), flush=True)

    save_json(dict(rows=rows, trajectories={str(k): v for k, v in trajs.items()},
                   config=vars(args)),
              os.path.join(RESULTS, f"n42_greedy_k{args.tag}.json"), seed=args.seed)

    def sub(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    print(f"\n[n42] K chosen per instance (OOD queries)")
    print(f"    {'starts':>7s}{'tau':>7s}{'mean K':>9s}{'K range':>10s}{'expl':>8s}"
          f"{'expl@K=3':>10s}")
    for ns in args.n_starts:
        for tau in args.taus:
            s = sub(kind="ood", n_starts=ns, tau=tau)
            if not s:
                continue
            k = np.array([r["k_greedy"] for r in s])
            print(f"    {ns:7d}{tau:7.2f}{k.mean():9.2f}{f'{k.min()}-{k.max()}':>10s}"
                  f"{np.mean([r['expl_greedy'] for r in s]):8.3f}"
                  f"{np.mean([r['expl_fixed'] for r in s]):10.3f}")

    fin = sub(kind="ood", n_starts=args.n_starts[-1], tau=args.taus[-1])
    if fin and "joint_greedy" in fin[0]:
        print(f"\n[n42] composition at greedy K vs fixed K={args.fixed_k}, same candidate pool")
        for tag in ("greedy", "fixed"):
            print(f"    {tag:8s} joint {np.mean([r[f'joint_{tag}'] for r in fin]):.3f}   "
                  f"F1 {np.mean([r[f'f1_{tag}'] for r in fin]):.3f}")

    print(f"\n[n42] as an in-distribution detector (AUROC, OOD vs seen; deployed router is "
          f"0.980 / 0.995)")
    for ns in args.n_starts:
        for tau in args.taus:
            o = sub(kind="ood", n_starts=ns, tau=tau)
            s = sub(kind="seen", n_starts=ns, tau=tau)
            if not o or not s:
                continue
            a_k = auroc([r["k_greedy"] for r in o], [r["k_greedy"] for r in s])
            a_e = auroc([-r["expl_greedy"] for r in o], [-r["expl_greedy"] for r in s])
            print(f"    starts {ns:2d} tau {tau:.2f}:  K_chosen {a_k:.3f}   "
                  f"residual(1-expl) {a_e:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for slot, tr in trajs.items():
        axes[0].plot(range(1, len(tr) + 1), tr, marker="o", ms=4, lw=1.6, alpha=0.85,
                     label=f"slot {slot}")
    axes[0].axhline(args.taus[-1], color="#B03A2E", ls=(0, (4, 3)), lw=1.2,
                    label=f"stop at {args.taus[-1]:.2f}")
    axes[0].axvline(args.fixed_k, color="#8A8A8A", ls=(0, (2, 3)), lw=1.2,
                    label=f"the fixed K={args.fixed_k}")
    axes[0].set_xlabel("concepts selected (greedily)")
    axes[0].set_ylabel("fraction of the query's variance explained")
    axes[0].set_title("Assembling the prototype, one concept at a time", fontsize=11)
    axes[0].grid(alpha=0.25, lw=0.6)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    for ns in args.n_starts:
        for tau in args.taus:
            o = sub(kind="ood", n_starts=ns, tau=tau)
            s = sub(kind="seen", n_starts=ns, tau=tau)
            if o:
                axes[1].scatter([tau + 0.002 * ns] * len(o), [r["k_greedy"] for r in o], s=26,
                                color="#1B7F4B", alpha=0.75,
                                label="OOD" if (ns == args.n_starts[0] and tau == args.taus[0]) else None)
            if s:
                axes[1].scatter([tau + 0.002 * ns] * len(s), [r["k_greedy"] for r in s], s=26,
                                marker="s", color="#B03A2E", alpha=0.75,
                                label="in-distribution" if (ns == args.n_starts[0] and tau == args.taus[0]) else None)
    axes[1].axhline(args.fixed_k, color="#8A8A8A", ls=(0, (2, 3)), lw=1.2)
    axes[1].set_xlabel("stopping threshold tau")
    axes[1].set_ylabel("K chosen")
    axes[1].set_title("Does K separate novel from familiar?  (robustness across tau, pool size)",
                      fontsize=11)
    axes[1].grid(alpha=0.25, lw=0.6)
    axes[1].legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG, f"f17_greedy_k{args.tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[n42] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
