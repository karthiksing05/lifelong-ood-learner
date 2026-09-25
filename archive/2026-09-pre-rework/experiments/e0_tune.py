"""E0 — how the defaults were chosen.  Run this before trusting E2/E3 numbers.

Two sweeps, deliberately separated, because they answer different questions and the first one bounds
the second:

  TEACHER sweep (K, tau).  Scores the probe reading of mu_T itself -- no sampling at all, so it is
  cheap and it isolates composition from rendering.  Reported against the PROBE CEILING: the same
  probes applied to REAL held-out images of the same OOD slots.  A teacher cannot be better than the
  ceiling, and samples drawn from a teacher cannot reliably be better than the teacher, so this
  number is the budget everything downstream spends.

  SAMPLER sweep (w0, u_start).  Only meaningful once the teacher is right; the first version of this
  sweep, run against a mis-tuned teacher, showed no signal for any setting and nearly sent the
  investigation after the sampler.  Scored on composition accuracy AND the geometry triple, because
  accuracy alone is maximized by copying the query.

Findings are written to results/e0_tune.json and the chosen values are the defaults in
`LifelongLearner.__init__` / `sample_teacher`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, base_parser, load_all, ood_queries         # noqa: E402

from lifelong_ood.compose import candidates_tree, poe_teacher          # noqa: E402
from lifelong_ood.evals import composition_accuracy, sample_geometry, save_json  # noqa: E402


def read_triple(probes, X):
    X = np.clip(np.atleast_2d(np.asarray(X, np.float32)), 0, 1)
    return (probes["digit"].predict(X), probes["fg"].predict(X), probes["bg"].predict(X))


def score_triple(probes, X, want_d, want_f, want_b):
    pd, pf, pb = read_triple(probes, X)
    ok = (pd == want_d) & (pf == want_f) & (pb == want_b)
    return dict(joint=float(ok.mean()), digit=float((pd == want_d).mean()),
                fg=float((pf == want_f).mean()), bg=float((pb == want_b).mean()))


def main():
    p = base_parser()
    p.add_argument("--n-slots", type=int, default=40, help="OOD slots for the teacher sweep")
    p.add_argument("--n-sampler-slots", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=16)
    args = p.parse_args()
    if args.quick:
        args.n_slots, args.n_sampler_slots, args.n_samples = 8, 3, 8
    data, learner, probes = load_all(args)
    snap = learner.field.snap
    qs = ood_queries(data, n_slots=args.n_slots, seed=args.seed)
    wd = np.array([q["d"] for q in qs]); wf = np.array([q["f"] for q in qs])
    wb = np.array([q["b"] for q in qs])
    out = {}

    # the ceiling: the probes' own reading of REAL held-out images of these slots
    ceil = score_triple(probes, np.stack([data["X"][q["query"]] for q in qs]), wd, wf, wb)
    out["probe_ceiling"] = ceil
    print(f"[e0] probe ceiling on real held-out queries: joint {ceil['joint']:.3f} "
          f"(digit {ceil['digit']:.2f} fg {ceil['fg']:.2f} bg {ceil['bg']:.2f})")

    # ── teacher sweep ────────────────────────────────────────────────────────
    pools = [(q, data["X"][q["query"]]) for q in qs]
    pools = [(q, xq) + candidates_tree(snap, xq)[:2] for q, xq in pools]
    grid = [(K, tau, ad) for K in (2, 3, 4, 6)
            for tau, ad in ((0.5, False), (0.05, False), (0.02, False),
                            (0.5, True), (0.05, True), (0.02, True))]
    rows = []
    print(f"\n[e0] TEACHER sweep on {len(qs)} OOD slots (mu_T probe reading, no sampling)")
    print(f"    {'K':>2s} {'tau':>6s} {'adapt':>5s} {'joint':>6s} {'digit':>6s} {'fg':>5s} "
          f"{'bg':>5s} {'w_max':>6s}")
    for K, tau, ad in grid:
        MU, sharp = [], []
        for (q, xq, M, V) in pools:
            t = poe_teacher(xq, M, V, K=K, tau=tau, adaptive_tau=ad)
            MU.append(t["mu"]); sharp.append(float(t["w"].max(0).mean()))
        s = score_triple(probes, np.stack(MU), wd, wf, wb)
        s.update(K=K, tau=tau, adaptive=ad, weight_sharpness=float(np.mean(sharp)),
                 frac_of_ceiling=s["joint"] / max(ceil["joint"], 1e-9))
        rows.append(s)
        print(f"    {K:2d} {tau:6.3f} {str(ad):>5s} {s['joint']:6.3f} {s['digit']:6.2f} "
              f"{s['fg']:5.2f} {s['bg']:5.2f} {s['weight_sharpness']:6.3f}")
    out["teacher_sweep"] = rows
    best = max(rows, key=lambda r: r["joint"])
    print(f"[e0] best teacher: K={best['K']} tau={best['tau']} adaptive={best['adaptive']} "
          f"-> joint {best['joint']:.3f} ({100 * best['frac_of_ceiling']:.0f}% of ceiling)")

    # ── sampler sweep, at the best teacher ───────────────────────────────────
    sq = qs[:args.n_sampler_slots]
    teachers = [learner.compose(data["X"][q["query"]], discovery="tree",
                                K=best["K"], tau=best["tau"]) for q in sq]
    # The teacher's own accuracy is the budget: samples drawn from it should not beat it, and how
    # much of it they KEEP is the sampler's score.  Anything that scores high on accuracy while
    # collapsing novelty/diversity has just copied the teacher mean and learned nothing renderable.
    tbudget = score_triple(probes, np.stack([t["mu"] for t in teachers]),
                           np.array([q["d"] for q in sq]), np.array([q["f"] for q in sq]),
                           np.array([q["b"] for q in sq]))
    out["teacher_budget_on_sampler_slots"] = tbudget
    srows = []
    print(f"\n[e0] SAMPLER sweep on {len(sq)} slots, {args.n_samples} samples each")
    print(f"    teacher budget on these slots: joint {tbudget['joint']:.3f} "
          f"(digit {tbudget['digit']:.2f} fg {tbudget['fg']:.2f} bg {tbudget['bg']:.2f})")
    print(f"    {'schedule':>9s} {'wmin':>5s} {'wmax':>5s} {'u0':>5s} {'joint':>6s} {'%tchr':>6s} "
          f"{'digit':>6s} {'fg':>5s} {'bg':>5s} {'novel':>7s} {'divers':>7s} {'gen':>7s}")
    configs = [("variance", 0.8, 2.0, us) for us in (0.4, 0.55, 0.7)] \
        + [("anneal", 0.8, w, us) for w in (1.5, 2.0) for us in (0.4, 0.55, 0.7)] \
        + [("const", w, w, 0.55) for w in (0.8, 1.0, 1.5, 2.0)] \
        + [("const", 1.0, 1.0, 0.9)]
    for sched, wmin, wmax, us in configs:
        A, G = [], []
        for qi, (q, t) in enumerate(zip(sq, teachers)):
            xq = data["X"][q["query"]]
            S = learner.sample_teacher(t, n=args.n_samples, seed=args.seed + qi,
                                       schedule=sched, wmin=wmin, wmax=wmax, u_start=us)
            A.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"]))
            G.append(sample_geometry(S, xq, data["X"][q["rest"][:100]]))
        m = lambda k, L: float(np.mean([x[k] for x in L]))   # noqa: E731
        r = dict(schedule=sched, wmin=wmin, wmax=wmax, u_start=us, ref_spread=m("ref_spread", G),
                 **{k: m(k, A) for k in ("joint", "digit", "fg", "bg")},
                 **{k: m(k, G) for k in ("novelty", "diversity", "gen_rmse")})
        r["frac_of_teacher"] = r["joint"] / max(tbudget["joint"], 1e-9)
        srows.append(r)
        print(f"    {sched:>9s} {wmin:5.1f} {wmax:5.1f} {us:5.2f} {r['joint']:6.3f} "
              f"{r['frac_of_teacher']:6.2f} {r['digit']:6.2f} {r['fg']:5.2f} {r['bg']:5.2f} "
              f"{r['novelty']:7.3f} {r['diversity']:7.3f} {r['gen_rmse']:7.3f}", flush=True)
    out["sampler_sweep"] = srows
    out["reference_spread"] = srows[0]["ref_spread"]
    # the sampler must both keep the composition AND stay off the query: rank by joint accuracy
    # among configs whose diversity is at least half the reference set's own spread
    live = [r for r in srows if r["diversity"] >= 0.5 * r["ref_spread"]] or srows
    bs = max(live, key=lambda r: (round(r["joint"], 3), r["diversity"]))   # ties -> more diverse
    print(f"[e0] reference set internal spread {out['reference_spread']:.3f} "
          f"(the yardstick for novelty/diversity)")
    print(f"[e0] best sampler: {bs['schedule']} wmin={bs['wmin']} wmax={bs['wmax']} "
          f"u_start={bs['u_start']} -> joint {bs['joint']:.3f} "
          f"({100 * bs['frac_of_teacher']:.0f}% of teacher), novelty {bs['novelty']:.3f}, "
          f"diversity {bs['diversity']:.3f}")
    out["chosen_sampler"] = bs
    save_json(out, os.path.join(RESULTS, "e0_tune.json"))
    print(f"[e0] wrote {RESULTS}/e0_tune.json")


if __name__ == "__main__":
    main()
