"""Does routing by DESCRIPTION LENGTH close the over-spend that five thresholds could not?

RESULTS 6c-bis closes five label-free routes to a better `t_vocab`, four of them for one reason: the
decision was posed as a classification -- "is this query out of vocabulary?" -- whose positive class
cannot be manufactured by the generator that has to answer it.

Cobweb does not pose it that way.  It applies each operator hypothetically and compares the resulting
partitions under category utility, which carries its own complexity penalty; there is no threshold
and no positive class.  `router.mdl_route` transplants that: encode the query batch under each tier,
pay the tier's real cost in bits, take the argmin.  See the block comment above `TIER_BITS`.

This scores it on N18's exact populations, so the numbers drop straight into 6c-bis's table:

    threshold cascade   overspend 0.00   underspend 0.10   0.90x oracle bytes
    by trial            overspend 0.80   underspend 0.00   2.00x oracle bytes

and adds the thing a threshold cannot have -- a sweep over BATCH SIZE.  Under (*) the fit term scales
with N and the cost term does not, so the same novelty should warrant a bigger tier when it arrives
as more examples.  If that curve is flat, the rule is not doing what it claims.

    python tests/diag_mdl_router.py --n-per-route 12 --batch-sizes 1,2,4,8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import backbone_gains                                       # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.evals import fit_probes                               # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_model               # noqa: E402
from lifelong_ood.neural.idealwords import (extend_basis, factorize,  # noqa: E402
                                            slot_factors)
from lifelong_ood.neural.router import (TIER_BITS, TIER_ROUTE, auroc,  # noqa: E402
                                        basis_assignments, mdl_crossover,
                                        batch_losses, calibrate, effective_dim,
                                        elbo_bits, mdl_crossover_bits, mdl_route,
                                        mdl_route_bits, route, route_signals,
                                        utility_route)

CACHE = os.path.join(HERE, "results", "cache")
ROUTES = ["in-distribution", "new-combination", "new-vocabulary"]
# what each route actually costs to serve, for the bytes-vs-oracle column (same ladder as 6c/N24)
ROUTE_BYTES = {"in-distribution": 0.0, "new-combination": 72.0, "new-vocabulary": 1024.0}


def extend_resid(field, fac, gains, X, lab, factor, steps, seed, u_grid, n_noise):
    """Fit ONE new direction for `factor` on this batch, return the residual under it.

    The other factors are held at the assignment ADDRESS already inferred, so nothing here reads a
    held-out label: the query's own best basis row supplies the context, exactly as it would in
    deployment.
    """
    labels = np.tile(np.asarray(lab, int), (len(X), 1))
    v, hist = extend_basis(field, np.asarray(X, np.float32), labels, fac, factor,
                           n_steps=steps, lr=0.05, batch=min(8, len(X)), seed=seed,
                           gains=gains, norm_match=True)
    c = np.asarray(fac["mu"], np.float64).copy()
    for f in range(len(fac["V"])):
        c = c + float(gains[f]) * (v if f == factor else fac["V"][f][int(lab[f])])
    m = float(batch_losses(field, X, c.reshape(1, -1), u_grid=u_grid,
                           n_noise=n_noise, seed=seed)[0])
    return m, c, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist_nobg3.pt"))
    p.add_argument("--held-bg", type=int, default=3)
    p.add_argument("--n-per-route", type=int, default=12)
    p.add_argument("--n-calib", type=int, default=24)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--batch-sizes", default="1,2,4,8")
    p.add_argument("--extend-steps", type=int, default=200)
    p.add_argument("--n-noise", type=int, default=4)
    p.add_argument("--n-t", type=int, default=96,
                   help="grid for the variational bound; converges from above")
    p.add_argument("--n-rep-deff", type=int, default=12)
    p.add_argument("--d-eff", type=float, default=0.0, help="override the estimate (0 = measure)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "diag_mdl_router.json"))
    args = p.parse_args()
    # ':' as well as ',': sbatch's --export=ALL,SCRIPT="..." parser SPLITS THE VALUE ON COMMAS, so
    # a comma-separated list silently arrives truncated at the first element and the rest become
    # stray environment variables.  Every run before 2026-08-13 06:50 measured N=1 only and the
    # header said so ("--batch-sizes 1") without anything failing.  Pass colons under sbatch.
    Ns = [int(x) for x in args.batch_sizes.replace(":", ",").split(",") if x.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_trained = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_trained]]
    field = load_field(args.ckpt, device=dev)
    K = args.held_bg

    fac = factorize(C, slot_factors(slots_trained))
    gains = backbone_gains(field, C, slots_trained, probes, data, seed=args.seed, steps=50, n=16)
    basis_conds, basis_labels = basis_assignments(fac, gains=gains, sizes=fac["sizes"])
    print(f"[mdl] backbone {os.path.basename(args.ckpt)}: {len(slots_trained)} trained slots, "
          f"R^2 {fac['r2']:.3f}, sizes {fac['sizes']} (no direction for bg={K})", flush=True)

    trained = set(int(s) for s in slots_trained)
    rng = np.random.default_rng(args.seed)
    pops = {r: [] for r in ROUTES}
    for s in np.unique(data["slot"]):
        if int(s) % 4 == K:
            pops["new-vocabulary"].append(int(s))
        elif int(s) in trained:
            pops["in-distribution"].append(int(s))
        else:
            pops["new-combination"].append(int(s))

    # basis rows the model does NOT already hold as a trained slot -- ADDRESS's actual candidate set.
    # Label-free: it is a lookup of which slots were trained, which the system supplied itself.
    untr = np.array([(int(r[0]) * 16 + int(r[1]) * 4 + int(r[2])) not in trained
                     for r in basis_labels])
    print(f"[mdl] ADDRESS candidates: {int(untr.sum())} of {len(basis_labels)} basis rows "
          f"(the other {len(basis_labels) - int(untr.sum())} ARE trained slots, so they belong to "
          f"the NONE tier and including them would let ADDRESS win by selection)", flush=True)

    u_grid = (0.15, 0.30, 0.45)          # the same band route_signals reads, averaged
    nk = dict(n_noise=args.n_noise, seed=args.seed)

    # ── D_eff, from trained slots under their OWN conditionings (label-free) ───────────────
    if args.d_eff > 0:
        d_eff = args.d_eff
        print(f"[mdl] D_eff {d_eff:.0f} (overridden)", flush=True)
    else:
        cal_slots = rng.permutation(pops["in-distribution"])[:8]
        ds = []
        for s in cal_slots:
            idx = np.where(data["slot"] == int(s))[0][:2]
            xs = [to_model(data["X"][i]).ravel() for i in idx]
            ci = C[list(slots_trained).index(int(s))].reshape(1, -1)
            ds.append(effective_dim(field, xs, ci, n_rep=args.n_rep_deff,
                                    u_grid=u_grid, seed=args.seed))
        d_eff = float(np.median(ds))
        print(f"[mdl] D_eff {d_eff:.0f}  (median over {len(ds)} trained slots, "
              f"spread {np.min(ds):.0f}-{np.max(ds):.0f}; nominal 3*32*32 = 3072)", flush=True)

    # ── the threshold cascade, calibrated exactly as n18 does, for the comparison row ──────
    calib_q = rng.permutation(pops["in-distribution"])[:args.n_calib]
    sig_calib = []
    for s in calib_q:
        i = int(np.where(data["slot"] == int(s))[0][7 % args.n_per_slot])
        sig_calib.append(route_signals(field, to_model(data["X"][i]).ravel(), C, basis_conds, **nk))
    thr = calibrate(sig_calib)           # no self-combos here -> the documented fallback
    print(f"[mdl] cascade thresholds t_novel {thr['t_novel']:.5f} "
          f"t_vocab {thr['t_vocab']:.5f} ({thr['calibrated_by']})", flush=True)

    # ── the sweep ─────────────────────────────────────────────────────────────────────────
    #
    # The SAME concepts at every batch size, drawn once.  Re-drawing inside the N loop would make the
    # promotion curve a comparison between different concepts at different N, which is exactly the
    # unbalanced-averaging mistake f31 already had to be fixed for: the curve would then move because
    # the population changed, not because the evidence did.
    picked = {t: list(rng.permutation(pops[t])[:args.n_per_route]) for t in ROUTES}
    rows = []
    for N in Ns:
        for truth in ROUTES:
            for s in picked[truth]:
                idx = np.where(data["slot"] == int(s))[0][:N]
                if len(idx) < N:
                    continue
                X = [to_model(data["X"][i]).ravel() for i in idx]

                # tier candidates.  NONE minimizes over the trained class embeddings; ADDRESS over
                # basis rows the model does NOT already hold as a trained slot -- restricted for a
                # reason, not for economy.  The full Cartesian product includes reconstructions of
                # every trained slot, so unrestricted ADDRESS nearly CONTAINS NONE and would "improve"
                # on it by selection over more candidates rather than by expressing anything new.
                # The restriction is also the tier's own definition: ADDRESS is for combinations of
                # known values that the model does not already have, and which those are is something
                # the system knows without any label.
                Lc = batch_losses(field, X, C, u_grid=u_grid, **nk)
                m_none, c_none = float(Lc.min()), C[int(Lc.argmin())]
                Lb = batch_losses(field, X, basis_conds[untr], u_grid=u_grid, **nk)
                j = int(Lb.argmin())
                m_addr, lab = float(Lb[j]), basis_labels[untr][j]
                c_addr = basis_conds[untr][j]

                # Cobweb evaluates every operator, so every factor is a candidate to extend --
                # which also means the rule discovers WHICH factor is missing rather than being told.
                best = None
                for f in range(len(fac["V"])):
                    m_f, c_f, h = extend_resid(field, fac, gains, X, lab, f, args.extend_steps,
                                               args.seed, u_grid, args.n_noise)
                    if best is None or m_f < best[0]:
                        best = (m_f, f, c_f)
                m_ext, f_ext, c_ext = best

                # the codelength the decision is actually taken on: the eps-matching part of the
                # variational bound, in bits, for the three SELECTED conditionings only (selection
                # stays on the cheap simple loss -- it is a ranking, not an accounting).
                # NO division by len(X): `elbo_bits` averages over the batch internally and already
                # returns PER-IMAGE bits.  Dividing again would make the fit term N*(code/N) = code,
                # cancelling the batch-size scaling that is the whole point of the rule -- and it
                # would do so silently, as a promotion curve that is flat for a plausible-looking
                # reason rather than as an error.
                code = {t: elbo_bits(field, X, c, n_t=args.n_t, n_noise=2, seed=args.seed)
                        for t, c in (("none", c_none), ("address", c_addr), ("extend", c_ext))}
                resid = dict(none=m_none, address=m_addr, extend=m_ext)
                tier, L = mdl_route_bits(code, N)
                tier_s, _ = mdl_route(resid, N, d_eff)          # the simple-loss reading, for contrast
                util, _ = utility_route(resid)
                sig = route_signals(field, X[0], C, basis_conds, **nk)
                rows.append(dict(N=N, truth=truth, slot=int(s), m_none=m_none, m_addr=m_addr,
                                 m_ext=m_ext, factor_extended=int(f_ext), code=code,
                                 mdl=TIER_ROUTE[tier], tier=tier, lengths=L,
                                 mdl_simple=TIER_ROUTE[tier_s], utility=TIER_ROUTE[util],
                                 n_star_addr=mdl_crossover_bits(code, lo="none", hi="address"),
                                 n_star_ext=mdl_crossover_bits(code, lo="address", hi="extend"),
                                 cascade=route(sig, thr)))
                print(f"[mdl] N={N:2d} {truth:16s} slot {int(s):3d} | m "
                      f"none {m_none:.5f} addr {m_addr:.5f} ext {m_ext:.5f} (f{f_ext}) | "
                      f"N* addr {rows[-1]['n_star_addr']:7.1f} ext {rows[-1]['n_star_ext']:7.1f} | "
                      f"mdl -> {TIER_ROUTE[tier]:16s} util -> {TIER_ROUTE[util]:16s} "
                      f"cascade -> {rows[-1]['cascade']}", flush=True)

    # ── scoring, in 6c-bis's columns ───────────────────────────────────────────────────────
    print("\n[mdl] " + "=" * 78)
    summary = []
    for N in Ns:
        R = [r for r in rows if r["N"] == N]
        if not R:
            continue
        for key in ("mdl", "mdl_simple", "utility", "cascade"):
            comb = [r for r in R if r["truth"] == "new-combination"]
            voc = [r for r in R if r["truth"] == "new-vocabulary"]
            ind = [r for r in R if r["truth"] == "in-distribution"]
            # over/under-spend against the LADDER, not against one adjacent pair.  The first version
            # of this counted underspend only as vocabulary->combination, which silently scored a rule
            # that sent every query to "store nothing" as underspend 0.00 -- the worst possible
            # policy reading as the safest.  Rank the routes and compare positions.
            rank = {r: i for i, r in enumerate(ROUTES)}
            over = np.mean([rank[r[key]] > rank[r["truth"]] for r in comb + ind]) \
                if comb or ind else float("nan")
            under = np.mean([rank[r[key]] < rank[r["truth"]] for r in voc + comb]) \
                if voc or comb else float("nan")
            acc = np.mean([r[key] == r["truth"] for r in R])
            spent = float(np.sum([ROUTE_BYTES[r[key]] for r in R]))
            oracle = float(np.sum([ROUTE_BYTES[r["truth"]] for r in R]))
            # under-spending is a capability failure, not a saving: count it separately
            idacc = np.mean([r[key] == "in-distribution" for r in ind]) if ind else float("nan")
            summary.append(dict(N=N, rule=key, overspend=float(over), underspend=float(under),
                                acc=float(acc), id_acc=float(idacc),
                                bytes_ratio=spent / oracle if oracle else float("nan")))
            print(f"[mdl] N={N:2d} {key:8s} overspend {over:.2f}  underspend {under:.2f}  "
                  f"acc {acc:.3f}  in-dist acc {idacc:.3f}  bytes {spent/oracle:.2f}x oracle")
    # ── the calibration-INVARIANT read ────────────────────────────────────────────────────
    #
    # N and D_eff enter (*) only as their product, so every N* below is proportional to 1/D_eff and
    # the SEPARATION between populations is untouched by the estimate.  This is the column to trust
    # if the D_eff moment estimator is doubted -- and it is directly comparable to 6c's AUROC 0.995,
    # which is the same separation measured on a statistic that then needed a threshold.
    n1 = [r for r in rows if r["N"] == min(Ns)]
    star = {t: np.array([r["n_star_ext"] for r in n1 if r["truth"] == t], float) for t in ROUTES}
    fin = {t: v[np.isfinite(v)] for t, v in star.items()}
    print("\n[mdl] batch size at which EXTEND becomes worth its 1 KB (median; inf = never):")
    for t in ROUTES:
        v = star[t]
        med = float(np.median(v)) if len(v) else float("nan")
        print(f"[mdl]   {t:16s} N* median {med:8.1f}   finite on {len(fin[t])}/{len(v)}")
    a = auroc(-star["new-vocabulary"][np.isfinite(star["new-vocabulary"])],
              -star["new-combination"][np.isfinite(star["new-combination"])])
    print(f"[mdl]   AUROC(vocabulary vs combination, on N*) {a:.3f}  "
          f"[6c's threshold statistic: 0.995] -- invariant to D_eff")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(rows=rows, summary=summary, d_eff=d_eff, thr=thr, n_star_auroc=float(a),
                   tier_bits=TIER_BITS, args=vars(args)),
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"[mdl] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
