"""N18 — the router: does the system decide FOR ITSELF which update an observation needs?

Every result in this repository so far was produced with the experimenter supplying the category of
the incoming query.  RESULTS 5q settled what to DO in each case -- a new COMBINATION of known factor
values is free, a new VOCABULARY item needs a weight update from real data -- but a lifelong learner
meets observations, not labelled categories, and the source paper names exactly this as open work
("A full mechanism for DETECTING, consolidating, and reusing genuinely new primitives").

One backbone gives all three routes at once.  `ddpm_colormnist_nobg3` was trained with background
value 3 withheld ENTIRELY as well as with the four held-out (fg, bg) pairs, so:

  in-distribution   trained slots                        -> nothing to do
  new-combination   the held-out (fg, bg) pairs, bg != 3 -> compose, zero parameters
  new-vocabulary    every slot with bg == 3              -> real-data weight update

The signal and its two negative results are documented in `neural/router.py`; the short version is
that the model's own denoising error under its best conditioning is monotone in novelty, that
normalising it by the unconditional error REVERSES it, and that the unconditional error alone is
just as discriminative -- so this is a density-based novelty detector and the `uncond` ablation row
below is there to stop it being read as a claim about the factor basis.

Reported four ways, because a router can look good for the wrong reason:
  * AUROC per pairwise decision -- threshold-free, independent of the calibration
  * the same AUROC for the `uncond` ablation -- the honest control
  * the confusion matrix at calibrated thresholds, with a sensitivity sweep
  * the COST of acting on the decision, split by what each kind of misroute actually costs:
    a new-vocabulary query sent down the free route yields a WRONG image (quality lost), and a
    new-combination query sent to the weight route yields a correct image but spends parameters it
    did not need (budget wasted).  Those are different failures and are reported separately.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains                       # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes, save_json  # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model           # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize, sample_factored_pool,  # noqa: E402
                                            slot_factors)
from lifelong_ood.neural.router import (auroc, basis_assignments, calibrate,  # noqa: E402
                                        calibrate_cost, leave_one_value_out,  # noqa: E402
                                        route, route_signals)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
ROUTES = ("in-distribution", "new-combination", "new-vocabulary")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist_nobg3.pt"))
    p.add_argument("--held-bg", type=int, default=3)
    p.add_argument("--n-per-route", type=int, default=40)
    p.add_argument("--n-calib", type=int, default=40)
    p.add_argument("--n-selfcombo", type=int, default=20,
                   help="model-generated samples from untrained basis combinations, used to "
                        "calibrate the new-vocabulary threshold without a label")
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--n-noise", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-endtoend", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-endtoend", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_per_route, args.n_calib, args.n_per_slot, args.n_selfcombo = 8, 8, 20, 4
        args.n_samples, args.steps, args.n_endtoend, args.n_noise = 8, 20, 2, 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_trained = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_trained]]
    field = load_field(args.ckpt, device=dev)
    K = args.held_bg

    labels = slot_factors(slots_trained)
    fac = factorize(C, labels)
    print(f"[n18] backbone {os.path.basename(args.ckpt)}: {len(slots_trained)} trained slots, "
          f"basis R^2 {fac['r2']:.3f}, factor sizes {fac['sizes']} "
          f"(no direction exists for bg={K})", flush=True)
    gains = backbone_gains(field, C, slots_trained, probes, data, seed=args.seed,
                           steps=args.steps, n=8 if args.quick else 16)
    basis_conds, basis_labels = basis_assignments(fac, gains=gains, sizes=fac["sizes"])

    trained = set(int(s) for s in slots_trained)
    rng = np.random.default_rng(args.seed)
    pops = {r: [] for r in ROUTES}
    for s in np.unique(data["slot"]):
        b_ = int(s) % 4
        if b_ == K:
            pops["new-vocabulary"].append(int(s))
        elif int(s) in trained:
            pops["in-distribution"].append(int(s))
        else:
            pops["new-combination"].append(int(s))
    for r in ROUTES:
        print(f"[n18]   {r:18s} {len(pops[r]):3d} slots", flush=True)

    def queries(slots, n, off=0):
        out = []
        for s in rng.permutation(slots)[:n]:
            idx = np.where(data["slot"] == s)[0]
            out.append(dict(slot=int(s), i=int(idx[off % len(idx)]),
                            d=int(s) // 16, f=(int(s) % 16) // 4, b=int(s) % 4))
        return out

    sig_kw = dict(n_noise=args.n_noise, seed=args.seed)

    def sig_of_image(x01):
        return route_signals(field, to_model(x01).ravel(), C, basis_conds, **sig_kw)

    # ── calibration 1: trained slots (label-free) ───────────────────────────
    calib_q = queries(pops["in-distribution"], args.n_calib, off=7)
    sig_calib = [sig_of_image(data["X"][q["i"]]) for q in calib_q]

    # ── calibration 2: the model's OWN samples from untrained combinations ──
    # The system can build these unaided -- it knows which slots it trained on, and the basis can
    # compose the rest.  This stands in for a labelled new-combination example.
    untrained_known = [(d, f, b) for d in range(10) for f in range(4)
                       for b in range(fac["sizes"][2])
                       if (d * 16 + f * 4 + b) not in trained]
    rng.shuffle(untrained_known)
    sig_self = []
    for (d_, f_, b_) in untrained_known[:args.n_selfcombo]:
        c = compose(fac, (d_, f_, b_), gains=gains)
        S = to_data(sample_factored_pool(field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                                         n=1, steps=args.steps, seed=args.seed))
        sig_self.append(sig_of_image(S[0]))
    thr = calibrate(sig_calib, sig_selfcombo=sig_self)
    print(f"[n18] thresholds: t_novel {thr['t_novel']:.5f} (trained slots), "
          f"t_vocab {thr['t_vocab']:.5f} ({thr['calibrated_by']})", flush=True)

    # ── calibration 3: the model's own samples from values the basis CANNOT express ──
    #
    # `calibrate` had only the NEGATIVE population (self-combinations) and had to take a quantile
    # inside it, which puts the threshold at 0.01142 -- barely above the real new-combination mean of
    # 0.01035 -- and sends 30% of combinations down the paid route. That single number is the whole
    # oracle-vs-measured gap in N24 (3.8M against 10.1M), even though the SIGNAL separates the two
    # classes at AUROC 0.995. The detector was never the problem; the operating point was.
    #
    # `synthetic_vocab_conds` manufactures the positive population by the same trick, on the axis the
    # negative one holds fixed: a direction orthogonal to the span of a factor's known value-vectors
    # is, by construction, a value the additive basis has no coordinate for. With both populations in
    # hand the threshold can sit BETWEEN them, and `calibrate_cost` places it where expected BYTES
    # are minimized rather than where a quantile falls -- because the two errors cost wildly
    # different amounts and an accuracy criterion prices them the same.
    # `synthetic_vocab_conds` was tried here first and FAILED: conditioning on an off-basis
    # direction decodes to a generic blur, a blur is easy to denoise, so the manufactured positives
    # scored BELOW the real negatives and cost-minimizing against them gave overspend 0.900 against
    # the quantile rule's 0.333.  `leave_one_value_out` replaces it and manufactures nothing: it
    # HIDES a value the system already has, refits the basis without it, and uses REAL images
    # carrying that value as positives -- images whose factor value the reduced basis provably
    # cannot express.  Choosing what to hide is a decision about its own representation, not a label.
    loo = leave_one_value_out(C, labels, seed=args.seed, gains=gains)
    thr_cost, cc = dict(thr), None
    if loo is not None:
        def sig_reduced(x01):
            return route_signals(field, to_model(x01).ravel(), C, loo["conds"], **sig_kw)
        rng_l = np.random.default_rng(args.seed)
        take = lambda idx: rng_l.permutation(idx)[:args.n_selfcombo]      # noqa: E731
        sig_loo_pos, sig_loo_neg = [], []
        for j in take(loo["pos"]):
            i = np.where(data["slot"] == int(slots_trained[j]))[0]
            if len(i):
                sig_loo_pos.append(sig_reduced(data["X"][i[0]]))
        for j in take(loo["neg"]):
            i = np.where(data["slot"] == int(slots_trained[j]))[0]
            if len(i):
                sig_loo_neg.append(sig_reduced(data["X"][i[0]]))
        if sig_loo_pos and sig_loo_neg:
            cc = calibrate_cost(sig_loo_neg, sig_loo_pos)
            thr_cost["t_vocab"] = cc["t_vocab"]
            thr_cost["calibrated_by"] = cc["calibrated_by"]
            print(f"[n18] leave-one-value-out: hid factor {loo['factor']} value {loo['value']} "
                  f"({len(sig_loo_pos)} pos, {len(sig_loo_neg)} neg under the reduced basis)",
                  flush=True)
            print(f"[n18] cost-calibrated t_vocab {cc['t_vocab']:.5f} "
                  f"(held-out-value overspend {cc['overspend']:.2f}, "
                  f"underspend {cc['underspend']:.2f}) -- vs quantile {thr['t_vocab']:.5f}",
                  flush=True)

    # The FALLBACK calibration, evaluated as a first-class arm rather than as a last resort.
    # `calibrate`'s docstring ranks `sig_selfcombo` above `vocab_mult` and calls the latter "an
    # explicit assumption".  Measured on the real populations the ranking inverts: the self-combo
    # quantile lands at 0.01142 and overspends 33% of new combinations, while 2x the trained-slot
    # quantile lands near 0.019 -- inside the separable gap (real combinations top out at 0.01758,
    # real vocabulary starts at 0.01813) -- and overspends none of them. Overspend is exactly what
    # N24 integrates into the 3.8M-vs-10.1M oracle gap, so which path is "preferred" is not a
    # documentation detail.
    thr_mult = calibrate(sig_calib)          # no sig_selfcombo -> vocab_mult path
    print(f"[n18] fallback t_vocab {thr_mult['t_vocab']:.5f} ({thr_mult['calibrated_by']})",
          flush=True)

    # ── evaluate ────────────────────────────────────────────────────────────
    rows = []
    for r in ROUTES:
        for q in queries(pops[r], args.n_per_route):
            s = sig_of_image(data["X"][q["i"]])
            rows.append(dict(truth=r, slot=q["slot"], pred=route(s, thr),
                             pred_cost=route(s, thr_cost),
                             pred_mult=route(s, thr_mult), **s))
        print(f"[n18] {r}: scored", flush=True)

    def vals(r, k):
        return [x[k] for x in rows if x["truth"] == r]

    def pair_auroc(kn, kv):
        return {
            "novel vs in-distribution":
                auroc(vals("new-combination", kn) + vals("new-vocabulary", kn),
                      vals("in-distribution", kn)),
            "new-vocabulary vs new-combination":
                auroc(vals("new-vocabulary", kv), vals("new-combination", kv)),
        }

    aur = pair_auroc("s_novel", "s_vocab")
    aur_abl = pair_auroc("uncond_novel", "uncond_vocab")     # the honest control
    conf = {t: {pp: int(sum(1 for x in rows if x["truth"] == t and x["pred"] == pp))
                for pp in ROUTES} for t in ROUTES}
    acc = float(np.mean([x["truth"] == x["pred"] for x in rows]))

    # ── sensitivity: how much does the operating point matter? ──────────────
    sweep = []
    for qn in (0.80, 0.85, 0.90, 0.95, 0.99):
        for qv in (0.80, 0.90, 0.95, 0.99):
            th = calibrate(sig_calib, sig_selfcombo=sig_self, q_novel=qn, q_vocab=qv)
            a = float(np.mean([x["truth"] == route(x, th) for x in rows]))
            sweep.append(dict(q_novel=qn, q_vocab=qv, accuracy=a))
    best = max(sweep, key=lambda z: z["accuracy"])
    print(f"[n18] calibrated accuracy {acc:.3f}; best over the quantile grid "
          f"{best['accuracy']:.3f} at q_novel={best['q_novel']} q_vocab={best['q_vocab']}",
          flush=True)

    # ── the cost of acting on the decision ──────────────────────────────────
    # Split by failure kind: a new-vocabulary query on the free route yields a WRONG image (quality
    # lost, measurable); a new-combination query on the weight route yields a correct image at
    # unnecessary parameter cost (budget wasted, counted not sampled).
    e2e = []
    if not args.skip_endtoend:
        def free_route_accuracy(q):
            """Render q by composing known factor values — what the free route would produce."""
            b_eff = min(q["b"], fac["sizes"][2] - 1)     # the basis has no direction for bg=K
            c = compose(fac, (q["d"], q["f"], b_eff), gains=gains)
            S = to_data(sample_factored_pool(field, torch.as_tensor(c, dtype=torch.float32,
                                                                    device=dev),
                                             n=args.n_samples, steps=args.steps,
                                             seed=args.seed)).astype(np.float32)
            return composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"]

        for r in ROUTES:
            for q in queries(pops[r], args.n_endtoend, off=3):
                s = sig_of_image(data["X"][q["i"]])
                pred = route(s, thr)
                e2e.append(dict(truth=r, pred=pred, slot=q["slot"],
                                free_acc=free_route_accuracy(q)))
        n_vocab = [x for x in e2e if x["truth"] == "new-vocabulary"]
        leaked = [x for x in n_vocab if x["pred"] != "new-vocabulary"]
        n_comb = [x for x in e2e if x["truth"] == "new-combination"]
        overspent = [x for x in n_comb if x["pred"] == "new-vocabulary"]
        cost = dict(
            n=len(e2e),
            vocab_sent_to_free=len(leaked) / max(len(n_vocab), 1),
            vocab_free_route_accuracy=float(np.mean([x["free_acc"] for x in n_vocab]))
            if n_vocab else float("nan"),
            combination_free_route_accuracy=float(np.mean([x["free_acc"] for x in n_comb]))
            if n_comb else float("nan"),
            combination_sent_to_weights=len(overspent) / max(len(n_comb), 1),
            wasted_params=int(len(overspent) * 551000))
        print(f"[n18] cost: {cost['vocab_sent_to_free']:.2f} of new-vocabulary queries would take "
              f"the free route (which scores {cost['vocab_free_route_accuracy']:.2f} against "
              f"{cost['combination_free_route_accuracy']:.2f} where it is correct); "
              f"{cost['combination_sent_to_weights']:.2f} of new combinations would spend "
              f"parameters they did not need", flush=True)

        # THE COMPARISON THIS EXPERIMENT NOW EXISTS FOR.  Same signals, same queries, same held-out
        # labels used for evaluation only -- the two rules differ solely in where t_vocab sits, and
        # `overspend` is the number that becomes N24's oracle-vs-measured gap.
        for nm, key in (("self-combo quantile  (the 'preferred' path)", "pred"),
                        ("cost-min  (refuted; see router.synthetic_vocab_conds)", "pred_cost"),
                        ("2x trained quantile  (the documented FALLBACK)", "pred_mult")):
            over = np.mean([x[key] == "new-vocabulary" for x in rows
                            if x["truth"] == "new-combination"]) if n_comb else float("nan")
            under = np.mean([x[key] != "new-vocabulary" for x in rows
                             if x["truth"] == "new-vocabulary"]) if n_vocab else float("nan")
            acc_r = float(np.mean([x["truth"] == x[key] for x in rows]))
            print(f"[n18]   {nm:52s} overspend {over:.3f}  underspend {under:.3f}  "
                  f"accuracy {acc_r:.3f}", flush=True)
        cost["overspend_quantile"] = float(np.mean([x["pred"] == "new-vocabulary" for x in rows
                                                    if x["truth"] == "new-combination"]))
        cost["overspend_costmin"] = float(np.mean([x["pred_cost"] == "new-vocabulary" for x in rows
                                                   if x["truth"] == "new-combination"]))
        cost["underspend_costmin"] = float(np.mean([x["pred_cost"] != "new-vocabulary" for x in rows
                                                    if x["truth"] == "new-vocabulary"]))
    else:
        cost = {}

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    cols = dict(zip(ROUTES, ["#4C78A8", "#F58518", "#54A24B"]))
    for r in ROUTES:
        axes[0].scatter(vals(r, "s_novel"), vals(r, "s_vocab"), s=16, alpha=0.75, label=r,
                        color=cols[r])
    axes[0].axvline(thr["t_novel"], ls="--", lw=1, color="0.4")
    axes[0].axhline(thr["t_vocab"], ls="--", lw=1, color="0.4")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(f"denoising error @ u=0.10  (novelty)")
    axes[0].set_ylabel(f"denoising error @ u=0.45  (vocabulary)")
    axes[0].set_title("One model-internal signal, two noise levels\n"
                      "(dashed = calibrated thresholds, no labels used)", fontsize=8)
    axes[0].legend(fontsize=7)
    M = np.array([[conf[t][pp] for pp in ROUTES] for t in ROUTES], float)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    axes[1].imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                         color="white" if Mn[i, j] > 0.5 else "black", fontsize=9)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels([r.split("-")[-1] for r in ROUTES],
                                                          fontsize=7)
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels([r.split("-")[-1] for r in ROUTES],
                                                          fontsize=7)
    axes[1].set_xlabel("routed to"); axes[1].set_ylabel("truth")
    axes[1].set_title(f"Self-routing, overall {acc:.2f}", fontsize=8)
    names = list(aur)
    xb = np.arange(len(names))
    axes[2].bar(xb - 0.2, [aur[k] for k in names], 0.4, label="best conditional")
    axes[2].bar(xb + 0.2, [aur_abl[k] for k in names], 0.4, label="unconditional (ablation)")
    axes[2].axhline(0.5, ls="--", color="0.4", lw=1)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(xb)
    axes[2].set_xticklabels(["novel vs\nin-dist", "vocab vs\ncombination"], fontsize=7)
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("The conditioning is NOT what decides:\nunconditional error does as well",
                      fontsize=8)
    axes[2].legend(fontsize=7)
    fig.suptitle("N18 — the system decides for itself which update an observation needs "
                 "(no probe, no label, no supervision)", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "n18_router.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    summ = dict(auroc=aur, auroc_uncond_ablation=aur_abl, confusion=conf, accuracy=acc,
                thresholds=thr, sweep=sweep, best_operating_point=best, cost=cost,
                signals={r: {k: float(np.mean(vals(r, k)))
                             for k in ("s_novel", "s_vocab", "uncond_novel", "uncond_vocab")}
                         for r in ROUTES})
    save_json(dict(rows=rows, e2e=e2e, **summ),
              os.path.join(RESULTS, "n18_router.json"), seed=args.seed)

    print(f"\n[n18] self-routing accuracy {acc:.3f} over {len(rows)} queries")
    print("    mean signal by route:")
    for r in ROUTES:
        s_ = summ["signals"][r]
        print(f"      {r:18s} u=0.10 {s_['s_novel']:.5f}   u=0.45 {s_['s_vocab']:.5f}")
    print("    AUROC (threshold-free):        conditional   unconditional-ablation")
    for k in aur:
        print(f"      {k:36s} {aur[k]:.3f}         {aur_abl[k]:.3f}")
    print("    confusion (row = truth):")
    for t in ROUTES:
        print(f"      {t:18s} " + "  ".join(f"{conf[t][pp]:3d}" for pp in ROUTES))
    print(f"[n18] wrote figures/n18_router.png and {RESULTS}/n18_router.json")


if __name__ == "__main__":
    main()
