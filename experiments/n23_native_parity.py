"""N23 — native parity: is a CONSOLIDATED concept as good as one the backbone was TRAINED on?

The project's own sentence is "adapt new instances into the manifold in a way that is similar to if
they were learned with the existing ideas".  Every experiment so far argues that indirectly --
preservation is bit-exact, accuracy beats baselines, forgetting is 0.000 -- but none of them
measures the thing the sentence actually claims, because the comparison it names has never been
available: what that slot WOULD have looked like had it never been held out.

`n1_train --all-slots` supplies it.  A backbone trained on all 160 ColorMNIST slots is not a method
and is not competing; it is the reference a consolidated concept is trying to reach.  The number
this experiment exists to produce is the ratio

    parity  =  consolidated / native

on each axis, for concepts the deployed backbone never saw and never retrained for.

Arms:
  `native`        the oracle backbone, conditioned on the slot's own trained class.  The reference.
  `ours`          the 120-slot backbone + composition + consolidation.  Never retrained.
  `paper`         a flat per-query LoRA distilled from the PoE teacher — the source paper's
                  consolidation step, which it performs and then discards.
  `real`          held-out images.  The data ceiling both backbones are approximating.

And the cost side, which is what makes parity meaningful rather than merely favourable: DRAG on the
rest of the manifold.  The oracle pays for its held-out slots by retraining on everything; `ours`
pays adapter parameters.  Seen-slot accuracy is therefore measured on both backbones before and
after, so a parity claim cannot hide damage done elsewhere.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries          # noqa: E402

from lifelong_ood.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, to_data,  # noqa: E402
                                 to_model)
from lifelong_ood.neural.idealwords import slot_factors                 # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def score(probes, S, q, R):
    a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
    pr = precision_recall(S, R)
    g = sample_geometry(S, R[0], R)
    return dict(joint=a["joint"], digit=a["digit"], fg=a["fg"], bg=a["bg"],
                precision=pr["precision"], recall=pr["recall"], f1=pr["f1"],
                content_ratio=g.get("content_ratio", float("nan")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--oracle", default=os.path.join(CACHE, "ddpm_colormnist_oracle.pt"))
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--n-seen-check", type=int, default=8, help="seen slots for the drag check")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the output file; --quick runs "
                                             "use one so a smoke can never overwrite a "
                                             "full result")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_cases, args.n_samples, args.n_per_slot = 2, 8, 20
        args.tag = args.tag or "_smoke"
        args.steps, args.inv_steps, args.epochs, args.n_seen_check = 20, 60, 4, 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    field = load_field(args.ckpt, device=dev)
    gains = backbone_gains(field, C, slots_seen, probes, data, seed=args.seed, steps=args.steps,
                           n=8 if args.quick else 16)

    ock = torch.load(args.oracle, map_location="cpu", weights_only=False)
    oracle = load_field(args.oracle, device=dev)
    oremap = ock["remap"]
    print(f"[n23] deployed backbone {len(remap)} classes | oracle {len(oremap)} classes "
          f"(all slots) | {len(qs)} cases | device {dev}", flush=True)
    missing = [q["slot"] for q in qs if int(q["slot"]) not in oremap]
    assert not missing, f"oracle is missing held-out slots {missing[:5]} — was it --all-slots?"

    rows = []

    # ── the reference: the slot as the backbone would have had it ───────────
    for q in qs:
        S = to_data(oracle.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                  clip=(-1, 1),
                                  branch=int(oremap[int(q["slot"])]))).astype(np.float32)
        rows.append(dict(arm="native", slot=q["slot"],
                         **score(probes, S, q, data["X"][q["rest"][:100]])))
    print(f"[n23] native reference done", flush=True)

    # ── ours: compose, consolidate, never retrain ───────────────────────────
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, slot_factors(slots_seen), gains=gains)
    t0 = time.time()
    for qi, q in enumerate(qs):
        L.learn_concept_factored(to_model(data["X"][q["query"]]).ravel(), f"c{q['slot']}",
                                 n_pool=48, epochs=args.epochs, inv_steps=args.inv_steps,
                                 seed=args.seed + qi)
    ours_wall = time.time() - t0
    for q in qs:
        S = to_data(L.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                     seed=args.seed)).astype(np.float32)
        rows.append(dict(arm="ours", slot=q["slot"],
                         **score(probes, S, q, data["X"][q["rest"][:100]])))
    print(f"[n23] ours: {len(qs)} concepts consolidated in {ours_wall:.0f}s, "
          f"{L.bank.n_params() / 1e6:.2f}M adapter params", flush=True)

    # ── the paper's consolidation: one flat LoRA per query, from the teacher ─
    field_p = load_field(args.ckpt, device=dev)
    Lp = NeuralLifelongLearner(field_p, steps=args.steps, seed=args.seed, device=dev, flat=True)
    for qi, q in enumerate(qs):
        xq = to_model(data["X"][q["query"]]).ravel()
        t = Lp.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4),
                       n_starts=8 if args.quick else 16, iters=30 if args.quick else 120)
        pool = Lp.sample_teacher(t, n=48, seed=args.seed + qi)
        Lp.consolidate(pool, f"p{q['slot']}", mu=t["mu"], epochs=args.epochs)
    for q in qs:
        S = to_data(Lp.sample_concept(f"p{q['slot']}", n=args.n_samples,
                                      seed=args.seed)).astype(np.float32)
        rows.append(dict(arm="paper", slot=q["slot"],
                         **score(probes, S, q, data["X"][q["rest"][:100]])))
    print(f"[n23] paper arm done", flush=True)

    # ── the data ceiling ────────────────────────────────────────────────────
    for q in qs:
        R = data["X"][q["rest"][:100]].astype(np.float32)
        rows.append(dict(arm="real", slot=q["slot"],
                         **score(probes, R[:args.n_samples], q, R[args.n_samples:])))

    # ── drag: what did each backbone do to the slots it already had? ────────
    # Read these three rows for different reasons.  `ours (after)` is expected to equal
    # `deployed (before)` EXACTLY -- the base weights are frozen and a consolidated concept's
    # adapter path is not active when sampling a base class -- so it is a consistency check on that
    # invariant, not evidence for it (the evidence is the bit-identical fingerprint test, 6b).
    # The informative comparison is the ORACLE: it bought its held-out slots by retraining on
    # everything, and this asks what that cost the slots it already had.
    rng = np.random.default_rng(args.seed)
    seen_check = rng.permutation(slots_seen)[:args.n_seen_check]
    drag = {}
    for tag, fld, rm, bank in (("deployed (before)", field, remap, None),
                               ("ours (after)", field, remap, L),
                               ("oracle", oracle, oremap, None)):
        accs = []
        for s in seen_check:
            d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
            S = to_data(fld.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                   clip=(-1, 1), branch=int(rm[int(s)]))).astype(np.float32)
            accs.append(composition_accuracy(probes, S, d_, f_, b_)["joint"])
        drag[tag] = float(np.mean(accs))
        print(f"[n23] seen-slot accuracy, {tag}: {drag[tag]:.3f}", flush=True)

    keys = ("joint", "digit", "fg", "bg", "precision", "recall", "f1", "content_ratio")
    arms = ["real", "native", "ours", "paper"]
    summary = {a: {k: float(np.nanmean([x[k] for x in rows if x["arm"] == a])) for k in keys}
               for a in arms}
    parity = {a: {k: (float(summary[a][k] / summary["native"][k])
                      if summary["native"][k] not in (0.0,) else float("nan"))
                  for k in ("joint", "f1", "recall", "content_ratio")}
              for a in ("ours", "paper")}

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    mk = ("joint", "f1", "recall")
    x = np.arange(len(mk))
    w = 0.2
    for i, a in enumerate(arms):
        axes[0].bar(x + (i - 1.5) * w, [summary[a][k] for k in mk], w, label=a)
    axes[0].set_xticks(x); axes[0].set_xticklabels(mk)
    axes[0].set_ylabel("score")
    axes[0].set_title("Held-out compositions: consolidated vs natively trained", fontsize=9)
    axes[0].legend(fontsize=7)
    pk = ["joint", "f1", "recall"]
    xp = np.arange(len(pk))
    for i, a in enumerate(("ours", "paper")):
        axes[1].bar(xp + (i - 0.5) * 0.35, [parity[a][k] for k in pk], 0.35, label=a)
    axes[1].axhline(1.0, ls="--", color="0.3", lw=1)
    axes[1].set_xticks(xp); axes[1].set_xticklabels(pk)
    axes[1].set_ylabel("consolidated / native")
    axes[1].set_title("Native-parity ratio (1.0 = as good as trained on it)", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.suptitle("N23 — parity against a backbone that was TRAINED on the held-out slots, "
                 "at zero retraining", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"n23_native_parity{args.tag}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    save_json(dict(rows=rows, summary=summary, parity=parity, drag=drag,
                   ours_params=int(L.bank.n_params()), ours_wall_seconds=ours_wall,
                   n_cases=len(qs)),
              os.path.join(RESULTS, f"n23_native_parity{args.tag}.json"), seed=args.seed)

    print(f"\n[n23] native parity over {len(qs)} held-out compositions")
    print(f"    {'arm':10s}{'joint':>7s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'div':>7s}")
    for a in arms:
        v = summary[a]
        print(f"    {a:10s}{v['joint']:7.2f}{v['digit']:7.2f}{v['fg']:5.2f}{v['bg']:5.2f}"
              f"{v['precision']:7.2f}{v['recall']:8.2f}{v['f1']:7.2f}{v['content_ratio']:7.2f}")
    print(f"\n    parity (consolidated / native):")
    for a in ("ours", "paper"):
        print(f"      {a:8s}" + "  ".join(f"{k} {parity[a][k]:.2f}"
                                          for k in ("joint", "f1", "recall")))
    print(f"\n    seen-slot accuracy (`ours (after)` matches `deployed (before)` by construction;")
    print(f"    the oracle row is what retraining on everything cost the slots it already had):")
    for k, v in drag.items():
        print(f"      {k:20s} {v:.3f}")
    print(f"[n23] wrote figures/n23_native_parity.png")


if __name__ == "__main__":
    main()
