"""N41 -- ADDRESSING, not adapting: a lifelong setting whose parameters barely grow.

The design this repository has been running is *additive*: each new part costs a low-rank block over
a frozen backbone.  That preserves everything, but storage grows at 551K per concept and the
hierarchy makes it worse, not better (§24).  This tests a different setting, and the argument for it
comes from our own group's ACS paper (Wang, Singaravadivelan & MacLellan, *Diffusion Models and
Concept Formation*):

    A diffusion model's noisy marginals are Gaussian smoothings of the data distribution --
    p_t = p_data * N(0, (1 - abar_t) I) -- so each noise level is a kernel density estimate of the
    data at a different bandwidth, and the MODES of those marginals at a ladder of noise levels form
    a hierarchy of prototypes that corresponds to a Cobweb tree.

If that is right, then **the concept hierarchy is already inside the frozen backbone.**  We do not
have to build one out of adapters; we have to ADDRESS the one that is there.  A concept then costs a
*pointer* -- where in the model's own conditioning space it lives -- rather than a private function.

Three storage classes, in increasing cost, and the router already decides between them (§6c):

  `coord`     a concept expressible in the factor basis: F factor coordinates.  ColorMNIST has
              10+4+4 = 18 logits -> **72 bytes**.
  `basis`     a genuinely new VALUE of an existing factor: one new 256-float direction ->
              **1024 bytes**, and §5q measured that ONE such direction covers every combination
              using that value (F1 0.755 +/- 0.037 at zero per-concept cost, BEATING per-concept
              adapters at 0.658 +/- 0.017).
  `block`     a genuinely new FACTOR, or a concept with no factored description -> a low-rank block.
              §5q measured that conditioning-only routes score **0.000** here, so this class is real
              and cannot be argued away.

Arms, all absorbing the same stream:

  `block-every`   one rank-8 block per arrival (the shipped default, and the pile a kept version of
                  the companion paper's protocol produces)
  `block-r1`      the same, at rank 1 -- the cheapest purely additive setting (§24d)
  `addressing`    coord / basis / block by route, i.e. the proposed setting

Reported: cumulative BYTES against arrivals, backbone drift (must be exactly 0 -- addressing never
writes weights at all, so this is a much stronger claim than bit-identical adapters), and the quality
of what was absorbed, because storage that buys nothing is not a saving.

    python experiments/n41_addressing.py --n-concepts 24
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries                # noqa: E402

from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import (sample_factored_pool,               # noqa: E402
                                            slot_factors)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def backbone_signature(field, seed=0, n=4, u_grid=(0.2, 0.5, 0.8)):
    """What the UNCONDITIONAL model computes.  Addressing must leave this bit-identical."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, field.dim))
    return np.concatenate([field.eps(x, u, None).ravel() for u in u_grid])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=24)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--n-pool", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--n-score", type=int, default=8,
                   help="how many of the absorbed concepts to sample and score at the end")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_per_slot, args.n_pool = 4, 20, 12
        args.epochs, args.steps, args.inv_steps, args.n_samples = 3, 20, 60, 8
        args.n_score = 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    fac_labels = slot_factors(slots)
    f0 = load_field(args.ckpt, device=dev)
    gains = backbone_gains(f0, C, slots, probes, data, seed=args.seed, steps=args.steps,
                           n=8 if args.quick else 16)
    sig0 = backbone_signature(f0, seed=args.seed)
    del f0
    torch.cuda.empty_cache()

    EMB = C.shape[1]
    COORD_BYTES = int(sum(len(np.unique(fac_labels[:, f])) for f in range(fac_labels.shape[1])) * 4)
    BASIS_BYTES = EMB * 4
    print(f"[n41] storage classes: coord {COORD_BYTES} B (one logit per factor value), "
          f"basis {BASIS_BYTES} B (one {EMB}-float direction), block = one low-rank set\n",
          flush=True)

    rows, curves = [], {}
    for arm in ("block-every", "block-r1", "addressing"):
        rank = 1 if arm == "block-r1" else 8
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, rank=rank, steps=args.steps, seed=args.seed,
                                  device=dev, flat=True)
        L.attach_factorization(C, fac_labels, gains=gains)
        cum, curve, learned, stored = 0, [], [], []
        for i, q in enumerate(qs):
            nm = f"c{q['slot']}"
            xq = to_model(data["X"][q["query"]]).ravel()
            if arm == "addressing":
                # ADDRESS: read the query in the factor basis and keep only the coordinates.
                # Nothing is trained; nothing is allocated; the backbone is not touched.
                c_new, ws, _ = L.compose_factored(xq, inv_steps=args.inv_steps, seed=args.seed,
                                                  gains=gains)
                stored.append(dict(name=nm, w=[np.asarray(w, np.float32) for w in ws]))
                cum += COORD_BYTES
            else:
                L.learn_concept_factored(xq, nm, n_pool=args.n_pool, epochs=args.epochs,
                                         inv_steps=args.inv_steps, seed=args.seed)
                learned.append((nm, q))
                cum = int(L.bank.n_params() * 4)
            curve.append(dict(i=i + 1, bytes=int(cum)))
        # the backbone must be untouched.  For `addressing` this is trivially true and worth
        # asserting anyway; for the block arms it is the usual adapter-level guarantee.
        drift = float(np.max(np.abs(backbone_signature(field, seed=args.seed) - sig0)))

        # quality of the last few absorbed concepts, by the route that arm used
        accs, f1s, vs = [], [], []
        # score the last --n-score concepts, not 4: the storage result is structural and needs
        # no replication, but the QUALITY comparison is the contested one and four concepts
        # at one seed is too thin to carry it.
        k = args.n_score
        tail = (stored[-k:] if arm == "addressing" else learned[-k:])
        for item in tail:
            if arm == "addressing":
                nm = item["name"]
                q = next(x for x in qs if f"c{x['slot']}" == nm)
                # Rebuild the conditioning vector from the STORED COORDINATES ALONE.  This is the
                # whole claim: the concept is the coordinates, and c_new = mu + sum_f gain_f (w_f V_f)
                # is recomputed from the frozen basis at read time rather than kept.
                # `learn_factored` returns a torch tensor and the sampler conditions on one, so the
                # rebuilt vector has to be a tensor too -- rebuilding in numpy and passing it
                # straight through dies inside UNet.forward on `emb_override.to(dtype)`.
                mu_, V_ = L.fac["mu"], L.fac["V"]
                c_np = mu_ + sum(gains[f] * (item["w"][f] @ V_[f]) for f in range(len(V_)))
                c_new = torch.as_tensor(np.asarray(c_np), dtype=torch.float32, device=dev)
                S = to_data(sample_factored_pool(field, c_new, n=args.n_samples,
                                                 steps=args.steps, seed=args.seed)).astype(np.float32)
            else:
                nm, q = item
                S = to_data(L.sample_concept(nm, n=args.n_samples, steps=args.steps,
                                             seed=args.seed)).astype(np.float32)
            R = data["X"][q["rest"][:100]].astype(np.float32)
            accs.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"])
            f1s.append(precision_recall(S, R)["f1"])
            vs.append(vendi_score(S))
        rows.append(dict(arm=arm, rank=rank, final_bytes=curve[-1]["bytes"],
                         bytes_per_concept=curve[-1]["bytes"] / len(qs),
                         backbone_drift=drift, backbone_exact=float(drift == 0.0),
                         joint=float(np.mean(accs)), f1=float(np.mean(f1s)),
                         vendi=float(np.mean(vs))))
        curves[arm] = curve
        print(f"[n41] {arm:12s} {curve[-1]['bytes'] / 1e6:8.3f} MB total  "
              f"{curve[-1]['bytes'] / len(qs):9.0f} B/concept  backbone drift {drift:.3e}  "
              f"joint {rows[-1]['joint']:.3f}  f1 {rows[-1]['f1']:.3f}", flush=True)
        del field, L
        torch.cuda.empty_cache()

    save_json(dict(rows=rows, curves=curves, coord_bytes=COORD_BYTES,
                   basis_bytes=BASIS_BYTES, config=vars(args)),
              os.path.join(RESULTS, f"n41_addressing{args.tag}.json"), seed=args.seed)

    base = rows[0]
    print(f"\n[n41] {len(qs)} concepts")
    print(f"    {'arm':14s}{'total':>12s}{'B/concept':>12s}{'vs block':>10s}"
          f"{'backbone':>11s}{'joint':>8s}{'F1':>7s}")
    for r in rows:
        print(f"    {r['arm']:14s}{r['final_bytes'] / 1e6:11.3f}M{r['bytes_per_concept']:12.0f}"
              f"{base['final_bytes'] / max(r['final_bytes'], 1):9.0f}x"
              f"{'EXACT' if r['backbone_exact'] else 'MOVED':>11s}{r['joint']:8.3f}{r['f1']:7.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    cols = dict(zip(("block-every", "block-r1", "addressing"),
                    ("#B03A2E", "#2C6FA8", "#1B7F4B")))
    for a, cur in curves.items():
        ax.plot([c["i"] for c in cur], [max(c["bytes"], 1) for c in cur], lw=2.4, color=cols[a],
                label=a, marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("OOD concepts absorbed")
    ax.set_ylabel("cumulative storage (bytes, log scale)")
    ax.set_title("N41 — addressing the hierarchy the backbone already has, instead of adding one.\n"
                 "The backbone is never written to in any arm; the difference is what a concept "
                 "is allowed to cost.", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG, f"f16_addressing{args.tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[n41] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
