"""N84 — one query, MANY prototypes: recovering coverage without giving up accuracy.

N83 measured the dial and found a CLIFF rather than a frontier: anchoring the reverse process to the
PoE prototype at ANY strength up to s=0.85 pins recall at 0.013-0.017, and only removing the anchor
entirely (s=1) restores it, at 0.637.  There is no intermediate operating point, because every one
of the n samples is anchored to THE SAME prototype and that basin is attractive enough to survive
85% noise.

The diagnosis names its own fix.  Their Stage 1 discovers MANY modes and their Eq. 8-9 submodular
step selects K COMPLEMENTARY ones -- and then Eq. 7 multiplies them into a single `mu_T`, which is
where the coverage goes.  A product of Gaussians is a Gaussian, so one query determines one image
(THEORY 4).  Their own 4.2 recovers diversity by running 100 different QUERIES.

This uses the selection for COVERAGE instead of for aggregation: run their discovery and their
submodular selection with K = n_samples, then anchor each sample to a DIFFERENT selected prototype.
One query, many prototypes, their machinery throughout -- only the aggregation step is dropped.

    poe-guided     one anchor  (their mu_T)              accurate, no coverage
    multi-proto    n anchors   (their Eq. 8-9 selection) <- this run
    address s=1    no anchor   (our composed conditional) coverage, less accurate

If multi-proto holds poe-guided's accuracy at materially higher recall, the coverage loss was never
intrinsic to the method -- it was the product step, and the fix costs no retraining.

    sbatch --export=ALL,SCRIPT="experiments/n84_celeba_multiproto.py" slurm/jobs/gpu.sbatch
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from common import CACHE                                              # noqa: E402

from lifelong_ood.data.celeba import load_celeba_factored             # noqa: E402
from lifelong_ood.data.factored import decode_slot, spread_targets    # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                save_json, vendi_score)
from lifelong_ood.instrument import check_instrument, min_detectable  # noqa: E402
from lifelong_ood.neural.idealwords import compose, cv_r2, factorize  # noqa: E402
from lifelong_ood.compose import (CELEBA_DISCOVERY,                   # noqa: E402
                                  PAPER_POE_CELEBA, candidates_ascent, poe_teacher)
from lifelong_ood.evals import precision_recall                       # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner,               # noqa: E402
                                 NeuralScoreField, UNet, sample_embedding, to_data, to_model)

RESULTS = os.path.join(HERE, "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_fac64.pt"))
    p.add_argument("--n-samples", type=int, default=24, help="per arm per cell")
    p.add_argument("--n-cells", type=int, default=4)
    p.add_argument("--dial", default="0.5:0.7:0.85",
                   help="COLON-separated u_start values; sbatch splits --export on commas")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features", default="inception")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck["split"]
    names = list(ck["names"])
    d = load_celeba_factored(img=int(ck["img"]), seed=int(sp["seed"]),
                             min_count=int(sp["min_count"]), n_holdout=int(sp["n_holdout"]),
                             # `max_rows=None` means "all rows" and is a legitimate stored value;
                             # int(None) is not.
                             max_rows=(None if sp.get("max_rows") is None
                                       else int(sp["max_rows"])),
                             n_per_class=int(sp["n_per_class"]),
                             as_uint8=False, binaries=list(ck["split"]["binaries"].split(":")))
    # 64px: the whole set is ~5 GB in float32, so no uint8 staging is needed.
    # CHUNKED, and in place.  `np.asarray(X, np.float32) / 255.0` allocates the 10 GB result AND
    # a 10 GB temporary for the division; on a contended node that swaps rather than runs.
    X, F, slot = (d["X"], np.asarray(d["factors"], int), np.asarray(d["slot"], int))
    remap = ck["remap"]
    seen_slots = np.array(sorted(int(k) for k in remap))
    # THE SPLIT MUST MATCH THE CHECKPOINT.  7kk is the whole reason this is asserted rather than
    # assumed: a loader default silently disagreeing with the trained split produced three separate
    # "findings" that were one bug.
    if sorted(set(int(s) for s in slot[np.asarray(d["seen"])])) != sorted(int(k) for k in remap):
        raise SystemExit("[n84] split mismatch against the checkpoint -- refusing to report")
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen_slots]]
    lab = np.asarray(decode_slot(seen_slots, ck["sizes"]), int)
    fac = factorize(C, lab)
    r2cv, _ = cv_r2(C, lab)
    n_free = 1 + sum(int(s) - 1 for s in ck["sizes"])
    print(f"[n84] {len(seen_slots)} seen / {len(ck['held'])} held out; basis R2 {fac['r2']:.3f} "
          f"in-sample, {r2cv:.3f} leave-one-out ({len(seen_slots)}/{n_free} = ratio "
          f"{len(seen_slots) / n_free:.1f})", flush=True)

    probes = fit_factor_probes(d, seed=args.seed, features=args.features)
    held = [int(h) for h in ck["held"] if (slot == int(h)).sum() >= 8]
    gate = check_instrument(tag="n84", real=d, probes=probes,
                            targets={h: tuple(int(v) for v in F[slot == h][0]) for h in held},
                            n_arm=args.n_samples, table=C, labels=lab, seed=args.seed)
    print(gate.summary("n80"), flush=True)
    readable = [h for h in held if gate.readable(h)]
    cells = spread_targets(readable,
                           {h: tuple(int(v) for v in F[slot == h][0]) for h in readable},
                           args.n_cells)
    print(f"[n84] {len(readable)} of {len(held)} held-out cells readable; judging {len(cells)}",
          flush=True)
    if not cells:
        print("[n84] no readable held-out cell -- nothing can be judged, and that IS the result")
        return 0

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])      # STRICT: a silent partial load is how N80 sampled
    field.model.eval()                          # from an untrained conditioning path
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)

    rows = []
    for h in cells:
        idx = np.where(slot == h)[0]
        tgt = tuple(int(v) for v in F[idx[0]])
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        # `to_data` on every generated row and `clip=(-1,1)` on `field.sample`: this field is
        # MODEL space and its base-class clip default is a DATA-range clamp (7ee's range fault).
        xq = to_model(X[idx[0]]).ravel()
        M, Vc, _meta = candidates_ascent(field, xq, seed=args.seed, **CELEBA_DISCOVERY)
        t_poe = poe_teacher(xq, M, Vc, K=L.cfg["K"], tau=L.cfg["tau"])
        proto = np.asarray(t_poe["mu"], np.float64).ravel()          # model space, like the field

        # THEIR selection, with K raised to the sample count: `sel` indexes the complementary
        # modes Eq. 8-9 picks.  `poe_teacher`'s `mu` -- the Eq. 7 product over them -- is computed
        # and DELIBERATELY unused for these arms; that product is where the coverage goes.
        t_wide = poe_teacher(xq, M, Vc, K=min(args.n_samples, len(M)), tau=L.cfg["tau"])
        sel = np.asarray(t_wide["sel"], int)
        anchors = np.asarray(M, np.float64)[sel]
        got = {
            "poe-guided (1 anchor)": to_data(L.sample_teacher(
                t_poe, n=args.n_samples, seed=args.seed, **PAPER_POE_CELEBA)),
        }
        for s in [float(v) for v in args.dial.split(":")]:
            # one sample per anchor, cycling if the pool is smaller than n_samples
            outs = []
            for i in range(args.n_samples):
                a_i = anchors[i % len(anchors)]
                outs.append(np.asarray(sample_embedding(
                    field, c_new, n=1, steps=args.steps, seed=args.seed + 7919 * i,
                    u_start=s, x_init=a_i), np.float64)[0])
            got[f"multi-proto s={s:g}"] = to_data(np.stack(outs))
        got["address s=1"] = to_data(sample_embedding(
            field, c_new, n=args.n_samples, steps=args.steps, seed=args.seed))
        got["unconditional"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed + 91, clip=(-1, 1), branch=None))
        print(f"[n84]   cell {h:>3}: {len(anchors)} distinct anchors from their selection",
              flush=True)

        for nm, S in got.items():
            S = np.asarray(S, np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, X[idx[:60]])
            rows.append(dict(cell=h, arm=nm, joint=float(a["joint"]),
                             recall=float(pr["recall"]), vendi=float(vendi_score(S))))
            print(f"[n84]   cell {h:>3} {nm:26s} joint {a['joint']:.3f} "
                  f"recall {pr['recall']:.3f} vendi {float(vendi_score(S)):.2f}", flush=True)
        R = X[idx[:args.n_samples]]
        rows.append(dict(cell=h, arm="ceiling", joint=float(factor_accuracy(probes, R, tgt)["joint"]),
                         vendi=float(vendi_score(R))))

    def m(arm):
        v = [r["joint"] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    n_pool = args.n_samples * len(cells)
    # Resolution is quoted against the FULL-COVERAGE end (s=1), which is the arm whose accuracy a
    # reader will compare to poe-guided's.
    full = "address s=1"
    res = min_detectable(max(m(full), 1e-6), n_pool)
    mr, mf = m(full) - m("poe-guided (1 anchor)"), m(full) - m("unconditional")
    def mm(arm, key):
        v = [r[key] for r in rows if r["arm"] == arm and key in r]
        return float(np.mean(v)) if v else float("nan")

    arms = ["poe-guided (1 anchor)"] \
        + [f"multi-proto s={float(v):g}" for v in args.dial.split(":")] \
        + ["address s=1", "unconditional", "ceiling"]
    print(f"\n[n84] {'arm':22s}{'joint':>8}{'recall':>8}{'vendi':>8}   (ceiling "
          f"{m('ceiling'):.3f})")
    for arm in arms:
        j, r = m(arm), mm(arm, "recall")
        print(f"[n84] {arm:22s}{j:>8.3f}{r:>8.3f}{mm(arm, 'vendi'):>8.2f}"
              f"   {j / max(m('ceiling'), 1e-9):>4.0%} of ceiling")
    # THE DIAGNOSTIC THIS RUN EXISTS FOR.
    poe = m("poe-guided (1 anchor)")
    # Is there an operating point that keeps their accuracy AND gains coverage?
    best = max(((m(a), mm(a, "recall"), a) for a in arms if a.startswith("multi-proto")),
               key=lambda z: (z[0] >= poe * 0.9) * z[1])
    print(f"[n84] poe-guided {poe:.3f} at recall {mm('poe-guided (1 anchor)', 'recall'):.3f}; "
          f"best dial point holding >=90% of that accuracy: {best[2]} "
          f"(joint {best[0]:.3f}, recall {best[1]:.3f})")

    save_json(dict(rows=rows, basis=dict(r2=float(fac["r2"]), r2_cv=float(r2cv),
                                         n_classes=len(seen_slots), n_free=n_free),
                   margin=dict(vs_poe=mr, vs_floor=mf, resolution=float(res), n=n_pool),
                   dial=[float(v) for v in args.dial.split(":")],
                   cells=cells, n_readable=len(readable), n_held=len(held),
                   instrument=gate.as_dict()),
              os.path.join(RESULTS, f"n84_celeba_multiproto{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
