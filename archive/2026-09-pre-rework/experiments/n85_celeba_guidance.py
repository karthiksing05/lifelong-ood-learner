"""N85 — sharpen the CONDITIONAL instead of anchoring it: guidance on the composed address.

Two runs have now closed the anchoring direction:

    N83  the dial is a CLIFF: recall 0.013-0.017 at every s <= 0.85, 0.637 only at s = 1
    N84  many anchors instead of one raise VENDI (1.25 -> 1.66) and leave RECALL at zero --
         the discovered prototypes are distinct from each other and collectively off the real
         distribution, which is 7ll's finding again (mode ascent finds density, not the class)

So coverage exists only in the model's own conditional, and anchoring destroys it however it is
applied.  The conditional already HAS the coverage (recall 0.637); what it lacks is accuracy
(19% of ceiling against poe-guided's 29%).  This attacks that side instead.

Ordinary classifier-free guidance against the null token sharpens a conditional toward its class.
It costs diversity -- but it starts from recall 0.637 rather than from 0.013, so there is room to
spend.  `interval=(lo, hi)` applies it ONLY inside a noise band: 5l-bis and Kynkaanniemi et al.
(arXiv:2404.07724) both find that guidance at LOW noise amplifies high-frequency error, which is
where the speckle in our samples came from -- and no accuracy or coverage metric here can see
speckle, so it has to be handled by construction rather than caught.

Read it as a FRONTIER: the useful outcome is a w where joint approaches poe-guided's 0.062 while
recall stays far above its 0.013.

    sbatch --export=ALL,SCRIPT="experiments/n85_celeba_guidance.py" slurm/jobs/gpu.sbatch
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
    p.add_argument("--dial", default="1.0:1.5:2.0:3.0",
                   help="COLON-separated guidance weights w; sbatch splits --export on commas")
    p.add_argument("--interval", default="0.3:1.0",
                   help="apply guidance only in this noise band (5l-bis: at low noise it "
                        "amplifies high-frequency error into speckle no metric here can see)")
    p.add_argument("--unused", default="",
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
        raise SystemExit("[n85] split mismatch against the checkpoint -- refusing to report")
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen_slots]]
    lab = np.asarray(decode_slot(seen_slots, ck["sizes"]), int)
    fac = factorize(C, lab)
    r2cv, _ = cv_r2(C, lab)
    n_free = 1 + sum(int(s) - 1 for s in ck["sizes"])
    print(f"[n85] {len(seen_slots)} seen / {len(ck['held'])} held out; basis R2 {fac['r2']:.3f} "
          f"in-sample, {r2cv:.3f} leave-one-out ({len(seen_slots)}/{n_free} = ratio "
          f"{len(seen_slots) / n_free:.1f})", flush=True)

    probes = fit_factor_probes(d, seed=args.seed, features=args.features)
    held = [int(h) for h in ck["held"] if (slot == int(h)).sum() >= 8]
    gate = check_instrument(tag="n85", real=d, probes=probes,
                            targets={h: tuple(int(v) for v in F[slot == h][0]) for h in held},
                            n_arm=args.n_samples, table=C, labels=lab, seed=args.seed)
    print(gate.summary("n80"), flush=True)
    readable = [h for h in held if gate.readable(h)]
    cells = spread_targets(readable,
                           {h: tuple(int(v) for v in F[slot == h][0]) for h in readable},
                           args.n_cells)
    print(f"[n85] {len(readable)} of {len(held)} held-out cells readable; judging {len(cells)}",
          flush=True)
    if not cells:
        print("[n85] no readable held-out cell -- nothing can be judged, and that IS the result")
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

        lo, hi = (float(v) for v in args.interval.split(":"))
        got = {
            "poe-guided (their Eq.14)": to_data(L.sample_teacher(
                t_poe, n=args.n_samples, seed=args.seed, **PAPER_POE_CELEBA)),
        }
        for w in [float(v) for v in args.dial.split(":")]:
            got[f"address w={w:g}"] = to_data(sample_embedding(
                field, c_new, n=args.n_samples, steps=args.steps, seed=args.seed,
                guidance=w, interval=(None if w == 1.0 else (lo, hi))))
        got["unconditional"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed + 91, clip=(-1, 1), branch=None))

        for nm, S in got.items():
            S = np.asarray(S, np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, X[idx[:60]])
            rows.append(dict(cell=h, arm=nm, joint=float(a["joint"]),
                             recall=float(pr["recall"]), vendi=float(vendi_score(S))))
            print(f"[n85]   cell {h:>3} {nm:26s} joint {a['joint']:.3f} "
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
    full = f"address w={float(args.dial.split(':')[0]):g}"
    res = min_detectable(max(m(full), 1e-6), n_pool)
    mr, mf = m(full) - m("poe-guided (their Eq.14)"), m(full) - m("unconditional")
    def mm(arm, key):
        v = [r[key] for r in rows if r["arm"] == arm and key in r]
        return float(np.mean(v)) if v else float("nan")

    arms = ["poe-guided (their Eq.14)"] \
        + [f"address w={float(v):g}" for v in args.dial.split(":")] \
        + ["unconditional", "ceiling"]
    print(f"\n[n85] {'arm':22s}{'joint':>8}{'recall':>8}{'vendi':>8}   (ceiling "
          f"{m('ceiling'):.3f})")
    for arm in arms:
        j, r = m(arm), mm(arm, "recall")
        print(f"[n85] {arm:22s}{j:>8.3f}{r:>8.3f}{mm(arm, 'vendi'):>8.2f}"
              f"   {j / max(m('ceiling'), 1e-9):>4.0%} of ceiling")
    # THE DIAGNOSTIC THIS RUN EXISTS FOR.
    poe = m("poe-guided (their Eq.14)")
    # Is there an operating point that keeps their accuracy AND gains coverage?
    best = max(((m(a), mm(a, "recall"), a) for a in arms if a.startswith("address")),
               key=lambda z: (z[0] >= poe * 0.9) * z[1])
    print(f"[n85] poe-guided {poe:.3f} at recall {mm('poe-guided (their Eq.14)', 'recall'):.3f}; "
          f"best dial point holding >=90% of that accuracy: {best[2]} "
          f"(joint {best[0]:.3f}, recall {best[1]:.3f})")

    save_json(dict(rows=rows, basis=dict(r2=float(fac["r2"]), r2_cv=float(r2cv),
                                         n_classes=len(seen_slots), n_free=n_free),
                   margin=dict(vs_poe=mr, vs_floor=mf, resolution=float(res), n=n_pool),
                   dial=[float(v) for v in args.dial.split(":")],
                   cells=cells, n_readable=len(readable), n_held=len(held),
                   instrument=gate.as_dict()),
              os.path.join(RESULTS, f"n85_celeba_guidance{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
