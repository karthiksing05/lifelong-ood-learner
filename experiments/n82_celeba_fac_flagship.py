"""N82 — the flagship pipeline on the FROM-SCRATCH 64px CelebA table (N81).

Same arms, same gate, same reporting as 7jj's Shapes3D run, so the two are directly comparable:

    address        compose(fac, factors) -> sample the model's own conditional
    nearest-class  the closest SEEN class embedding: the control for "the model already had it"
    unconditional  the floor
    ceiling        REAL images of the held-out combination, scored by the same probes

**Read the margins POOLED, not per cell.**  7nn measured why: at 24 samples an arm resolves margins
no finer than ~0.19, and this dataset's whole band is ~0.2 wide.  Pooling k cells multiplies the
effective sample count by k, and the pooled resolution is what the verdict is checked against.

**Probes are fitted on INCEPTION features.**  `diag_celeba_attr_readability.py` measured 26 of 40
attributes readable in pixel space against 36 in Inception, and CelebA is the dataset where that gap
decides whether anything can be judged at all.

**THE FIRST NUMBER TO READ IS `nearest-class` AGAINST `ceiling`.**  A trained class embedding must
reproduce its own combination before any composition number means anything.  N80 reported ADDRESS
margins twice on a backbone where that check reads 0.000-0.031 against a 0.240 ceiling, and every
conclusion drawn from those runs had to be withdrawn.  This gate comes first here.

    sbatch --export=ALL,SCRIPT="experiments/n82_celeba_fac_flagship.py" slurm/jobs/gpu.sbatch
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
    p.add_argument("--n-cells", type=int, default=6)
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
        raise SystemExit("[n82] split mismatch against the checkpoint -- refusing to report")
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen_slots]]
    lab = np.asarray(decode_slot(seen_slots, ck["sizes"]), int)
    fac = factorize(C, lab)
    r2cv, _ = cv_r2(C, lab)
    n_free = 1 + sum(int(s) - 1 for s in ck["sizes"])
    print(f"[n82] {len(seen_slots)} seen / {len(ck['held'])} held out; basis R2 {fac['r2']:.3f} "
          f"in-sample, {r2cv:.3f} leave-one-out ({len(seen_slots)}/{n_free} = ratio "
          f"{len(seen_slots) / n_free:.1f})", flush=True)

    probes = fit_factor_probes(d, seed=args.seed, features=args.features)
    held = [int(h) for h in ck["held"] if (slot == int(h)).sum() >= 8]
    gate = check_instrument(tag="n82", real=d, probes=probes,
                            targets={h: tuple(int(v) for v in F[slot == h][0]) for h in held},
                            n_arm=args.n_samples, table=C, labels=lab, seed=args.seed)
    print(gate.summary("n80"), flush=True)
    readable = [h for h in held if gate.readable(h)]
    cells = spread_targets(readable,
                           {h: tuple(int(v) for v in F[slot == h][0]) for h in readable},
                           args.n_cells)
    print(f"[n82] {len(readable)} of {len(held)} held-out cells readable; judging {len(cells)}",
          flush=True)
    if not cells:
        print("[n82] no readable held-out cell -- nothing can be judged, and that IS the result")
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
        # THE COMPANION PAPER'S OWN METHOD, as the ACCURACY control.  `nearest-class` only asks
        # whether a trained class works; `poe-guided` asks whether the backbone can be made to
        # generate this held-out combination AT ALL, by a method already known to score 0.958 joint
        # on ColorMNIST (at recall 0.000 -- it re-emits one query, THEORY 4).  If it lands high here
        # and ADDRESS does not, the fault is in our composition path, not the backbone, the probes
        # or the table.  If it lands at the floor too, the backbone cannot render the combination
        # and no composer could.
        #
        # w0 = 0.5, the paper's Table 8 CelebA value: guiding CelebA at ColorMNIST's 1.2 is 2.4x the
        # published strength and is recorded as a large part of why this arm looked broken here.
        xq = to_model(X[idx[0]]).ravel()
        M, Vc, _meta = candidates_ascent(field, xq, seed=args.seed, **CELEBA_DISCOVERY)
        t_poe = poe_teacher(xq, M, Vc, K=L.cfg["K"], tau=L.cfg["tau"])
        got = {
            "poe-guided (their Eq.14)": to_data(L.sample_teacher(
                t_poe, n=args.n_samples, seed=args.seed, **PAPER_POE_CELEBA)),
            "query-only": to_data(L.sample_teacher(
                L.baseline_teacher(xq, "query_only"), n=args.n_samples, seed=args.seed,
                **PAPER_POE_CELEBA)),
            "address": to_data(sample_embedding(field, c_new, n=args.n_samples,
                                                steps=args.steps, seed=args.seed)),
            "nearest-class": to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                  seed=args.seed, clip=(-1, 1),
                                                  # C is reordered by seen_slots, so j_near is an
                                                  # index into C, NOT a row of the model's table.
                                                  branch=int(remap[int(seen_slots[j_near])]))),
            "unconditional": to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                  seed=args.seed + 91, clip=(-1, 1), branch=None)),
        }
        for nm, S in got.items():
            S = np.asarray(S, np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, X[idx[:60]])
            rows.append(dict(cell=h, arm=nm, joint=float(a["joint"]),
                             recall=float(pr["recall"]), vendi=float(vendi_score(S))))
            print(f"[n82]   cell {h:>3} {nm:26s} joint {a['joint']:.3f} "
                  f"recall {pr['recall']:.3f} vendi {float(vendi_score(S)):.2f}", flush=True)
        R = X[idx[:args.n_samples]]
        rows.append(dict(cell=h, arm="ceiling", joint=float(factor_accuracy(probes, R, tgt)["joint"]),
                         vendi=float(vendi_score(R))))

    def m(arm):
        v = [r["joint"] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    n_pool = args.n_samples * len(cells)
    res = min_detectable(max(m("address"), 1e-6), n_pool)
    mr, mf = m("address") - m("nearest-class"), m("address") - m("unconditional")
    def mm(arm, key):
        v = [r[key] for r in rows if r["arm"] == arm and key in r]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n[n82] {'arm':28s}{'joint':>8}{'recall':>8}{'vendi':>8}")
    for arm in ("poe-guided (their Eq.14)", "query-only", "address", "nearest-class",
                "unconditional", "ceiling"):
        print(f"[n82] {arm:28s}{m(arm):>8.3f}{mm(arm, 'recall'):>8.3f}{mm(arm, 'vendi'):>8.2f}")
    # THE DIAGNOSTIC THIS RUN EXISTS FOR.
    poe = m("poe-guided (their Eq.14)")
    print(f"[n82] poe-guided reaches {poe:.3f} = {poe / max(m('ceiling'), 1e-9):.0%} of ceiling; "
          f"ADDRESS {m('address') / max(m('ceiling'), 1e-9):.0%}")
    print("[n82]   " + ("their method CAN render these cells -- the gap is OUR composition path"
                        if poe > m("unconditional") + res else
                        "their method cannot either -- the BACKBONE cannot render the combination, "
                        "and no composer could"))
    cond = m("nearest-class") - m("unconditional")
    print(f"[n82] CONDITIONING CHECK: nearest-class {m('nearest-class'):.3f} vs ceiling "
          f"{m('ceiling'):.3f} ({m('nearest-class') / max(m('ceiling'), 1e-9):.0%} of it), "
          f"over the floor {cond:+.3f}")
    if cond <= res:
        print("[n82]   A TRAINED CLASS CANNOT REPRODUCE ITS OWN COMBINATION. Nothing below is a "
              "composition result -- this is what N80 measured twice without checking.")
    print(f"[n82] margin vs retrieval {mr:+.3f} | vs floor {mf:+.3f} | resolution {res:.3f} "
          f"at n={n_pool} pooled over {len(cells)} cells")
    ok_r, ok_f = mr > res, mf > res
    print(f"[n82] VERDICT: beats retrieval {'YES' if ok_r else 'NO'} | "
          f"beats floor {'YES' if ok_f else 'NO'}  (a margin below resolution is not a result)")
    save_json(dict(rows=rows, basis=dict(r2=float(fac["r2"]), r2_cv=float(r2cv),
                                         n_classes=len(seen_slots), n_free=n_free),
                   margin=dict(vs_retrieval=mr, vs_floor=mf, resolution=float(res), n=n_pool),
                   cells=cells, n_readable=len(readable), n_held=len(held),
                   instrument=gate.as_dict()),
              os.path.join(RESULTS, f"n82_celeba_fac_flagship{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
