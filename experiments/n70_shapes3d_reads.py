"""N70 -- close the loop on Shapes3D: replace the GROUND-TRUTH factors with a real read.

The Shapes3D case study composes its ADDRESS row from `decode_slot(slot)` -- the held-out
combination's own factor values, handed to it.  So the figure demonstrates that the backbone can
RENDER an unseen combination given its coordinates, which is a real claim, but not the one
`docs/THE_TASK.md` makes ("given ONE example ... read the composition into the conditioning space").
Two of the other three case studies have the same shortcut; this closes it on the one where the
backbone is pretrained and 64 px, i.e. the setting closest to a policy.

Scored against the ground-truth arm as a CEILING rather than as a peer -- the point is how much is
lost by having to discover the coordinates instead of being told them:

  address-oracle    compose(fac, TRUE factors)                the current figure, now labelled
  address-muT       compose(fac, probe(mu_T))                 their discovery, our generation
  address-probe     compose(fac, probe(query))                the trivial read
  nearest-class     the closest trained embedding             the "did it just retrieve" control
  unconditional     the floor

On Color-MNIST every read agreed (12/12) and the arms collapsed onto each other; the interesting
question there was answered by the collapse itself.  Shapes3D differs in the way that matters: the
probes are strong on floor_hue and shape but `object_hue` drops out once `wall_hue` is pinned
(5t-bis), so the read has somewhere to fail, and the backbone is one whose conditioning had to be
trained at a 50x learning rate against an already-minimised loss.

    python experiments/n70_shapes3d_reads.py --n-cases 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.compose import (PAPER_COLORMNIST, candidates_ascent,  # noqa: E402
                                  poe_teacher)
from lifelong_ood.data.factored import decode_slot, load_factored      # noqa: E402
from lifelong_ood.evals import precision_recall, save_json, vendi_score  # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize          # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field                # noqa: E402

ARMS = ("address-oracle", "address-muT", "address-muT-agnostic", "address-probe",
        "nearest-class", "unconditional")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_shapes3d_pre2.pt"))
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_cases, args.n_samples, args.steps = 2, 6, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, ck = load_ve_field(args.ckpt, device=dev)
    sp = ck["split"]
    data = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                         value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                         fix_nuisance=sp["fix_nuisance"])
    X, slots = np.asarray(data["X"], np.float32), np.asarray(data["slot"])
    names, sizes = data["names"], data["sizes"]
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema_cemb"].numpy()[[remap[int(s)] for s in seen]]
    fac = factorize(C, decode_slot(seen, sizes))
    held = np.array(sorted(set(int(s) for s in np.unique(slots)) - set(seen.tolist())))
    print(f"[n70] basis R2 {fac['r2']:.3f} | {len(held)} held-out slots | factors {names}",
          flush=True)

    from lifelong_ood.evals import fit_factor_probes
    probes = fit_factor_probes(data, seed=args.seed)
    readable = [n for n in names if n in probes]
    # REPORT THE INSTRUMENT.  A read can only be as good as the probe doing it, and 5t-bis records
    # that `object_hue` drops out on this split once `wall_hue` is pinned.  `fit_factor_probes`
    # admits a probe at min_val_acc=0.55, which is a low bar for a 10-value factor, so a factor can
    # be "readable" and still be wrong most of the time.  Print the held-out accuracy per factor so
    # a failed READ is attributable to the probe rather than to the method.
    val = {n: float(probes[n].acc(np.asarray(data["X"], np.float32),
                                  np.asarray(data["factors"])[:, i]))
           for i, n in enumerate(names) if n in probes}
    print(f"[n70] readable factors and their accuracy on REAL images: "
          + ", ".join(f"{n} {v:.3f}" for n, v in val.items()), flush=True)
    # SCORE ONLY WHAT THE INSTRUMENT CAN READ.  `object_hue` measures 0.239 against a 0.20 chance
    # baseline for a 5-value factor -- it is not readable from pixels on this split, which 5t-bis
    # already records ("object_hue drops out once wall_hue is pinned ... so `joint` is a two-factor
    # joint").  Including it makes the joint read 0/8 by construction and turns this experiment into
    # a test of the probe.  n58 scores the same two-factor joint, so restricting here also keeps the
    # two Shapes3D numbers comparable.
    USABLE_AT = 0.60
    scored = [n for n, v in val.items() if v >= USABLE_AT]
    dropped = [(n, v) for n, v in val.items() if v < USABLE_AT]
    print(f"[n70] scoring over {scored}; DROPPED {[(str(n), round(v, 3)) for n, v in dropped]} "
          f"as unreadable (chance is 1/size)", flush=True)
    if not scored:
        print("[n70] no factor is readable on this split -- nothing here can be measured")
        return 1

    sidx = [i for i, n in enumerate(names) if n in scored]

    def read(img_row):
        """Full-length factor tuple, but only `scored` positions are ever compared."""
        x = np.asarray(img_row, np.float32).reshape(1, -1)
        return tuple(int(probes[n].predict(x)[0]) if n in probes else 0 for n in names)

    def sub(t):
        return tuple(t[i] for i in sidx)

    disc = dict(PAPER_COLORMNIST)
    if args.quick:
        disc.update(u_grid=(0.1, 0.3), n_starts=8, iters=40)
    rows, reads = [], []
    for ci, s in enumerate(held[:args.n_cases]):
        idx = np.where(slots == int(s))[0]
        tgt = tuple(int(v) for v in decode_slot(np.array([int(s)]), sizes)[0])
        xq = X[int(idx[0])]
        R = X[idx[1:1 + 40]]
        got = {}
        got["address-oracle"] = np.asarray(field.sample(n=args.n_samples, steps=args.steps,
                                                        seed=args.seed + ci,
                                                        emb_override=compose(fac, tgt)))
        r_probe = read(xq)
        got["address-probe"] = np.asarray(field.sample(n=args.n_samples, steps=args.steps,
                                                       seed=args.seed + ci,
                                                       emb_override=compose(fac, r_probe)))
        # THE PAPER'S ACTUAL PIPELINE.  Until the VE bridge grew eps/score/laplace_diag/
        # mode_ascent, `candidates_ascent` could not run on this backbone at all, and this arm was a
        # 4-start single-noise-level unweighted mean of modes standing in for Eq. 7 -- not a fair
        # test of their method.  It now runs Eq. 4 discovery, Eq. 8-9 selection and the Eq. 7
        # product, at their ColorMNIST-scale budget since their CelebA column collapses (N68).
        M, Vv, meta = candidates_ascent(field, xq, seed=args.seed + 17 * ci, **disc)
        t = poe_teacher(xq, M, Vv, K=3, tau=0.05)
        r_mu = read(t["mu"])
        # and the SAME product restricted to query-agnostic experts, because on both other backbones
        # selection returned only query-seeded ones and a teacher anchored at the query is the
        # `query-only` baseline wearing the method's name (5e-quater).
        elig = np.where(np.asarray(meta["src"]) == "random")[0]
        if len(elig) >= 3:
            t_ag = poe_teacher(xq, M, Vv, K=3, tau=0.05, eligible=elig)
            r_ag = read(t_ag["mu"])
        else:
            r_ag = r_mu
        got["address-muT-agnostic"] = np.asarray(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed + ci,
            emb_override=compose(fac, r_ag)))
        sel_src = list(np.asarray(meta["src"])[t["sel"]])
        got["address-muT"] = np.asarray(field.sample(n=args.n_samples, steps=args.steps,
                                                     seed=args.seed + ci,
                                                     emb_override=compose(fac, r_mu)))
        d2 = ((C - compose(fac, tgt)[None]) ** 2).sum(1)
        got["nearest-class"] = np.asarray(field.sample(n=args.n_samples, steps=args.steps,
                                                       seed=args.seed + ci,
                                                       branch=remap[int(seen[int(np.argmin(d2))])]))
        got["unconditional"] = np.asarray(field.sample(n=args.n_samples, steps=args.steps,
                                                       seed=args.seed + 71 * (ci + 1)))
        reads.append(dict(slot=int(s), target=list(tgt), muT=list(r_mu), probe=list(r_probe),
                          muT_agnostic=list(r_ag), selected_from=[str(x) for x in sel_src],
                          agnostic_correct=bool(sub(r_ag) == sub(tgt)),
                          muT_correct=bool(sub(r_mu) == sub(tgt)),
                          probe_correct=bool(sub(r_probe) == sub(tgt))))
        print(f"[n70] case {ci + 1} slot {int(s)} target {tgt} | mu_T {r_mu} "
              f"{'OK' if sub(r_mu) == sub(tgt) else 'WRONG'} | probe {r_probe} "
              f"{'OK' if sub(r_probe) == sub(tgt) else 'WRONG'}  "
              f"(scored on {[str(x) for x in scored]})", flush=True)
        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            hits = [float((probes[n].predict(S) == tgt[i]).mean())
                    for i, n in enumerate(names) if n in scored]
            pr = precision_recall(S, R)
            rows.append(dict(arm=arm, slot=int(s), joint=float(np.prod(hits)) if hits else 0.0,
                             per_factor=float(np.mean(hits)) if hits else 0.0,
                             recall=pr["recall"], precision=pr["precision"],
                             vendi=float(vendi_score(S))))

    print(f"\n[n70] {'=' * 72}")
    print(f"{'arm':18s}{'joint':>9s}{'per-factor':>12s}{'recall':>9s}{'precision':>11s}{'vendi':>8s}")
    summ = {}
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm]
        summ[arm] = {k: float(np.mean([r[k] for r in g])) for k in
                     ("joint", "per_factor", "recall", "precision", "vendi")}
        print(f"{arm:18s}{summ[arm]['joint']:>9.3f}{summ[arm]['per_factor']:>12.3f}"
              f"{summ[arm]['recall']:>9.3f}{summ[arm]['precision']:>11.3f}"
              f"{summ[arm]['vendi']:>8.2f}")
    n = len(reads)
    n_mu = sum(r["muT_correct"] for r in reads)
    n_ag = sum(r["agnostic_correct"] for r in reads)
    n_pq = sum(r["probe_correct"] for r in reads)
    from collections import Counter
    prov = Counter(x for r in reads for x in r["selected_from"])
    print(f"\n[n70] READ accuracy over {n} slots -- mu_T {n_mu}/{n} | "
          f"mu_T query-agnostic {n_ag}/{n} | probe-query {n_pq}/{n}")
    print(f"[n70] selected experts came from: {dict(prov)}  "
          f"(all-query means the teacher is anchored at the query -- 5e-quater)")
    print("[n70] VERDICT")
    o, m = summ["address-oracle"]["joint"], summ["address-muT"]["joint"]
    fl = summ["unconditional"]["joint"]
    if n_ag > max(n_mu, n_pq):
        print("[n70]   THE COMPLEMENTARITY CONSTRAINT is what pays: forcing selection onto experts")
        print("[n70]   that DISAGREE with the query reads better than both the plain aggregate and")
        print("[n70]   a direct probe. Borrow Eq. 8-9 with `eligible=` set, not the aggregate alone.")
    elif o <= max(fl, 0.05):
        # `m >= 0.8 * o` is vacuously true when the oracle itself is at the floor, which is how an
        # earlier run printed "THE LOOP CLOSES" on reads that were 0/2.  Gate on the ceiling being
        # a ceiling first.
        print("[n70]   INCONCLUSIVE: the ORACLE arm is at the unconditional floor "
              f"({o:.3f} vs {fl:.3f}), so there is no ceiling to compare a read against. Raise "
              "--n-samples/--steps before reading anything into the arms.")
    elif sum(r["muT_correct"] for r in reads) == 0 and sum(r["probe_correct"] for r in reads) == 0:
        print("[n70]   THE READ FAILS on every slot, so the loop cannot close here regardless of")
        print("[n70]   the arms. Check the per-factor probe accuracies above first -- if a probe")
        print("[n70]   cannot read a factor off a REAL image, no discovery route can supply it.")
    elif m >= 0.8 * o:
        print("[n70]   THE LOOP CLOSES: discovering the coordinates costs little against being told")
        print("[n70]   them, so the Shapes3D case study can show a real one-example read.")
    else:
        print("[n70]   Discovery costs a lot here against the oracle. The figure must either show")
        print("[n70]   the read arm WITH that gap stated, or stop calling itself a one-example loop.")
    save_json(dict(rows=rows, summary=summ, reads=reads, basis_r2=float(fac["r2"]),
                   config=vars(args)),
              os.path.join(RESULTS, "n70_shapes3d_reads.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
