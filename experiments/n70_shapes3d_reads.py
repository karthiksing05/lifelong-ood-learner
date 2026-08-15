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

from lifelong_ood.data.factored import decode_slot, load_factored      # noqa: E402
from lifelong_ood.evals import precision_recall, save_json, vendi_score  # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize          # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field                # noqa: E402

ARMS = ("address-oracle", "address-muT", "address-probe", "nearest-class", "unconditional")


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
    weak = [n for n, v in val.items() if v < 0.8]
    if weak:
        print(f"[n70] WARNING probes below 0.80 on real images: {weak} -- a read that misses these "
              f"is the instrument failing, not the method", flush=True)

    def read(img_row):
        x = np.asarray(img_row, np.float32).reshape(1, -1)
        return tuple(int(probes[n].predict(x)[0]) if n in probes else 0 for n in names)

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
        # mu_T needs mode ascent on the VE-bridged field; reuse n58's helper so the two figures
        # describe the same operation rather than two similar ones
        from n58_shapes3d_results import mode_ascent
        M, src = mode_ascent(field, xq.reshape(64, 64, 3).transpose(2, 0, 1), 0.30,
                            n_starts=4, seed=args.seed + 17 * ci)
        # the paper aggregates a SELECTED subset; with 4 modes and no Laplace covariance here the
        # honest reduction is the mean of the query-agnostic ones, which is the same "aggregate
        # several trained modes" operation without pretending to be Eq. 7
        keep = [i for i, q in enumerate(src) if q == "random"] or list(range(len(M)))
        mu_T = np.asarray(M)[keep].mean(0)
        r_mu = read(mu_T)
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
                          muT_correct=bool(r_mu == tgt), probe_correct=bool(r_probe == tgt)))
        print(f"[n70] case {ci + 1} slot {int(s)} target {tgt} | mu_T {r_mu} "
              f"{'OK' if r_mu == tgt else 'WRONG'} | probe {r_probe} "
              f"{'OK' if r_probe == tgt else 'WRONG'}", flush=True)
        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            hits = [float((probes[n].predict(S) == tgt[i]).mean())
                    for i, n in enumerate(names) if n in probes]
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
    print(f"\n[n70] READ accuracy over {n} slots -- mu_T {sum(r['muT_correct'] for r in reads)}/{n}"
          f" | probe-query {sum(r['probe_correct'] for r in reads)}/{n}")
    print("[n70] VERDICT")
    o, m = summ["address-oracle"]["joint"], summ["address-muT"]["joint"]
    fl = summ["unconditional"]["joint"]
    if o <= max(fl, 0.05):
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
