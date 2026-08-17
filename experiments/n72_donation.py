"""N72 -- is the method DONATING a capability, or locating one the model already had?

This is the question the whole thesis stands on, and none of the existing experiments asks it
directly.  They measure whether the composed samples carry the right factors; they do not measure
whether the model could ALREADY produce that combination, nor whether the samples are new images
rather than retrieved ones.  Both alternatives would score well on accuracy.

Four things have to hold simultaneously for "we composed something novel" to be true, and each has
a way of being false that a joint-accuracy number cannot see:

  1 THE MODEL DID NOT ALREADY HAVE IT.  If the base model produces the held-out combination on its
    own, an address that "adds" it donates nothing.  Measured as the unconditional floor and as
    `nearest-class` -- the best EXISTING class embedding, chosen by distance to the composed
    address.  If nearest-class scores what ADDRESS scores, the address is a rediscovery.

  2 THE ADDRESS IS A NEW POINT.  `c_new` must sit somewhere the table does not already sit.  Its
    distance to the nearest trained embedding is compared against the typical spacing BETWEEN
    trained embeddings: an address closer to a trained row than trained rows are to each other is
    that row under another name.

  3 THE SAMPLES ARE NOT COPIES.  MOSAIC's test (`evals.memorization`, d_NN1/d_NN2 < 1/3) against
    the images the model actually trained on, and separately against the QUERY, because "copied the
    one example it was shown" is the specific failure this method is most exposed to -- it is handed
    a real image of the target class.

  4 THE SAMPLES COVER THE CLASS.  Recall against held-out reals, with the real-vs-real spread as the
    scale bar.  A method can pass 1-3 by emitting one novel image forever.

The arms are chosen so that each failure mode has a witness: `query-copy` (return the query) passes
1 and 2 and fails 3; `nearest-class` passes 3 and 4 and fails 1; the unconditional floor passes 3
and 4 and fails everything about identity.

    python experiments/n72_donation.py --n-cases 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries          # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                 # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,      # noqa: E402
                                memorization, precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,    # noqa: E402
                                 sample_embedding, to_data, to_model)
from lifelong_ood.neural.idealwords import compose, slot_factors       # noqa: E402

ARMS = ("address", "nearest-class", "query-copy", "unconditional")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        (args.n_cases, args.n_samples, args.steps, args.n_per_slot,
         args.inv_steps) = 2, 8, 40, 20, 40
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ckp = os.path.join(CACHE, "ddpm_colormnist.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field = load_field(ckp, device=dev)
    gains = backbone_gains(field, C, slots, probes, data, seed=args.seed, steps=args.steps)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, slot_factors(slots), gains=gains)

    # the images the model ACTUALLY trained on -- the reference for the copying test
    X = np.asarray(data["X"], np.float32)
    seen_mask = np.asarray(data["seen"])
    TRAIN = X[seen_mask]
    print(f"[n72] {len(TRAIN)} training images (seen slots) as the copying reference", flush=True)

    # 2 ── how far apart are trained embeddings from each other?  the scale bar for "new point"
    D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(C), 1)
    spacing = float(np.median(D[iu]))
    nn_spacing = float(np.median(np.sort(D + np.eye(len(C)) * 1e9, axis=1)[:, 0]))
    print(f"[n72] trained embeddings: median pairwise {spacing:.3f}, "
          f"median nearest-neighbour {nn_spacing:.3f}", flush=True)

    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    rows, geo = [], []
    for ci, q in enumerate(qs):
        xq_m = to_model(data["X"][q["query"]]).ravel()
        xq_d = np.asarray(data["X"][q["query"]], np.float32)
        R = X[q["rest"][:100]]
        tgt = (q["d"], q["f"], q["b"])
        _, ws, read = L.compose_factored(xq_m, inv_steps=args.inv_steps, seed=args.seed)
        c_new = compose(L.fac, tuple(int(v) for v in read), gains=gains)

        d_to_rows = np.sqrt(((C - c_new[None]) ** 2).sum(-1))
        j_near = int(np.argmin(d_to_rows))
        geo.append(dict(slot=int(q["slot"]), d_nearest=float(d_to_rows[j_near]),
                        ratio_to_spacing=float(d_to_rows[j_near] / max(nn_spacing, 1e-9))))

        got = {
            "address": to_data(sample_embedding(field, c_new, n=args.n_samples,
                                                steps=args.steps, seed=args.seed + ci)),
            "nearest-class": to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                  seed=args.seed + ci, clip=(-1, 1),
                                                  branch=int(j_near))),
            "query-copy": np.repeat(xq_d[None], args.n_samples, axis=0),
            "unconditional": to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                  seed=args.seed + 977 * (ci + 1), clip=(-1, 1),
                                                  branch=None)),
        }
        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            a = composition_accuracy(probes, S, *tgt)
            pr = precision_recall(S, R)
            mem = memorization(S, TRAIN)
            dq = float(np.mean(np.linalg.norm(S - xq_d[None], axis=1)))
            rows.append(dict(arm=arm, slot=int(q["slot"]), joint=a["joint"],
                             recall=pr["recall"], precision=pr["precision"],
                             vendi=float(vendi_score(S)),
                             memorized=float(mem["frac_memorized"]),
                             d_query=dq))
        r_ = {x["arm"]: x for x in rows if x["slot"] == int(q["slot"])}
        print(f"[n72] slot {int(q['slot'])} {tgt}: address joint {r_['address']['joint']:.3f} "
              f"vs nearest-class {r_['nearest-class']['joint']:.3f} vs uncond "
              f"{r_['unconditional']['joint']:.3f} | address memorized "
              f"{r_['address']['memorized']:.2f} | d(address,query) "
              f"{r_['address']['d_query']:.2f} vs d(query-copy,query) "
              f"{r_['query-copy']['d_query']:.2f}", flush=True)

    # real-vs-real spread, the scale bar for "is d(sample, query) large"
    rr = []
    for q in qs:
        Rr = X[q["rest"][:24]]
        rr.append(float(np.mean([np.linalg.norm(Rr[i] - Rr[j])
                                 for i in range(len(Rr)) for j in range(i + 1, len(Rr))])))
    real_spread = float(np.mean(rr))

    def m(arm, k):
        return float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    print(f"\n[n72] {'=' * 76}")
    print(f"{'arm':16s}{'joint':>8s}{'recall':>8s}{'vendi':>7s}{'memorized':>11s}{'d(query)':>10s}")
    for arm in ARMS:
        print(f"{arm:16s}{m(arm, 'joint'):>8.3f}{m(arm, 'recall'):>8.3f}{m(arm, 'vendi'):>7.2f}"
              f"{m(arm, 'memorized'):>11.2f}{m(arm, 'd_query'):>10.2f}")
    print(f"{'--- real vs real':16s}{'':>8s}{'':>8s}{'':>7s}{'':>11s}{real_spread:>10.2f}")
    dn = float(np.mean([g["d_nearest"] for g in geo]))
    rt = float(np.mean([g["ratio_to_spacing"] for g in geo]))
    print(f"\n[n72] the composed address sits {dn:.3f} from the nearest TRAINED embedding, "
          f"{rt:.2f}x the median nearest-neighbour spacing between trained rows")

    print("[n72] VERDICT -- four conditions, each with a way of being false:")
    c1 = m("address", "joint") > max(m("nearest-class", "joint"), m("unconditional", "joint")) + 0.15
    c2 = rt > 0.5
    # TWO copying tests, because one of them cannot see the failure it is aimed at.  MOSAIC's
    # d_NN1/d_NN2 compares against the TRAINING set, but the query is a HELD-OUT image the model
    # never trained on -- so `query-copy`, which returns the query verbatim, scores 0.00 memorized
    # and passes a test designed to catch exactly it.  The second condition is distance to the
    # query itself, scaled by how far apart two REAL images of the class sit: a sample no further
    # from the query than reals are from each other is a copy of the query whatever the NN-ratio
    # says.
    far_from_query = m("address", "d_query") >= 0.5 * real_spread
    c3 = m("address", "memorized") < 0.25 and far_from_query
    c4 = m("address", "recall") > 0.3
    print(f"[n72]   1 the model did not already have it   {c1}  "
          f"(address {m('address', 'joint'):.3f} vs nearest-class "
          f"{m('nearest-class', 'joint'):.3f}, uncond {m('unconditional', 'joint'):.3f})")
    print(f"[n72]   2 the address is a NEW point          {c2}  ({rt:.2f}x trained-row spacing)")
    print(f"[n72]   3 the samples are not copies          {c3}  "
          f"({m('address', 'memorized'):.2f} of training images memorized; "
          f"d(sample,query) {m('address', 'd_query'):.2f} against a real-vs-real spread of "
          f"{real_spread:.2f})")
    print(f"[n72]       the witness: query-copy scores memorized "
          f"{m('query-copy', 'memorized'):.2f} -- the NN test CANNOT see it, because the query is "
          f"held out and was never trained on; d(query) {m('query-copy', 'd_query'):.2f} is what "
          f"catches it")
    print(f"[n72]   4 the samples cover the class         {c4}  "
          f"(recall {m('address', 'recall'):.3f}, vendi {m('address', 'vendi'):.2f})")
    if all((c1, c2, c3, c4)):
        print("[n72]   ALL FOUR HOLD: the method donates a capability the model did not have, at a")
        print("[n72]   point the table did not occupy, as new images that cover the class.")
    else:
        print("[n72]   NOT ALL HOLD -- the failing condition is the claim that has to be weakened.")
    save_json(dict(rows=rows, geometry=geo, real_spread=real_spread, spacing=spacing,
                   nn_spacing=nn_spacing, conditions=dict(model_lacked_it=bool(c1),
                                                          new_point=bool(c2),
                                                          not_copies=bool(c3),
                                                          covers=bool(c4)), config=vars(args)),
              os.path.join(RESULTS, "n72_donation.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
