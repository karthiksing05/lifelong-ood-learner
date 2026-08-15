"""N67 -- the paper's DISCOVERY with our GENERATION.  Does the combination beat both parents?

The two methods in this repository are disjoint pipelines, which the case-study figures hid:

  * Wang et al. DISCOVER by mode ascent (Eq. 4), SELECT K complementary experts by submodular
    coverage (Eq. 8-9), AGGREGATE them into a per-dimension product (Eq. 7/10), and then GENERATE by
    guiding the sampler with that teacher (Eq. 12-14).
  * Ours never runs any of that.  `compose_factored` does textual inversion of the FACTOR BASIS
    against the query itself (`learn_factored(field, targets=xq, ...)`, ~18 parameters), then
    samples the model's own conditional.  `learn_concept_factored`'s docstring claims "mode ascent
    supplies the prototypes the factorization is read against"; it does not, and that claim is
    corrected by this file existing.

Each has the half the other lacks.  Their discovery is principled and their selected prototypes are
complementary trained modes; their GENERATION collapses, because a product of Gaussians is one
Gaussian and a deterministic sampler turns one query into one image (recall 0.000, vendi 1.06).
Ours generates from the model's multimodal conditional and covers the class, but identifies the
composition by optimising directly against the query, which uses no prototype at all.

So the arms split the pipeline at the seam and cross the halves over:

    mode ascent (their budget) -> submodular select K -> per-dim PoE -> mu_T
      -> PROBE mu_T for the factor values          <- the only thing taken from the teacher
      -> c_new = mu + sum_f g_f (w_f . V_f)        <- our factor basis
      -> sample p_theta(x | c_new)                 <- our generation, multimodal by construction

Reading mu_T is legitimate rather than a shortcut: RESULTS 5e-quater already measures that "mu_T
already probe-reads the correct composition", so the identity is genuinely present in their
aggregate and only the SAMPLING of it is what destroys diversity.

FIVE ARMS, one query each, so the comparison is like-for-like:

  poe-guided           their discovery + their generation
  ours-inversion       query inversion + our generation, SOFT per-factor weights (what we ship)
  ours-inversion-hard  query inversion + our generation, HARD argmax composition
  ours-prototypes      their discovery + our generation
  unconditional        the floor

The fourth arm exists because of what the first run measured: mu_T read the composition correctly
5/6 and inversion 6/6, yet the mu_T arm scored HIGHER joint accuracy (0.465 vs 0.340).  The gain
therefore is NOT in the reading -- it is that a hard `compose(fac, values)` places the point more
cleanly than `learn_factored`'s soft weights, which spread mass over values the composition does not
want.  If that is the mechanism, the most reliable read composed hard should beat both parents.

    python experiments/n67_best_of_both.py --n-cases 6 --n-samples 24
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries          # noqa: E402

from lifelong_ood.compose import (PAPER_COLORMNIST, PAPER_POE_COLORMNIST,  # noqa: E402
                                  candidates_ascent, poe_teacher)
from lifelong_ood.data.datasets import load_colormnist                 # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,      # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,    # noqa: E402
                                 sample_embedding, to_data, to_model)
from lifelong_ood.neural.idealwords import compose, slot_factors       # noqa: E402

ARMS = ("poe-guided", "ours-inversion", "ours-inversion-hard", "ours-prototypes", "unconditional")


def read_factors(probes, img_row):
    """Probe ONE image for (digit, fg, bg).  -> tuple of ints.

    This is the whole interface between their half and ours: the teacher mean is an image, and the
    factor basis is indexed by factor VALUES, so a probe is what converts one into the other.
    """
    x = np.asarray(img_row, np.float32).reshape(1, -1)
    return (int(probes["digit"].predict(x)[0]), int(probes["fg"].predict(x)[0]),
            int(probes["bg"].predict(x)[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cases", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_cases, args.n_samples, args.steps, args.n_per_slot = 1, 6, 40, 20
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
    fac = L.fac

    disc = dict(PAPER_COLORMNIST)
    if args.quick:
        disc.update(u_grid=(0.05, 0.2), n_starts=8, iters=40)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    rows, reads = [], []

    for ci, q in enumerate(qs):
        xq = to_model(data["X"][q["query"]]).ravel()
        R = np.asarray(data["X"], np.float32)[q["rest"][:100]]
        tgt = (q["d"], q["f"], q["b"])
        got = {}

        # their half: discovery -> selection -> aggregate, then their generation
        M, V, meta = candidates_ascent(field, xq, seed=args.seed, **disc)
        t = poe_teacher(xq, M, V, K=L.cfg["K"], tau=L.cfg["tau"])
        got["poe-guided"] = to_data(L.sample_teacher(t, n=args.n_samples, seed=args.seed + ci,
                                                     **PAPER_POE_COLORMNIST))

        # the hybrid: read THEIR aggregate, generate with OURS
        read_p = read_factors(probes, to_data(t["mu"]))
        got["ours-prototypes"] = to_data(sample_embedding(
            field, compose(fac, read_p, gains=gains), n=args.n_samples, steps=args.steps,
            seed=args.seed + ci))

        # what we ship today: soft per-factor weights from inversion against the query
        c_i, ws, read_i = L.compose_factored(xq, seed=args.seed)
        got["ours-inversion"] = to_data(sample_embedding(field, c_i, n=args.n_samples,
                                                         steps=args.steps, seed=args.seed + ci))

        # the same read, composed HARD -- see the module docstring
        got["ours-inversion-hard"] = to_data(sample_embedding(
            field, compose(fac, tuple(int(v) for v in read_i), gains=gains), n=args.n_samples,
            steps=args.steps, seed=args.seed + ci))

        got["unconditional"] = to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                    seed=args.seed + 977 * (ci + 1),
                                                    clip=(-1, 1), branch=None))

        reads.append(dict(slot=int(q["slot"]), target=list(tgt), from_mu_T=list(read_p),
                          from_inversion=[int(v) for v in read_i],
                          mu_T_correct=bool(tuple(read_p) == tgt),
                          inversion_correct=bool(tuple(read_i) == tgt)))
        print(f"[n67] case {ci + 1}/{len(qs)} slot {int(q['slot'])} target {tgt} | "
              f"mu_T {read_p} {'OK' if tuple(read_p) == tgt else 'WRONG'} | "
              f"inversion {tuple(read_i)} "
              f"{'OK' if tuple(read_i) == tgt else 'WRONG'}", flush=True)

        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            a = composition_accuracy(probes, S, *tgt)
            pr = precision_recall(S, R)
            rows.append(dict(arm=arm, slot=int(q["slot"]), joint=a["joint"],
                             precision=pr["precision"], recall=pr["recall"],
                             vendi=float(vendi_score(S))))
            print(f"[n67]     {arm:20s} joint {a['joint']:.3f}  recall {pr['recall']:.3f}  "
                  f"vendi {float(vendi_score(S)):.2f}", flush=True)

    print(f"\n[n67] {'=' * 70}")
    print(f"{'arm':22s}{'joint':>9s}{'recall':>9s}{'precision':>11s}{'vendi':>8s}")
    summ = {}
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm]
        if not g:
            continue
        summ[arm] = {k: float(np.mean([r[k] for r in g])) for k in
                     ("joint", "recall", "precision", "vendi")}
        print(f"{arm:22s}{summ[arm]['joint']:>9.3f}{summ[arm]['recall']:>9.3f}"
              f"{summ[arm]['precision']:>11.3f}{summ[arm]['vendi']:>8.2f}")

    n_mu = sum(1 for r in reads if r["mu_T_correct"])
    n_in = sum(1 for r in reads if r["inversion_correct"])
    print(f"\n[n67] READ accuracy -- mu_T {n_mu}/{len(reads)} | inversion {n_in}/{len(reads)}")
    print("[n67] VERDICT")
    best = max((a for a in ARMS if a.startswith("ours")), key=lambda a: summ[a]["joint"])
    p_ = summ["poe-guided"]
    print(f"[n67]   best 'ours' arm on joint accuracy: {best} ({summ[best]['joint']:.3f}), "
          f"recall {summ[best]['recall']:.3f} vs poe-guided's {p_['recall']:.3f}")
    if summ[best]["recall"] > p_["recall"] + 0.15:
        print("[n67]   The seam holds: identity is recoverable without their generation, and")
        print("[n67]   coverage is recoverable without their teacher.")
    if summ["ours-inversion-hard"]["joint"] > summ["ours-inversion"]["joint"] + 0.05:
        print("[n67]   HARD beats SOFT composition on the same read -- `learn_factored`'s soft")
        print("[n67]   per-factor weights are leaking mass onto values the composition rejects,")
        print("[n67]   and the shipped arm should take the argmax it already computes.")
    save_json(dict(rows=rows, summary=summ, reads=reads, config=vars(args)),
              os.path.join(RESULTS, "n67_best_of_both.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
