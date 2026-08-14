"""Reproduce the companion paper's Table 1 under ITS protocol, not ours.

Every composition number this repository publishes was measured one way: 24 samples from ONE query,
scored against held-out images in a downsampled-pixel feature space.  The paper measures a different
way, and the difference is not cosmetic:

    §4.2   "For each OOD class, we run the full framework ... independently on each of the 100
            queries x_q^(i)"
    Tab.8  "Saved gens per query per teacher: 16"          -> 1600 samples per OOD class
    §4.2   precision/recall in **Inception-V3** feature space, k-NN estimator, k=3
    §4.2   TWO reference sets per class:
              Faithfulness   the 100 query images themselves
              Generalization 100 other held-out images of the class, excluding the query

Under our protocol their PoE arm scores recall 0.001; they report 80.7.  `RESULTS` §5e explains that
gap as a property of their method -- "guiding toward a unimodal Gaussian collapses every trajectory
onto its mean" -- and `tests/diag_teacher_variance.py` confirms the per-query collapse is real
(vendi 1.06 from 24 samples, at every covariance scale).  But their sampler is DDIM with eta=0
(Table 8), so it is deterministic given the teacher: within one query the only source of diversity is
the initial noise, and their method would collapse per query too.  Their diversity can only come from
the 100 DIFFERENT teachers, one per query.  If so, both statements are true at once -- the per-query
collapse is real AND their recall is real -- and the 0.001 we publish as theirs answers a question
they never asked.

This runs their five arms under their protocol so the comparison is like-for-like:

    top-1        condition on the single nearest trained class          (their retrieval floor)
    top-3        add the guidance residuals of the 3 nearest classes    (Composable Diffusion)
    query-only   N(x_q, sigma^2 I) through the same sampler             (their spread-the-query floor)
    PoE          Eq. 12-14, the analytic teacher                        (their main arm)
    +LoRA        distil the PoE pool into (c_new, LoRA) and sample it   (their best arm)
    ours         the factored route: the teacher LOCATES the concept, the ideal-words basis reads
                 it, and the model's own conditional GENERATES it

`ours` is here because a reproduction that leaves our method out answers only half the question.  It
and `+LoRA` both train per query, so both are run on `--distill-queries` of them and every other arm
is re-scored at that same count for a paired block -- recall depends on how many queries are pooled,
so an arm evaluated on fewer of them is being handicapped on the axis under test.

Absolute values still will not match theirs: our backbone is smaller than their Table 5 config, we
have no CLIP checkpoint (`cos` substitutes an Inception image-image cosine and is labelled as such),
and Appendix E already warns their own FIDs are small-N.  What SHOULD reproduce is the ordering and
the separation -- PoE and +LoRA far above top-k, and query-only at the floor.  That is the claim this
script tests.

    python experiments/n49_paper_protocol.py --n-classes 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains                              # noqa: E402

from lifelong_ood.compose import query_only_teacher                            # noqa: E402
from lifelong_ood.data.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,              # noqa: E402
                                paper_metrics, save_json, vendi_score)
from lifelong_ood.neural import (load_field, teacher_init_alpha,               # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.composable import compose_classes                     # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                        # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner          # noqa: E402

# their Table 8, ColorMNIST column -- the same constants `compose.PAPER_POE_COLORMNIST` records
PAPER_POE = dict(schedule="variance", var_scale=1.0, w0=1.2, wmax=2.0, wmin=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-classes", type=int, default=8,
                   help="OOD classes to evaluate; the paper uses all 40")
    p.add_argument("--n-queries", type=int, default=100, help="their §4.2")
    p.add_argument("--per-query", type=int, default=16, help="their Table 8")
    p.add_argument("--distill-queries", type=int, default=8,
                   help="queries given the arms that TRAIN per query (+LoRA, ours); one adapter "
                        "per query is the paper's protocol and costs a training run each, so this "
                        "is smaller than --n-queries and every arm is re-scored at this count too")
    p.add_argument("--pool-size", type=int, default=256, help="their Table 7, N_pool")
    p.add_argument("--steps", type=int, default=50, help="their Table 8, DDIM 50 steps")
    p.add_argument("--inv-steps", type=int, default=400, help="inversion steps for `ours`")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.5,
                   help="CFG weight when the MANIFOLD generates the pool for `ours`; n9 calibrates "
                        "this on TRAINED slots, and overriding it is what collapsed that arm once")
    p.add_argument("--n-per-slot", type=int, default=260)
    p.add_argument("--arms", nargs="+",
                   default=["top-1", "top-3", "query-only", "PoE", "+LoRA", "ours"])
    p.add_argument("--space", default="inception", choices=["inception", "pixel"])
    p.add_argument("--seed", type=int, default=42, help="their Table 8")
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_classes, args.n_queries, args.per_query = 2, 12, 4
        args.distill_queries, args.pool_size, args.tag = 2, 32, args.tag or "_smoke"
        args.steps, args.inv_steps, args.epochs = 20, 60, 4

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=0)
    probes = fit_probes(data, seed=0, epochs=8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    Call = ck["ema"]["cemb.weight"].numpy()
    null_row = Call[field.model.null_id]
    # `ours` reads the composition in the ideal-words basis, which is defined over the SEEN slots'
    # embeddings in slot order -- not over the raw table, whose rows are in remap order.
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = Call[[remap[int(x)] for x in slots_seen]]
    gains = backbone_gains(field, C, slots_seen, probes, data, seed=args.seed,
                           steps=args.steps, n=8 if args.quick else 16)

    slots = np.unique(data["slot"][~data["seen"]])
    rng = np.random.default_rng(args.seed)
    slots = rng.permutation(slots)[:args.n_classes]
    print(f"[n49] {len(slots)} OOD classes x {args.n_queries} queries x {args.per_query} gens "
          f"= {args.n_queries * args.per_query} samples per class; metric space={args.space}\n")

    rows = []
    for si, slot in enumerate(slots):
        idx = np.where(data["slot"] == slot)[0]
        nq = min(args.n_queries, max(2, len(idx) - 100))
        q_idx, r_idx = idx[:nq], idx[nq:nq + 100]
        if len(r_idx) < 10:                       # not enough held-out images for Generalization
            print(f"[n49] slot {slot}: only {len(idx)} images, skipping")
            continue
        d0, f0, b0 = (int(data["digit"][q_idx[0]]), int(data["fg"][q_idx[0]]),
                      int(data["bg"][q_idx[0]]))
        REF = {"Faithful.": data["X"][q_idx], "General.": data["X"][r_idx]}
        xqs = [to_model(data["X"][i]).ravel() for i in q_idx]
        L = NeuralLifelongLearner(field, device=dev, seed=args.seed)
        print(f"[n49] class {si + 1}/{len(slots)}  slot {slot}  "
              f"(digit {d0}, fg {f0}, bg {b0})  {nq} queries")

        # ── the four analytic-teacher arms, one teacher per query ────────────
        T = L.compose_batch(xqs, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120)
        pools = {}
        if "PoE" in args.arms:
            pools["PoE"] = L.sample_teacher_batch(T, n=args.per_query, seed=args.seed,
                                                  u_start=1.0, steps=args.steps, **PAPER_POE)
        if "query-only" in args.arms:
            pools["query-only"] = L.sample_teacher_batch(
                [query_only_teacher(x) for x in xqs], n=args.per_query, seed=args.seed,
                u_start=1.0, steps=args.steps, **PAPER_POE)
        # top-k condition on TRAINED classes, so they need no teacher -- but they are still run
        # per query, because which classes are nearest is a function of the query.
        if "top-1" in args.arms or "top-3" in args.arms:
            A = [teacher_init_alpha(field, t, x, seed=args.seed) for t, x in zip(T, xqs)]
            if "top-1" in args.arms:
                pools["top-1"] = [field.sample(n=args.per_query, steps=args.steps, clip=(-1, 1),
                                               seed=args.seed + i, branch=int(np.argmax(a)))
                                  for i, a in enumerate(A)]
            if "top-3" in args.arms:
                pools["top-3"] = [compose_classes(field, C, np.argsort(-a)[:3], n=args.per_query,
                                                  steps=args.steps, seed=args.seed + i,
                                                  null_emb=null_row)
                                  for i, a in enumerate(A)]
        # ── the two arms that TRAIN per query, on a subset (see --distill-queries) ──
        nl = min(args.distill_queries, nq)
        if "+LoRA" in args.arms:
            lp = []
            for i in range(nl):
                pool = L.sample_teacher(T[i], n=args.pool_size, seed=args.seed + i, u_start=1.0,
                                        steps=args.steps, **PAPER_POE)
                Li = NeuralLifelongLearner(field, device=dev, seed=args.seed)
                Li.consolidate(to_data(pool), f"ood{slot}q{i}", mu=T[i]["mu"], epochs=args.epochs,
                               batch=16, lr=1e-3, seed=args.seed)
                lp.append(Li.sample_concept(f"ood{slot}q{i}", n=args.per_query, steps=args.steps,
                                            seed=args.seed + i, guidance=0.8))
                # each query gets its OWN adapter, per their §4.2 ("independently on each query"):
                # constructing Li rebinds field.bank, so the next iteration starts from a clean one
            pools["+LoRA"] = lp
        if "ours" in args.arms:
            op = []
            for i in range(nl):
                Oi = NeuralLifelongLearner(field, device=dev, seed=args.seed, flat=True)
                Oi.attach_factorization(C, slot_factors(slots_seen), gains=gains)
                Oi.learn_concept_factored(xqs[i], f"c{slot}q{i}", n_pool=48, epochs=args.epochs,
                                          inv_steps=args.inv_steps, seed=args.seed + i,
                                          guidance=args.guidance)
                op.append(Oi.sample_concept(f"c{slot}q{i}", n=args.per_query, steps=args.steps,
                                            seed=args.seed + i))
            pools["ours"] = op

        trains = {"+LoRA", "ours"}
        for arm, P in pools.items():
            P = [to_data(np.asarray(x)).astype(np.float32) for x in P]
            # Recall depends on HOW MANY queries were pooled (that is this script's whole finding),
            # so every arm is also scored at `nl` -- the per-query-training budget -- and only rows
            # with the same n_queries may be compared.
            #
            # The n_queries=1 block is the one that decides our own claim.  Their PoE needs 100
            # teachers to cover a class because each one collapses; if `ours` covers from a SINGLE
            # query, that is the contribution, stated on the axis the paper itself uses -- and it is
            # a sharper claim than the F1 margin the repository currently leads with.
            for n_use in sorted({1, nl, len(P)}):
                if n_use > len(P):
                    continue
                S = np.concatenate(P[:n_use])
                for name, R in REF.items():
                    Rn = R[:n_use] if name == "Faithful." else R
                    m = paper_metrics(S, Rn, space=args.space, device=dev)
                    a = composition_accuracy(probes, S, d0, f0, b0)
                    rows.append(dict(arm=arm, slot=int(slot), reference=name,
                                     n_queries=int(n_use), trains_per_query=arm in trains,
                                     joint=a["joint"], vendi=float(vendi_score(S[:256])), **m))
            print(f"    {arm:12s} " + "  ".join(
                f"{r['reference']}(q{r['n_queries']}) P{r['precision']:.3f} R{r['recall']:.3f} "
                f"F{r['f1']:.3f}" for r in rows[-2:]))

    # ── their Table 1 layout, one block per query budget ─────────────────────
    budgets = sorted({r["n_queries"] for r in rows})
    for nq in budgets:
        have = [a for a in args.arms if any(r["arm"] == a and r["n_queries"] == nq for r in rows)]
        if not have:
            continue
        print(f"\n[n49] Table 1 reproduction -- {nq} queries pooled per class "
              f"({args.space} features, {len(slots)} classes, mean over classes)")
        hdr = f"  {'Method':12s}"
        for m in ("fid", "cos", "precision", "recall", "f1"):
            hdr += f" | {m + ' F.':>10s} {m + ' G.':>10s}"
        print(hdr)
        for arm in have:
            line = f"  {arm:12s}"
            for m in ("fid", "cos", "precision", "recall", "f1"):
                for ref in ("Faithful.", "General."):
                    v = [r[m] for r in rows if r["arm"] == arm and r["reference"] == ref
                         and r["n_queries"] == nq]
                    line += f" | {np.mean(v):10.3f}" if v else f" | {'--':>10s}"
            print(line)
    print("\n[n49] rows are comparable only WITHIN a block: recall is a pooled quantity and rises\n"
          "      with the number of queries, so an arm evaluated on fewer of them is handicapped.")
    print("\n[n49] the paper's Table 1, ColorMNIST / 40 OOD compositions (Faithful./General., %):")
    print("       PoE    P 75.5/67.0  R 80.7/69.8  F1 75.3/63.0")
    print("       +LoRA  P 89.5/82.2  R 88.0/78.8  F1 87.6/76.3")
    print("       top-1  P  9.5/ 5.3  R 42.2/38.8  F1  6.2/ 6.5")
    print("       top-3  P 11.0/15.8  R 30.0/30.5  F1  5.1/ 5.3")
    print("       q-only P  0.5/ 0.0  R  2.5/ 1.2  F1  0.0/ 0.0")
    # Their Table 2 is the sanity check on any low recall measured above.  On CelebA -- the harder
    # dataset, and the one whose backbone is a fine-tune of a real pretrained model -- THEIR OWN PoE
    # recall is 12.0/9.0 and their best arm reaches only 19.9/16.8.  A recall well below their
    # ColorMNIST 80.7 is therefore inside the range their method produces, not prima facie a bug.
    # Their Table 2 also shows query-only WINNING Faithfulness (F1 64.0) and collapsing on
    # Generalization (20.2), which is the signature of an arm that reproduces the query itself.
    print("\n[n49] the paper's Table 2, CelebA / 23 OOD compositions -- note the low recalls:")
    print("       PoE    P 75.8/74.5  R 12.0/ 9.0  F1 20.3/15.6")
    print("       +LoRA  P 46.2/52.7  R 19.9/16.8  F1 26.9/24.7")
    print("       q-only P 80.6/48.2  R 53.6/13.4  F1 64.0/20.2   <- wins Faithfulness by copying xq")
    print("       top-1  P 39.4/43.1  R  1.4/ 1.0  F1  1.9/ 1.5")
    print("       top-3  P 45.3/48.9  R  0.4/ 0.7  F1  0.6/ 1.0")

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n49_paper_protocol{args.tag}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
