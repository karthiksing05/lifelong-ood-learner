"""N50 -- are the representations as good as a model trained on everything from the start?

Every composition result so far scores what the model GENERATES.  This asks the other question, and
it is the one that decides whether incremental absorption is a real substitute for joint training:
**are the internal representations of a frozen backbone plus consolidated concepts as informative as
those of a backbone that saw every composition during training?**

Three backbones, one benchmark, identical probes:

    base      `ddpm_colormnist.pt` -- 120 slots, the held-out compositions never trained
    ours      the same weights, frozen, with the held-out concepts consolidated into the bank
    oracle    `ddpm_colormnist_oracle.pt` -- 160 slots, trained on everything from the get-go

The representation of an image is the model's own epsilon-prediction error under each conditioning it
can express -- a vector of "how well does each concept I know explain this?".  That is the quantity
the router already reads (`conditional_losses`), it needs no architecture surgery, and it is defined
identically for all three arms, which a hidden-activation probe would not be: `ours` and `oracle`
have different numbers of conditioning rows and `ours` has adapters the oracle lacks.

Two readouts, both standard:

    linear probe   multinomial logistic regression on the representation -> the factor value
    k-NN           k=5, cosine, fit on SEEN slots and evaluated on HELD-OUT compositions

Scored per factor (digit / fg / bg) and jointly.  The comparison that matters is `ours` against
`oracle`: if consolidating a concept produces a representation as separable as training on it did,
then the incremental route costs 72 bytes and loses nothing measurable.  `base` is the floor that
says how much of any gap is the backbone rather than the route.

    python experiments/n50_representation.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains                              # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import SoftmaxProbe, fit_probes, save_json             # noqa: E402
from lifelong_ood.neural import load_field, to_model                           # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                        # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner          # noqa: E402
from lifelong_ood.neural.router import conditional_losses                      # noqa: E402

FACTORS = ("digit", "fg", "bg")


def represent(field, X01, conds, u_grid=(0.15, 0.45), n_noise=2, seed=0):
    """-> [N, M] the model's epsilon error for each image under each conditioning.

    Low error under a conditioning means "this concept explains this image", so the row is a profile
    over everything the model can express.  Identical in shape and meaning across the three arms even
    though their conditioning sets differ in size, because the probe is fit per arm.
    """
    out = []
    for x01 in X01:
        L = conditional_losses(field, to_model(x01).ravel(), conds, u_grid=u_grid,
                               n_noise=n_noise, seed=seed)
        out.append(np.asarray(L, np.float64))
    R = np.asarray(out)
    # standardize per column: the arms differ in how hard their conditionings are on average, and the
    # question is whether the PATTERN separates the factors, not whether one arm has lower loss.
    mu, sd = R.mean(0, keepdims=True), R.std(0, keepdims=True) + 1e-9
    return (R - mu) / sd


def knn_acc(Rtr, ytr, Rte, yte, k=5):
    """Cosine k-NN.  Fit on seen slots, evaluated on held-out compositions."""
    A = Rtr / (np.linalg.norm(Rtr, axis=1, keepdims=True) + 1e-9)
    B = Rte / (np.linalg.norm(Rte, axis=1, keepdims=True) + 1e-9)
    S = B @ A.T
    idx = np.argsort(-S, 1)[:, :k]
    pred = np.array([np.bincount(ytr[row]).argmax() for row in idx])
    return float((pred == yte).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slot", type=int, default=12)
    p.add_argument("--n-concepts", type=int, default=8, help="held-out concepts consolidated")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_per_slot, args.n_concepts, args.epochs, args.inv_steps = 4, 2, 4, 60
        args.tag = args.tag or "_smoke"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    seen = np.asarray(data["seen"], bool)
    held = np.unique(data["slot"][~seen])[:args.n_concepts]
    rng = np.random.default_rng(args.seed)

    # rows: a balanced sample of SEEN images (to fit the readouts) and HELD-OUT images (to test)
    tr_idx = rng.permutation(np.where(seen)[0])[:600]
    te_idx = np.concatenate([np.where(data["slot"] == s)[0][:args.n_per_slot] for s in held])
    print(f"[n50] {len(tr_idx)} seen images to fit the readouts, {len(te_idx)} held-out images "
          f"over {len(held)} compositions\n", flush=True)

    rows = []
    for arm in ("base", "ours", "oracle"):
        ckpt = os.path.join(CACHE, "ddpm_colormnist_oracle.pt" if arm == "oracle"
                            else "ddpm_colormnist.pt")
        if not os.path.exists(ckpt):
            print(f"[n50] {arm}: missing {ckpt}, skipping")
            continue
        field = load_field(ckpt, device=dev)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        remap = ck["remap"]
        slots_tr = np.array(sorted(remap))
        C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_tr]]

        # MATCH THE NUMBER OF CONDITIONINGS ACROSS ARMS.
        #
        # The representation is one column per conditioning, so an arm with more of them hands the
        # readouts more capacity for a reason that has nothing to do with representation quality.
        # The oracle trained on all 160 slots and would otherwise get 160 columns against `ours`'s
        # 122 -- and it would win partly on width. So every arm is restricted to the SAME 120 seen
        # slots, plus one column for each held-out composition: the oracle's own trained class for
        # it, and for `ours` the concept consolidated from a single query. That is the like-for-like
        # question -- a trained conditioning against an absorbed one, everything else equal.
        seen_rows = [i for i, s in enumerate(slots_tr) if bool(seen[data["slot"] == int(s)][0])]
        C, slots_tr = C[seen_rows], slots_tr[seen_rows]      # keep rows and labels in step
        conds = C
        if arm == "oracle":
            held_rows = [remap[int(s)] for s in held if int(s) in remap]
            conds = np.concatenate([C, ck["ema"]["cemb.weight"].numpy()[held_rows]])

        if arm == "ours":
            # consolidate each held-out composition, then let the bank's own embeddings join the
            # conditioning set -- that is what "the representation after absorption" means here.
            gains = backbone_gains(field, C, slots_tr, probes, data, seed=args.seed,
                                   steps=50, n=8 if args.quick else 16)
            L = NeuralLifelongLearner(field, device=dev, seed=args.seed, flat=True)
            L.attach_factorization(C, slot_factors(slots_tr), gains=gains)
            extra = []
            for j, s in enumerate(held):
                i = np.where(data["slot"] == s)[0][0]
                xq = to_model(data["X"][i]).ravel()
                c_new, _ = L.learn_concept_factored(xq, f"c{s}", n_pool=48, epochs=args.epochs,
                                                    inv_steps=args.inv_steps, seed=args.seed + j,
                                                    guidance=args.guidance)
                extra.append(np.asarray(c_new.detach().cpu().numpy(), np.float32).ravel())
                print(f"[n50]   consolidated c{s} ({j + 1}/{len(held)})", flush=True)
            conds = np.concatenate([C, np.stack(extra)])

        Rtr = represent(field, data["X"][tr_idx], conds, seed=args.seed)
        Rte = represent(field, data["X"][te_idx], conds, seed=args.seed)
        print(f"[n50] {arm}: representation {Rte.shape[1]}-d over {len(conds)} conditionings",
              flush=True)

        res = dict(arm=arm, dim=int(Rte.shape[1]), n_conds=int(len(conds)))
        joint_lin, joint_knn = None, None
        for f in FACTORS:
            ytr, yte = data[f][tr_idx].astype(int), data[f][te_idx].astype(int)
            pr = SoftmaxProbe(int(max(ytr.max(), yte.max())) + 1, seed=args.seed,
                              epochs=60).fit(Rtr, ytr)
            lin = pr.predict(Rte)
            res[f"lin_{f}"] = float((lin == yte).mean())
            res[f"knn_{f}"] = knn_acc(Rtr, ytr, Rte, yte)
            joint_lin = (lin == yte) if joint_lin is None else (joint_lin & (lin == yte))
        res["lin_joint"] = float(joint_lin.mean())
        rows.append(res)
        print(f"  {arm:7s} linear  " + "  ".join(f"{f} {res[f'lin_{f}']:.3f}" for f in FACTORS) +
              f"  joint {res['lin_joint']:.3f}")
        print(f"  {arm:7s} k-NN    " + "  ".join(f"{f} {res[f'knn_{f}']:.3f}" for f in FACTORS),
              flush=True)

    print("\n[n50] `ours` against `oracle` is the comparison: if a consolidated concept yields a\n"
          "      representation as separable as one the backbone was TRAINED on, incremental\n"
          "      absorption costs 72 bytes and loses nothing the readouts can see. `base` says how\n"
          "      much of any gap belongs to the backbone rather than to the route.")
    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n50_representation{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
