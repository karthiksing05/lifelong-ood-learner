"""N43 -- what conditioning scheme buys, and what our architecture reveals about conditional models.

§13 found that score addition works on CelebA (factored conditioning) and fails on ColorMNIST
(one entangled row per combination), and concluded the *granularity of the conditionals* decides it.
That comparison confounded two things: the conditioning scheme AND the dataset.  This runs both
schemes on the SAME data, the same UNet, the same recipe and the same epoch count, so the only
difference is how the model is told what to draw.

**The mechanism the literature proposes.**  Recent work (Apple, *Local Mechanisms of Compositional
Generalization in Conditional Diffusion*, arXiv:2509.16447) proves an equivalence between
compositional structure and **local conditional scores** -- scores with *sparse dependencies on both
pixels and conditioners*.  Models that generalize compositionally have them; models that fail do not;
and a causal intervention enforcing locality restores generalization.  Related results: compositional
ability emerges multiplicatively with training (arXiv:2501.13241), improves with training in a way
that is not gradual (arXiv:2502.12089 -- compare our own N35 competence ladder), and linear score
composition is only justified when the conditionals' supports factorize or are orthogonal, which is
exactly the failure §13 measured.

So this measures the proposed mechanism directly, on two backbones that differ only in conditioning:

  A. CONDITIONER LOCALITY.  Perturb one factor's conditioning contribution and measure where the
     score changes.  Two numbers per factor:
       `concentration`  fraction of the total change carried by the top 10% of pixels -- how SPARSE
                        the dependence is spatially
       `selectivity`    change on the pixels that factor owns, over change on the pixels it does not
                        (foreground vs background masks come from the image itself)
     Prediction: factored conditioning is more local and more selective on both.

  B. WHAT OUR ARCHITECTURE GETS.  ideal-words R^2 (does the embedding space factorize at all?), and
     every composition arm on both backbones.  §13 predicts score addition FLIPS -- 0.000 on the
     entangled backbone, working on the factored one -- and this is the first test of that where the
     dataset is held fixed.

  C. WHAT THE FACTORED BACKBONE REVEALS ABOUT OUR COMPOSER.  Only here do we know the GROUND TRUTH
     per-factor embeddings: the model was trained with one vector per (factor, value) and they are
     readable off `model.attr`.  Our composer estimates exactly that object by ridge regression on
     class embeddings, so for the first time its estimate can be scored against the truth rather than
     against downstream accuracy.  That is a validation the entangled backbone makes impossible.

    python experiments/n43_conditioning.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries                # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.ddpm import CH                                       # noqa: E402
from lifelong_ood.neural.idealwords import factorize, slot_factors            # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
FACTOR_NAMES = ("digit", "fg", "bg")


def conditioner_locality(field, X, embs, deltas, img, seed=0, u=0.3, top=0.10):
    """Where does the score move when ONE factor's conditioning is perturbed?

    `embs` is the base conditioning per image; `deltas[f]` a perturbation of factor f's contribution.
    -> per factor: spatial concentration of |Delta eps| and its selectivity for that factor's pixels.

    The masks come from the images themselves: a pixel is FOREGROUND if it is brighter than the
    image's median, which on ColorMNIST is the digit stroke.  So "does perturbing the fg conditioner
    move fg pixels more than bg pixels" is answerable without any labels.
    """
    out = {}
    base = field.eps(X, u, None, emb=embs)
    A = np.asarray(X, np.float64).reshape(len(X), img, img, CH)
    lum = A.mean(-1)
    fg_mask = (lum > np.median(lum.reshape(len(X), -1), axis=1)[:, None, None])
    for f, d in enumerate(deltas):
        pert = field.eps(X, u, None, emb=embs + d)
        dd = np.abs(pert - base).reshape(len(X), img, img, CH).mean(-1)   # [N, H, W]
        flat = dd.reshape(len(X), -1)
        tot = flat.sum(1) + 1e-12
        k = max(1, int(top * flat.shape[1]))
        conc = np.sort(flat, axis=1)[:, -k:].sum(1) / tot                 # top-10% share
        infg = (dd * fg_mask).reshape(len(X), -1).sum(1)
        inbg = (dd * ~fg_mask).reshape(len(X), -1).sum(1)
        nfg = fg_mask.reshape(len(X), -1).sum(1) + 1e-12
        nbg = (~fg_mask).reshape(len(X), -1).sum(1) + 1e-12
        sel = (infg / nfg) / ((inbg / nbg) + 1e-12)                       # per-pixel ratio
        out[FACTOR_NAMES[f]] = dict(concentration=float(conc.mean()),
                                    selectivity=float(np.median(sel)),
                                    magnitude=float(tot.mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--factored-ckpt",
                   default=os.path.join(CACHE, "ddpm_colormnist_factored.pt"))
    p.add_argument("--n-cases", type=int, default=6)
    p.add_argument("--n-probe", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_probe, args.n_per_slot = 2, 16, 20
        args.steps, args.inv_steps, args.epochs, args.n_samples = 20, 60, 3, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)[:args.n_cases]
    rng = np.random.default_rng(args.seed)
    pidx = rng.choice(np.where(data["seen"])[0], args.n_probe, replace=False)
    Xp = np.stack([to_model(data["X"][i]).ravel() for i in pidx])

    rows, locality, recovery = [], {}, {}
    for scheme, ck_path in (("class", args.class_ckpt), ("factored", args.factored_ckpt)):
        if not os.path.exists(ck_path):
            print(f"[n43] {scheme}: {ck_path} missing -- skipping", flush=True)
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        remap = ck["remap"]
        slots = np.array(sorted(remap))
        labels = slot_factors(slots)
        field = load_field(ck_path, device=dev)
        model = field.model
        img = field.img

        # ---- the conditioning vectors, per scheme -------------------------
        if scheme == "factored":
            attr = model.attr.detach().cpu().numpy()          # [n_factors, n_values, edim] TRUTH
            C = np.stack([attr[np.arange(labels.shape[1]), labels[i]].sum(0)
                          for i in range(len(slots))])
        else:
            C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
            attr = None

        fac = factorize(C, labels)
        print(f"\n[n43] === {scheme} conditioning ===  ideal-words R^2 {fac['r2']:.4f}", flush=True)

        # ---- A. conditioner locality --------------------------------------
        sl = data["slot"][pidx]
        rowsC = np.stack([C[int(np.where(slots == s)[0][0])] for s in sl])
        scale = float(np.mean(np.linalg.norm(C, axis=1))) * 0.5
        deltas = []
        for f in range(labels.shape[1]):
            if scheme == "factored":
                # swap this factor's TRUE vector for another value's -- the cleanest perturbation
                v = labels[[int(np.where(slots == s)[0][0]) for s in sl]][:, f]
                alt = (v + 1) % attr.shape[1]
                deltas.append(attr[f][alt] - attr[f][v])
            else:
                d = np.asarray(fac["V"][f][1] - fac["V"][f][0], np.float64)
                deltas.append(np.repeat(d[None], len(sl), 0))
            deltas[-1] = deltas[-1] / (np.linalg.norm(deltas[-1], axis=1, keepdims=True) + 1e-12) * scale
        loc = conditioner_locality(field, Xp, rowsC, deltas, img, seed=args.seed)
        locality[scheme] = loc
        for f, v in loc.items():
            print(f"[n43]   perturb {f:6s}: top-10% share {v['concentration']:.3f}   "
                  f"fg/bg selectivity {v['selectivity']:.3f}   |Δε| {v['magnitude']:.2f}", flush=True)

        # ---- C. does our composer recover the TRUE factor vectors? ---------
        if attr is not None:
            errs = {}
            for f in range(labels.shape[1]):
                nv = len(np.unique(labels[:, f]))
                Vt = attr[f][:nv]
                Ve = np.asarray(fac["V"][f])[:nv]
                Vt = Vt - Vt.mean(0, keepdims=True)
                Ve = Ve - Ve.mean(0, keepdims=True)
                cs = [float(np.dot(Vt[i], Ve[i]) /
                            (np.linalg.norm(Vt[i]) * np.linalg.norm(Ve[i]) + 1e-12))
                      for i in range(nv)]
                errs[FACTOR_NAMES[f]] = float(np.mean(cs))
            recovery = errs
            print(f"[n43]   ridge-estimated vs TRUE factor vectors, cosine: "
                  + "  ".join(f"{k} {v:+.3f}" for k, v in errs.items()), flush=True)

        # ---- B. composition arms ------------------------------------------
        gains = backbone_gains(field, C, slots, probes, data, seed=args.seed, steps=args.steps,
                               n=8 if args.quick else 16)
        for q in qs:
            R = data["X"][q["rest"][:100]].astype(np.float32)
            xq = to_model(data["X"][q["query"]]).ravel()
            L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
            L.attach_factorization(C, labels, gains=gains)
            for arm in ("ours", "score-add"):
                try:
                    if arm == "ours":
                        nm = f"c{q['slot']}"
                        L.learn_concept_factored(xq, nm, n_pool=32, epochs=args.epochs,
                                                 inv_steps=args.inv_steps, seed=args.seed)
                        S = to_data(L.sample_concept(nm, n=args.n_samples, steps=args.steps,
                                                     seed=args.seed)).astype(np.float32)
                    else:
                        # per-FACTOR conditionals added in score space -- the operator §13 says
                        # needs factored conditionals to work at all
                        cs = [np.asarray(fac["mu"]) + gains[f] * np.asarray(fac["V"][f][q[k]])
                              for f, k in enumerate(("d", "f", "b"))]
                        def guide(x, u, cs=cs):
                            e0 = field.eps(x, u, None)
                            return e0 + sum(field.eps(x, u, None, emb=c) - e0 for c in cs)
                        S = to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                 seed=args.seed, guide_eps=guide,
                                                 guide_w=lambda u: 1.0,
                                                 clip=(-1, 1))).astype(np.float32)
                    a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                    pr = precision_recall(S, R)
                    rows.append(dict(scheme=scheme, arm=arm, slot=int(q["slot"]), joint=a["joint"],
                                     digit=a["digit"], fg=a["fg"], bg=a["bg"],
                                     recall=pr["recall"], f1=pr["f1"], vendi=vendi_score(S),
                                     r2=float(fac["r2"])))
                except Exception as exc:
                    print(f"[n43]   {scheme}/{arm} slot {q['slot']} FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            del L
            torch.cuda.empty_cache()
            print(f"[n43]   slot {q['slot']}: " + "  ".join(
                f"{r['arm']} joint {r['joint']:.2f} f1 {r['f1']:.2f}"
                for r in rows[-2:]), flush=True)
        del field
        torch.cuda.empty_cache()

    save_json(dict(rows=rows, locality=locality, recovery=recovery, config=vars(args)),
              os.path.join(RESULTS, f"n43_conditioning{args.tag}.json"), seed=args.seed)

    print(f"\n[n43] composition by conditioning scheme")
    print(f"    {'scheme':10s}{'arm':11s}{'R^2':>7s}{'joint':>8s}{'recall':>8s}{'F1':>7s}")
    for scheme in ("class", "factored"):
        for arm in ("ours", "score-add"):
            sel = [r for r in rows if r["scheme"] == scheme and r["arm"] == arm]
            if not sel:
                continue
            g = lambda k: float(np.mean([r[k] for r in sel]))
            print(f"    {scheme:10s}{arm:11s}{sel[0]['r2']:7.3f}{g('joint'):8.3f}"
                  f"{g('recall'):8.3f}{g('f1'):7.3f}")
    if len(locality) == 2:
        print(f"\n[n43] conditioner locality (the arXiv:2509.16447 mechanism)")
        print(f"    {'factor':8s}{'concentration':>26s}{'selectivity':>24s}")
        print(f"    {'':8s}{'class':>13s}{'factored':>13s}{'class':>12s}{'factored':>12s}")
        for f in FACTOR_NAMES:
            c, fa = locality["class"][f], locality["factored"][f]
            print(f"    {f:8s}{c['concentration']:13.3f}{fa['concentration']:13.3f}"
                  f"{c['selectivity']:12.3f}{fa['selectivity']:12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
