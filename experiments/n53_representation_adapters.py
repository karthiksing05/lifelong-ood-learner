"""N53 -- representational strength, with the ADAPTERS ACTUALLY SWITCHED ON.

N50 asked whether an absorbed concept's representation is as informative as a trained one's and
answered "no, and it barely beats no adaptation".  That answer is withdrawn (RESULTS 6e).  Its
readout calls `conditional_losses(field, x, conds)`, which substitutes an EMBEDDING and nothing else.
Consolidation produces an embedding **and a LoRA adapter**, and an adapter only enters a forward pass
inside `bank.use(name)`, which sets `state["path"]`.  N50 never opens it, so `state["path"]` stayed
empty -- the frozen base -- and every trained adapter was inert for the whole measurement.

So N50 measured "do twelve extra EMBEDDING columns help a probe", and 6e-bis proves that must be no:
`c_new` lies in the affine span of the seen embeddings at residual 0.0000, so it adds a point and no
axis.  The adapter is the part that CAN add an axis -- a LoRA changes the function from (x_t, u, c)
to epsilon and is confined to no span -- and it is the part that was switched off.

This file changes exactly one thing: each consolidated concept's column is computed INSIDE its own
`bank.use(name)`, so the column reflects the concept as actually stored.  Everything else -- arms,
matched conditioning counts, per-column standardization, readouts fit on seen slots only -- is N50's.

    ours-embed    embedding only, adapters closed      <- reproduces N50, as the control
    ours-full     embedding + that concept's adapter   <- what "absorbed" actually means
    base          no adaptation
    oracle        trained on all 160

If `ours-full` beats `ours-embed`, the representation is carried by the adapter and N50's conclusion
was an artefact of the readout.  If they tie, the conclusion survives correction and the method
genuinely does not enrich representations -- which matters, because a lifelong scheme meant for
embodied agents and diffusion policies has to build representations, not only render OOD samples.

    python experiments/n53_representation_adapters.py --n-concepts 8
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS, backbone_gains                      # noqa: E402

from lifelong_ood.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.evals import fit_probes                              # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                # noqa: E402
from lifelong_ood.evals import SoftmaxProbe, save_json                 # noqa: E402
from lifelong_ood.neural import NeuralLifelongLearner                  # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_model              # noqa: E402
from lifelong_ood.neural.router import conditional_losses              # noqa: E402


def represent(field, X01, cols, u_grid=(0.15, 0.45), n_noise=2, seed=0):
    """-> [N, M] epsilon error per image per column.

    `cols` is a list of (conditioning_vector, ctx_factory).  `ctx_factory` is a zero-argument
    callable returning a context manager -- `bank.use(name)` for a consolidated concept, or a null
    context for a plain conditioning.  THIS is the difference from N50, which had no context at all
    and therefore measured every column on the frozen base.
    """
    import contextlib
    out = []
    for x01 in X01:
        x = to_model(x01).ravel()
        row = []
        for c, ctx in cols:
            with (ctx() if ctx is not None else contextlib.nullcontext()):
                L = conditional_losses(field, x, np.asarray(c).reshape(1, -1),
                                       u_grid=u_grid, n_noise=n_noise, seed=seed)
            row.append(float(L[0]))
        out.append(row)
    R = np.asarray(out, np.float64)
    mu, sd = R.mean(0, keepdims=True), R.std(0, keepdims=True) + 1e-9
    return (R - mu) / sd


def readouts(Rtr, ytr, Rte, yte, seed=0, epochs=200):
    """Linear probe + 1-NN on the representation.  -> dict."""
    out = {}
    for key, C in (("digit", 10), ("fg", 4), ("bg", 4)):
        p = SoftmaxProbe(C, seed=seed, epochs=epochs).fit(Rtr, ytr[key])
        out[f"lin_{key}"] = float((p.predict(Rte) == yte[key]).mean())
    hit = np.ones(len(Rte), bool)
    for key in ("digit", "fg", "bg"):
        p = SoftmaxProbe({"digit": 10}.get(key, 4), seed=seed, epochs=epochs).fit(Rtr, ytr[key])
        hit &= (p.predict(Rte) == yte[key])
    out["lin_joint"] = float(hit.mean())
    d = ((Rte[:, None, :] - Rtr[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(1)
    for key in ("digit", "fg", "bg"):
        out[f"knn_{key}"] = float((ytr[key][nn] == yte[key]).mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--oracle", default=os.path.join(CACHE, "ddpm_colormnist_oracle.pt"))
    p.add_argument("--n-per-slot", type=int, default=16)
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ock = torch.load(args.oracle, map_location="cpu", weights_only=False)
    rb, ro = ck["remap"], ock["remap"]
    seen = np.array(sorted(rb))
    C = ck["ema"]["cemb.weight"].numpy()[[rb[int(s)] for s in seen]]
    held_all = np.array(sorted(set(int(s) for s in ro) - set(int(s) for s in seen)))
    rng = np.random.default_rng(args.seed)
    held = np.sort(rng.permutation(held_all)[:args.n_concepts])
    print(f"[n53] {len(seen)} seen, {len(held)} held-out concepts consolidated", flush=True)

    te = np.where(np.isin(data["slot"], held))[0]
    tr = np.where(np.isin(data["slot"], seen))[0]
    rng.shuffle(tr)
    tr = tr[:min(len(tr), 1200)]
    ytr = {k: np.asarray(data[k])[tr] for k in ("digit", "fg", "bg")}
    yte = {k: np.asarray(data[k])[te] for k in ("digit", "fg", "bg")}
    print(f"[n53] readout fit on {len(tr)} SEEN images, tested on {len(te)} held-out", flush=True)
    probes = fit_probes(data, seed=args.seed, epochs=30)

    rows = []
    for arm in ("base", "ours-embed", "ours-full", "oracle"):
        field = load_field(args.ckpt if arm != "oracle" else args.oracle, device=dev)
        cols = []
        if arm == "oracle":
            Co = ock["ema"]["cemb.weight"].numpy()
            cols = [(Co[ro[int(s)]], None) for s in seen] + [(Co[ro[int(s)]], None) for s in held]
        elif arm == "base":
            # matched width: base gets its own conditionings twice rather than fewer columns
            cols = [(c, None) for c in C] + [(C[i], None) for i in range(len(held))]
        else:
            L = NeuralLifelongLearner(field, steps=50, seed=args.seed, device=dev)
            # compose_factored needs the basis attached first -- N50 does this and I omitted it.
            gains = backbone_gains(field, C, seen, probes, data, seed=args.seed, steps=50, n=8)
            L.attach_factorization(C, slot_factors(seen), gains=gains)
            names = []
            for j, s in enumerate(held):
                idx = np.where(data["slot"] == int(s))[0]
                xq = to_model(data["X"][int(idx[0])]).ravel()
                nm = f"c{int(s)}"
                L.learn_concept_factored(xq, nm, n_pool=48, epochs=args.epochs,
                                         inv_steps=args.inv_steps, seed=args.seed + j)
                names.append(nm)
                print(f"[n53]   {arm}: consolidated {nm} ({j + 1}/{len(held)})", flush=True)
            cols = [(c, None) for c in C]
            for nm in names:
                emb = L.bank.embedding(nm)
                v = np.asarray(emb.detach().cpu().numpy(), np.float32).ravel()
                # THE ONE DIFFERENCE: ours-full opens the concept's adapter for its own column.
                ctx = (lambda n=nm: L.bank.use(n)) if arm == "ours-full" else None
                cols.append((v, ctx))

        Rtr = represent(field, data["X"][tr], cols, seed=args.seed)
        Rte = represent(field, data["X"][te], cols, seed=args.seed)
        r = readouts(Rtr, ytr, Rte, yte, seed=args.seed)
        rows.append(dict(arm=arm, n_cols=len(cols), **r))
        print(f"[n53] {arm:11s} ({len(cols)} cols)  linear digit {r['lin_digit']:.3f} "
              f"fg {r['lin_fg']:.3f} bg {r['lin_bg']:.3f} joint {r['lin_joint']:.3f} | "
              f"kNN digit {r['knn_digit']:.3f}", flush=True)

    print("\n[n53] " + "=" * 70)
    g = {r["arm"]: r for r in rows}
    if "ours-embed" in g and "ours-full" in g:
        d = g["ours-full"]["lin_digit"] - g["ours-embed"]["lin_digit"]
        print(f"[n53] adapter contribution (ours-full - ours-embed) on linear digit: {d:+.3f}")
        print(f"[n53] ours-full vs base: "
              f"{g['ours-full']['lin_digit'] - g['base']['lin_digit']:+.3f}")
        print(f"[n53] gap to oracle remaining: "
              f"{g['oracle']['lin_digit'] - g['ours-full']['lin_digit']:+.3f}")
    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n53_representation_adapters{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
