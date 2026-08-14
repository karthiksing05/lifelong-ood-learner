"""Which model-internal signal actually separates the three routes?  Measured, not assumed.

The first version of n18 normalised the best conditional loss by the UNCONDITIONAL loss of the same
query, on the reasoning that dividing out per-image denoising difficulty makes the ratio comparable
across queries.  The smoke run said that reasoning is wrong -- and wrong with the sign reversed
(AUROC 0.29, i.e. novel queries scored as MORE in-distribution than trained ones).

The plausible reading is that the normaliser is not neutral: for a query carrying a factor value the
backbone never saw, the unconditional branch is ALSO bad at it, so the denominator inflates by more
than the numerator and the ratio falls.  A ratio to a quantity that is itself sensitive to novelty
cannot measure novelty.

Rather than patch the ratio and re-guess, this enumerates the candidate signals over a proper query
set and reports AUROC for each, per noise level, so n18 can be built on whichever one measures
rather than on whichever one sounds right.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.neural import load_field, to_model                    # noqa: E402
from lifelong_ood.neural.idealwords import factorize, slot_factors      # noqa: E402
from lifelong_ood.neural.router import (auroc, basis_assignments, conditional_losses,  # noqa: E402
                                        null_conditioning)

HELD_BG = 3
N_PER_ROUTE = int(os.environ.get("DIAG_N", 40))
U_LEVELS = [0.10, 0.20, 0.30, 0.45, 0.60, 0.80]
ROUTES = ("in-distribution", "new-combination", "new-vocabulary")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=60, seed=0)
    ckp = os.path.join(CACHE, "ddpm_colormnist_nobg3.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_trained = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_trained]]
    field = load_field(ckp, device=dev)
    fac = factorize(C, slot_factors(slots_trained))
    basis_conds, _ = basis_assignments(fac, sizes=fac["sizes"])
    null = null_conditioning(field.model).reshape(1, -1)
    print(f"[diag] {len(C)} trained classes, {len(basis_conds)} basis assignments, "
          f"basis sizes {fac['sizes']}", flush=True)

    trained = set(int(s) for s in slots_trained)
    pops = {r: [] for r in ROUTES}
    for s in np.unique(data["slot"]):
        b_ = int(s) % 4
        if b_ == HELD_BG:
            pops["new-vocabulary"].append(int(s))
        elif int(s) in trained:
            pops["in-distribution"].append(int(s))
        else:
            pops["new-combination"].append(int(s))

    rng = np.random.default_rng(0)
    allc = np.concatenate([null, C, basis_conds])
    n_cls = len(C)
    per_route = {r: [] for r in ROUTES}
    for r in ROUTES:
        for s in rng.permutation(pops[r])[:N_PER_ROUTE]:
            idx = np.where(data["slot"] == s)[0]
            xq = to_model(data["X"][idx[0]]).ravel()
            rec = {}
            for u in U_LEVELS:
                L = conditional_losses(field, xq, allc, u_grid=(u,), n_noise=4, seed=0)
                unc, cls, bas = float(L[0]), L[1:1 + n_cls], L[1 + n_cls:]
                rec[u] = dict(uncond=unc, best_class=float(cls.min()), best_basis=float(bas.min()),
                              mean_class=float(cls.mean()), std_class=float(cls.std()),
                              mean_basis=float(bas.mean()), std_basis=float(bas.std()))
            per_route[r].append(rec)
        print(f"[diag] {r}: {len(per_route[r])} queries", flush=True)

    def sig(rec, name, u):
        d = rec[u]
        if name == "best_class_abs":
            return d["best_class"]
        if name == "best_basis_abs":
            return d["best_basis"]
        if name == "uncond_abs":
            return d["uncond"]
        if name == "rel_class(ratio)":
            return d["best_class"] / d["uncond"]
        if name == "rel_basis(ratio)":
            return d["best_basis"] / d["uncond"]
        if name == "gap_class(unc-best)":
            return d["uncond"] - d["best_class"]
        if name == "gap_basis(unc-best)":
            return d["uncond"] - d["best_basis"]
        if name == "basis_minus_class":
            return d["best_basis"] - d["best_class"]
        if name == "margin_class(z)":
            return (d["mean_class"] - d["best_class"]) / max(d["std_class"], 1e-9)
        if name == "margin_basis(z)":
            return (d["mean_basis"] - d["best_basis"]) / max(d["std_basis"], 1e-9)
        raise KeyError(name)

    names = ["best_class_abs", "best_basis_abs", "uncond_abs", "rel_class(ratio)",
             "rel_basis(ratio)", "gap_class(unc-best)", "gap_basis(unc-best)",
             "basis_minus_class", "margin_class(z)", "margin_basis(z)"]

    print("\nAUROC for 'is NOVEL' (new-combination + new-vocabulary vs in-distribution)")
    print(f"{'signal':24s}" + "".join(f"{u:>9.2f}" for u in U_LEVELS))
    best = []
    for nm in names:
        row = []
        for u in U_LEVELS:
            pos = ([sig(r, nm, u) for r in per_route["new-combination"]]
                   + [sig(r, nm, u) for r in per_route["new-vocabulary"]])
            neg = [sig(r, nm, u) for r in per_route["in-distribution"]]
            row.append(auroc(pos, neg))
        print(f"{nm:24s}" + "".join(f"{v:9.3f}" for v in row))
        best.append((max(abs(v - 0.5) for v in row), nm, "novel"))

    print("\nAUROC for 'is NEW VOCABULARY' (vs new-combination — the hard one)")
    print(f"{'signal':24s}" + "".join(f"{u:>9.2f}" for u in U_LEVELS))
    for nm in names:
        row = []
        for u in U_LEVELS:
            pos = [sig(r, nm, u) for r in per_route["new-vocabulary"]]
            neg = [sig(r, nm, u) for r in per_route["new-combination"]]
            row.append(auroc(pos, neg))
        print(f"{nm:24s}" + "".join(f"{v:9.3f}" for v in row))
        best.append((max(abs(v - 0.5) for v in row), nm, "vocab"))

    print("\nmean signal by route, at each noise level (u, signal, ID / COMBO / VOCAB)")
    for nm in ("best_class_abs", "gap_class(unc-best)", "margin_class(z)", "margin_basis(z)",
               "basis_minus_class"):
        for u in (0.20, 0.45):
            v = [float(np.mean([sig(r, nm, u) for r in per_route[k]])) for k in ROUTES]
            print(f"  u={u:.2f} {nm:22s} {v[0]:9.4f} {v[1]:9.4f} {v[2]:9.4f}")

    best.sort(reverse=True)
    print("\nmost discriminative signals overall:")
    for d, nm, task in best[:6]:
        print(f"  |AUROC-0.5| = {d:.3f}  {nm}  ({task})")


if __name__ == "__main__":
    main()
