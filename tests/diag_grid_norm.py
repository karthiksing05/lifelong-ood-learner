"""Is the composed address OFF-MANIFOLD because its NORM is atypical?

The symptom, consistent across every CelebA-grid run and every backbone: ADDRESS wins on factor
accuracy and loses badly on realism.

    arm             joint   vendi   precision
    address         0.194   13.6    0.000
    nearest-class   0.125    3.5    0.200
    REAL images     0.414    2.6    0.750

Vendi far ABOVE the data's own and precision far BELOW a trained class is not "more diverse" -- it
is samples the data would not produce.  It is also the thing that has looked wrong in the CelebA
case studies from the beginning.

THE SUSPECT.  `compose` returns `mu + sum_f V_f[value_f]`, and the per-factor vectors are CENTRED,
so summing them can land well inside the shell the trained embeddings occupy.  A conditioning vector
of atypical norm is off-distribution for a model that only ever saw embeddings on that shell, and
the sampler then behaves as it does off-distribution: high variance, low realism.  If that is the
cause it is also trivially fixable -- rescale the composed point to the trained norm, which changes
the direction not at all and is therefore still an ADDRESS, still ~0 parameters.

Two arms, everything else identical:

    address         compose(fac, target)                      -- as reported everywhere
    address-normed  the same point rescaled to the MEDIAN norm of the trained embedding rows

If normalisation recovers precision without costing joint accuracy, the composer should adopt it and
every grid number should be re-measured.  If it does not, the realism gap is the backbone's and must
be reported as such -- a 9.3M-parameter UNet trained for half an hour on 64x64 faces is not expected
to be photorealistic, and `nearest-class` scoring only 0.20 precision says the ceiling is low for
every generated arm, not just for ours.

    python tests/diag_grid_norm.py --ckpt results/cache/ddpm_celeba_grid.pt
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

from lifelong_ood.data.celeba import load_celeba_grid                 # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                precision_recall, vendi_score)
from lifelong_ood.neural import NeuralScoreField, UNet, sample_embedding  # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid.pt"))
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})
    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    d = load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                         n_per_cell=int(sp.get("n_per_slot", 3000)),
                         n_holdout=int(sp.get("n_holdout") or (12 - len(seen))),
                         max_rows=sp.get("max_rows"))
    X, F, slot = np.asarray(d["X"], np.float32), np.asarray(d["factors"]), np.asarray(d["slot"])
    probes = fit_factor_probes(d, seed=args.seed)
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    lab = np.stack([F[slot == int(s)][0] for s in seen])
    fac = factorize(C, lab)

    norms = np.linalg.norm(C, axis=1)
    med = float(np.median(norms))
    print(f"[norm] trained embedding norms: median {med:.3f}, range "
          f"{norms.min():.3f}-{norms.max():.3f} over {len(C)} seen cells", flush=True)
    print(f"[norm] mu norm {np.linalg.norm(fac['mu']):.3f}; per-factor vector norms "
          + " | ".join(f"{n}: {np.linalg.norm(v, axis=1).mean():.3f}"
                       for n, v in zip(d["names"], fac["V"])), flush=True)

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    rows = []
    for ci, h in enumerate(d["held"]):
        idx = np.where(slot == int(h))[0]
        if not len(idx):
            continue
        tgt = tuple(int(v) for v in F[idx[0]])
        R = X[idx[:60]]
        c = compose(fac, tgt)
        nc = float(np.linalg.norm(c))
        c_n = c * (med / max(nc, 1e-9))
        print(f"[norm] cell {h} {tgt}: ||composed|| {nc:.3f} = {nc / med:.2f}x the trained median",
              flush=True)
        for tag, cc in (("address", c), ("address-normed", c_n)):
            S = np.asarray(sample_embedding(field, cc, n=args.n_samples, steps=args.steps,
                                            seed=args.seed + ci), np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, R)
            j = float(a["joint"])
            rows.append(dict(arm=tag, cell=int(h), joint=j if j == j else 0.0,
                             precision=pr["precision"], recall=pr["recall"],
                             vendi=float(vendi_score(S))))
        # the instrument's own reading on this cell, so nothing below is read against 1.0
        S = X[idx[60:60 + args.n_samples]] if len(idx) >= 60 + args.n_samples \
            else X[idx[-args.n_samples:]]
        a = factor_accuracy(probes, S, tgt)
        pr = precision_recall(np.asarray(S, np.float32), R)
        rows.append(dict(arm="real", cell=int(h), joint=float(a["joint"]),
                         precision=pr["precision"], recall=pr["recall"],
                         vendi=float(vendi_score(np.asarray(S, np.float32)))))

    def m(arm, k):
        v = [r[k] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n[norm] {'=' * 60}")
    print(f"{'arm':16s}{'joint':>9s}{'precision':>11s}{'recall':>9s}{'vendi':>8s}")
    for arm in ("address", "address-normed", "real"):
        print(f"{arm:16s}{m(arm, 'joint'):>9.3f}{m(arm, 'precision'):>11.3f}"
              f"{m(arm, 'recall'):>9.3f}{m(arm, 'vendi'):>8.2f}")

    dp = m("address-normed", "precision") - m("address", "precision")
    dj = m("address-normed", "joint") - m("address", "joint")
    print("\n[norm] VERDICT")
    if dp > 0.05 and dj > -0.05:
        print(f"[norm]   NORMALISING THE ADDRESS RECOVERS REALISM: precision {dp:+.3f} at a joint")
        print(f"[norm]   cost of {dj:+.3f}. The composed point was off the shell the trained")
        print("[norm]   embeddings occupy, and rescaling changes its DIRECTION not at all -- so")
        print("[norm]   this is still an address and still ~0 parameters. Adopt it and re-measure.")
    elif dp > 0.05:
        print(f"[norm]   Normalising buys precision {dp:+.3f} but costs joint {dj:+.3f}. A")
        print("[norm]   trade, not a fix -- report both rather than silently picking one.")
    else:
        print(f"[norm]   NORM IS NOT THE CAUSE: precision moves {dp:+.3f}. The realism gap belongs")
        print("[norm]   to the backbone, and `nearest-class` -- a TRAINED class -- scoring low")
        print("[norm]   precision too is the evidence for that. Report the gap as the backbone's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
