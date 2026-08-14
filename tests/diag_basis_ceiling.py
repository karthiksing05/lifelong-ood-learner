"""Does ADDRESS fail on CelebA because its factor basis is too weak to express the class?

N48 (RESULTS 5e-sexies): on the paper's own CelebA backbone, `ours (ADDRESS)` scores joint 0.131
against unconditional sampling's 0.128 -- it does not clear the floor. The proposed reason is that
ideal-words fits CelebA's 51 seen classes over 14 attributes at **R^2 = 0.620**, against 0.826 on
ColorMNIST, and ADDRESS composes `c_new = mu + sum_f g_f(w_f . V_f)` strictly inside that basis, so
R^2 bounds what the route can express.

That was recorded as *consistent with the numbers but not demonstrated*, and the obvious test --
"does accuracy track R^2 across backbones?" -- needs backbones we do not have. This is the cheap
version, and it is sharper because it varies the basis quality WITHIN one backbone:

    For each SEEN class, the basis reconstruction residual ||c_true - compose(fac, combo)|| is
    known exactly, because a seen class HAS a true embedding. Sample the model at the RECONSTRUCTED
    embedding and score the composition. If ADDRESS is limited by the basis, per-class accuracy
    should fall as the per-class residual rises -- and it should approach the accuracy of the TRUE
    embedding where the residual is near zero.

Two controls make the answer readable:

    true         sample at the class's own trained embedding      -- the ceiling
    address      sample at the basis reconstruction of it         -- what ADDRESS can reach
    uncond       sample with no conditioning                      -- the floor

No held-out label is touched: every class here is one the backbone trained on, which is what makes
the residual computable at all.

    python tests/diag_basis_ceiling.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.data.celeba import load_celeba                                    # noqa: E402
from lifelong_ood.evals import attr_accuracy, fit_attr_probes, probe_usable    # noqa: E402
from lifelong_ood.neural import to_data                                        # noqa: E402
from lifelong_ood.neural.idealwords import compose as fac_compose, factorize   # noqa: E402
from lifelong_ood.neural.pretrained import load_paper_backbone                 # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(CACHE, "ddpm_celeba_paper128.pt")
    if not os.path.exists(ckpt):
        print(f"[basis] missing {ckpt} -- run n47 first")
        return 1
    field, img, cemb, remap = load_paper_backbone(ckpt, device=dev, batch=4)
    data = load_celeba(n_per_class=100, img=img, seed=0, min_count=25, max_rows=25000)
    seen = np.asarray(data["seen"], bool)
    probes = fit_attr_probes(dict(X=np.asarray(data["X"]), attrs=data["attrs"], img=img,
                                  seen=seen), seed=0, epochs=30)
    usable = [a for a in probes["_attrs"] if probe_usable(probes, a)]

    seen_cls = np.unique(np.asarray(data["cls"])[seen])
    rows_seen = [remap[int(c)] for c in seen_cls if int(c) in remap]
    keep = [int(c) for c in seen_cls if int(c) in remap]
    lab = np.stack([np.asarray(data["combos"][c], int) for c in keep])
    fac = factorize(cemb[rows_seen], lab)
    print(f"[basis] ideal-words over {len(keep)} seen classes x {lab.shape[1]} attributes: "
          f"R^2 = {fac['r2']:.3f}\n")

    # per-class residual, then the classes spanning its range
    resid = np.array([float(np.linalg.norm(cemb[remap[c]] - fac_compose(fac, np.asarray(
        data["combos"][c], int)))) / (np.linalg.norm(cemb[remap[c]]) + 1e-9) for c in keep])
    order = np.argsort(resid)
    pick = [order[0], order[len(order) // 4], order[len(order) // 2],
            order[3 * len(order) // 4], order[-1]]

    print(f"  {'class':>6s} {'resid':>7s} {'true':>7s} {'address':>8s} {'uncond':>7s}   "
          f"(joint accuracy at each conditioning)")
    out = []
    for j, i in enumerate(pick):
        c = keep[i]
        combo = np.asarray(data["combos"][c], int)
        got = {}
        for nm, emb in (("true", cemb[remap[c]]),
                        ("address", fac_compose(fac, combo)), ("uncond", None)):
            S = to_data(field.sample(n=24, steps=50, seed=1234 + j, clip=(-1, 1),
                                     emb=emb)).astype(np.float32)
            got[nm] = float(attr_accuracy(probes, S, combo, attrs=usable).get("joint", np.nan))
        print(f"  {c:6d} {resid[i]:7.3f} {got['true']:7.3f} {got['address']:8.3f} "
              f"{got['uncond']:7.3f}")
        out.append((resid[i], got["true"], got["address"], got["uncond"]))

    A = np.array(out)
    t, ad, un = A[:, 1].mean(), A[:, 2].mean(), A[:, 3].mean()
    print(f"\n  mean over these classes: true {t:.3f}  address {ad:.3f}  uncond {un:.3f}")
    # THE PRECONDITION, CHECKED BEFORE ANY CORRELATION IS PRINTED.  The residual/accuracy
    # correlation only means something if conditioning on the TRUE embedding beats conditioning on
    # nothing -- otherwise there is no ceiling for the basis to be falling short of, and the
    # correlation is over a column of noise.  The first run of this printed -0.698 under exactly
    # that condition, which reads as support for a conclusion the data cannot carry.
    if t <= un:
        print(f"\n  CONTROL FAILED: the true embedding ({t:.3f}) does not beat unconditional "
              f"({un:.3f}).\n"
              f"  The ceiling is at the floor, so this metric cannot rank ANY conditioning on this\n"
              f"  backbone -- the basis is exonerated, and no accuracy number here (ours or a\n"
              f"  baseline's) is interpretable. Correlation suppressed: it would be over noise.")
    elif len(A) > 2 and np.std(A[:, 0]) > 1e-9:
        r = float(np.corrcoef(A[:, 0], A[:, 2])[0, 1])
        print(f"  corr(residual, ADDRESS joint) = {r:+.3f}   (negative = the basis is the limit)")
    print("\n[basis] if ADDRESS sits near `uncond` while `true` sits well above it, the CONDITIONING\n"
          "        works and the BASIS is what fails -- the composed embedding is not close enough\n"
          "        to any real one. If `true` is also near the floor, the backbone's conditioning is\n"
          "        the limit and the basis is exonerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
