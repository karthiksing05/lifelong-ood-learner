"""Do the CelebA prototypes COLLAPSE?  (raised from the case-study figure, 2026-08-15)

The CelebA panels show three selected prototypes that are near-copies of one another -- same face,
same pose, same studio background -- while the three real held-out images are three different
people.  If that is what the pool contains, the whole composition mechanism is disabled: Eq. 8-9
exists to pick experts that DISAGREE, and Eq. 7's per-dimension product over three copies of one
mode is just that mode.  It would explain a CelebA composition that tracks a generic face rather
than the held-out attribute combination.

Measured, against ColorMNIST as the control (its case-study prototypes are visibly different from
each other, so it is the shape a healthy pool has):

  pool size after dedup            how many distinct modes ascent actually found
  pairwise RMSE among selected K   the quantity that decides whether the product can compose
  pairwise RMSE across the pool    whether the collapse is in SELECTION or in DISCOVERY
  spread of real images            the scale bar -- "similar" only means anything against it

The distinction the last two draw is the actionable one.  If the POOL is diverse and the SELECTED
three are not, the fault is in selection.  If the pool itself has collapsed, no selection rule can
help and the ascent settings are what need to change.

    python tests/diag_prototype_diversity.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.compose import (PAPER_CELEBA, PAPER_COLORMNIST,      # noqa: E402
                                  candidates_ascent, poe_teacher)
from lifelong_ood.neural.ddpm import load_field, to_model              # noqa: E402


def rmse_pairs(A, cap=24, seed=0):
    """Mean pairwise per-dimension RMSE -- comparable across resolutions, unlike raw L2.

    Subsamples at RANDOM rather than taking the first `cap`.  `candidates_ascent` emits candidates
    grouped by noise level and, within a level, query-seeded before random-seeded, so a prefix slice
    of a large pool is entirely query-seeded low-noise chains -- which made a 64-start hybrid pool
    read as LESS diverse (0.14x) than an 8-start one (0.75x) when it is strictly a superset.
    """
    if len(A) < 2:
        return float("nan")
    if len(A) > cap:
        A = A[np.random.default_rng(seed).choice(len(A), cap, replace=False)]
    d = [float(np.sqrt(np.mean((A[i] - A[j]) ** 2)))
         for i in range(len(A)) for j in range(i + 1, len(A))]
    return float(np.mean(d))


def run(name, ckpt, loader, disc, K=3, seed=0, dev="cuda"):
    p = os.path.join(CACHE, ckpt)
    if not os.path.exists(p):
        print(f"[div] SKIP {name}: no checkpoint at {p}")
        return None
    field = load_field(p, device=dev)
    X, held_rows = loader()
    xq = to_model(X[held_rows[0]]).ravel()
    M, V, meta = candidates_ascent(field, xq, seed=seed, **disc)
    t = poe_teacher(xq, M, V, K=K, tau=0.05)
    sel = M[t["sel"]]
    real = np.stack([to_model(X[i]).ravel() for i in held_rows[:6]])
    out = dict(pool=len(M), n_raw=int(meta["n_raw"]),
               d_selected=rmse_pairs(sel), d_pool=rmse_pairs(M[:24]),
               d_real=rmse_pairs(real),
               src=[str(s) for s in np.asarray(meta["src"])[t["sel"]]])
    print(f"\n[div] {name}: pool {out['pool']} distinct of {out['n_raw']} raw chains")
    print(f"[div]   pairwise RMSE  selected-3 {out['d_selected']:.4f} | "
          f"pool {out['d_pool']:.4f} | REAL held-out images {out['d_real']:.4f}")
    print(f"[div]   selected chains came from: {out['src']}")
    r = out["d_selected"] / max(out["d_real"], 1e-9)
    print(f"[div]   selected spread is {r:.2f}x the spread of real images of the class")
    if r < 0.5:
        where = "SELECTION" if out["d_pool"] > 1.5 * out["d_selected"] else "DISCOVERY"
        print(f"[div]   -> COLLAPSED, and the collapse is in {where}.")
    else:
        print("[div]   -> prototypes are as varied as real images of the class; no collapse.")
    del field
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def cm():
        from lifelong_ood.data.datasets import load_colormnist
        d = load_colormnist(n_per_slot=20, seed=0)
        ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu",
                        weights_only=False)
        slots = np.asarray(d["slot"])
        held = [s for s in np.unique(slots) if s not in set(int(k) for k in ck["remap"])]
        return np.asarray(d["X"], np.float32), list(np.where(slots == int(held[0]))[0])

    def cel():
        # The 128px CelebA backbone records its holdout as `hold_ids` / `hold_combos` rather than a
        # `seen` mask, and the loader is keyed on (n_per_class, img, min_count) -- read them off the
        # checkpoint instead of guessing, because a mismatched loader would silently score a
        # DIFFERENT class partition than the model was trained with (the fault 5t-bis records).
        from lifelong_ood.data.celeba import load_celeba
        ck = torch.load(os.path.join(CACHE, "ddpm_celeba128.pt"), map_location="cpu",
                        weights_only=False)
        d = load_celeba(img=int(ck.get("img", 128)), min_count=int(ck.get("min_count", 120)),
                        seed=int(ck.get("seed", 0)))
        cls = np.asarray(d["cls"] if "cls" in d else d["slot"])
        # `or []` on a numpy array raises rather than falling back -- the truth value of an array
        # is ambiguous.  Normalise explicitly.
        _h = ck.get("hold_ids")
        hold = [] if _h is None else [int(h) for h in np.asarray(_h).ravel()]
        pick = next((h for h in hold if (cls == h).any()), int(np.unique(cls)[0]))
        print(f"[div]   CelebA held-out class {pick} "
              f"({int((cls == pick).sum())} real images, 0 training rows)")
        return np.asarray(d["X"], np.float32), list(np.where(cls == pick)[0])

    a = run("ColorMNIST (control)", "ddpm_colormnist.pt", cm, dict(PAPER_COLORMNIST), dev=dev)
    b = run("CelebA", "ddpm_celeba128.pt", cel, dict(PAPER_CELEBA), dev=dev)
    if a and b:
        print(f"\n[div] {'=' * 66}")
        print(f"[div] selected-prototype spread relative to real images: "
              f"ColorMNIST {a['d_selected'] / max(a['d_real'], 1e-9):.2f}x, "
              f"CelebA {b['d_selected'] / max(b['d_real'], 1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
