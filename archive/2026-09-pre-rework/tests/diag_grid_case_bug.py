"""Two faults the h4 case study makes visible, checked numerically rather than by eye.

The figure shows `ours (ADDRESS)` rows that look IDENTICAL across four different held-out cells,
and every generated row looks colour-destroyed while the real row does not.  Two independent
suspects, and they have to be separated before either is fixed:

  1. RANGE.  `sample_embedding` and `field.sample` return MODEL space [-1,1] -- `n71` wraps both in
     `to_data` for exactly this reason.  The figure calls `np.clip(S, 0, 1)` on them directly, which
     would throw away the whole negative half.  If n73 also omits `to_data`, then every generated
     arm has been SCORED by probes trained on [0,1] images while living in [-1,1], and the joint
     accuracies in RESULTS 7ee are measurements of that mismatch rather than of composition.

  2. CONDITIONING.  If `sample_embedding` returns the same images for different composed points,
     the composed embedding is not reaching the model at all and every ADDRESS number is really an
     unconditional number.

Both are cheap to settle and neither can be settled by looking at the picture.

    python tests/diag_grid_case_bug.py --ckpt results/cache/ddpm_celeba_grid_h4.pt
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
from lifelong_ood.neural import (NeuralScoreField, UNet,              # noqa: E402
                                 sample_embedding, to_data)
from lifelong_ood.neural.idealwords import compose, factorize         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid_h4.pt"))
    p.add_argument("--n", type=int, default=6)
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
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    fac = factorize(C, np.stack([F[slot == int(s)][0] for s in seen]))
    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    print(f"[bug] REAL images range [{X.min():.3f}, {X.max():.3f}]  <- the probes' space", flush=True)

    # ── 1. what space do the samplers return? ────────────────────────────────
    held = [h for h in d["held"] if (slot == int(h)).any()]
    S0 = np.asarray(sample_embedding(field, compose(fac, tuple(int(v) for v in F[
        np.where(slot == int(held[0]))[0][0]])), n=args.n, steps=args.steps, seed=args.seed),
        np.float32)
    U0 = np.asarray(field.sample(n=args.n, steps=args.steps, seed=args.seed, branch=None),
                    np.float32)
    print(f"[bug] sample_embedding range [{S0.min():.3f}, {S0.max():.3f}], "
          f"mean {S0.mean():+.3f}")
    print(f"[bug] field.sample      range [{U0.min():.3f}, {U0.max():.3f}], "
          f"mean {U0.mean():+.3f}")
    print(f"[bug] to_data(sample)   range [{to_data(S0).min():.3f}, {to_data(S0).max():.3f}], "
          f"mean {to_data(S0).mean():+.3f}")
    model_space = S0.min() < -0.15
    print(f"[bug] VERDICT 1: samplers return {'MODEL [-1,1]' if model_space else 'DATA [0,1]'} space."
          + ("  n73 and the case figure use them WITHOUT to_data, so both are wrong."
             if model_space else "  no conversion needed."))
    if model_space:
        frac = float((S0 < 0).mean())
        print(f"[bug]   {frac:.1%} of sample pixels are negative -- clip(S,0,1) destroys all of "
              f"them, and probes trained on [0,1] see a distribution shifted by -1 and scaled by 2.")

    # ── 2. does the composed point actually change the samples? ──────────────
    print(f"\n[bug] {'cell':>6s}{'||c_new||':>11s}{'d(c, c_first)':>15s}{'d(img, img_first)':>19s}")
    ref_c, ref_S = None, None
    for h in held:
        idx = np.where(slot == int(h))[0]
        tgt = tuple(int(v) for v in F[idx[0]])
        c = compose(fac, tgt)
        S = np.asarray(sample_embedding(field, c, n=args.n, steps=args.steps, seed=args.seed),
                       np.float32)
        if ref_c is None:
            ref_c, ref_S = c, S
        print(f"[bug] {h:>6d}{np.linalg.norm(c):>11.4f}"
              f"{np.linalg.norm(c - ref_c):>15.4f}"
              f"{float(np.linalg.norm(S - ref_S)):>19.4f}")

    # the same question against a TRAINED branch, which is known to respond
    A = np.asarray(field.sample(n=args.n, steps=args.steps, seed=args.seed,
                                branch=remap[int(seen[0])]), np.float32)
    B = np.asarray(field.sample(n=args.n, steps=args.steps, seed=args.seed,
                                branch=remap[int(seen[-1])]), np.float32)
    print(f"[bug] reference: two different TRAINED branches, same seed -> "
          f"d(img) {float(np.linalg.norm(A - B)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
