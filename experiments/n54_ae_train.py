"""N54 -- the ColorMNIST autoencoder the DiT track needs, verified by PROBES rather than by MSE.

Step one of asking whether the three-tier architecture is a property of the METHOD or of the
pixel-space UNet it was developed on.  A DiT works on latent patches and conditions through adaLN
modulation; before it can be trained, ColorMNIST needs a latent space.

**The acceptance test is not reconstruction MSE.**  An autoencoder can post a respectable MSE while
losing the background colour, and background is one of the three factors every composition claim is
scored on -- so a lossy-in-the-wrong-way AE would quietly cap every downstream number and look like a
method failure.  This fits the AE, then reads the SAME digit/fg/bg probes over reconstructions that
the composition metrics use.  If a factor does not survive the round trip, the DiT track is not worth
starting and this says so.

    python experiments/n54_ae_train.py --epochs 40
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                   # noqa: E402
from lifelong_ood.neural.latent import decode, encode, train_ae        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slot", type=int, default=200)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--zdim", type=int, default=4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fg-weight", type=float, default=1.0,
                   help="upweight pixels that differ from the image's modal colour. The first AE "
                        "reached mse 0.00044 and still lost FOREGROUND (probe 1.000 -> 0.835): the "
                        "digit stroke is a few hundred pixels against ~2800 of background, so "
                        "per-pixel MSE barely sees it. Capacity alone does not fix a weighting "
                        "problem.")
    p.add_argument("--out", default=os.path.join(CACHE, "ae_colormnist.pt"))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    seen = np.asarray(data["seen"], bool)
    X = np.asarray(data["X"], np.float32)
    print(f"[n54] {len(X)} images, fitting the AE on the {int(seen.sum())} SEEN ones only -- the "
          f"held-out compositions must not reach the autoencoder either", flush=True)

    m, scale, hist = train_ae(X[seen], epochs=args.epochs, batch=args.batch, lr=args.lr,
                              zdim=args.zdim, base=args.base, device=dev, seed=args.seed,
                              fg_weight=args.fg_weight)
    print(f"[n54] final mse {hist[-1]:.5f}; latent scale {scale:.4f} "
          f"(latents normalised to unit variance)", flush=True)
    if hist[-1] > 0.01:
        # A collapsed AE predicts a constant, whose mse is the data variance (~0.075 here). Say so
        # loudly rather than letting a black reconstruction grid be the first sign.
        print(f"[n54] *** DIVERGED: mse {hist[-1]:.5f} is at data-variance scale, i.e. the decoder "
              f"is predicting a constant. Lower --lr or --fg-weight; do NOT read the probes below "
              f"as a statement about the latent space.", flush=True)

    Z = encode(m, X, scale=scale, device=dev)
    R = decode(m, Z, scale=scale, device=dev)
    mse = float(np.mean((R - X) ** 2))
    print(f"[n54] round-trip mse over ALL images (incl. held-out): {mse:.5f}", flush=True)

    # THE ACCEPTANCE TEST: do the factors survive the round trip?
    probes = fit_probes(data, seed=args.seed, epochs=30)
    rows = []
    for key in ("digit", "fg", "bg"):
        y = np.asarray(data[key])
        a_real = float((probes[key].predict(X.astype(np.float32)) == y).mean())
        a_rec = float((probes[key].predict(R.astype(np.float32)) == y).mean())
        rows.append(dict(factor=key, real=a_real, recon=a_rec, drop=a_real - a_rec))
        print(f"[n54]   {key:6s} probe on REAL {a_real:.3f} -> on RECONSTRUCTION {a_rec:.3f}  "
              f"(drop {a_real - a_rec:+.3f})", flush=True)
    worst = max(r["drop"] for r in rows)
    ok = worst < 0.05
    print(f"\n[n54] {'PASS' if ok else 'FAIL'}: worst factor drop {worst:+.3f} "
          f"({'under' if ok else 'OVER'} the 0.05 bar)")
    if not ok:
        print("[n54] a factor does not survive the autoencoder -- the DiT track would be capped by")
        print("[n54] the latent space, not by the method. Raise zdim or base before continuing.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(args.seed)
        pick = rng.permutation(len(X))[:12]
        fig, axes = plt.subplots(2, 12, figsize=(13, 2.6))
        for k, i in enumerate(pick):
            for r_, (lbl, A) in enumerate((("real", X), ("reconstruction", R))):
                axes[r_][k].imshow(np.clip(A[i].reshape(32, 32, 3), 0, 1))
                axes[r_][k].axis("off")
            axes[0][k].set_title(f"d{int(data['digit'][i])} f{int(data['fg'][i])} "
                                 f"b{int(data['bg'][i])}", fontsize=6)
        axes[0][0].set_ylabel("real", rotation=0, ha="right", va="center", fontsize=8)
        axes[1][0].set_ylabel("recon", rotation=0, ha="right", va="center", fontsize=8)
        for r_ in (0, 1):
            axes[r_][0].axis("on"); axes[r_][0].set_xticks([]); axes[r_][0].set_yticks([])
        fig.suptitle(
            f"ColorMNIST autoencoder: 32x32x3 -> {args.zdim}x8x8, mse {mse:.5f}.\n"
            + "  ".join(f"{r['factor']} probe {r['real']:.3f} -> {r['recon']:.3f} "
                        f"({r['drop']:+.3f})" for r in rows)
            + "\nAccepted on PROBES, not on MSE: an AE can post a fine MSE and still lose a factor "
              "every composition claim is scored on.", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.84])
        o = os.path.join(HERE, "figures", "dit", "d1_ae_reconstruction.png")
        os.makedirs(os.path.dirname(o), exist_ok=True)
        fig.savefig(o, dpi=140, bbox_inches="tight")
        print(f"[n54] wrote {o}")
    except Exception as e:                                             # noqa: BLE001
        print(f"[n54] recon figure failed: {e}")

    torch.save(dict(model=m.state_dict(), scale=scale, zdim=args.zdim, base=args.base,
                    img=32, ch=3, mse=mse, probe_rows=rows, config=vars(args)), args.out)
    print(f"[n54] wrote {args.out}")
    save_json(dict(rows=rows, mse=mse, scale=scale, hist=hist[-5:], config=vars(args)),
              os.path.join(RESULTS, "n54_ae.json"), seed=args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
