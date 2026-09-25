"""At which noise level does an eps-loss comparison actually identify a backbone's DATA RANGE?

`instrument.convention_probe` decides the range by feeding a real image as [0,1] and as [-1,1] and
taking whichever denoises better. On the CelebA grid backbone it answered "[0,1]" for a model we know
is model-space -- `train_ddpm` calls `to_model(X01)`, and its sampler's output spans [-1,1] with 49%
negative mass.

The suspected reason is that the test is confounded by SIGNAL MAGNITUDE: a [0,1] image has half the
amplitude of its [-1,1] counterpart, so at moderate noise `x_t` sits closer to pure noise and
predicting eps from it is trivially easier. If so, the comparison only discriminates at LOW u, where
the signal dominates and feeding the wrong range actually hurts.

This sweeps u to find where the answer flips, so the gate's operating point is measured rather than
guessed -- the same discipline `router.calibrate_u_novel` applies to its own threshold.

    python tests/diag_range_probe.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.celeba import load_celeba_grid                 # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet                # noqa: E402
from lifelong_ood.score import cosine_abar                            # noqa: E402


def main():
    ck_path = os.path.join(HERE, "results", "cache", "ddpm_celeba_grid.pt")
    if not os.path.exists(ck_path):
        print("[range] no celeba grid checkpoint; skipping")
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    d = load_celeba_grid(img=64, seed=0, n_per_cell=200, n_holdout=3, max_rows=40000)
    X = np.asarray(d["X"], np.float32)[:8]
    f = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                         layout=ck.get("layout"))
    f.model.load_state_dict(ck["ema"])
    f.model.eval()
    t = torch.as_tensor(X, device=dev).reshape(8, 64, 64, 3).permute(0, 3, 1, 2).contiguous()

    print("[range] this backbone is MODEL space by construction (train_ddpm calls to_model), so")
    print("[range] '[-1,1]' is the right answer at every u the test is trustworthy at.\n")
    print(f"[range] {'u':>6s}{'loss[0,1]':>12s}{'loss[-1,1]':>12s}   {'eps says':>8s}"
          f"        {'rel[0,1]':>11s}{'rel[-1,1]':>11s}   {'recon says':>8s}")
    for u in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        a = float(cosine_abar(np.array([u]))[0])

        def loss(x, a=a, u=u):
            g = torch.Generator(device=dev).manual_seed(0)
            tot = 0.0
            for _ in range(4):
                e = torch.randn(x.shape, generator=g, device=dev)
                xt = (a ** 0.5) * x + ((1 - a) ** 0.5) * e
                uu = torch.full((len(x),), float(u), device=dev)
                tot += float(((f.model(xt, uu) - e) ** 2).mean())
            return tot / 4

        def recon(x, a=a, u=u):
            """RELATIVE x0-reconstruction error, which is magnitude-fair.

            eps-loss is not: a [0,1] image has half the amplitude of its [-1,1] counterpart, so
            x_t sits closer to pure noise and predicting eps from it is trivially easier. That is
            why the eps comparison answers '[0,1]' at every u for a model we know is model-space.
            Reconstructing x0 and measuring the error RELATIVE to the signal removes the scale.
            """
            g = torch.Generator(device=dev).manual_seed(0)
            tot = 0.0
            for _ in range(4):
                e = torch.randn(x.shape, generator=g, device=dev)
                xt = (a ** 0.5) * x + ((1 - a) ** 0.5) * e
                uu = torch.full((len(x),), float(u), device=dev)
                x0h = (xt - ((1 - a) ** 0.5) * f.model(xt, uu)) / (a ** 0.5)
                tot += float(((x0h - x).pow(2).mean() / x.pow(2).mean()).item())
            return tot / 4

        l01, l11 = loss(t), loss(t * 2 - 1)
        r01, r11 = recon(t), recon(t * 2 - 1)
        print(f"[range] {u:>6.2f}{l01:>12.5f}{l11:>12.5f}   "
              f"{'[0,1]' if l01 < l11 else '[-1,1]':>8s}"
              f"{'  ' if l11 < l01 else '  WRONG'}"
              f"{r01:>11.4f}{r11:>11.4f}   "
              f"{'[0,1]' if r01 < r11 else '[-1,1]':>8s}"
              f"{'' if r11 < r01 else '  WRONG'}")
    print("\n[range] Use the lowest u at which the answer is correct: there the signal dominates,")
    print("[range] and a wrong range genuinely hurts rather than merely shrinking the input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
