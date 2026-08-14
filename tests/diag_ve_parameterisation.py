"""What does the pretrained Shapes3D VE model actually PREDICT?  Measure, do not guess.

`eurecom-ds/scoresdeve-ema-shapes3d-64` is a Score-SDE VARIANCE-EXPLODING model
(`ScoreSdeVeScheduler`, sigma 0.01 .. 90.0).  Our entire stack is variance-preserving with a cosine
schedule.  Feeding it `u * 1000` as a timestep is meaningless to it, and the symptom was unambiguous:
fine-tuning sat at eps-loss 1.0000 for 1200 steps, which is exactly what predicting zero scores.

There is an exact correspondence between the two, so the weights ARE usable:

    VP:  x_t = sqrt(a) x0 + sqrt(1-a) eps
    /sqrt(a):  x_t / sqrt(a) = x0 + sqrt((1-a)/a) eps        <- VE form with sigma = sqrt((1-a)/a)

so a VP sample at abar=a is a VE sample at sigma=sqrt((1-a)/a) after dividing by sqrt(a).  What is NOT
determined by that algebra is what the network returns -- VE implementations variously output the
score (-eps/sigma), the noise itself, or a sigma-scaled variant -- nor whether it wants images in
[0,1] or [-1,1].  Four candidate readings times two input ranges, scored by how well each recovers
the KNOWN eps that was added.  The winner is the one to build on; if none wins, the weights are not
usable through our stack and that is worth knowing before spending GPU hours on it.

    python tests/diag_ve_parameterisation.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.factored import load_factored                        # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import UNet2DModel
    mid = "eurecom-ds/scoresdeve-ema-shapes3d-64"
    unet = UNet2DModel.from_pretrained(mid, subfolder="unet").to(dev).eval()
    print(f"[ve] {mid}: sample_size {unet.config.sample_size}, "
          f"{sum(q.numel() for q in unet.parameters()) / 1e6:.1f}M params", flush=True)

    d = load_factored("shapes3d", n_per_slot=4, seed=0, value_stride=2, n_holdout=25,
                      fix_nuisance=["wall_hue"])
    X = np.asarray(d["X"], np.float32)[:16].reshape(-1, 64, 64, 3)
    x01 = torch.from_numpy(X).permute(0, 3, 1, 2).to(dev)
    g = torch.Generator(device=dev).manual_seed(0)

    print(f"\n{'range':>8} {'sigma':>8} " + " ".join(f"{k:>12}" for k in
                                                     ("out", "-out*sig", "out*sig", "-out")))
    print("-" * 74)
    best = (None, 1e9)
    for rng_name, x0 in (("[0,1]", x01), ("[-1,1]", x01 * 2 - 1)):
        for sigma in (0.05, 0.2, 1.0, 5.0, 20.0):
            e = torch.randn(x0.shape, generator=g, device=dev)
            xs = x0 + sigma * e
            with torch.no_grad():
                out = unet(xs, torch.full((len(xs),), float(sigma), device=dev)).sample
            cands = {"out": out, "-out*sig": -out * sigma, "out*sig": out * sigma, "-out": -out}
            row = []
            for k, v in cands.items():
                m = float(((v - e) ** 2).mean())
                row.append(m)
                if m < best[1]:
                    best = ((rng_name, sigma, k), m)
            print(f"{rng_name:>8} {sigma:>8.2f} " + " ".join(f"{m:>12.4f}" for m in row),
                  flush=True)

    print(f"\n[ve] best reading: {best[0]} at MSE {best[1]:.4f}")
    print("[ve] eps ~ N(0,1), so MSE 1.0 means 'predicts nothing'. A usable reading is well under 1.")
    if best[1] > 0.8:
        print("[ve] *** NO reading recovers eps. These weights are not usable through a VP stack "
              "without the model's own scheduler; do not spend GPU on fine-tuning them this way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
