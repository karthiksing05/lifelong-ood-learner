"""The paper's own backbone, fed a real image vs a permuted one.  This should be decisive.

`google/ddpm-celebahq-256` has FIXED weights fitted to real images.  Until now this repository handed
it a plain `.reshape(-1, 3, H, W)` of an HWC row -- a pixel permutation (tests/diag_layout.py).  On a
backbone we train ourselves that costs the convolutional prior, because training and sampling share
the permutation.  On a PRETRAINED one it is a different and worse thing: scrambled input to a frozen
function.

The companion paper's entire CelebA setup is a fine-tune of this model, so if we cannot get clean
faces out of it unmodified, nothing downstream of it can replicate their figures.

Three checks, cheapest first:

  1. UNCONDITIONAL SAMPLES from the field under each layout.  The pretrained model should produce
     photorealistic faces.  If `permuted` does not and `hwc` does, the layout is the blocker.
  2. DENOISING A REAL IMAGE.  Noise a real face to a middling level and take one Tweedie step.
     Under the correct layout that should return something close to the face; under a permutation it
     cannot, because the filters are reading scrambled neighbourhoods.
  3. THE PIPELINE'S OWN REFERENCE.  `DDPMPipeline` sampling from the same weights, which is what
     the model is supposed to look like.

    python tests/diag_pretrained_layout.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
MODEL = "google/ddpm-celebahq-256"


def main():
    from lifelong_ood.data.celeba import load_celeba
    from lifelong_ood.neural import to_data, to_model
    from lifelong_ood.neural.pretrained import load_pretrained
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    per_layout = {}
    for lay in ("permuted", "hwc"):
        field, img = load_pretrained(MODEL, device=dev, batch=4, layout=lay)
        print(f"[pre-layout] layout={lay:9s} img={img}", flush=True)
        S = to_data(field.sample(n=4, steps=50, seed=0, clip=(-1, 1))).astype(np.float32)
        per_layout[lay] = dict(samples=S, img=img)

        # 2. one-step denoise of a REAL face at a middling noise level
        data = load_celeba(n_per_class=4, img=img, seed=0, min_count=25, max_rows=4000)
        x0 = to_model(np.asarray(data["X"])[:4])
        rng = np.random.default_rng(0)
        u = 0.4
        from lifelong_ood.score import cosine_abar
        ab = float(cosine_abar(np.array([u]))[0])
        e = rng.standard_normal(x0.shape)
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * e
        eh = field.eps(xt, u)
        x0h = (xt - np.sqrt(1 - ab) * eh) / max(np.sqrt(ab), 1e-8)
        err = float(np.sqrt(np.mean((x0h - x0) ** 2)))
        # a useless denoiser returns roughly the noisy input; that is the number to beat
        base = float(np.sqrt(np.mean((xt - x0) ** 2)))
        per_layout[lay]["denoise"] = (np.clip(to_data(x0h), 0, 1).astype(np.float32),
                                      np.clip(to_data(x0), 0, 1).astype(np.float32))
        rows.append((lay, err, base))
        print(f"[pre-layout]   one-step denoise RMSE {err:.4f}  (no-op baseline {base:.4f}, "
              f"{'BETTER' if err < base else 'WORSE than doing nothing'})", flush=True)
        del field
        torch.cuda.empty_cache()

    # 3. the pipeline's own output, as the reference for what these weights should look like
    from diffusers import DDPMPipeline
    pipe = DDPMPipeline.from_pretrained(MODEL).to(dev)
    with torch.no_grad():
        ref = np.asarray(pipe(batch_size=4, num_inference_steps=50,
                              generator=torch.Generator(dev).manual_seed(0),
                              output_type="np").images, np.float32)
    print(f"\n[pre-layout] reference DDPMPipeline output: {ref.shape}")

    print("\n[pre-layout] one-step denoise RMSE (lower is better; the no-op baseline is the bar):")
    for lay, err, base in rows:
        print(f"    {lay:9s} {err:.4f}   vs {base:.4f}   ratio {err / base:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = per_layout["hwc"]["img"]
    panels = [("reference\n(DDPMPipeline)", [r for r in ref]),
              ("field sample\nlayout=permuted", [s.reshape(img, img, 3)
                                                 for s in per_layout["permuted"]["samples"]]),
              ("field sample\nlayout=hwc", [s.reshape(img, img, 3)
                                            for s in per_layout["hwc"]["samples"]]),
              ("real (target)", [x.reshape(img, img, 3)
                                 for x in per_layout["hwc"]["denoise"][1]]),
              ("1-step denoise\nlayout=permuted", [x.reshape(img, img, 3)
                                                   for x in per_layout["permuted"]["denoise"][0]]),
              ("1-step denoise\nlayout=hwc", [x.reshape(img, img, 3)
                                              for x in per_layout["hwc"]["denoise"][0]])]
    fig, axes = plt.subplots(len(panels), 4, figsize=(8.5, 2.1 * len(panels)))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(panels):
        for c in range(4):
            axes[r][c].axis("off")
            if c < len(imgs):
                axes[r][c].imshow(np.clip(np.asarray(imgs[c], np.float32), 0, 1))
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s_ in axes[r][0].spines.values():
            s_.set_visible(False)
    fig.suptitle(f"{MODEL} — fixed pretrained weights, fed a permuted row vs a real image.\n"
                 f"The companion paper's CelebA setup is a fine-tune of this model.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    o = os.path.join(FIG, "extra", "f25_pretrained_layout.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[pre-layout] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
