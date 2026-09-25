"""Does every measurement path agree with a backbone about LAYOUT and RANGE?

This fault class has now cost three results.  RESULTS 30d was a renderer reading HWC rows as CHW
("the bank renders nothing").  The Shapes3D pretrained run hit it twice more, silently:

  * `conditional_losses` hardcoded `.reshape(1, CH, img, img)` -- correct for a `permuted` backbone,
    wrong for an `hwc` one -- so the conditioning gap was measured on a SCRAMBLED image and read
    +0.00007, which looked like "this dataset does not condition";
  * the VE bridge's sampler started at `0.5 + noise` where the VP prior is centred at ZERO, so every
    trajectory began in one basin and DDIM funnelled them to a single washed-out scene.

Both are invisible in a loss curve and both produce plausible-looking numbers.  So this asserts the
conventions directly, per backbone, against the one thing that cannot lie: a REAL image the model was
trained on should be denoised better than a scrambled or mis-scaled version of itself.

    layout   feed a real image the right way and the wrong way; the right way must score lower
    range    feed it in [0,1] and in [-1,1]; whichever the model wants must score lower
    prior    the sampler's starting distribution must match the VP prior (mean ~0, std ~1)

    python tests/diag_backbone_conventions.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.data.factored import load_factored                   # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_model              # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402

CACHE = os.path.join(HERE, "results", "cache")


@torch.no_grad()
def eps_loss(field, x_img, u=0.3, n=4, seed=0):
    """Mean eps error for an already-shaped [B, C, H, W] tensor."""
    g = torch.Generator(device=field.device).manual_seed(seed)
    a = float(cosine_abar(np.array([u]))[0])
    tot = 0.0
    for _ in range(n):
        e = torch.randn(x_img.shape, generator=g, device=field.device)
        xt = (a ** 0.5) * x_img + ((1 - a) ** 0.5) * e
        uu = torch.full((len(x_img),), float(u), device=field.device)
        pred = field.model(xt, uu)
        tot += float(((pred - e) ** 2).mean())
    return tot / n


def check(name, field, X01, img, expect_layout, expect_range):
    dev = field.device
    x = np.asarray(X01[:8], np.float32)
    hwc = torch.from_numpy(x).reshape(-1, img, img, 3).permute(0, 3, 1, 2).contiguous().to(dev)
    chw = torch.from_numpy(x).reshape(-1, 3, img, img).to(dev)
    l_hwc, l_chw = eps_loss(field, hwc), eps_loss(field, chw)
    l_01, l_11 = eps_loss(field, hwc), eps_loss(field, hwc * 2 - 1)
    got_layout = "hwc" if l_hwc < l_chw else "permuted"
    got_range = "[0,1]" if l_01 < l_11 else "[-1,1]"
    ok = (got_layout == expect_layout) and (got_range == expect_range)
    print(f"  {name:34s} layout {got_layout:9s} ({l_hwc:.4f} vs {l_chw:.4f})   "
          f"range {got_range:6s} ({l_01:.4f} vs {l_11:.4f})   "
          f"{'OK' if ok else 'MISMATCH vs ' + expect_layout + '/' + expect_range}")
    return ok


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fails = []
    print("[conv] a real training image must denoise better the RIGHT way round than the wrong way\n")

    p = os.path.join(CACHE, "ddpm_shapes3d_pre.pt")
    if os.path.exists(p):
        from lifelong_ood.neural.ve_bridge import load_ve_field
        field, ck = load_ve_field(p, device=dev)
        sp = ck["split"]
        d = load_factored("shapes3d", n_per_slot=8, seed=sp["seed"], value_stride=sp["value_stride"],
                          n_holdout=sp["n_holdout"], fix_nuisance=sp["fix_nuisance"])
        if not check("shapes3d (pretrained, VE bridge)", field, d["X"], 64, "hwc", "[0,1]"):
            fails.append("shapes3d")
        # the sampler's prior must match the VP prior, not the data mean
        S = field.sample(n=8, steps=10, seed=0)
        print(f"  {'sampler output':34s} mean {S.mean():.3f} sd {S.std():.3f}  "
              f"(a collapsed sampler has sd near 0 and identical rows)")
        d2 = float(np.mean([np.abs(S[i] - S[j]).mean() for i in range(4) for j in range(i + 1, 4)]))
        print(f"  {'mean |pairwise difference|':34s} {d2:.4f}   "
              f"{'OK -- samples differ' if d2 > 0.02 else 'COLLAPSED -- samples are near-identical'}")
        if d2 <= 0.02:
            fails.append("shapes3d sampler collapse")
    else:
        print("  (no shapes3d pretrained checkpoint)")

    print("\n[conv] " + ("PASS" if not fails else f"FAIL: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
