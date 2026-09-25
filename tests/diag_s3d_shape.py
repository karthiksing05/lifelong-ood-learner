"""Why do generated Shapes3D samples of TRAINED classes read shape at ~0.42?

c1/c2's Shapes3D smoke measured, at theta_0, joint 0.38 with wall 0.85 / object 1.00 / shape 0.42 on
the backbone's own trained classes -- at 50 and at 200 DDIM steps -- which caps every Shapes3D number.
The n78 flagship read shape ~1.0 on composed addresses but never scored trained classes directly.
Separate the two suspects:

  sampler   the VE bridge's own `field.sample(branch=cid)`  vs  cobweb_update.ddim_sample(cls=cid)
  probes    c_bench's probes (120/cell rows, +grey)          vs  n78's recipe: fit_factor_probes on
                                                                  the checkpoint's own split (40/cell)

and save a grid (real | field sampler | ddim_sample) to look at.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from c_bench import load_bench                                        # noqa: E402

from lifelong_ood.data.factored import load_factored                  # noqa: E402
from lifelong_ood.evals import factor_accuracy, fit_factor_probes     # noqa: E402
from lifelong_ood.neural.cobweb_update import ddim_sample             # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field               # noqa: E402


def main():
    dev = "cuda"
    B = load_bench("shapes3d", seed=0, device=dev)
    ck = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "results", "cache", "ddpm_s3d_full.pt"),
                    map_location="cpu", weights_only=False)
    field, _ = load_ve_field(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "results", "cache", "ddpm_s3d_full.pt"), device=dev)
    sp = ck["split"]
    d = load_factored("shapes3d", n_per_slot=int(sp["n_per_slot"]), seed=int(sp.get("seed", 0)),
                      value_stride=int(sp["value_stride"]), fix_nuisance=sp["fix_nuisance"],
                      n_holdout=0, rows="legacy")
    d["seen"] = np.isin(np.asarray(d["slot"]), np.array(sorted(int(k) for k in ck["remap"])))
    p78 = fit_factor_probes(d, seed=0)
    rng = np.random.default_rng(0)
    pick = [B.seen[i] for i in rng.permutation(len(B.seen))[:8]]
    res, grid = [], []
    for cid, tup in pick:
        Sf = np.asarray(field.sample(n=16, steps=200, seed=1, branch=cid), np.float32)
        Sm = B.to_d(ddim_sample(B.model, 16, steps=200, seed=1, cls=cid, clip=B.clip,
                                layout=B.layout, img=B.img, device=dev))
        R = B.images(tup)[:16]
        for name, S in (("real", R), ("field.sample", Sf), ("ddim_sample", Sm)):
            ab = B.acc(S, tup)
            a78 = factor_accuracy(p78, S, tup)
            res.append((tup, name, ab["joint"], ab["shape"], a78.get("joint"), a78.get("shape")))
        grid.append((R[:4], Sf[:4], Sm[:4]))
        print(f"[diag] class {cid} {tup}: " + " | ".join(
            f"{r[1]} bench j/shape {r[2]:.2f}/{r[3]:.2f} n78 j/shape {r[4]:.2f}/{r[5]:.2f}"
            for r in res[-3:]), flush=True)
    for name in ("real", "field.sample", "ddim_sample"):
        rr = [r for r in res if r[1] == name]
        print(f"[diag] {name:13s} bench joint {np.mean([r[2] for r in rr]):.3f} shape "
              f"{np.mean([r[3] for r in rr]):.3f} | n78-probes joint {np.mean([r[4] for r in rr]):.3f}"
              f" shape {np.mean([r[5] for r in rr]):.3f}", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(grid), 12, figsize=(12, len(grid) * 1.1))
    for i, (R, Sf, Sm) in enumerate(grid):
        for j, img in enumerate(list(R) + list(Sf) + list(Sm)):
            axes[i, j].imshow(np.clip(img.reshape(64, 64, 3), 0, 1))
            axes[i, j].axis("off")
    axes[0, 1].set_title("real", fontsize=8)
    axes[0, 5].set_title("field.sample", fontsize=8)
    axes[0, 9].set_title("ddim_sample", fontsize=8)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "c",
                       "diag_s3d_shape.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print("[diag] saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
