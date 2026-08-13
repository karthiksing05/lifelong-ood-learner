"""What does the Shapes3D backbone actually GENERATE?  Look before theorising.

N52 reports every arm at chance (floor_hue 0.099 against a 0.100 chance rate, shape 0.250 against
0.250) and -- the tell -- **precision 0.000 on the UNCONDITIONAL arm too**.  Misconditioning cannot
explain that: an unconditional sampler that produced plausible Shapes3D scenes would still land near
real images.  Precision zero everywhere says the samples are not Shapes3D at all, which points at the
pipeline (layout, normalisation, sampler) rather than at conditioning.

Writes a grid of real images beside generated ones, at both layout interpretations, so the failure is
visible rather than inferred.  RESULTS 30d was exactly this: "the bank renders nothing" turned out to
be a renderer reading HWC rows as CHW, and it cost days before anyone looked at a picture.

    python tests/diag_shapes3d_render.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.evals import save_grid                               # noqa: E402
from lifelong_ood.factored import load_factored                        # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data               # noqa: E402

FIG = os.path.join(HERE, "figures")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(HERE, "results", "cache", "ddpm_shapes3d.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(f"[render] checkpoint layout={ck.get('layout')} img={ck.get('img')} "
          f"n_classes={ck.get('n_classes')}", flush=True)

    data = load_factored("shapes3d", n_per_slot=8, seed=0)
    X = np.asarray(data["X"], np.float32)
    img = int(data["img"])
    print(f"[render] real X: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]", flush=True)

    field = load_field(ckpt, device=dev)
    remap = ck["remap"]
    slots = np.array(sorted(remap))[:8]
    S = to_data(field.sample(n=8, steps=50, seed=0, clip=(-1, 1),
                             branch=remap[int(slots[0])]))
    S = np.asarray(S, np.float32)
    print(f"[render] generated: {S.shape}, range [{S.min():.3f}, {S.max():.3f}]", flush=True)

    rows = []
    rows.append(("real (as stored, HWC)", X[:8].reshape(8, img, img, 3)))
    rows.append(("generated, read HWC", S[:8].reshape(8, img, img, 3)))
    rows.append(("generated, read CHW", S[:8].reshape(8, 3, img, img).transpose(0, 2, 3, 1)))
    for name, A in rows:
        print(f"[render] {name:26s} mean {A.mean():.3f} sd {A.std():.3f} "
              f"per-channel mean {np.round(A.reshape(-1, 3).mean(0), 3)}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(rows), 8, figsize=(12, 1.7 * len(rows)))
        for r, (name, A) in enumerate(rows):
            for k in range(8):
                axes[r][k].imshow(np.clip(A[k], 0, 1))
                axes[r][k].axis("off")
            axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        fig.suptitle("Shapes3D backbone: real vs generated, both layout readings.\n"
                     "N52 scores every arm at chance with precision 0.000 including the "
                     "unconditional one — this shows whether the samples are images at all.",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out = os.path.join(FIG, "diag_shapes3d_render.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"[render] wrote {out}")
    except Exception as e:                                             # noqa: BLE001
        print(f"[render] plotting failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
