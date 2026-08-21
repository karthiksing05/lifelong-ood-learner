"""Does classifier-free guidance make the CelebA conditioning visible?

Every sample drawn from this backbone so far used guidance 1.0 -- the raw conditional -- and two
different TRAINED classes came out pixel-identical (diag_celeba_n79_samples.png).  The class-specific
signal is present in the table (100% of its norm after centring, pairwise |cos| 0.196) and the
network is not using it.

n79 trained with `p_drop=0.1`, i.e. it was trained FOR guidance, and none was applied.  CFG is the
standard amplifier for exactly this:

    eps = eps_null + w * (eps_class - eps_null)

If two classes separate at w > 1, the conditioning is real and merely quiet, and every earlier
"joint 0.000" was measured at the one guidance setting that shows the least.  If they stay identical
at w = 7, the pathway carries no class information and no sampling choice will recover it.

    python tests/diag_celeba_cfg.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.neural.pretrained import load_pretrained            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def sample_cfg(field, emb_c, emb_n, n=3, steps=50, seed=0, w=1.0):
    """DDIM on the model's native grid with classifier-free guidance."""
    sch = field.scheduler
    sch.set_timesteps(steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, field.img, field.img, generator=g).to(field.device)
    ec = torch.as_tensor(np.asarray(emb_c, np.float32), device=field.device).reshape(1, -1)
    en = torch.as_tensor(np.asarray(emb_n, np.float32), device=field.device).reshape(1, -1)
    for t in sch.timesteps:
        tt = t.to(field.device)
        e_c = field.model.forward_t(x, tt, emb_override=ec.expand(len(x), -1))
        e_n = field.model.forward_t(x, tt, emb_override=en.expand(len(x), -1))
        eps = e_n + w * (e_c - e_n)
        x = sch.step(eps, t, x).prev_sample
    return x.permute(0, 2, 3, 1).reshape(len(x), -1).cpu().numpy()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(HERE, "results/cache/ddpm_celeba_factored.pt"),
                    map_location="cpu", weights_only=False)
    field, _ = load_pretrained(ck["model"], device=dev, batch=4)
    field.unet.load_state_dict(ck["ema"], strict=True)
    field.unet.eval()
    C = np.asarray(ck["ema_cemb"])
    null = C[int(ck["null_id"])]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = [1.0, 3.0, 5.0, 7.0]
    fig, axes = plt.subplots(2 * len(ws), 3, figsize=(7.5, 2.5 * 2 * len(ws)))
    r = 0
    for w in ws:
        for ci in (0, 5):
            S = sample_cfg(field, C[ci], null, n=3, steps=50, seed=0, w=w)
            d = float(np.abs(S - sample_cfg(field, C[5 if ci == 0 else 0], null,
                                            n=3, steps=50, seed=0, w=w)).mean())
            print(f"[cfg] w={w:.0f} class {ci}: range [{S.min():.2f},{S.max():.2f}] "
                  f"| mean |class0 - class5| = {d:.4f}", flush=True)
            D = np.clip((S + 1) / 2, 0, 1)
            for k in range(3):
                axes[r][k].imshow(D[k].reshape(256, 256, 3)); axes[r][k].axis("off")
            axes[r][0].set_ylabel(f"w={w:g}, class {ci}", rotation=0, ha="right", va="center",
                                  fontsize=8)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            r += 1
    fig.suptitle("CelebA conditioning under classifier-free guidance: do two TRAINED classes "
                 "separate?", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    o = os.path.join(HERE, "figures/initial/extra/diag_celeba_cfg.png")
    fig.savefig(o, dpi=100, bbox_inches="tight")
    print(f"[cfg] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
