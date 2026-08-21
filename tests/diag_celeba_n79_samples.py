"""What does the N79-fine-tuned CelebA backbone actually produce?

N80 reported joint 0.000 for EVERY arm -- address, nearest-class and unconditional alike -- against
a real-image ceiling of 0.240.  A trained class embedding that cannot reproduce its own combination
is not a composition result; it means the samples or the weights are wrong.  The weights were ruled
out (450/450 keys load through `load_pretrained`), so this looks at the pictures, which is the step
that has settled every previous instance of this in the repository.

No data loading, no probes: checkpoint in, PNG out.

    python tests/diag_celeba_n79_samples.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.neural.pretrained import load_pretrained            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(HERE, "results/cache/ddpm_celeba_factored.pt"),
                    map_location="cpu", weights_only=False)
    field, _ = load_pretrained(ck["model"], device=dev, batch=4)
    C = np.asarray(ck["ema_cemb"])
    print(f"[diag] ckpt {ck['steps']} steps, {ck['n_classes']} classes, null {ck['null_id']}",
          flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [("PRETRAINED, no fine-tune, unconditional", None, None),
            ("N79 weights, unconditional (null emb)", "ema", C[int(ck["null_id"])]),
            ("N79 weights, a TRAINED class", "ema", C[0]),
            ("N79 weights, another TRAINED class", "ema", C[5])]
    fig, axes = plt.subplots(len(arms), 4, figsize=(10, 2.6 * len(arms)))
    for r, (label, which, emb) in enumerate(arms):
        if which == "ema":
            field.unet.load_state_dict(ck["ema"], strict=True)
        field.unet.eval()
        S = np.asarray(field.sample(n=4, steps=50, seed=0, emb=emb), np.float32)
        print(f"[diag] {label:42s} range [{S.min():.2f}, {S.max():.2f}] mean {S.mean():.3f}",
              flush=True)
        D = np.clip((S + 1.0) / 2.0, 0, 1)
        for k in range(4):
            axes[r][k].imshow(D[k].reshape(256, 256, 3)); axes[r][k].axis("off")
        axes[r][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
    fig.suptitle("N79 CelebA backbone: did fine-tuning damage it, or is conditioning inert?",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    o = os.path.join(HERE, "figures/initial/extra/diag_celeba_n79_samples.png")
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"[diag] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
