"""Does a CNN probe read CelebA attributes a linear probe cannot -- and how big a grid does that buy?

The grid this benchmark can support is bounded by the attributes an INSTRUMENT can read, not by the
attributes CelebA labels.  On the linear probe only four pass a 0.10 margin (Male, Smiling,
Brown_Hair, Blond_Hair), and two of those are mutually exclusive hair colours, so the largest
complete grid is Male x Smiling x Brown_Hair = 8 cells.  ColorMNIST has 160.

A linear probe on a low-pass of 64px pixels is a weak reader of face attributes.  The multi-label
CNN written for the multi-object detector reached 0.946 exact-set where the linear probe was at
chance, so the question is whether the same architecture unlocks attributes here.  Every extra
readable attribute DOUBLES the grid, so this is worth answering before a backbone is trained
against it -- training first and discovering the benchmark cannot measure success is the Shapes3D
`object_hue` mistake.

Reports held-out accuracy and margin over base rate for both readers, on the same split.

    python tests/diag_celeba_cnn_probe.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from lifelong_ood.data.celeba import ATTRS, load_celeba                # noqa: E402

CANDS = ["Male", "Smiling", "Wavy_Hair", "Black_Hair", "Brown_Hair", "Blond_Hair",
         "Oval_Face", "Big_Lips", "No_Beard", "Big_Nose"]
IMG = 64


class AttrCNN(torch.nn.Module):
    """The multi-object detector's trunk, retargeted at CelebA attributes."""

    def __init__(self, n_out, base=32):
        super().__init__()

        def blk(i, o):
            return [torch.nn.Conv2d(i, o, 3, 1, 1), torch.nn.BatchNorm2d(o), torch.nn.ReLU(),
                    torch.nn.Conv2d(o, o, 3, 2, 1), torch.nn.BatchNorm2d(o), torch.nn.ReLU()]
        self.trunk = torch.nn.Sequential(*blk(3, base), *blk(base, base * 2),
                                         *blk(base * 2, base * 4), *blk(base * 4, base * 4))
        self.head = torch.nn.Linear(base * 8, n_out)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_celeba(n_per_class=400, img=IMG, seed=0, min_count=20)
    X = np.asarray(d["X"], np.float32)
    A = np.asarray(d["attrs"])
    idx = np.arange(len(X))
    np.random.default_rng(0).shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    cols = [ATTRS.index(n) for n in CANDS]
    Y = A[:, cols].astype(np.float32)
    print(f"[cnn] {len(tr)} train / {len(te)} held-out images, {len(CANDS)} attributes", flush=True)

    def chw(z):
        return torch.as_tensor(z.reshape(-1, IMG, IMG, 3)).permute(0, 3, 1, 2)

    torch.manual_seed(0)
    net = AttrCNN(len(CANDS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    xt = chw(X[tr]).to(dev)
    yt = torch.as_tensor(Y[tr], device=dev)
    steps = max(40, len(tr) // 256)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60 * steps)
    rng = np.random.default_rng(0)
    for ep in range(60):
        for _ in range(steps):
            b = torch.as_tensor(rng.integers(0, len(xt), 256), device=dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        P = torch.sigmoid(net(chw(X[te]).to(dev))).cpu().numpy()

    print(f"\n{'attribute':14s}{'CNN acc':>10}{'base':>8}{'margin':>9}{'  readable?':>12}")
    good = []
    for k, n in enumerate(CANDS):
        y = Y[te][:, k]
        acc = float(((P[:, k] > 0.5) == (y > 0.5)).mean())
        base = float(max(y.mean(), 1 - y.mean()))
        ok = acc - base >= 0.10
        if ok:
            good.append(n)
        print(f"{n:14s}{acc:>10.3f}{base:>8.3f}{acc - base:>+9.3f}"
              f"{('  YES' if ok else '  no'):>12}")
    print(f"\n[cnn] readable at margin >= 0.10: {good}")
    print(f"[cnn] a complete grid over the readable set bounds the benchmark: "
          f"{2 ** len([g for g in good if 'Hair' not in g])} cells from the non-hair attributes "
          f"alone, times the hair factor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
