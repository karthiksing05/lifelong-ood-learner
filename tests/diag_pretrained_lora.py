"""Find a LoRA setting that ADDS an attribute to a pretrained DDPM without degrading it.

N25 v1/v2 both produced the same failure: the concept is partly learned (sunglasses do appear) while
the image quality collapses into painterly, patchy faces.  The unconditional row from the same model
is clean, so the degradation is entirely ours.

Three candidate causes, and this sweep separates them:

  `lr`     our PathLoRA has no alpha/r scaling, so the update magnitude is whatever the optimizer
           makes it.  A rate tuned for a 6.8M UNet trained from scratch may simply be too large for
           a converged 113.7M one.
  `t_min`  we train at ALL noise levels, including the low ones where fine texture is decided.  But
           our images are CelebA (178px aligned, upsampled to 256) and the model is CelebA-HQ --
           different crop, different sharpness.  Training at low t therefore teaches it OUR blur,
           which is exactly what "painterly and patchy" looks like.  An attribute like eyeglasses is
           a semantic feature settled at HIGH noise, so it should be learnable from high t alone.
  `rank`   capacity; more rank is more room to overfit 128 images.

Renders `Eyeglasses only` for each setting beside the untouched model, so the trade is visible
rather than inferred.

    python tests/diag_pretrained_lora.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from n25_pretrained_case import load_hires, show, to_data, to_model                # noqa: E402
from lifelong_ood.evals import SoftmaxProbe, vendi_score                           # noqa: E402
from lifelong_ood.neural.adapters import HierarchicalAdapterBank                    # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained                          # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def train(field, bank, name, X01, img, epochs, batch, lr, t_min, seed=0):
    emb_p = bank._emb_extra[name]
    with torch.no_grad():
        emb_p.zero_()
    params = [q for q in bank.trainable(name) if q is not emb_p]
    opt = torch.optim.Adam(params, lr=lr)
    X = torch.from_numpy(to_model(X01)).to(field.device).reshape(-1, 3, img, img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = len(field._abar)
    ab = torch.from_numpy(field._abar).float().to(field.device)
    field.model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i0 in range(0, len(X), batch):
            x0 = X[perm[i0:i0 + batch].to(field.device)]
            ts = torch.randint(int(t_min), T, (len(x0),), generator=g).to(field.device)
            a = ab[ts].view(-1, 1, 1, 1)
            e = torch.randn(x0.shape, generator=g).to(field.device)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            with bank.activate(name):
                pred = field.model.forward_t(xt, ts, emb_override=None)
            loss = torch.nn.functional.mse_loss(pred, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    field.model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    p.add_argument("--attrs", nargs=2, default=["Eyeglasses", "Smiling"])
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--probe-img", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 2e-4])
    p.add_argument("--t-mins", type=int, nargs="+", default=[0, 300, 600])
    p.add_argument("--ranks", type=int, nargs="+", default=[4])
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, img = load_pretrained(args.model, device=dev, batch=args.batch)
    A, B = args.attrs
    data = load_hires(args.attrs, img=img, n_each=args.n_train, seed=args.seed,
                      probe_img=args.probe_img)
    PX, PY = data["probe_X"], data["probe_Y"]
    cut = int(0.9 * len(PX))
    probe = SoftmaxProbe(2, seed=args.seed, epochs=30).fit(PX[:cut], PY[:cut, 0].astype(int))
    yv = PY[cut:, 0].astype(int); pv = probe.predict(PX[cut:])
    print(f"[diag] probe {A}: TPR {float((pv[yv==1]==1).mean()):.2f}\n", flush=True)

    P = args.probe_img

    def to_probe(X01):
        n = len(X01)
        Z = np.asarray(X01, np.float32).reshape(n, 3, img, img)
        f = max(img // P, 1)
        return Z.reshape(n, 3, P, f, P, f).mean((3, 5)).reshape(n, -1)

    base = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1)))
    panels = [(f"pretrained (untouched)\n{A} {float((probe.predict(to_probe(base))==1).mean()):.2f}",
               base)]
    print(f"    {'lr':>8s}{'t_min':>7s}{'rank':>6s}{A[:10]:>12s}{'vendi':>8s}{'sec':>7s}")
    for lr in args.lrs:
        for tm in args.t_mins:
            for rk in args.ranks:
                t0 = time.time()
                f2, _ = load_pretrained(args.model, device=dev, batch=args.batch)
                bank = HierarchicalAdapterBank(f2.model, rank=rk, device=dev, flat=True,
                                               targets=("linear",))
                f2.bank = bank
                bank.add_concept(A, to_model(data["a_only"]).mean(0))
                train(f2, bank, A, data["a_only"], img, args.epochs, args.batch, lr, tm,
                      seed=args.seed)
                S = to_data(f2.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                      clip=(-1, 1), branch=A))
                hit = float((probe.predict(to_probe(S)) == 1).mean())
                vd = vendi_score(S)
                print(f"    {lr:8.0e}{tm:7d}{rk:6d}{hit:12.2f}{vd:8.2f}{time.time()-t0:7.0f}",
                      flush=True)
                panels.append((f"lr {lr:.0e}, t>={tm}, r{rk}\n{A} {hit:.2f}", S))
                del f2, bank
                torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = len(panels), args.n_samples
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol + 2.2, 1.75 * nrow))
    axes = np.atleast_2d(axes)
    for r, (lab, S) in enumerate(panels):
        for c in range(ncol):
            ax = axes[r][c]; ax.axis("off")
            if c < len(S):
                ax.imshow(show(S[c], img))
        axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for sp in axes[r][0].spines.values():
            sp.set_visible(False)
    fig.suptitle(f"adapting {args.model} for `{A}`: does it add the attribute without melting?",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "diag_pretrained_lora.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
