"""Why does the adapter bank turn a photorealistic backbone into noise?  (RESULTS 30d)

n26 consolidates concepts into `google/ddpm-celebahq-256` and its samples are rainbow noise.  Two
facts narrow this a long way before any new measurement:

  * **the pool is REAL IMAGES.**  n26 trains on `data[a]`, real CelebA photographs with the
    attribute.  So "the adapter faithfully learned a bad pool" is ruled out -- the pool is as good
    as a pool gets.
  * **the adapter reaches a training loss of 0.0002**, so it is fitting those real images almost
    exactly, while the SAMPLES are noise.  Fitting the denoising objective on real data and then
    sampling noise is the signature of a delta that is fine at the noise levels it was trained on
    and destructive elsewhere -- n26 trains only at high noise (`t_min`), which is deliberate
    (RESULTS 14) but leaves the low-noise end of the trajectory unconstrained.

So the hypothesis to test is not "is the adapter learning" but **how much perturbation the frozen
model tolerates**, and where.  This sweeps the knobs one at a time on ONE concept and reports what
the samples actually look like:

    rank      how many directions the delta may use
    lr        how far it moves
    epochs    how long
    t_min     whether the low-noise end of the trajectory is trained at all

`tv_ratio` against real faces is the readout: ~1.0 is real-image texture, and the noise we are
chasing sits far above it.  The `pretrained, no adapter` row is the control -- that number is what
the backbone does untouched, and any setting that cannot get near it is destroying the model.

    python tests/diag_pretrained_adapter.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    from lifelong_ood.celeba import load_celeba
    from lifelong_ood.evals import speckle
    from lifelong_ood.neural import to_data, to_model
    from lifelong_ood.neural.adapters import HierarchicalAdapterBank
    from lifelong_ood.neural.pretrained import load_pretrained
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, img = load_pretrained("google/ddpm-celebahq-256", device=dev, batch=4)
    data = load_celeba(n_per_class=8, img=img, seed=0, min_count=25, max_rows=6000)
    X = np.asarray(data["X"])
    pool = X[:32]
    R = X[:128].astype(np.float32)

    base = to_data(field.sample(n=6, steps=50, seed=0, clip=(-1, 1))).astype(np.float32)
    b = speckle(base, R, img=img)
    print(f"[adapter] CONTROL  pretrained, no adapter        tv {b.get('tv_ratio', float('nan')):.3f}"
          f"   <- anything far from this is destroying the model\n", flush=True)

    rows, panels = [], {"no adapter": base}
    settings = [("rank 8, lr 1e-3, 40ep  (n26)", dict(rank=8, lr=1e-3, epochs=40)),
                ("rank 8, lr 1e-4, 40ep",        dict(rank=8, lr=1e-4, epochs=40)),
                ("rank 8, lr 1e-5, 40ep",        dict(rank=8, lr=1e-5, epochs=40)),
                ("rank 1, lr 1e-3, 40ep",        dict(rank=1, lr=1e-3, epochs=40)),
                ("rank 8, lr 1e-3,  5ep",        dict(rank=8, lr=1e-3, epochs=5))]
    for label, kw in settings:
        t0 = time.time()
        bank = HierarchicalAdapterBank(field.model, rank=kw["rank"], device=dev, flat=True,
                                       targets=("linear",),
                                       exclude=("time_emb_proj", "time_embedding"))
        field.bank = bank
        bank.add_concept("c", to_model(pool).mean(0))
        # the same objective n26 uses, inlined so the knobs are visible
        opt = torch.optim.Adam([p for p in bank.parameters() if p.requires_grad], lr=kw["lr"])
        Xp = torch.from_numpy(to_model(pool).astype(np.float32)).to(dev)
        Xp = Xp.reshape(-1, img, img, 3).permute(0, 3, 1, 2).contiguous()   # layout=hwc
        g = torch.Generator(device="cpu").manual_seed(0)
        abar = field._abar
        with bank.activate("c"):
            for ep in range(kw["epochs"]):
                idx = torch.randperm(len(Xp), generator=g)[:8].to(dev)
                x0 = Xp[idx]
                t = torch.randint(600, 1000, (len(x0),), device=dev)     # n26's high-noise window
                a = torch.from_numpy(abar[t.cpu().numpy()]).float().to(dev).view(-1, 1, 1, 1)
                e = torch.randn(x0.shape, device=dev)
                xt = a.sqrt() * x0 + (1 - a).sqrt() * e
                pred = field.model.forward_t(xt, int(t[0]))
                loss = torch.nn.functional.mse_loss(pred, e)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            S = to_data(field.sample(n=6, steps=50, seed=0, clip=(-1, 1),
                                     branch="c")).astype(np.float32)
        sk = speckle(S, R, img=img)
        tv = float(sk.get("tv_ratio", np.nan))
        rows.append((label, tv, float(loss)))
        panels[label] = S
        print(f"[adapter] {label:32s} tv {tv:.3f}   final loss {float(loss):.5f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
        field.bank = None
        del bank
        torch.cuda.empty_cache()

    print(f"\n[adapter] control tv {b.get('tv_ratio', float('nan')):.3f}.  A setting near it "
          f"preserves the model; one far above it is the failure RESULTS 30d records.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 6, figsize=(10, 1.8 * len(panels)))
    axes = np.atleast_2d(axes)
    for r, (name, S) in enumerate(panels.items()):
        for c in range(6):
            axes[r][c].axis("off")
            if c < len(S):
                axes[r][c].imshow(np.clip(np.asarray(S[c], np.float32).reshape(img, img, 3), 0, 1))
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s_ in axes[r][0].spines.values():
            s_.set_visible(False)
    fig.suptitle("How much adapter can a pretrained backbone take before it stops rendering?\n"
                 "Pool is REAL images; the top row is the untouched model.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    o = os.path.join(FIG, "f27_pretrained_adapter.png")
    fig.savefig(o, dpi=120, bbox_inches="tight")
    print(f"[adapter] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
