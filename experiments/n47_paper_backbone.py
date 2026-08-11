"""N47 -- the companion paper's CelebA backbone, built the way they built it.

Their Table 5 CelebA column is not "a diffusion model trained on CelebA".  It is:

    google/ddpm-celebahq-256, sample_size overridden to 128, FINE-TUNED,
    with a class-embedding table sized |C|+1 (52 rows -> 51 classes),
    AdamW 2e-4 / wd 1e-6 / clip 1.0, batch 128, 500k steps, bf16, EMA,
    null-token dropout 0.1, LINEAR beta with T=1000, epsilon-prediction.

We were training from scratch at 64px over 1205 conditioning rows on a cosine schedule.  Every one
of those is a deviation, and together they are why our faces never looked like theirs.  The largest
by far is the first: they START from a model that already renders photorealistic faces and only have
to teach it *conditioning*; we were asking a small network to learn faces and conditioning at once
on an eighth of the compute.

This builds their backbone.  Two things it deliberately keeps from our side, both stated:

  * `--img 128` matches them, but the run length cannot: 500k steps at batch 128 is ~4 GPU-days on
    four RTX6000s.  We fine-tune for a fixed wall-clock budget instead.  That is legitimate here in
    a way it would not be for a from-scratch run, because fine-tuning a photorealistic model to
    accept conditioning is a far smaller ask than learning the image distribution.
  * The class-embedding table feeds the time embedding additively, which is what diffusers'
    `class_embed_type` does and what the paper describes.  N15 concluded that channel was too weak
    -- but it optimized ONE vector against a FROZEN network, and it measured through the broken
    layout (RESULTS 30b).  Here the network is fine-tuned WITH the table, which is the paper's
    setting and a different question.

    python experiments/n47_paper_backbone.py --hours 10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE                                                       # noqa: E402


def linear_abar(T=1000, beta_start=1e-4, beta_end=0.02):
    """The paper's schedule: LINEAR beta, T=1000.  Ours is cosine, which is a documented deviation
    and one that changes what a given `u` means -- their timestep grids do not transfer."""
    b = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    return np.cumprod(1.0 - b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--n-per-class", type=int, default=300)
    p.add_argument("--max-rows", type=int, default=25000)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(CACHE, "ddpm_celeba_paper128.pt"))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.hours, args.img, args.batch, args.n_per_class = 0.05, 64, 4, 20
        args.out = os.path.join(CACHE, "ddpm_celeba_paper128_smoke.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    from diffusers import UNet2DModel
    from lifelong_ood.celeba import load_celeba
    from lifelong_ood.neural.pretrained import _CondTimeEmbedding

    data = load_celeba(n_per_class=args.n_per_class, img=args.img, seed=args.seed,
                       min_count=args.min_count, max_rows=args.max_rows)
    seen = np.asarray(data["seen"], bool)
    X = np.asarray(data["X"])[seen]
    cls = np.asarray(data["cls"])[seen]
    uniq = np.unique(cls)
    remap = {int(c): i for i, c in enumerate(uniq)}
    y = np.array([remap[int(c)] for c in cls], np.int64)
    n_classes = len(uniq)
    print(f"[n47] {len(X)} images, {n_classes} seen classes -> class table of {n_classes + 1} rows "
          f"(the paper's CelebA column is 52)", flush=True)

    # ── their backbone ───────────────────────────────────────────────────────
    unet = UNet2DModel.from_pretrained(args.model)
    unet.config.sample_size = args.img
    unet.sample_size = args.img
    d_emb = int(unet.time_embedding.linear_2.out_features)
    # class-embedding table sized |C|+1, the extra row reserved for the CFG null token, added to the
    # time embedding -- the paper's description, and what diffusers' class_embed_type does
    cemb = torch.nn.Embedding(n_classes + 1, d_emb).to(dev)
    torch.nn.init.normal_(cemb.weight, std=0.02)
    with torch.no_grad():
        cemb.weight[n_classes].zero_()          # null token = no class signal
    null_id = n_classes
    # Wrap the time embedding rather than reimplementing UNet2DModel.forward.  That
    # forward has a `skip_sample` path and its block signatures move between diffusers
    # versions; a wrapper that ADDS to the time embedding is the same operation the
    # native class_embed_type performs, is what the paper describes, and is already the
    # mechanism the rest of this repository conditions a pretrained UNet through.
    unet.time_embedding = _CondTimeEmbedding(unet.time_embedding)
    unet = unet.to(dev).train()

    abar = torch.from_numpy(linear_abar(1000)).float().to(dev)
    params = list(unet.parameters()) + list(cemb.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler_dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ema = {k: v.detach().clone().float() for k, v in unet.state_dict().items()}
    fkeys = [k for k, v in unet.state_dict().items() if v.dtype.is_floating_point]
    ema_f = [ema[k] for k in fkeys]
    mod_f = [unet.state_dict()[k] for k in fkeys]
    ema_c = cemb.weight.detach().clone().float()

    # TRUE NCHW.  The whole point: these weights were fitted to real images.
    Xt = torch.from_numpy(X.astype(np.float32)).reshape(-1, args.img, args.img, 3)
    Xt = Xt.permute(0, 3, 1, 2).contiguous()
    Yt = torch.from_numpy(y)
    n = len(Xt)
    g = torch.Generator().manual_seed(args.seed)
    t0, step, tot, nb = time.time(), 0, 0.0, 0
    budget = args.hours * 3600.0
    print(f"[n47] fine-tuning for {args.hours:.2f} h at {args.img}px, batch {args.batch}, "
          f"linear beta T=1000, bf16", flush=True)
    while time.time() - t0 < budget:
        idx = torch.randint(0, n, (args.batch,), generator=g)
        x0 = (Xt[idx].to(dev) * 2.0 - 1.0)
        c = Yt[idx].to(dev).clone()
        c[torch.rand(len(c), device=dev) < args.p_drop] = null_id       # CFG dropout
        t = torch.randint(0, 1000, (len(x0),), device=dev)
        a = abar[t].view(-1, 1, 1, 1)
        noise = torch.randn(x0.shape, device=dev)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        with torch.autocast(device_type="cuda", dtype=scaler_dtype, enabled=(dev == "cuda")):
            unet.time_embedding.set_extra(cemb(c))
            out = unet(xt, t).sample
            loss = torch.nn.functional.mse_loss(out.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            dk = min(args.ema_decay, (1.0 + step) / (10.0 + step))
            torch._foreach_mul_(ema_f, dk)
            torch._foreach_add_(ema_f, mod_f, alpha=1 - dk)
            ema_c.mul_(dk).add_(cemb.weight.detach().float(), alpha=1 - dk)
        tot += float(loss); nb += 1; step += 1
        if step % 200 == 0:
            el = (time.time() - t0) / 60
            print(f"    step {step:6d}  loss {tot / max(nb,1):.4f}  ({el:.1f} min / "
                  f"{args.hours*60:.0f})", flush=True)
            tot, nb = 0.0, 0
    print(f"[n47] {step} steps in {(time.time()-t0)/60:.1f} min", flush=True)
    torch.save(dict(ema=ema, ema_cemb=ema_c.cpu(), n_classes=n_classes, null_id=null_id,
                    remap=remap, img=args.img, model=args.model, layout="hwc",
                    schedule="linear", T=1000, steps=step, dataset="celeba",
                    split=dict(loader="load_celeba", n_per_class=args.n_per_class, img=args.img,
                               seed=args.seed, min_count=args.min_count, max_rows=args.max_rows)),
               args.out)
    print(f"[n47] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
