"""N57 -- Shapes3D on a PRETRAINED unconditional backbone, the CelebA/N47 recipe.

Three from-scratch attempts at a conditional Shapes3D model all failed the same way: the samples were
excellent and the label was ignored (seen-slot joint 0.000 -> 0.070 -> 0.062, and the null token
explained training images BETTER than their own class embedding).  N47's diagnosis for CelebA applies
exactly -- asking one network to learn the image distribution AND the conditioning at once is a much
harder problem than teaching conditioning to a model that already renders the domain, and on a
marginal as easy as flat-shaded rendered scenes the loss is minimised almost entirely without the
label.

`eurecom-ds/scoresdeve-ema-shapes3d-64` is the missing piece: a `UNet2DModel` at 64px trained
UNCONDITIONALLY on Shapes3D (`num_class_embeds: null`).  It already renders the domain.  So this does
to it exactly what N47 does to `google/ddpm-celebahq-256`:

  * bolt on a class-embedding table of |C|+1 rows, the extra one reserved for the CFG null token
  * add it to the TIME EMBEDDING via `_CondTimeEmbedding`, which is the same operation diffusers'
    `class_embed_type` performs and the same place our own UNet accepts `emb_override`
  * fine-tune both together, so the network learns to USE the label rather than to render

It also keeps the fix the from-scratch runs converged on: `wall_hue` is PINNED, because a composition
(floor_hue, object_hue, shape) otherwise varies internally over wall_hue x scale x orientation = 1200
combinations and wall hue repaints half the image -- within-class variance swamps the label.

    python experiments/n57_shapes3d_pretrained.py --hours 1.5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.data.factored import load_factored                        # noqa: E402
from lifelong_ood.neural.ve_bridge import VEToVPUNet                    # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="eurecom-ds/scoresdeve-ema-shapes3d-64")
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--n-per-slot", type=int, default=200)
    p.add_argument("--value-stride", type=int, default=2)
    p.add_argument("--n-holdout", type=int, default=25)
    p.add_argument("--fix-nuisance", nargs="*", default=["wall_hue"])
    p.add_argument("--hours", type=float, default=1.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--p-drop", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(CACHE, "ddpm_shapes3d_pre.pt"))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import UNet2DModel

    data = load_factored("shapes3d", n_per_slot=args.n_per_slot, seed=args.seed,
                         value_stride=args.value_stride, n_holdout=args.n_holdout,
                         fix_nuisance=args.fix_nuisance)
    seen = np.asarray(data["seen"], bool)
    slots = np.asarray(data["slot"])
    uniq = np.unique(slots[seen])
    remap = {int(s): i for i, s in enumerate(sorted(uniq))}
    n_classes = len(uniq)
    X = np.asarray(data["X"], np.float32)[seen]
    y = np.array([remap[int(s)] for s in slots[seen]], np.int64)
    print(f"[n57] shapes3d {data['names']} sizes {data['sizes']}, wall_hue pinned; "
          f"{len(X)} images, {n_classes} seen compositions, "
          f"{len(np.unique(slots)) - n_classes} held out", flush=True)

    unet = UNet2DModel.from_pretrained(args.model, subfolder="unet")
    unet.config.sample_size = args.img
    unet.sample_size = args.img
    # THE BRIDGE.  This checkpoint is Score-SDE VARIANCE-EXPLODING; our stack is variance-preserving
    # with a cosine schedule.  The first version of this file fed it `u * 1000` as a timestep and
    # read its output as epsilon -- all three of (timestep, output convention, input range) wrong at
    # once -- and eps-loss sat at exactly 1.0000 for 1200 steps, i.e. the pretrained weights
    # contributed nothing.  `tests/diag_ve_parameterisation.py` measured the right reading:
    # images in [0,1], sigma as the timestep, eps = -output * sigma (MSE 0.0002-0.0005 against a
    # known eps, against 0.64-441 for every alternative).
    model = VEToVPUNet(unet, n_classes=n_classes).to(dev).train()
    cemb = model.cemb
    null_id = model.null_id
    d_emb = model.edim
    print(f"[n57] pretrained {args.model}: "
          f"{sum(q.numel() for q in unet.parameters()) / 1e6:.1f}M params, time-embed dim {d_emb}; "
          f"class table {n_classes + 1} x {d_emb}; VE<->VP bridge active", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = {k: v.detach().clone().float() for k, v in unet.state_dict().items()}
    fkeys = [k for k, v in unet.state_dict().items() if v.dtype.is_floating_point]
    ema_c = cemb.weight.detach().clone().float()

    # [0,1], NOT [-1,1]: the probe found the model's inputs are in [0,1] (MSE 0.0002 there
    # against 0.9165 for the same reading on [-1,1]).
    Xt = torch.from_numpy(X).reshape(-1, args.img, args.img, 3).permute(0, 3, 1, 2).contiguous()
    Yt = torch.from_numpy(y)
    g = torch.Generator().manual_seed(args.seed)
    t0, step, tot, nb = time.time(), 0, 0.0, 0
    budget = args.hours * 3600.0
    while time.time() - t0 < budget:
        idx = torch.randint(0, len(Xt), (args.batch,), generator=g)
        x0 = Xt[idx].to(dev)
        c = Yt[idx].to(dev).clone()
        c[torch.rand(len(c), device=dev) < args.p_drop] = null_id
        u = torch.rand(len(x0), device=dev)
        a = torch.from_numpy(cosine_abar(u.cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
        e = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * e
        pred = model(xt, u, c=c)
        loss = torch.nn.functional.mse_loss(pred, e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            sd = unet.state_dict()
            for k in fkeys:
                ema[k].mul_(0.999).add_(sd[k].float(), alpha=0.001)
            ema_c.mul_(0.999).add_(cemb.weight.detach().float(), alpha=0.001)
        tot += float(loss); nb += 1; step += 1
        if step % 200 == 0:
            print(f"    step {step}: loss {tot / nb:.4f}  "
                  f"({(time.time() - t0) / 60:.0f}/{budget / 60:.0f} min)", flush=True)
            tot, nb = 0.0, 0
    torch.save(dict(ema=ema, ema_cemb=ema_c.cpu(), n_classes=n_classes, null_id=null_id,
                    remap=remap, img=args.img, model=args.model, layout="hwc",
                    schedule="cosine", T=1000, steps=step, dataset="shapes3d",
                    sizes=[int(v) for v in data["sizes"]], names=list(data["names"]),
                    split=dict(n_per_slot=args.n_per_slot, value_stride=args.value_stride,
                               n_holdout=args.n_holdout, fix_nuisance=args.fix_nuisance,
                               seed=args.seed)),
               args.out)
    print(f"[n57] {step} steps in {(time.time() - t0) / 60:.1f} min -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
