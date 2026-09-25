"""N79 — CelebA the way Shapes3D was done: a PRETRAINED backbone with a table big enough to matter.

7ee retired the CelebA grid, and the retirement was right about the grid and wrong about the
dataset.  The grid is 12 cells over 5 free vectors (ratio 1.8) trained FROM SCRATCH with a density
holdout.  7jj works because of four things it has and the grid does not: a pretrained backbone, a
table at ratio 18.2, the class table on its own 50x learning rate, and a CONDITIONING holdout.

Three diagnostics chose this design before anything was trained, which is the 7ee lesson applied in
advance:

  `diag_celeba_design.py`          the arithmetic: CelebA CAN reach 153 classes / 11 free = 13.9,
                                   but it gets there through `Attractive` and `Pointy_Nose`.
  `diag_celeba_attr_readability.py` the readability, and a correction to its own first answer -- the
                                   +0.10-over-base bar admits 7 of 40 and is ARITHMETICALLY
                                   IMPOSSIBLE for the 16 whose base rate exceeds 0.90.  By AUROC,
                                   36 of 40 are readable.
  `diag_celeba_table_design.py`    both at once, on E[joint] = product of per-factor balanced
                                   accuracies.  hair(4) x Male x Smiling x Bangs: 30 classes /
                                   7 free = ratio 4.3 at E[joint] 0.341.

`hair` is CATEGORICAL, which is the move that buys CelebA Shapes3D's SHAPE: same free-vector cost as
three binaries, but it multiplies one accuracy into the joint instead of three.

**What this cannot match, stated up front.** Shapes3D is ratio 18.2 at E[joint] ~0.86; this is 4.3
at 0.341.  CelebA is worse on both axes at once and no design fixes that -- its factors are binary,
so classes only come from more factors, and each one multiplies the joint down again.  Ratio 4.3 is
just past the bar where in-sample and cross-validated R^2 begin to agree, so the basis claim here is
supportable but not comfortable, and the per-cell margins will need POOLING across cells to clear
their own resolution (7nn).

    sbatch --export=ALL,SCRIPT="experiments/n79_celeba_pretrained.py --hours 2" slurm/jobs/gpu.sbatch
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
from common import CACHE                                              # noqa: E402

from lifelong_ood.data.celeba import load_celeba_factored             # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained            # noqa: E402
from lifelong_ood.score import cosine_abar                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    p.add_argument("--img", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=30000)
    p.add_argument("--n-per-class", type=int, default=400)
    p.add_argument("--min-count", type=int, default=200)
    p.add_argument("--n-holdout", type=int, default=6)
    p.add_argument("--binaries", default="Male:Smiling:Bangs:Wearing_Lipstick",
                   help="COLON-separated (sbatch splits --export on commas). The basis is fit on "
                        "SEEN classes, and the holdout comes out of the same budget: three binaries "
                        "give 30 cells -> 24 seen / 7 free = 3.4, below the bar 7ee blamed. Four "
                        "give 44 -> 38 seen / 8 free = 4.8, at E[joint] 0.341 -> 0.299.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--emb-lr-mult", type=float, default=50.0,
                   help="5t-bis: a strong pretrained backbone reaches low loss in minutes and "
                        "starves the conditioning pathway. 50x took Shapes3D ADDRESS 0.094 -> 0.583.")
    p.add_argument("--freeze-backbone", type=int, default=1,
                   help="TRAIN ONLY THE CLASS TABLE.  The first run put all 113.7M backbone "
                        "parameters in the optimiser at 1e-5 -- the comment said 'frozen', the code "
                        "did not -- and over 18,865 steps the backbone absorbed the conditioning "
                        "while the table COLLAPSED onto a common vector: 98%% of each embedding's "
                        "norm shared, so null / class-0 / class-5 samples came out pixel-identical. "
                        "It also degraded the pretrained model (green cast, soft). Frozen, the "
                        "weights stay bit-identical and the table has to carry the conditioning.")
    p.add_argument("--wd", type=float, default=0.01,
                   help="ON THE CLASS TABLE, and it is load-bearing rather than hygiene.  Adding the "
                        "same vector to every class embedding shifts the time embedding uniformly "
                        "and carries NO class information -- it is a gauge freedom -- but nothing "
                        "penalises it, so at 50x LR it grows until the class-specific part is 2% of "
                        "the norm and samples come out pixel-identical across classes.  Measured "
                        "with wd=0: shared 0.17 -> 0.86 by step 2200, frozen backbone or not.  "
                        "Decay suppresses it preferentially because the common mode buys almost no "
                        "loss reduction while the class-specific part is the only thing that does.")
    p.add_argument("--centre-table", type=int, default=1,
                   help="Project the common mode out of the class table after every step.\n"
                        "The mean of the class embeddings is a GAUGE FREEDOM: adding the same "
                        "vector to every class shifts the time embedding uniformly and carries no "
                        "class information, so nothing in the loss opposes it and at 50x LR it grew "
                        "until the class-specific part was 2% of the norm and every class sampled "
                        "IDENTICALLY. Weight decay is the wrong tool -- AdamW's decoupled decay "
                        "shrinks by lr*wd = 5e-6 per step, measured too small to matter (shared "
                        "0.30 by step 600 either way). Centring removes it exactly, every step, and "
                        "costs nothing: `factorize` subtracts this same mean before fitting, so the "
                        "quantity being deleted is one composition never uses.")
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(CACHE, "ddpm_celeba_factored.pt"))
    p.add_argument("--smoke", action="store_true", help="tiny run: shapes and one loss step only")
    args = p.parse_args()
    if args.smoke:
        args.max_rows, args.n_per_class, args.min_count = 600, 20, 10
        args.n_holdout, args.hours, args.batch = 2, 0.02, 2
        args.out = args.out.replace(".pt", "_smoke.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    data = load_celeba_factored(img=args.img, seed=args.seed, min_count=args.min_count,
                                n_holdout=args.n_holdout, max_rows=args.max_rows,
                                n_per_class=args.n_per_class, as_uint8=True,
                                binaries=args.binaries.split(":"))
    X, slot, seen = data["X"], np.asarray(data["slot"]), np.asarray(data["seen"])
    live = [int(s) for s in data["live"]]
    seen_slots = sorted(set(int(s) for s in slot[seen]))
    remap = {int(s): i for i, s in enumerate(seen_slots)}
    n_classes = len(remap)
    print(f"[n79] loaded in {time.time() - t0:.0f}s: {len(X)} images at {args.img}px, "
          f"{len(live)} live classes, {n_classes} seen / {len(data['held'])} held out; "
          f"factors {data['names']} sizes {data['sizes']}", flush=True)
    n_free = 1 + sum(int(sz) - 1 for sz in data["sizes"])
    print(f"[n79] BASIS ratio {len(remap)} seen / {n_free} free = "
          f"{len(remap) / n_free:.1f} (Shapes3D flagship 14.4; the retired grid 1.8)", flush=True)
    if n_classes < 4:
        print("[n79] too few seen classes to train a table -- widen --max-rows or lower --min-count")
        return 2

    # ── the backbone, frozen except through its conditioning path ───────────
    field, img = load_pretrained(args.model, device=dev, batch=4)
    shim = field.model                       # CondUNetShim: forward(x, u, emb_override=...)
    unet = field.unet
    d_emb = int(shim.emb_dim)
    cemb = torch.nn.Embedding(n_classes + 1, d_emb).to(dev)
    torch.nn.init.normal_(cemb.weight, std=0.02)
    null_id = n_classes
    with torch.no_grad():
        cemb.weight[null_id].zero_()
    print(f"[n79] {args.model}: {sum(q.numel() for q in unet.parameters()) / 1e6:.1f}M params, "
          f"time-embed dim {d_emb}; class table {n_classes + 1} x {d_emb}", flush=True)

    emb_params = list(cemb.parameters())
    if args.freeze_backbone:
        for q in unet.parameters():
            q.requires_grad_(False)
        groups = [{"params": emb_params, "lr": args.lr * args.emb_lr_mult}]
        print(f"[n79] backbone FROZEN ({sum(q.numel() for q in unet.parameters()) / 1e6:.1f}M "
              f"params, 0 trainable); class table lr {args.lr * args.emb_lr_mult:.2e} on "
              f"{sum(q.numel() for q in emb_params)} parameters", flush=True)
    else:
        groups = [{"params": [q for q in unet.parameters() if q.requires_grad], "lr": args.lr},
                  {"params": emb_params, "lr": args.lr * args.emb_lr_mult}]
        print(f"[n79] backbone TRAINING at {args.lr:.2e}; class table "
              f"{args.emb_lr_mult:g}x that -- see --freeze-backbone for why this collapsed",
              flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)

    ema = {k: v.detach().clone().float() for k, v in unet.state_dict().items()}
    fkeys = [k for k, v in unet.state_dict().items() if v.dtype.is_floating_point]
    ema_c = cemb.weight.detach().clone().float()

    tr = np.where(seen)[0]
    y = np.array([remap[int(s)] for s in slot[tr]])
    Xt = torch.from_numpy(X[tr])             # uint8 [N, img*img*3]
    Yt = torch.from_numpy(y)
    g = torch.Generator().manual_seed(args.seed)
    step, tot, nb, t1 = 0, 0.0, 0, time.time()
    budget = args.hours * 3600.0

    def _save(path):
        torch.save(dict(ema=ema, ema_cemb=ema_c.cpu(), n_classes=n_classes, null_id=null_id,
                        remap=remap, img=args.img, model=args.model, layout="hwc",
                        schedule="cosine", T=1000, steps=step, dataset="celeba_factored",
                        sizes=[int(v) for v in data["sizes"]], names=list(data["names"]),
                        held=[int(h) for h in data["held"]],
                        split=dict(min_count=args.min_count, n_per_class=args.n_per_class,
                                   n_holdout=args.n_holdout, max_rows=args.max_rows,
                                   img=args.img, seed=args.seed)), path)

    while time.time() - t1 < budget:
        idx = torch.randint(0, len(Xt), (args.batch,), generator=g)
        x0 = (Xt[idx].to(dev).float() / 255.0).reshape(-1, args.img, args.img, 3)
        x0 = x0.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0      # this backbone is [-1, 1]
        c = Yt[idx].to(dev).clone()
        c[torch.rand(len(c), device=dev) < args.p_drop] = null_id
        u = torch.rand(len(x0), device=dev)
        a = torch.from_numpy(cosine_abar(u.cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
        e = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * e
        pred = shim(xt, u, emb_override=cemb(c))
        loss = torch.nn.functional.mse_loss(pred, e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + emb_params, 1.0)
        opt.step()
        if args.centre_table:
            with torch.no_grad():
                W = cemb.weight[:n_classes]
                W -= W.mean(0, keepdim=True)      # null row left alone: it is the reference
        with torch.no_grad():
            if not args.freeze_backbone:
                sd = unet.state_dict()
                for k in fkeys:
                    ema[k].mul_(0.999).add_(sd[k].float(), alpha=0.001)
            ema_c.mul_(0.999).add_(cemb.weight.detach().float(), alpha=0.001)
        tot += float(loss); nb += 1; step += 1
        if step % 200 == 0:
            # PAIRWISE COSINE of the class table is the early warning `shapes3d.py` records: a
            # 400-class table that ignored its labels sat at 0.007, and the eps-loss looked fine
            # throughout (5t-ter: the loss gap is ~0.0001 at every rung, including the dead one).
            #
            # CENTRED, though.  The raw number read 0.80 here and looks like collapse; it is not.
            # 90% of each embedding's norm is a COMMON component the table learns first, and
            # `factorize` subtracts the mean before fitting anything -- so the raw cosine measures
            # the one direction composition never uses.  Centred it reads 0.115, i.e. the classes
            # are well separated in the space that matters.  Both are printed so the difference
            # stays visible rather than becoming folklore.
            with torch.no_grad():
                T = ema_c[:n_classes]
                W = torch.nn.functional.normalize(T, dim=1)
                cos = float((W @ W.T).fill_diagonal_(0).abs().mean())
                Wc = torch.nn.functional.normalize(T - T.mean(0, keepdim=True), dim=1)
                cosc = float((Wc @ Wc.T).fill_diagonal_(0).abs().mean())
                shared = float(T.mean(0).norm() / T.norm(dim=1).mean())
            # `shared` IS THE FAILURE SIGNAL, and the centred cosine hid it: direction without
            # magnitude is nothing.  At shared 0.98 every class embedding is the same vector plus
            # 2% noise, the samples are pixel-identical across classes, and the centred cosine
            # still reads a healthy 0.185 because it normalises exactly the magnitude that
            # vanished.  Anything above ~0.9 means the table has collapsed.
            flag = "  <- TABLE COLLAPSING (class-specific part vanishing)" if shared > 0.90 else ""
            print(f"[n79] step {step}: loss {tot / max(nb, 1):.4f} | table |cos| raw {cos:.3f} "
                  f"centred {cosc:.3f} | shared {shared:.2f} class-specific {1 - shared:.2f}"
                  f"{flag} | {(time.time() - t1) / 60:.1f} min", flush=True)
            tot, nb = 0.0, 0
            _save(args.out)
        if args.smoke and step >= 2:
            break
    _save(args.out)
    print(f"[n79] wrote {args.out} after {step} steps, {(time.time() - t1) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
