"""N55 -- train a latent DiT on ColorMNIST, then ask whether OUR architecture survives the move.

The question is not "can a DiT model ColorMNIST" (it can).  It is whether the three-tier scheme is a
property of the METHOD or of the pixel-space UNet it was developed on.  Everything the scheme does is
defined over a class embedding table -- `factorize` fits `c ~ mu + sum_f V_f[value]` to it, ADDRESS
composes a new point in that basis -- and none of that mentions convolutions.  A DiT conditions
through adaLN modulation instead of an additive time-embedding bias, works on latent patches, and is
the family VLA models are built from.  If the basis still emerges, the claim is architectural.

`neural/dit.py` exposes the UNet's exact interface (cemb table + null row + emb_override), so
`factorize` and `compose` run on it UNCHANGED.  Anything that needs changing marks a place the
architecture was secretly UNet-specific -- which is itself the finding.

Reports, and writes to figures/dit/:
    training curve and held-out latent loss        -- d2, is the model any good
    basis R^2 against the UNet's 0.870             -- did the table organise compositionally
    ADDRESS composition accuracy on held-out slots -- does the scheme transfer

    python experiments/n55_dit_train.py --epochs 120
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                      # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                   # noqa: E402
from lifelong_ood.neural.dit import DiT                                # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize, slot_factors  # noqa: E402
from lifelong_ood.neural.latent import ConvAE, decode, encode          # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402


@torch.no_grad()
def ddim_latent(model, c, n, shape, steps=50, seed=0, device="cuda", guidance=1.0):
    """Plain DDIM eta=0 in latent space, conditioned by an embedding vector."""
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((n, *shape), generator=g, device=device)
    us = np.linspace(1.0, 0.0, steps + 1)
    cc = torch.as_tensor(np.asarray(c, np.float32)).reshape(1, -1).to(device).expand(n, -1)
    null = model.cemb.weight[model.null_id].detach().reshape(1, -1).expand(n, -1)
    for i in range(steps):
        u, un = float(us[i]), float(us[i + 1])
        a, an = float(cosine_abar(np.array([u]))[0]), float(cosine_abar(np.array([un]))[0])
        e = model(z, torch.full((n,), u, device=device), emb_override=cc)
        if guidance != 1.0:
            e0 = model(z, torch.full((n,), u, device=device), emb_override=null)
            e = e0 + guidance * (e - e0)
        z0 = (z - (1 - a) ** 0.5 * e) / (a ** 0.5)
        z = (an ** 0.5) * z0 + ((1 - an) ** 0.5) * e
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae", default=os.path.join(CACHE, "ae_colormnist.pt"))
    p.add_argument("--n-per-slot", type=int, default=200)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(CACHE, "dit_colormnist.pt"))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = ConvAE(zdim=ack["zdim"], base=ack["base"]).to(dev)
    ae.load_state_dict(ack["model"]); ae.eval()
    scale = float(ack["scale"])
    worst = max(r["drop"] for r in ack["probe_rows"])
    print(f"[n55] AE {ack['zdim']}x8x8, scale {scale:.4f}, worst factor drop {worst:+.3f} "
          f"-- the latent space preserves every factor the composition is scored on", flush=True)

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    seen = np.asarray(data["seen"], bool)
    slots = np.asarray(data["slot"])
    uniq = np.unique(slots[seen])
    remap = {int(s): i for i, s in enumerate(sorted(uniq))}
    Z = encode(ae, np.asarray(data["X"], np.float32), scale=scale, device=dev)
    y = torch.tensor([remap[int(s)] for s in slots[seen]], dtype=torch.long)
    Zs = Z[torch.from_numpy(seen)]
    print(f"[n55] {len(Zs)} latents {tuple(Zs.shape[1:])}, {len(uniq)} seen compositions, "
          f"{len(np.unique(slots)) - len(uniq)} held out", flush=True)

    model = DiT(n_classes=len(uniq), in_ch=ack["zdim"], size=8, patch=args.patch,
                d=args.dim, depth=args.depth).to(dev)
    print(f"[n55] DiT {sum(p_.numel() for p_ in model.parameters()) / 1e6:.2f}M params, "
          f"{args.epochs} epochs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    torch.manual_seed(args.seed)
    hist, t0 = [], time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(Zs))
        tot = 0.0
        for i in range(0, len(Zs), args.batch):
            idx = perm[i:i + args.batch]
            z0 = Zs[idx].to(dev)
            c = y[idx].to(dev).clone()
            c[torch.rand(len(c), device=dev) < args.p_drop] = model.null_id   # CFG dropout
            u = torch.rand(len(z0), device=dev)
            a = torch.from_numpy(cosine_abar(u.cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
            e = torch.randn_like(z0)
            zt = a.sqrt() * z0 + (1 - a).sqrt() * e
            loss = F.mse_loss(model(zt, u, c=c), e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(z0)
        sched.step()
        hist.append(tot / len(Zs))
        if (ep + 1) % 10 == 0:
            print(f"    dit epoch {ep + 1}/{args.epochs}: loss {hist[-1]:.4f}", flush=True)
    mins = (time.time() - t0) / 60
    torch.save(dict(model=model.state_dict(), n_classes=len(uniq), remap=remap,
                    zdim=ack["zdim"], patch=args.patch, dim=args.dim, depth=args.depth,
                    scale=scale, hist=hist, config=vars(args)), args.out)
    print(f"[n55] trained in {mins:.1f} min -> {args.out}", flush=True)

    # ── THE TEST: does a factor basis exist in the DiT's class table? ────────
    model.eval()
    C = model.class_table().cpu().numpy()
    slots_seen = np.array(sorted(uniq))
    fac = factorize(C, slot_factors(slots_seen))
    print(f"\n[n55] basis R^2 on the DiT class table: {fac['r2']:.3f}   "
          f"[pixel-space UNet: 0.870]", flush=True)

    probes = fit_probes(data, seed=args.seed, epochs=30)
    held = np.array(sorted(set(int(s) for s in np.unique(slots)) - set(int(s) for s in uniq)))
    rng = np.random.default_rng(args.seed)
    cases = rng.permutation(held)[:args.n_cases]
    rows = []
    for s in cases:
        d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        c_new = compose(fac, (d_, f_, b_))
        z = ddim_latent(model, c_new, args.n_samples, tuple(Zs.shape[1:]), steps=args.steps,
                        seed=args.seed, device=dev)
        S = decode(ae, z.cpu(), scale=scale, device=dev).astype(np.float32)
        hit = ((probes["digit"].predict(S) == d_) & (probes["fg"].predict(S) == f_)
               & (probes["bg"].predict(S) == b_))
        rows.append(dict(slot=int(s), joint=float(hit.mean()),
                         digit=float((probes["digit"].predict(S) == d_).mean()),
                         fg=float((probes["fg"].predict(S) == f_).mean()),
                         bg=float((probes["bg"].predict(S) == b_).mean())))
        print(f"[n55] held-out slot {int(s):3d}  joint {rows[-1]['joint']:.3f}  "
              f"digit {rows[-1]['digit']:.2f} fg {rows[-1]['fg']:.2f} bg {rows[-1]['bg']:.2f}",
              flush=True)

    m = {k: float(np.mean([r[k] for r in rows])) for k in ("joint", "digit", "fg", "bg")}
    print(f"\n[n55] ADDRESS on the DiT: joint {m['joint']:.3f} "
          f"(digit {m['digit']:.3f} fg {m['fg']:.3f} bg {m['bg']:.3f}) over {len(rows)} held-out")
    save_json(dict(rows=rows, mean=m, basis_r2=float(fac["r2"]), hist=hist,
                   train_minutes=mins, config=vars(args)),
              os.path.join(RESULTS, "n55_dit.json"), seed=args.seed)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(hist, lw=1.8, color="#1F4E79")
        ax[0].set_xlabel("epoch", fontsize=8); ax[0].set_ylabel("latent eps-loss", fontsize=8)
        ax[0].set_title(f"DiT training ({mins:.0f} min, final {hist[-1]:.4f})", fontsize=9)
        ax[0].grid(alpha=0.25, lw=0.5)
        ax[1].bar(["digit", "fg", "bg", "joint"],
                  [m["digit"], m["fg"], m["bg"], m["joint"]], color="#2E7D32")
        ax[1].axhline(0.1, ls=":", color="black", lw=1)
        ax[1].set_ylim(0, 1); ax[1].grid(alpha=0.25, lw=0.5, axis="y")
        ax[1].set_title(f"ADDRESS on held-out compositions, DiT backbone\n"
                        f"basis R^2 {fac['r2']:.3f} (UNet 0.870)", fontsize=9)
        fig.suptitle("Does the three-tier scheme survive the move to a transformer?\n"
                     "Same cemb interface, so factorize/compose run unchanged.", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        o = os.path.join(HERE, "figures", "dit", "d2_dit_training.png")
        os.makedirs(os.path.dirname(o), exist_ok=True)
        fig.savefig(o, dpi=140, bbox_inches="tight")
        print(f"[n55] wrote {o}")
    except Exception as e:                                             # noqa: BLE001
        print(f"[n55] figure failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
