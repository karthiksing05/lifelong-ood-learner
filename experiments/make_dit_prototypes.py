"""D5 -- the PROTOTYPES a DiT discovers, and D4 re-rendered with both backbones calibrated.

Two things.

(1) PROTOTYPES.  Stage 1 of the loop is mode ascent on the model's own score field: from a noised
start, climb log p_t and return the mode.  On the UNet those modes are pictures (f9's `prototypes`
row).  A DiT's score lives in LATENT space, so the ascent runs there and the modes are decoded
through the autoencoder to be seen.  Same procedure, same hybrid initialisation -- half the chains
started from the noised query and half from pure noise, because a pool grown only from the query
explores one basin and composition needs experts that disagree with it.

(2) A FAIRNESS FIX to D4.  The first version composed BOTH backbones with `compose(fac, ...)` and no
gains.  That is not neutral: 5q measures the UNet under-responding to background (bg direction norm
0.55 against 0.79 for digit), which is the entire reason `backbone_gains` exists, and the render
showed it -- red backgrounds where purple belonged, a cyan 2, a green 6.  The DiT needed no such
correction.  Comparing a calibrated model against an uncalibrated one would make the transfer claim
look better than it is, so both are calibrated here by the same routine and the gains are printed.

    python experiments/make_dit_prototypes.py
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, backbone_gains                               # noqa: E402

from lifelong_ood.data.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import fit_probes                              # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data               # noqa: E402
from lifelong_ood.neural.dit import DiT                                # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize,        # noqa: E402
                                            sample_factored_pool, slot_factors)
from lifelong_ood.neural.latent import ConvAE, decode, encode          # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402
from n55_dit_train import ddim_latent                                  # noqa: E402

FIG = os.path.join(HERE, "figures", "dit")


def latent_mode_ascent(model, z_q, u, n_starts=8, iters=120, lr=0.05, seed=0, init="hybrid",
                       device="cuda"):
    """Adam ascent on log p_t in LATENT space -> x0-space modes. The DiT analogue of
    `BaseScoreField.mode_ascent`, same parameterisation (Adam, no clamp, no Tweedie projection)."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = float(cosine_abar(np.array([u]))[0])
    zq = torch.as_tensor(np.asarray(z_q, np.float32), device=device).reshape(1, *z_q.shape)
    half = n_starts // 2 if init == "hybrid" else (n_starts if init == "query" else 0)
    starts = []
    if half:
        e = torch.randn((half, *zq.shape[1:]), generator=g, device=device)
        starts.append((a ** 0.5) * zq.expand(half, -1, -1, -1) + ((1 - a) ** 0.5) * e)
    if n_starts - half:
        starts.append(torch.randn((n_starts - half, *zq.shape[1:]), generator=g, device=device))
    zt = torch.cat(starts).clone().requires_grad_(True)
    src = (["query"] * half) + (["random"] * (n_starts - half))
    opt = torch.optim.Adam([zt], lr=lr)
    null = model.cemb.weight[model.null_id].detach().reshape(1, -1)
    for _ in range(iters):
        eps = model(zt, torch.full((len(zt),), float(u), device=device),
                    emb_override=null.expand(len(zt), -1))
        # grad log p_t = -eps / sqrt(1-abar); ascend it
        loss = (eps.detach() * zt).sum() / max((1 - a) ** 0.5, 1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        eps = model(zt, torch.full((len(zt),), float(u), device=device),
                    emb_override=null.expand(len(zt), -1))
        z0 = (zt - ((1 - a) ** 0.5) * eps) / (a ** 0.5)      # Tweedie to clean-latent space
    return z0.detach().cpu(), src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=os.path.join(CACHE, "dit_colormnist.pt"))
    p.add_argument("--ae", default=os.path.join(CACHE, "ae_colormnist.pt"))
    p.add_argument("--unet", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-slot", type=int, default=40)
    p.add_argument("--u-grid", type=float, nargs="*", default=[0.05, 0.15, 0.30])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(FIG, exist_ok=True)

    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = ConvAE(zdim=ack["zdim"], base=ack["base"]).to(dev)
    ae.load_state_dict(ack["model"]); ae.eval()
    scale = float(ack["scale"])
    dck = torch.load(args.dit, map_location="cpu", weights_only=False)
    dit = DiT(n_classes=dck["n_classes"], in_ch=dck["zdim"], size=8, patch=dck["patch"],
              d=dck["dim"], depth=dck["depth"]).to(dev)
    dit.load_state_dict(dck["model"]); dit.eval()
    slots_seen = np.array(sorted(int(k) for k in dck["remap"]))
    Cd = dit.class_table().cpu().numpy()
    fac_d = factorize(Cd, slot_factors(slots_seen))

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    X = np.asarray(data["X"], np.float32)
    Z = encode(ae, X, scale=scale, device=dev)
    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"])) - set(slots_seen.tolist())))
    rng = np.random.default_rng(args.seed)
    cases = rng.permutation(held)[:args.n_cases]

    uck = torch.load(args.unet, map_location="cpu", weights_only=False)
    ur = uck["remap"]; us = np.array(sorted(ur))
    Cu = uck["ema"]["cemb.weight"].numpy()[[ur[int(s)] for s in us]]
    ufield = load_field(args.unet, device=dev)
    fac_u = factorize(Cu, slot_factors(us))
    gu = backbone_gains(ufield, Cu, us, probes, data, seed=args.seed, steps=args.steps, n=16)
    print(f"[proto] UNet gains {gu}  (5q: the backbone under-responds to background)", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def acc(S, d_, f_, b_):
        S = np.asarray(S, np.float32)
        return float(((probes["digit"].predict(S) == d_) & (probes["fg"].predict(S) == f_)
                      & (probes["bg"].predict(S) == b_)).mean())

    # ── D5: prototypes discovered by the DiT ────────────────────────────────
    nrow = len(cases) * 2
    fig, axes = plt.subplots(nrow, args.n_show, figsize=(1.4 * args.n_show + 4.2, 1.45 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for s in cases:
        d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        i0 = int(np.where(data["slot"] == int(s))[0][0])
        zq = Z[i0].numpy()
        M, src, us_ = [], [], []
        for u in args.u_grid:
            z0, sr = latent_mode_ascent(dit, zq, u, n_starts=4, seed=args.seed, device=dev)
            M.append(z0); src += sr; us_ += [u] * len(z0)
        M = torch.cat(M)
        P = decode(ae, M, scale=scale, device=dev).astype(np.float32)
        for k in range(args.n_show):
            axes[r][k].axis("off")
            axes[r][k].imshow(np.clip(X[np.where(data["slot"] == int(s))[0][k]].reshape(32, 32, 3),
                                      0, 1))
        axes[r][0].set_ylabel(f"real - HELD OUT\n{slot_name(d_, f_, b_)}", rotation=0,
                              ha="right", va="center", fontsize=6.5)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        r += 1
        for k in range(args.n_show):
            axes[r][k].axis("off")
            if k < len(P):
                axes[r][k].imshow(np.clip(P[k].reshape(32, 32, 3), 0, 1))
                axes[r][k].set_title(f"{src[k]}-init u={us_[k]:g}", fontsize=5.5)
        axes[r][0].set_ylabel("prototypes\n(latent mode ascent)", rotation=0, ha="right",
                              va="center", fontsize=6.5)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        r += 1
        print(f"[proto] slot {int(s)}: {len(P)} modes decoded", flush=True)
    fig.suptitle("Prototypes a DiT discovers: Adam ascent on log p_t in LATENT space, decoded "
                 "through the autoencoder.\nHybrid initialisation -- half the chains from the "
                 "noised query, half from pure noise, so the pool is not confined to the query's "
                 "own basin.\nSame procedure as the UNet's f9 `prototypes` row; only the space it "
                 "runs in differs.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    o = os.path.join(FIG, "d5_dit_prototypes.png")
    fig.savefig(o, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[proto] wrote {o}")

    # ── D4 re-render, BOTH calibrated ───────────────────────────────────────
    rows = ["real", "UNet ADDRESS", "DiT ADDRESS"]
    fig, axes = plt.subplots(len(cases) * len(rows), args.n_show,
                             figsize=(1.4 * args.n_show + 4.0, 1.4 * len(cases) * len(rows)))
    axes = np.atleast_2d(axes)
    r = 0
    for ci, s in enumerate(cases):
        d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        idx = np.where(data["slot"] == int(s))[0][:args.n_show]
        got = {
            "real": X[idx],
            "UNet ADDRESS": to_data(sample_factored_pool(
                ufield, torch.as_tensor(compose(fac_u, (d_, f_, b_), gains=gu),
                                        dtype=torch.float32, device=dev),
                n=args.n_show, steps=args.steps, seed=args.seed)),
            "DiT ADDRESS": decode(ae, ddim_latent(dit, compose(fac_d, (d_, f_, b_)), args.n_show,
                                                  (dck["zdim"], 8, 8), steps=args.steps,
                                                  seed=args.seed, device=dev).cpu(),
                                  scale=scale, device=dev),
        }
        for nm in rows:
            S = np.asarray(got[nm], np.float32)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(32, 32, 3), 0, 1))
            lab = f"real - HELD OUT\n{slot_name(d_, f_, b_)}" if nm == "real" else nm
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            axes[r][args.n_show - 1].set_title(f"joint {acc(S, d_, f_, b_):.2f}", fontsize=6.5,
                                               loc="right")
            r += 1
    fig.suptitle(f"The same held-out compositions through BOTH backbones, BOTH CALIBRATED.\n"
                 f"UNet R^2 {fac_u['r2']:.3f} with gains {gu} (5q: it under-responds to "
                 f"background); DiT R^2 {fac_d['r2']:.3f} at unit gain.\nThe first version of this "
                 f"figure gave the UNet no gains, which flattered the DiT.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    o = os.path.join(FIG, "d4_unet_vs_dit.png")
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"[proto] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
