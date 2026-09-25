"""D3/D4 -- what the DiT actually GENERATES for a held-out composition, beside the UNet.

N55 reports basis R^2 0.832 and ADDRESS joint 0.807 on the DiT.  Those are numbers; this is the
picture, built to the same standard as f9 and f26 -- every block NAMES the withheld combination and
its training-image count, so the split is verifiable rather than trusted.

    D3  the DiT alone: real / ADDRESS / nearest-class / unconditional, per held-out composition
    D4  the SAME held-out compositions through both backbones, so the transfer claim is visible
        rather than only tabulated

    python experiments/make_dit_case.py --n-cases 4
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.data.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import fit_probes                              # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data               # noqa: E402
from lifelong_ood.neural.dit import DiT                                # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize,        # noqa: E402
                                            sample_factored_pool, slot_factors)
from lifelong_ood.neural.latent import ConvAE, decode                  # noqa: E402
from n55_dit_train import ddim_latent                                  # noqa: E402

FIG = os.path.join(HERE, "figures", "initial", "dit")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=os.path.join(CACHE, "dit_colormnist.pt"))
    p.add_argument("--ae", default=os.path.join(CACHE, "ae_colormnist.pt"))
    p.add_argument("--unet", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-slot", type=int, default=40)
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
    remap = dck["remap"]
    slots_seen = np.array(sorted(int(k) for k in remap))
    Cd = dit.class_table().cpu().numpy()
    fac_d = factorize(Cd, slot_factors(slots_seen))
    print(f"[dit-case] DiT basis R^2 {fac_d['r2']:.3f}", flush=True)

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"]))
                           - set(int(s) for s in slots_seen)))
    rng = np.random.default_rng(args.seed)
    cases = rng.permutation(held)[:args.n_cases]

    # UNet side, for D4
    uck = torch.load(args.unet, map_location="cpu", weights_only=False)
    ur = uck["remap"]
    us = np.array(sorted(ur))
    Cu = uck["ema"]["cemb.weight"].numpy()[[ur[int(s)] for s in us]]
    ufield = load_field(args.unet, device=dev)
    fac_u = factorize(Cu, slot_factors(us))
    print(f"[dit-case] UNet basis R^2 {fac_u['r2']:.3f}", flush=True)

    def acc(S, d_, f_, b_):
        S = np.asarray(S, np.float32)
        return dict(joint=float(((probes["digit"].predict(S) == d_)
                                 & (probes["fg"].predict(S) == f_)
                                 & (probes["bg"].predict(S) == b_)).mean()),
                    digit=float((probes["digit"].predict(S) == d_).mean()),
                    fg=float((probes["fg"].predict(S) == f_).mean()),
                    bg=float((probes["bg"].predict(S) == b_).mean()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── D3: the DiT on its own ───────────────────────────────────────────────
    order = ["real", "ours (ADDRESS)", "nearest-class", "unconditional"]
    fig, axes = plt.subplots(len(cases) * len(order), args.n_show,
                             figsize=(1.4 * args.n_show + 4.0, 1.4 * len(cases) * len(order)))
    axes = np.atleast_2d(axes)
    r = 0
    for ci, s in enumerate(cases):
        d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        idx = np.where(data["slot"] == int(s))[0][:args.n_show]
        n_train = int(np.sum((data["slot"] == int(s)) & data["seen"]))
        c_new = compose(fac_d, (d_, f_, b_))
        zsh = (dck["zdim"], 8, 8)
        got = {
            "real": np.asarray(data["X"], np.float32)[idx],
            "ours (ADDRESS)": decode(ae, ddim_latent(dit, c_new, args.n_show, zsh,
                                                     steps=args.steps, seed=args.seed,
                                                     device=dev).cpu(), scale=scale, device=dev),
            "nearest-class": decode(ae, ddim_latent(
                dit, Cd[int(np.argmin(((Cd - c_new[None]) ** 2).sum(1)))], args.n_show, zsh,
                steps=args.steps, seed=args.seed, device=dev).cpu(), scale=scale, device=dev),
            "unconditional": decode(ae, ddim_latent(
                dit, dit.cemb.weight[dit.null_id].detach().cpu().numpy(), args.n_show, zsh,
                steps=args.steps, seed=args.seed + 7919 * (ci + 1), device=dev).cpu(),
                scale=scale, device=dev),
        }
        for nm in order:
            S = np.asarray(got[nm], np.float32)
            a = acc(S, d_, f_, b_)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(32, 32, 3), 0, 1))
            lab = (f"real - HELD OUT\n{slot_name(d_, f_, b_)}\nslot {int(s)} - "
                   f"{n_train} training images" if nm == "real" else nm)
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp in axes[r][0].spines.values():
                sp.set_visible(False)
            axes[r][args.n_show - 1].set_title(
                f"joint {a['joint']:.2f}  digit {a['digit']:.2f} fg {a['fg']:.2f} "
                f"bg {a['bg']:.2f}", fontsize=6.5, loc="right")
            r += 1
            print(f"[dit-case] slot {int(s):3d} {nm:16s} joint {a['joint']:.3f}", flush=True)
    fig.suptitle(f"ColorMNIST on a LATENT DiT - one held-out composition per block, named so the "
                 f"split is verifiable.\nADDRESS composes it from the DiT's own class table "
                 f"(basis R^2 {fac_d['r2']:.3f}); `nearest-class` is the retrieval floor and "
                 f"`unconditional` is the null token.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    o = os.path.join(FIG, "d3_case_colormnist_dit.png")
    fig.savefig(o, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[dit-case] wrote {o}")

    # ── D4: same compositions, both backbones ───────────────────────────────
    rows = ["real", "UNet ADDRESS", "DiT ADDRESS"]
    fig, axes = plt.subplots(len(cases) * len(rows), args.n_show,
                             figsize=(1.4 * args.n_show + 4.0, 1.4 * len(cases) * len(rows)))
    axes = np.atleast_2d(axes)
    r = 0
    for ci, s in enumerate(cases):
        d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        idx = np.where(data["slot"] == int(s))[0][:args.n_show]
        cu = compose(fac_u, (d_, f_, b_))
        cd = compose(fac_d, (d_, f_, b_))
        got = {
            "real": np.asarray(data["X"], np.float32)[idx],
            "UNet ADDRESS": to_data(sample_factored_pool(
                ufield, torch.as_tensor(cu, dtype=torch.float32, device=dev),
                n=args.n_show, steps=args.steps, seed=args.seed)),
            "DiT ADDRESS": decode(ae, ddim_latent(dit, cd, args.n_show, (dck["zdim"], 8, 8),
                                                  steps=args.steps, seed=args.seed,
                                                  device=dev).cpu(), scale=scale, device=dev),
        }
        for nm in rows:
            S = np.asarray(got[nm], np.float32)
            a = acc(S, d_, f_, b_)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(32, 32, 3), 0, 1))
            lab = (f"real - HELD OUT\n{slot_name(d_, f_, b_)}" if nm == "real" else nm)
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp in axes[r][0].spines.values():
                sp.set_visible(False)
            axes[r][args.n_show - 1].set_title(f"joint {a['joint']:.2f}", fontsize=6.5, loc="right")
            r += 1
    fig.suptitle(f"The same held-out compositions through BOTH backbones.\n"
                 f"UNet basis R^2 {fac_u['r2']:.3f} - DiT basis R^2 {fac_d['r2']:.3f}. "
                 f"`factorize` and `compose` are identical code in both columns; only the backbone "
                 f"and the space it works in differ.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    o = os.path.join(FIG, "d4_unet_vs_dit.png")
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"[dit-case] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
