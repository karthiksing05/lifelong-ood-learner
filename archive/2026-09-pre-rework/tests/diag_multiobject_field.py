"""Is the multi-object backbone broken, or is the path that READS it broken?

N64 on the 220-epoch backbone: the object-membership basis is excellent (R^2 0.931, CV 0.875) while
generation is pure noise and the model scores 0.000 conditioned on its OWN trained row.  Those two
cannot both be facts about the same model -- a table that factorises that cleanly was learned by a
network that was fitting something.  So this asks which half is lying, before any more GPU-hours go
into "training it longer".

The measurement is the eps-loss of REAL TRAINING IMAGES under their OWN trained class, computed
through `conditional_losses` -- the same layout-aware path the router and the sampler use:

  ~0.01-0.10  the field agrees with the checkpoint; the model learned the domain, and a bad sample
              is then the SAMPLER's problem (clip range, step count, prior)
  ~1.0        the field and the checkpoint disagree about a convention.  A loss pinned at exactly
              1.0 means the model is predicting nothing useful about the noise, which is what
              happens when an image is handed over in a layout or range it was never trained on --
              the VE/VP dialect mismatch (5t-bis) read exactly 1.0000 for 1200 steps.

Both LAYOUTS are tried against the same checkpoint, because the checkpoint records `hwc` and this is
the first backbone in the repository trained that way -- the ColorMNIST backbone kept `permuted` for
continuity, so an `hwc` fault would never have shown up before.  If `permuted` scores far better
than the `hwc` the checkpoint claims, the training/sampling convention is the bug and no amount of
training will fix it.

    python tests/diag_multiobject_field.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.data.multiobject import load_multiobject             # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet                 # noqa: E402
from lifelong_ood.neural.ddpm import to_model                        # noqa: E402
from lifelong_ood.neural.router import (conditional_losses,            # noqa: E402
                                        null_conditioning)

U_GRID = (0.1, 0.3, 0.6, 0.9)


def main():
    ck_path = os.path.join(CACHE, "ddpm_multiobject_v2.pt")
    if not os.path.exists(ck_path):
        ck_path = os.path.join(CACHE, "ddpm_multiobject.pt")
    if not os.path.exists(ck_path):
        print(f"SKIP: no multi-object checkpoint at {ck_path}")
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sp = ck["split"]
    print(f"[diag] {os.path.basename(ck_path)}: layout={ck.get('layout')!r} "
          f"epochs={ck.get('epochs_done')}")
    d = load_multiobject(n_per_scene=20, seed=sp["seed"], n_holdout=sp["n_holdout"])
    X, scene = d["X"], d["scene"]
    remap = ck["remap"]
    uniq = np.array(sorted(int(k) for k in remap))
    C = ck["ema"]["cemb.weight"].numpy()

    worst = {}
    own_rec, nul_rec = [], []
    for lay in ("hwc", "permuted"):
        field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                                 layout=lay)
        field.model.load_state_dict(ck["ema"])
        field.model.eval()
        null = null_conditioning(field.model).reshape(1, -1)
        own, other, nul = [], [], []
        for s in uniq[:8]:
            i = int(np.where(scene == int(s))[0][0])
            row = C[[remap[int(s)]]]
            L = conditional_losses(field, to_model(X[i]).ravel(),
                                   np.concatenate([null, row]),
                                   u_grid=U_GRID, n_noise=4, seed=0)
            nul.append(float(L[0]))
            own.append(float(L[1]))
        if lay == ck.get("layout"):
            own_rec, nul_rec = list(own), list(nul)
        tag = "  <-- what the checkpoint records" if lay == ck.get("layout") else ""
        print(f"[diag] layout={lay:9s} eps-loss under OWN class {np.mean(own):.4f} "
              f"| under null {np.mean(nul):.4f}{tag}")
        worst[lay] = float(np.mean(own))
        del field
        if dev == "cuda":
            torch.cuda.empty_cache()

    rec = ck.get("layout")
    best = min(worst, key=worst.get)
    print(f"[diag] layout recorded={rec}, best-scoring={best} -- "
          + ("consistent" if best == rec else "MISMATCH: the reader disagrees with the checkpoint"))

    # --- PER-u, and against a REFERENCE backbone -------------------------------------------
    # An absolute eps-loss means nothing on its own: eps-prediction is trivial at high noise
    # (x_t IS approximately eps) and impossible at low noise, so a number is only interpretable
    # against the same number from a backbone known to work.  ColorMNIST is that reference -- its
    # router scores AUROC 1.000, which is only possible if its class conditionals genuinely differ.
    def per_u(field, Xr, slots_r, uniq_r, Cr, remap_r, tag, prep_r=lambda z: z):
        null = null_conditioning(field.model).reshape(1, -1)
        print(f"[diag] {tag}: per-u OWN vs NULL")
        for u in U_GRID:
            o, n = [], []
            for s in uniq_r[:8]:
                idx = np.where(np.asarray(slots_r) == int(s))[0]
                if not len(idx):
                    continue
                L = conditional_losses(field, prep_r(Xr[int(idx[0])]),
                                       np.concatenate([null, Cr[[remap_r[int(s)]]]]),
                                       u_grid=(u,), n_noise=4, seed=0)
                n.append(float(L[0]))
                o.append(float(L[1]))
            print(f"[diag]     u={u:<4g} own {np.mean(o):.4f}  null {np.mean(n):.4f}  "
                  f"gap {np.mean(n) - np.mean(o):+.4f}")

    f2 = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev, layout=rec)
    f2.model.load_state_dict(ck["ema"])
    f2.model.eval()
    per_u(f2, X, scene, uniq, C, remap, "multi-object",
          prep_r=lambda z: to_model(z).ravel())
    del f2
    if dev == "cuda":
        torch.cuda.empty_cache()

    try:
        from lifelong_ood.data.datasets import load_colormnist
        # NOT `import to_model` here: a function-local import of a name that is also imported at
        # module level makes that name local for the WHOLE function, so the use of `to_model`
        # 60 lines earlier raises UnboundLocalError.
        from lifelong_ood.neural.ddpm import load_field
        rp = os.path.join(CACHE, "ddpm_colormnist_nobg3.pt")
        rck = torch.load(rp, map_location="cpu", weights_only=False)
        rfield = load_field(rp, device=dev)
        rrem = rck["remap"]
        runiq = np.array(sorted(rrem))
        rdata = load_colormnist(n_per_slot=4, seed=0)
        rX = np.stack([to_model(x).ravel() for x in np.asarray(rdata["X"], np.float32)])
        per_u(rfield, rX, np.asarray(rdata["slot"]), runiq,
              rck["ema"]["cemb.weight"].numpy(), rrem, "REFERENCE ColorMNIST (router AUROC 1.000)")
        del rfield
    except Exception as e:                                          # noqa: BLE001
        print(f"[diag] reference backbone unavailable: {type(e).__name__}: {e}")

    # --- is the class table actually carrying anything? ------------------------------------
    # N64 reports the object-membership basis at R^2 0.931, which reads as a strongly structured
    # table.  But R^2 is a RATIO to the table's own variance, so a COLLAPSED table -- every row
    # nearly identical -- is fit almost perfectly by any linear model and scores high while meaning
    # nothing.  The eps-loss gap above is ~0.001, i.e. the model barely distinguishes its own class
    # from the null token, and those two readings cannot both be healthy.  So measure the scale.
    Cs = C[[remap[int(s)] for s in uniq]]
    ctr = Cs - Cs.mean(0)
    nrm = np.linalg.norm(ctr, axis=1)
    unit = ctr / np.maximum(nrm[:, None], 1e-12)
    cos = unit @ unit.T
    off = cos[~np.eye(len(cos), dtype=bool)]
    print(f"[diag] class table: row-norm {np.linalg.norm(Cs, axis=1).mean():.4f}, "
          f"centred-row-norm {nrm.mean():.4f} (spread/scale {nrm.mean() / max(np.linalg.norm(Cs, axis=1).mean(), 1e-12):.3f}), "
          f"mean |pairwise cos| {np.abs(off).mean():.3f}")

    print()
    if worst[rec] > 0.5:
        print(f"[diag] FAULT: at its own layout the model explains almost nothing about the noise "
              f"({worst[rec]:.4f}).")
        if best != rec and worst[best] < worst[rec] * 0.7:
            print(f"[diag]   layout={best} scores {worst[best]:.4f} on the SAME weights -- the "
                  f"convention is the bug, not the training.")
        return 1
    gap = float(np.mean(nul_rec) - np.mean(own_rec))
    print(f"[diag] at layout={rec}: OWN class {np.mean(own_rec):.4f} vs NULL "
          f"{np.mean(nul_rec):.4f} -- conditioning gap {gap:+.4f}")
    if gap < 0.01:
        print("[diag] FAULT: THE MODEL IS ESSENTIALLY UNCONDITIONAL.  Its own class row barely")
        print("[diag]   beats the null token, so nothing downstream of the class table can work,")
        print("[diag]   and N64's basis R^2 must be re-read against the table's SCALE above before")
        print("[diag]   it is believed -- a collapsed table fits any linear model.")
        return 1
    print("[diag] OK: the field agrees with the checkpoint and conditioning is live.")
    print("[diag]   A noisy SAMPLE is then the sampler's problem -- clip range, steps, or prior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
