"""Which CelebA attributes does the factored conditioning actually control?

The aggregate margin over unconditional is small (+0.024 at guidance 3 on the 15-epoch backbone),
but that is a mean over twelve attributes of very different difficulty, and a mean can hide a strong
signal on a few behind no signal on the rest.  This measures each attribute separately, by the only
comparison that isolates conditioning: generate with the attribute ON and with it OFF, everything
else held fixed, and ask the probe how often it moved.

`flip` is that quantity -- P(probe reads ON | conditioned ON) - P(probe reads ON | conditioned OFF).
It is the per-attribute controllability of the model, it needs no held-out label, and unlike
"accuracy against a target combination" it cannot be inflated by an attribute's base rate.  An
attribute with flip 0 is one the conditioning does not control at any guidance scale.

This decides two things: whether the CelebA claims should be stated over all attributes or over the
controllable ones, and whether `Smiling x Big_Nose` -- the held-out pair for the strong split -- is
a pair the backbone can actually express.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                # noqa: E402
from n19_celeba_train import build_probes                               # noqa: E402

from lifelong_ood.celeba import ATTRS, load_celeba_comp                 # noqa: E402
from lifelong_ood.neural import load_field, sample_embedding, to_data   # noqa: E402
from lifelong_ood.neural.idealwords import compose_attr                 # noqa: E402

CKPT = os.environ.get("DIAG_CKPT", os.path.join(CACHE, "ddpm_celeba_factored_e15.pt"))
GUID = float(os.environ.get("DIAG_GUIDANCE", "3.0"))
N = int(os.environ.get("DIAG_N", 24))
N_CTX = int(os.environ.get("DIAG_NCTX", 3))          # contexts (other attributes) per attribute


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    img = ck["img"]
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = img, 3 * img * img
    data = load_celeba_comp(img=img, min_count=ck.get("min_count", 60),
                            n_holdout=ck.get("n_holdout", len(ck["hold_ids"])),
                            holdout_pair=ck.get("holdout_pair"),
                            holdout_attr=ck.get("holdout_attr"),
                            seed=ck.get("seed", 0), cache_dir=CACHE)
    sp = data["split"]
    probes = build_probes(data, 0, 30)
    field = load_field(CKPT, device=dev, batch=64)
    model = field.model
    print(f"[diag] {os.path.basename(CKPT)} at guidance {GUID}, {N} samples x {N_CTX} contexts",
          flush=True)

    rng = np.random.default_rng(0)
    tr = np.array(sorted(ck["remap"], key=lambda z: ck["remap"][z]))
    ctxs = [sp["all_combos"][int(j)].copy() for j in rng.permutation(tr)[:N_CTX]]

    rows = []
    for k, a in enumerate(ATTRS):
        val = probes.get(f"{a}_val_acc", 0.0)
        if val < 0.65:
            rows.append((a, val, float("nan"), float("nan"), float("nan")))
            continue
        on_rate, off_rate = [], []
        for cb in ctxs:
            for v, acc in ((1, on_rate), (0, off_rate)):
                c2 = cb.copy(); c2[k] = v
                c = compose_attr(model, c2)
                S = to_data(sample_embedding(field, c.detach(), n=N, steps=50, seed=0,
                                             guidance=GUID)).astype(np.float32)
                acc.append(float((probes[a].predict(S) == 1).mean()))
        on_m, off_m = float(np.mean(on_rate)), float(np.mean(off_rate))
        rows.append((a, val, on_m, off_m, on_m - off_m))
        print(f"  {a:20s} probe {val:.3f} | ON {on_m:.3f}  OFF {off_m:.3f}  "
              f"flip {on_m - off_m:+.3f}", flush=True)

    ok = [r for r in rows if r[4] == r[4]]
    ok.sort(key=lambda r: -r[4])
    print(f"\n  ranked by controllability (flip):")
    for a, val, on, off, fl in ok:
        bar = "#" * max(0, int(round(fl * 40)))
        print(f"    {a:20s} {fl:+.3f}  {bar}")
    strong = [a for a, _, _, _, fl in ok if fl >= 0.15]
    print(f"\n  {len(strong)}/{len(ok)} attributes reach flip >= 0.15: {', '.join(strong)}")
    pair = ("Smiling", "Big_Nose")
    for a in pair:
        m = [r for r in rows if r[0] == a]
        if m:
            print(f"  held-out-pair attribute {a:12s} flip {m[0][4]:+.3f}")
    print("\n  If the strong set is small, the CelebA claims should be stated over the")
    print("  CONTROLLABLE attributes and say so, rather than over a 12-way mean that averages")
    print("  a real signal against attributes the conditioning does not reach.")


if __name__ == "__main__":
    main()
