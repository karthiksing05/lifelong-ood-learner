"""N60 -- axes 3 and 4 on the DiT: does absorbing damage anything, and what does it cost?

N55/N56 established composition on a latent DiT: basis R^2 0.832 against the pixel UNet's 0.870, and
a full loop that reads the composition from ONE query at 0.917 and scores joint 0.661.  Both are
axis-1/2 results.  The thesis rests on axes 3 and 4 -- preservation and cost -- and neither has been
measured on a transformer.

WHY THIS IS A REAL PORTABILITY QUESTION AND NOT A FORMALITY.  The preservation guarantee everywhere
else in this repository is *structural*: a frozen ancestor path plus a private conditioning vector
cannot be written by a later concept.  That argument was developed against a UNet, and its
implementation (`HierarchicalAdapterBank`, the sealing discipline) injects LoRA into convolutions.  A
DiT has no convolutions in its blocks -- it is attention and MLPs modulated by adaLN.  So:

  * for ADDRESS the argument should carry over untouched, because ADDRESS writes NO parameters at
    all: it stores factor coordinates and composes a point in the existing basis.  That is
    architecture-independent by construction, and this measures whether the implementation honours it.
  * the retention risk that IS architecture-independent is the shared factor basis being refit as
    vocabulary grows -- an early concept degrading with no weight having moved.  N59 found that does
    not happen on Shapes3D; whether it happens on a transformer is a separate fact.

Same protocol and same metrics as N5 (ColorMNIST), N22 (CelebA) and N59 (Shapes3D), so the four sit
in one table: the accuracy matrix, ACC / BWT / FGT, an exact-unchanged flag, and a sha256 over every
parameter before and after.

    python experiments/n60_dit_lifelong.py --n-concepts 12
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                 # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                   # noqa: E402
from lifelong_ood.neural.dit import DiT                                # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize, slot_factors  # noqa: E402
from lifelong_ood.neural.latent import ConvAE, decode                  # noqa: E402
from n55_dit_train import ddim_latent                                  # noqa: E402


def weight_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=os.path.join(CACHE, "dit_colormnist.pt"))
    p.add_argument("--ae", default=os.path.join(CACHE, "ae_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-slot", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = ConvAE(zdim=ack["zdim"], base=ack["base"]).to(dev)
    ae.load_state_dict(ack["model"]); ae.eval()
    scale = float(ack["scale"])
    dck = torch.load(args.dit, map_location="cpu", weights_only=False)
    dit = DiT(n_classes=dck["n_classes"], in_ch=dck["zdim"], size=8, patch=dck["patch"],
              d=dck["dim"], depth=dck["depth"]).to(dev)
    dit.load_state_dict(dck["model"]); dit.eval()
    slots_seen = np.array(sorted(int(k) for k in dck["remap"]))
    C = dit.class_table().cpu().numpy()
    print(f"[n60] DiT {sum(q.numel() for q in dit.parameters()) / 1e6:.2f}M params, "
          f"{len(slots_seen)} seen compositions", flush=True)

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"])) - set(slots_seen.tolist())))
    rng = np.random.default_rng(args.seed)
    stream = rng.permutation(held)[:args.n_concepts]
    zsh = (ack["zdim"], 8, 8)
    h0 = weight_hash(dit)

    def score(slot, fac):
        d_, f_, b_ = int(slot) // 16, (int(slot) % 16) // 4, int(slot) % 4
        c = compose(fac, (d_, f_, b_))
        z = ddim_latent(dit, c, args.n_samples, zsh, steps=args.steps, seed=args.seed, device=dev)
        S = decode(ae, z.cpu(), scale=scale, device=dev).astype(np.float32)
        return float(((probes["digit"].predict(S) == d_) & (probes["fg"].predict(S) == f_)
                      & (probes["bg"].predict(S) == b_)).mean())

    acc = [[] for _ in stream]
    absorbed = []
    for t, slot in enumerate(stream):
        absorbed.append(int(slot))
        fac = factorize(C, slot_factors(slots_seen))
        for j in range(t + 1):
            acc[j].append(score(absorbed[j], fac))
        print(f"[n60] after {t + 1:2d} absorbed: " +
              " ".join(f"{acc[j][-1]:.2f}" for j in range(t + 1)), flush=True)

    identical = (h0 == weight_hash(dit))
    ACC = float(np.mean([r[-1] for r in acc]))
    FGT = float(np.mean([max(r) - r[-1] for r in acc]))
    BWT = float(np.mean([r[-1] - r[0] for r in acc if len(r) > 1]))
    exact = float(np.mean([1.0 if abs(r[-1] - r[0]) < 1e-9 else 0.0 for r in acc]))
    print("\n[n60] " + "=" * 66)
    print(f"[n60] DiT bit-identical after {len(stream)} absorptions: "
          f"{'YES' if identical else 'NO'}   (sha256 over every parameter)")
    print(f"[n60] ACC {ACC:.3f}   BWT {BWT:+.3f}   FGT {FGT:.3f}   "
          f"rows exactly unchanged {exact:.3f}")
    print("[n60]   [ColorMNIST UNet N5: bit-identical 1.00, FGT 0.000]")
    print("[n60]   [Shapes3D      N59: bit-identical YES,  FGT 0.000]")
    print(f"[n60] storage: {len(stream)} x 72 B = {len(stream) * 72} B against a "
          f"{sum(q.numel() for q in dit.parameters()) * 4 / 1e6:.1f} MB backbone")
    save_json(dict(acc_matrix=acc, ACC=ACC, BWT=BWT, FGT=FGT, bit_identical=bool(identical),
                   rows_exact=exact, n_concepts=len(stream), bytes=len(stream) * 72,
                   config=vars(args)),
              os.path.join(RESULTS, "n60_dit_lifelong.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
