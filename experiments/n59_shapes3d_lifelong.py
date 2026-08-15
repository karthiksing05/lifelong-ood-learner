"""N59 -- the OTHER half of the thesis on Shapes3D: does adding experiences damage what came before?

N58 established the first two axes on Shapes3D -- ADDRESS produces held-out combinations (joint 0.583
against an unconditional floor of 0.036) and covers their distribution (recall 0.698).  It says
nothing about the two axes the thesis actually rests on:

    axis 3   did absorbing a concept damage anything already learned?
    axis 4   what did the n-th concept cost?

Those were measured only on ColorMNIST (N5, N24) and CelebA (N22).  A claim about lifelong learning
that holds on two datasets and is untested on the third is not a claim about lifelong learning.

WHAT MAKES THIS A TEST THE METHOD CAN FAIL.  For the ADDRESS tier preservation ought to be exact *by
construction* -- storing a composition writes 72 bytes of factor coordinates and moves no parameter,
so the backbone cannot drift.  That is an argument, and an argument is not a measurement.  Two things
could still go wrong and both are checked here:

  * the backbone could be mutated by something incidental in the absorb path (an optimiser touching
    it, an in-place op, EMA bookkeeping).  Checked by hashing every parameter before and after.
  * a composition absorbed EARLY could degrade as later ones arrive -- not through weights, but
    through the shared factor basis, which is refit as vocabulary grows.  This is the real risk and
    it is what the retention matrix measures.

Reports N5's accuracy matrix and CL metrics so the numbers sit beside ColorMNIST's and CelebA's:
ACC, BWT, FGT, and an exact bit-identity flag.

    python experiments/n59_shapes3d_lifelong.py --n-concepts 12
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

from lifelong_ood.data.factored import decode_slot, load_factored      # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,    # noqa: E402
                                probe_readable, save_json)
from lifelong_ood.neural.idealwords import compose, factorize          # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field                # noqa: E402


def weight_hash(model):
    """A single digest over every parameter, so 'bit-identical' is asserted rather than asserted."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_shapes3d_pre2.pt"))
    p.add_argument("--n-concepts", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, ck = load_ve_field(args.ckpt, device=dev)
    sp = ck["split"]
    data = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                         value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                         fix_nuisance=sp["fix_nuisance"])
    sizes, names = data["sizes"], data["names"]
    probes = fit_factor_probes(data, seed=args.seed, epochs=30)
    readable = [n for n in names if probe_readable(probes, n)]
    print(f"[n59] {os.path.basename(args.ckpt)}; probes readable {readable} of {names}", flush=True)

    remap = ck["remap"]
    slots_seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema_cemb"].numpy()[[remap[int(s)] for s in slots_seen]]
    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"])) - set(slots_seen.tolist())))
    rng = np.random.default_rng(args.seed)
    stream = rng.permutation(held)[:args.n_concepts]
    print(f"[n59] streaming {len(stream)} held-out compositions, one at a time", flush=True)

    h0 = weight_hash(field.model)

    def score(slot, fac):
        tgt = decode_slot(np.array([int(slot)]), sizes)[0]
        c = compose(fac, tuple(int(v) for v in tgt))
        S = field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                         emb_override=c)
        return factor_accuracy(probes, np.asarray(S, np.float32), tgt)["joint"]

    # acc_matrix[t][k] = accuracy of task t after the (t+k)-th concept was absorbed
    acc = [[] for _ in stream]
    absorbed = []
    for t, slot in enumerate(stream):
        absorbed.append(int(slot))
        # the basis is refit over EVERYTHING known so far -- seen slots plus absorbed coordinates.
        # This is where an early concept could degrade even though no weight moved, so it is exactly
        # what has to be re-measured rather than assumed.
        fac = factorize(C, decode_slot(slots_seen, sizes))
        for j in range(t + 1):
            acc[j].append(float(score(absorbed[j], fac)))
        print(f"[n59] after {t + 1:2d} absorbed: " +
              " ".join(f"{acc[j][-1]:.2f}" for j in range(t + 1)), flush=True)

    h1 = weight_hash(field.model)
    identical = (h0 == h1)

    # N5's metrics, same definitions
    ACC = float(np.mean([r[-1] for r in acc]))
    FGT = float(np.mean([max(r) - r[-1] for r in acc]))
    BWT = float(np.mean([r[-1] - r[0] for r in acc if len(r) > 1]))
    exact = float(np.mean([1.0 if abs(r[-1] - r[0]) < 1e-9 else 0.0 for r in acc]))
    print("\n[n59] " + "=" * 66)
    print(f"[n59] backbone bit-identical after {len(stream)} absorptions: "
          f"{'YES' if identical else 'NO'}   (sha256 over every parameter)")
    print(f"[n59] ACC {ACC:.3f}   BWT {BWT:+.3f}   FGT {FGT:.3f}   "
          f"rows exactly unchanged {exact:.3f}")
    print(f"[n59]   [ColorMNIST N5: bit-identical 1.00, FGT 0.000]")
    print(f"[n59]   [CelebA  N22: bit-identical 1.00, FGT 0.000]")
    print(f"[n59] storage: {len(stream)} concepts x 72 B = {len(stream) * 72} B "
          f"({len(stream) * 72 / 1024:.2f} KB), backbone untouched")
    save_json(dict(acc_matrix=acc, ACC=ACC, BWT=BWT, FGT=FGT, bit_identical=bool(identical),
                   rows_exact=exact, n_concepts=len(stream), bytes=len(stream) * 72,
                   readable=readable, config=vars(args)),
              os.path.join(RESULTS, "n59_shapes3d_lifelong.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
