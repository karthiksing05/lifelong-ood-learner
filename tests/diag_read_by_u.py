"""At which noise levels can the known factors of a concept that carries a NEW value still be read?

C4: under the new background, the eps-loss read of the OTHER factors is right for 0.15 of new-value
concepts (0.55 with the background left out; digit exact, foreground 0.55, foreground 1 never read).
Every read so far pools levels.  This scores the read level by level, for three populations:

  in-dist    trained classes, full read                       (the read works: reference)
  new-comb   untrained combinations, full read                (the read works: reference)
  new-value  the background LEFT OUT, digit and fg read       (the broken case)

and prints per level: digit and fg accuracy.  If some band reads fg under the new value, a band chosen
on trained classes alone (label-free) might; if none does, the limit is the backbone's conditioning.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from c_bench import load_bench                                        # noqa: E402

from lifelong_ood.neural.cobweb_update import candidate_losses         # noqa: E402

U = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def main():
    dev = "cuda"
    B = load_bench("colormnist", seed=0, device=dev)
    model = B.model
    mu = np.asarray(B.fac["mu"], np.float32)
    V = [np.asarray(v, np.float32) for v in B.fac["V"]]
    rng = np.random.default_rng(0)
    pops = ([("in-dist", B.seen[i][1]) for i in rng.permutation(len(B.seen))[:24]]
            + [("new-comb", t) for t in B.held_comb]
            + [("new-value", t) for t in B.new_pairs])
    out = {p: {"digit": [], "fg": []} for p in ("in-dist", "new-comb", "new-value")}
    for pop, tup in pops:
        X = B.images(tup)
        Xv = B.to_m(X[64:64 + 16])
        # new-value: background LEFT OUT; the others: the full basis read, scored on digit and fg
        if pop == "new-value":
            cl = list(itertools.product(range(len(V[0])), range(len(V[1]))))
            E = np.stack([mu + V[0][d] + V[1][f] for d, f in cl])
        else:
            cl = list(itertools.product(range(len(V[0])), range(len(V[1])), range(len(V[2]))))
            E = np.stack([mu + V[0][d] + V[1][f] + V[2][b] for d, f, b in cl])
        L = candidate_losses(model, Xv, U, embs=E.astype(np.float32), seed=4242, per_u=True,
                             n_noise=4, batch=1024, layout=B.layout, img=B.img, device=dev)
        k = L.argmin(0)                                              # best candidate per level
        out[pop]["digit"].append([cl[i][0] == tup[0] for i in k])
        out[pop]["fg"].append([cl[i][1] == tup[1] for i in k])
    print("level  " + "  ".join(f"{u:5.2f}" for u in U))
    for pop in out:
        for fac in ("digit", "fg"):
            a = np.mean(out[pop][fac], 0)
            print(f"{pop:9s} {fac:5s} " + "  ".join(f"{x:5.2f}" for x in a))
    # per true fg value under the new background, best level
    nv = [t for p, t in pops if p == "new-value"]
    fga = np.array(out["new-value"]["fg"])
    for f in range(len(V[1])):
        m = np.array([t[1] == f for t in nv])
        print(f"new-value fg={f}: " + "  ".join(f"{x:5.2f}" for x in fga[m].mean(0)))
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    sys.exit(main())
