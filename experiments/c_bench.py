"""The two benchmarks the c-series runs on, behind one interface, so every c-experiment runs the
SAME code path on both -- a result that holds on one and not the other is then a fact about the data,
not about a second implementation.

Each `Bench` has a frozen backbone and a factor basis fitted to its class table, and one factor
VALUE the backbone has never rendered:

  colormnist  ddpm_colormnist_nobg3.pt, a 90-class UNet trained with background 3 withheld
              entirely (RESULTS 5q Phase B).  New value: bg = 3.
  shapes3d    ddpm_s3d_full.pt, the eurecom VE backbone with a 360-class table (n57).  Every
              Shapes3D hue is in its pretraining data, so the new value is one Shapes3D does not
              contain: a GREY wall (`data/novel_value.py`).  NOTE the factor this repository calls
              `floor_hue` is the image's WALL in this split (loader bug, d8298e1; rows="legacy" is
              what the backbone was trained on), which is why the grey surface is the wall.

Interface (all image rows are flat HWC in DATA space [0, 1]):

  .model .layout .img .clip .device     backbone + sampler conventions (`clip` is MODEL space)
  .to_m(X01) / .to_d(Xm)                data <-> model space
  .fac                                  dict(mu, V[list], sizes, r2) over the trained classes
  .names .new_f .new_j                  factor names; which factor gets the new value, its index
  .seen                                 [(class_id, tuple)] trained classes
  .held_comb                            [tuple] untrained combinations of KNOWN values
  .new_pairs                            [tuple] combinations that use the new value
  .images(tup)                          real rows of a composition (recoloured for a grey wall)
  .acc(S01, tup)                        dict(joint, <factor>...) from probes fitted on real images
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.evals import SoftmaxProbe                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "results", "cache")


class ColorMNISTBench:
    name = "colormnist"
    names = ("digit", "fg", "bg")

    def __init__(self, seed=0, n_per_slot=200, device="cuda",
                 ckpt=os.path.join(CACHE, "ddpm_colormnist_nobg3.pt"), probe_epochs=30):
        from lifelong_ood.data.datasets import load_colormnist, slot_id
        from lifelong_ood.evals import composition_accuracy, fit_probes
        from lifelong_ood.neural import load_field, to_data, to_model
        from lifelong_ood.neural.idealwords import factorize, slot_factors
        self._slot_id, self._acc = slot_id, composition_accuracy
        self.to_m = lambda X: to_model(X).astype(np.float32)
        self.to_d = lambda X: to_data(X).astype(np.float32)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        remap = {int(k): int(v) for k, v in ck["remap"].items()}
        seen = np.array(sorted(remap))
        field = load_field(ckpt, device=device)
        self.model, self.layout, self.img, self.device = field.model.eval(), field.layout, field.img, device
        self.clip = (-1.0, 1.0)
        lab = slot_factors(seen)
        if (lab[:, 2] == 3).any():
            raise SystemExit("[bench] this backbone saw bg=3; the new-value test needs one that did not")
        C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
        self.fac = factorize(C, lab)
        self.new_f, self.new_j = 2, len(self.fac["V"][2])            # bg, index 3
        self.seen = [(remap[int(s)], tuple(int(x) for x in r)) for s, r in zip(seen, lab)]
        seen_t = {t for _, t in self.seen}
        self.held_comb = [(d, f, b) for d in range(10) for f in range(4) for b in range(3)
                          if (d, f, b) not in seen_t]
        self.new_pairs = [(d, f, 3) for d in range(10) for f in range(4)]
        self.data = load_colormnist(n_per_slot=n_per_slot, seed=seed)
        self.probes = fit_probes(self.data, seed=seed, epochs=probe_epochs)

    def images(self, tup):
        return self.data["X"][np.where(self.data["slot"] == self._slot_id(*tup))[0]]

    def acc(self, S, tup):
        return self._acc(self.probes, S, *tup)


class Shapes3DBench:
    name = "shapes3d"
    names = ("wall", "object", "shape")      # 'wall' = the factor the repo labels floor_hue here

    def __init__(self, seed=0, device="cuda", ckpt=os.path.join(CACHE, "ddpm_s3d_full.pt"),
                 probe_epochs=30, n_probe=20):
        from lifelong_ood.data.factored import decode_slot, load_factored
        from lifelong_ood.data.novel_value import grey_floor
        from lifelong_ood.neural.idealwords import factorize
        from lifelong_ood.neural.ve_bridge import load_ve_field
        self.to_m = lambda X: np.asarray(X, np.float32)               # the bridge is data-space
        self.to_d = lambda X: np.clip(np.asarray(X, np.float32), 0.0, 1.0)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        remap = {int(k): int(v) for k, v in ck["remap"].items()}
        seen = np.array(sorted(remap))
        field, _ = load_ve_field(ckpt, device=device)
        self.model, self.layout, self.img, self.device = field.model.eval(), ck["layout"], ck["img"], device
        self.clip = (0.0, 1.0)
        sp = ck["split"]
        # ALL 120 images of every cell (8 scales x 15 orientations), not the 40 the table was trained
        # on: a held-out cell needs disjoint train / val / reference sets.  Same legacy row mapping,
        # so the same consistent relabelling the backbone saw (test_shapes3d_rows.py).
        d = load_factored("shapes3d", n_per_slot=120, seed=int(sp.get("seed", 0)),
                          value_stride=int(sp["value_stride"]), fix_nuisance=sp["fix_nuisance"],
                          n_holdout=0, rows="legacy")
        self.sizes = [int(v) for v in d["sizes"]]
        self.X, self.F, self.slot = d["X"], np.asarray(d["factors"], int), np.asarray(d["slot"], int)
        lab = np.asarray(decode_slot(seen, self.sizes), int)
        C = np.asarray(ck["ema_cemb"])[[remap[int(s)] for s in seen]]
        self.fac = factorize(C, lab)
        self.new_f, self.new_j = 0, self.sizes[0]                    # wall, index 10 = GREY
        self.seen = [(remap[int(s)], tuple(int(x) for x in r)) for s, r in zip(seen, lab)]
        seen_t = {t for _, t in self.seen}
        allc = {tuple(int(x) for x in r) for r in self.F}
        self.held_comb = sorted(t for t in allc if t not in seen_t)
        self._grey, self._grey_fn = {}, grey_floor
        # probe pool = the first n_probe rows of every cell, concept pool = the rest (disjoint)
        order = np.argsort(self.slot, kind="stable")
        rank = np.empty(len(order), int)
        for s in np.unique(self.slot):
            w = order[self.slot[order] == s]
            rank[w] = np.arange(len(w))
        self._probe_rows = rank < n_probe
        self.new_pairs = [(self.new_j, o, s) for o in range(self.sizes[1]) for s in range(self.sizes[2])]
        self._fit_probes(seed, probe_epochs)

    def _grey_rows(self, o, s, pool):
        """Recoloured rows of every cell (k, o, s) whose wall hue k is separable (k != object, != 0)."""
        key = (o, s, pool)
        if key not in self._grey:
            m = (self.F[:, 1] == o) & (self.F[:, 2] == s) & (self._probe_rows == (pool == "probe"))
            G, keep = self._grey_fn(self.X[m], self.F[m, 0], self.F[m, 1], img=self.img)
            self._grey[key] = G
        return self._grey[key]

    def _fit_probes(self, seed, epochs):
        P = self._probe_rows
        Xp, Fp = self.X[P], self.F[P]
        G, Gy = [], []
        for o in range(self.sizes[1]):
            for s in range(self.sizes[2]):
                g = self._grey_rows(o, s, "probe")
                G.append(g); Gy.append(np.tile([self.new_j, o, s], (len(g), 1)))
        Xa = np.concatenate([Xp, np.concatenate(G)])
        Fa = np.concatenate([Fp, np.concatenate(Gy)])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xa))
        cut = int(0.9 * len(perm))
        tr, va = perm[:cut], perm[cut:]
        self.probes = {}
        for k, (nm, C) in enumerate(zip(self.names, (self.sizes[0] + 1, self.sizes[1], self.sizes[2]))):
            pr = SoftmaxProbe(C, seed=seed, epochs=epochs).fit(Xa[tr], Fa[tr, k])
            self.probes[nm] = pr
            self.probes[f"{nm}_val_acc"] = pr.acc(Xa[va], Fa[va, k])

    def images(self, tup):
        f0, o, s = tup
        if f0 == self.new_j:
            return self._grey_rows(o, s, "concept")
        m = (self.F[:, 0] == f0) & (self.F[:, 1] == o) & (self.F[:, 2] == s) & ~self._probe_rows
        return self.X[m]

    def acc(self, S, tup):
        hits = [self.probes[nm].predict(S) == v for nm, v in zip(self.names, tup)]
        out = {nm: float(h.mean()) for nm, h in zip(self.names, hits)}
        out["joint"] = float(np.logical_and.reduce(hits).mean())
        return out


def load_bench(name, **kw):
    if name == "colormnist":
        return ColorMNISTBench(**kw)
    if name == "shapes3d":
        return Shapes3DBench(**kw)
    raise SystemExit(f"unknown bench {name!r}")
