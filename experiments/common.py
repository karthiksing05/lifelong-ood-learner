"""Shared experiment plumbing: build-or-load the base learner and the attribute probes.

Every experiment starts from the same object -- a GaussianCobweb fit incrementally on the 120 SEEN
ColorMNIST slots -- so E1/E2/E3 results are about one manifold, not three.  The fitted tree and the
probes are cached to results/cache/ keyed by (n_per_slot, seed, prior_var).
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood.datasets import load_colormnist                     # noqa: E402
from lifelong_ood.evals import fit_probes                             # noqa: E402
from lifelong_ood.lifelong import LifelongLearner                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
CACHE = os.path.join(RESULTS, "cache")


def base_parser(**over):
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slot", type=int, default=over.get("n_per_slot", 40))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prior-var", type=float, default=0.05)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--mode", choices=["leaves", "frontier"], default="leaves")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--quick", action="store_true", help="tiny settings for a smoke run")
    return p


def fresh_learner(args, data):
    """Build (or load from cache when the backend allows pickling) a learner fit on the seen slots.

    The cobweb-private backend is a live C++ object: it neither pickles nor deepcopies, so with that
    backend the tree is refit from the stream each call -- C++ ifit is fast enough that this is
    cheaper than any serialization scheme, and it is also what E3 needs to give each consolidation
    strategy its own untouched base."""
    os.makedirs(CACHE, exist_ok=True)
    key = f"n{args.n_per_slot}_s{args.seed}_pv{args.prior_var}"
    tpath = os.path.join(CACHE, f"tree_{key}.pkl")
    if os.path.exists(tpath):
        with open(tpath, "rb") as fh:
            tree = pickle.load(fh)
        print(f"[common] loaded cached tree {tpath} ({tree.n_nodes} nodes)")
        return LifelongLearner(tree=tree, min_count=args.min_count, mode=args.mode,
                               steps=args.steps, seed=args.seed)
    learner = LifelongLearner(prior_var=args.prior_var, min_count=args.min_count,
                              mode=args.mode, steps=args.steps, seed=args.seed)
    Xseen = data["X"][data["seen"]]
    print(f"[common] fitting {getattr(learner.tree, 'backend', 'numpy')} tree "
          f"on {len(Xseen)} seen instances ...", flush=True)
    learner.fit(Xseen, verbose=True)
    print(f"[common] tree: {learner.tree.n_nodes} nodes")
    try:
        with open(tpath, "wb") as fh:
            pickle.dump(learner.tree, fh)
        print(f"[common] cached {tpath}")
    except TypeError:
        os.path.exists(tpath) and os.remove(tpath)
    return learner


def load_all(args):
    """-> (data dict, learner fit on seen slots, probes)."""
    os.makedirs(CACHE, exist_ok=True)
    if args.quick:
        args.n_per_slot = min(args.n_per_slot, 12)
    key = f"n{args.n_per_slot}_s{args.seed}_pv{args.prior_var}"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    learner = fresh_learner(args, data)

    ppath = os.path.join(CACHE, f"probes_{key}.pkl")
    if os.path.exists(ppath):
        with open(ppath, "rb") as fh:
            probes = pickle.load(fh)
    else:
        print("[common] training attribute probes ...", flush=True)
        probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
        with open(ppath, "wb") as fh:
            pickle.dump(probes, fh)
    print(f"[common] probe val acc: digit={probes['digit_val_acc']:.3f} "
          f"fg={probes['fg_val_acc']:.3f} bg={probes['bg_val_acc']:.3f}")
    return data, learner, probes


def ood_queries(data, n_slots=None, seed=0):
    """One held-out query per OOD slot (plus the rest of the slot as a generalization reference)."""
    rng = np.random.default_rng(seed)
    slots = np.unique(data["slot"][~data["seen"]])
    if n_slots:
        slots = rng.permutation(slots)[:n_slots]
    out = []
    for s in slots:
        idx = np.where(data["slot"] == s)[0]
        q = idx[rng.integers(len(idx))]
        out.append(dict(slot=int(s), query=q, rest=idx[idx != q],
                        d=int(data["digit"][q]), f=int(data["fg"][q]), b=int(data["bg"][q])))
    return out


def backbone_gains(field, C, slots_seen, probes, data, factor=2, grid=(1.0, 1.5, 2.0, 2.5, 3.0),
                   n_slots=6, n=16, steps=50, seed=0, verbose=True):
    """Per-factor gain calibrated on THIS backbone, using seen slots only.

    The gain corrects the model's under-response to a factor -- background, in ColorMNIST.  It was
    previously hardcoded to 2.0, which was tuned against one checkpoint; a retrained backbone has a
    different embedding table (ideal-words R^2 0.816 -> 0.746 across two runs of the same command)
    and the constant does not transfer.  Calibrating here makes the composer a function of the
    backbone it is given rather than of a checkpoint that happened to exist.

    Only SEEN slots are used, so no held-out label is touched: composing a training slot from the
    factorization and asking whether the model reproduces it is label-free.
    """
    from lifelong_ood.evals import composition_accuracy
    from lifelong_ood.neural.idealwords import calibrate_gains, slot_factors

    def score_fn(S, d, f, b):
        return composition_accuracy(probes, S, d, f, b)["joint"]

    from lifelong_ood.neural.idealwords import factorize
    fac = factorize(C, slot_factors(slots_seen))
    g, sc = calibrate_gains(field, fac, probes, data, slots_seen, grid=grid, factor=factor,
                            n_slots=n_slots, n=n, steps=steps, seed=seed, score_fn=score_fn)
    gains = [1.0] * len(fac["V"])
    gains[factor] = g
    if verbose:
        print(f"[calib] ideal-words R^2 {fac['r2']:.3f}; factor-{factor} gain {g} "
              f"(seen-slot joint accuracy {sc:.2f}) -> gains {gains}", flush=True)
    return gains
