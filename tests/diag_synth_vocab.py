"""Does `synthetic_vocab_conds` actually manufacture something the router reads as NEW VOCABULARY?

It was supposed to supply the positive population that `calibrate` never had, so the threshold could
sit BETWEEN two manufactured classes instead of inside one.  Run in N18 it did the opposite: the
cost-minimizing threshold came out at 0.00352 against the quantile rule's 0.01142, with manufactured
overspend 1.00 -- every real combination sent to the paid route.

That can only happen if the manufactured "new vocabulary" samples score LOWER on `s_vocab` than the
manufactured combinations do.  In other words the positive class is not positive, and pushing the
threshold below it to avoid underspend drags it below the negatives too.

Two candidate reasons, and they call for opposite fixes:

  * THE EMBEDDING IS NOT ACTUALLY OFF-BASIS.  Projecting out the span of a factor's centred
    value-vectors leaves a direction with no coordinate in that factor -- but `compose` adds `mu`
    and the other factors, and the result may still land inside the basis's overall span, so the
    router can explain it with an existing conditioning.
  * THE SAMPLE IS OFF-MANIFOLD JUNK.  Conditioning on a vector the model never saw can produce a
    generic blur, and a blur is EASY to denoise -- `s_vocab` is a minimum of denoising error over
    conditionings, so an uninformative image scores LOW, not high. This would make the manufactured
    positives look maximally in-distribution, which is exactly the observed failure.

This prints the `s_vocab` distribution of every population on one axis -- the two manufactured ones
and the three REAL labelled ones (labels used for reporting only) -- so the manufacture can be judged
against the thing it is standing in for, at several scales of the off-basis direction.

    python tests/diag_synth_vocab.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize, sample_factored_pool, slot_factors
from lifelong_ood.neural.router import (basis_assignments, route_signals,      # noqa: E402
                                        synthetic_vocab_conds)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = os.path.join(CACHE, "ddpm_colormnist_nobg3.pt")
    if not os.path.exists(ck_path):
        print(f"[synth] missing {ck_path}")
        return 1
    data = load_colormnist(n_per_slot=40, seed=0)
    field = load_field(ck_path, device=dev)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_seen]]
    fac = factorize(C, slot_factors(slots_seen))
    basis_conds, _ = basis_assignments(fac, sizes=fac["sizes"])
    print(f"[synth] basis R^2 {fac['r2']:.3f}, factor sizes {fac['sizes']}\n")

    def sig(x01):
        return route_signals(field, to_model(x01).ravel(), C, basis_conds, n_noise=4,
                             seed=0)["s_vocab"]

    trained = set(int(s) for s in slots_seen)
    pops = {}
    # real populations, by their true label -- reported, never used to calibrate
    for name, keep in (("real in-distribution", lambda s: s in trained),
                       ("real new-combination", lambda s: s not in trained
                        and int(data["bg"][data["slot"] == s][0]) != 3),
                       ("real new-vocabulary", lambda s: s not in trained
                        and int(data["bg"][data["slot"] == s][0]) == 3)):
        slots = [s for s in np.unique(data["slot"]) if keep(int(s))][:10]
        vals = []
        for s in slots:
            i = np.where(data["slot"] == s)[0][0]
            vals.append(sig(data["X"][i]))
        pops[name] = np.array(vals)

    # manufactured: samples from TRAINED slots.  This is the missing measurement, and it is the one
    # that makes the calibration correctable.  `calibrate` takes a quantile of the MANUFACTURED
    # combination population, but the diagnostic below shows that population sits at 0.00725 against
    # the real one's 0.00977 -- model samples are systematically EASIER to denoise than photographs,
    # so a quantile of them lands ~26% low and 33% of real combinations end up above the threshold.
    # The system can measure that optimism without any label: it has its own samples from trained
    # slots AND real images of those same slots, and the ratio between them is the correction.
    vals = []
    for s_ in list(trained)[:10]:
        d_, f_, b_ = int(s_) // 16, (int(s_) % 16) // 4, int(s_) % 4
        c = compose(fac, (d_, f_, min(b_, fac["sizes"][2] - 1)))
        S = to_data(sample_factored_pool(field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                                         n=1, steps=50, seed=0))
        vals.append(sig(S[0]))
    pops["manufactured in-distribution"] = np.array(vals)

    # manufactured: combinations the basis CAN express, over untrained slots
    untrained_known = [(d, f, b) for d in range(10) for f in range(4) for b in range(fac["sizes"][2])
                       if (d * 16 + f * 4 + b) not in trained][:10]
    vals = []
    for combo in untrained_known:
        c = compose(fac, combo)
        S = to_data(sample_factored_pool(field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                                         n=1, steps=50, seed=0))
        vals.append(sig(S[0]))
    pops["manufactured combination"] = np.array(vals)

    # manufactured: values the basis CANNOT express, at several magnitudes
    for sc in (0.5, 1.0, 2.0, 4.0):
        vals = []
        for cv in synthetic_vocab_conds(fac, n=10, seed=0, scale=sc):
            S = to_data(sample_factored_pool(field, torch.as_tensor(cv, dtype=torch.float32,
                                                                    device=dev),
                                             n=1, steps=50, seed=0))
            vals.append(sig(S[0]))
        pops[f"manufactured vocab x{sc:g}"] = np.array(vals)

    print(f"  {'population':30s} {'n':>3s} {'mean':>9s} {'median':>9s} {'min':>9s} {'max':>9s}")
    for k, v in pops.items():
        print(f"  {k:30s} {len(v):3d} {v.mean():9.5f} {np.median(v):9.5f} {v.min():9.5f} "
              f"{v.max():9.5f}")
    mi, ri = pops["manufactured in-distribution"], pops["real in-distribution"]
    print(f"\n  SELF-SAMPLE OPTIMISM: manufactured in-distribution {mi.mean():.5f} vs real "
          f"{ri.mean():.5f}  ->  ratio {mi.mean() / max(ri.mean(), 1e-9):.3f}")
    mc = pops["manufactured combination"]
    q90 = float(np.quantile(mc, 0.90))
    corrected = q90 * (ri.mean() / max(mi.mean(), 1e-9))
    print(f"  the current rule: quantile(manufactured combos, 0.90) = {q90:.5f}")
    print(f"  corrected for that optimism:                          {corrected:.5f}")
    rc, rv = pops["real new-combination"], pops["real new-vocabulary"]
    print(f"\n  the gap to straddle: real combinations {rc.mean():.5f} -> real vocabulary "
          f"{rv.mean():.5f}")
    print(f"  real combinations span up to {rc.max():.5f}; real vocabulary starts at {rv.min():.5f} "
          f"-- so ANY threshold in ({rc.max():.5f}, {rv.min():.5f}) is error-free on this sample.")
    print("  a manufactured vocabulary population is USABLE only if it sits at or above the real\n"
          "  vocabulary mean; below the real COMBINATION mean it is worse than having none, because\n"
          "  minimizing cost against it pushes the threshold underneath the negatives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
