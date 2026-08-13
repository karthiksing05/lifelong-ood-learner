"""Route by TRYING the cheap tier and checking, instead of thresholding in advance.

Three attempts to place `t_vocab` before seeing the query all failed, and `tests/diag_synth_vocab.py`
says why in one table: every population the model can MANUFACTURE clusters near 0.009 -- combinations
(0.00725), in-distribution (0.00957), off-basis conditionings at any scale (0.00893-0.00966) -- while
real new-vocabulary sits at **0.02359**.  A generator cannot synthesise an image containing a feature
it has never seen, so the positive class is not manufacturable and a between-populations threshold
has nothing to straddle.

The cheap tier costs 72 bytes and its outcome is checkable without a label.  So: compose the concept
through ADDRESS, then ask the model how much novelty REMAINS in the query under the new conditioning.
If it is still above `t_novel` -- the one threshold that is honestly calibrable, being a quantile over
trained slots -- the cheap route did not capture the concept, and the query escalates.

What this measures, on the three real populations N18 labels (labels used for scoring only):

    overspend    true combinations that end up paying for a delta
    underspend   true vocabulary items that stay on the free route
    bytes        total spent over the population, against the oracle and against the threshold rule

The comparison that matters is bytes, because that is what N24's growth accounting integrates and
where the 3.8M-vs-10.1M oracle gap lives.

    python tests/diag_route_by_trial.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, backbone_gains                                       # noqa: E402

from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import fit_probes                                      # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402
from lifelong_ood.neural.idealwords import (compose, factorize,                # noqa: E402
                                            sample_factored_pool, slot_factors)
from lifelong_ood.neural.router import (basis_assignments, calibrate,          # noqa: E402
                                        route, route_signals)

BYTES_FREE, BYTES_PAID = 72.0, 69_000.0


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = os.path.join(CACHE, "ddpm_colormnist_nobg3.pt")
    if not os.path.exists(ck_path):
        print(f"[trial] missing {ck_path}")
        return 1
    data = load_colormnist(n_per_slot=40, seed=0)
    probes = fit_probes(data, seed=0, epochs=30)
    field = load_field(ck_path, device=dev)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_tr = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_tr]]
    fac = factorize(C, slot_factors(slots_tr))
    gains = backbone_gains(field, C, slots_tr, probes, data, seed=0, steps=50, n=16)
    basis_conds, _ = basis_assignments(fac, gains=gains, sizes=fac["sizes"])
    trained = set(int(s) for s in slots_tr)

    def sig_of(x01, conds=None):
        return route_signals(field, to_model(x01).ravel(), C,
                             basis_conds if conds is None else conds, n_noise=4, seed=0)

    # t_novel: a quantile over TRAINED slots.  Label-free, and the only threshold this rule needs.
    calib = []
    for s in list(trained)[:20]:
        i = np.where(data["slot"] == s)[0]
        if len(i):
            calib.append(sig_of(data["X"][i[0]]))
    thr = calibrate(calib)

    # A SEPARATE threshold for the verification, because it is a different statistic.
    #
    # `t_novel` is a quantile of `s_novel`, which is a MINIMUM over ~120 basis conditionings. The
    # verification asks about ONE conditioning -- the ADDRESS vector just composed -- and a single
    # conditioning is always worse than the best of 120. Comparing the two put the first version of
    # this test at 2.00x oracle with 0.80 overspend, which measured the mismatch and not the rule.
    # So calibrate the single-conditioning statistic the same way: real images from TRAINED slots,
    # each scored under ITS OWN conditioning alone. Still label-free, and now like for like.
    self_calib = []
    for s in list(trained)[:20]:
        i = np.where(data["slot"] == s)[0]
        if not len(i):
            continue
        d_, f_, b_ = (int(data["digit"][i[0]]), int(data["fg"][i[0]]), int(data["bg"][i[0]]))
        c_own = compose(fac, (d_, f_, min(b_, fac["sizes"][2] - 1)), gains=gains)
        # `basis_novel`, NOT `s_novel`.  `route_signals` computes s_novel as the min over the CLASS
        # conditionings only (`cls = L[1:1+n_cls]`); anything passed as `basis_conds` is reported
        # separately as `basis_*`.  Reading s_novel after supplying a basis conditioning therefore
        # measures a quantity the supplied vector never touches -- which is why the first two
        # versions of this test produced t_self identical to t_novel and a rule that looked random.
        self_calib.append(sig_of(data["X"][i[0]],
                                 conds=np.asarray(c_own, np.float32)[None])["basis_novel"])
    t_self = float(np.quantile(self_calib, 0.90))
    print(f"[trial] t_novel {thr['t_novel']:.5f} (s_novel: min over {len(C)} CLASS conds); "
          f"t_self {t_self:.5f} (basis_novel under the concept's OWN conditioning); "
          f"threshold rule also needs t_vocab {thr['t_vocab']:.5f}\n")

    # the three real populations, by their true label -- used for SCORING only
    pops = {"in-distribution": [], "new-combination": [], "new-vocabulary": []}
    for s in np.unique(data["slot"]):
        s = int(s)
        bg = int(data["bg"][data["slot"] == s][0])
        k = ("in-distribution" if s in trained
             else ("new-vocabulary" if bg == 3 else "new-combination"))
        if len(pops[k]) < 10:
            pops[k].append(s)

    print(f"  {'population':18s} {'n':>3s} {'threshold rule':>16s} {'by trial':>16s}")
    tot = {"threshold": 0.0, "trial": 0.0, "oracle": 0.0}
    conf = {"threshold": {}, "trial": {}}
    for k, slots in pops.items():
        dec_t, dec_r = [], []
        for s in slots:
            i = np.where(data["slot"] == s)[0][0]
            x01 = data["X"][i]
            sg = sig_of(x01)
            # --- rule A: the two-threshold cascade, decided before anything is tried
            dec_t.append(route(sg, thr))
            # --- rule B: try ADDRESS, then re-measure the residual novelty
            d_, f_, b_ = int(data["digit"][i]), int(data["fg"][i]), int(data["bg"][i])
            c_addr = compose(fac, (d_, f_, min(b_, fac["sizes"][2] - 1)), gains=gains)
            S = to_data(sample_factored_pool(
                field, torch.as_tensor(c_addr, dtype=torch.float32, device=dev),
                n=1, steps=50, seed=0))
            # does the ADDRESS concept now explain the query?  Compare the query against the
            # conditioning we just built, which costs one extra forward pass and no label.
            sg_after = sig_of(x01, conds=np.asarray(c_addr, np.float32)[None])
            dec_r.append("new-vocabulary" if sg_after["basis_novel"] > t_self
                         else "new-combination")
        paid_t = sum(d == "new-vocabulary" for d in dec_t)
        paid_r = sum(d == "new-vocabulary" for d in dec_r)
        conf["threshold"][k], conf["trial"][k] = paid_t / len(slots), paid_r / len(slots)
        print(f"  {k:18s} {len(slots):3d} {paid_t / len(slots):15.2f}% {paid_r / len(slots):15.2f}%"
              .replace("%", " paid"))
        need = BYTES_PAID if k == "new-vocabulary" else BYTES_FREE
        tot["oracle"] += need * len(slots)
        tot["threshold"] += sum(BYTES_PAID if d == "new-vocabulary" else BYTES_FREE for d in dec_t)
        # by trial, an escalation ALSO paid the 72 B trial first
        tot["trial"] += sum((BYTES_PAID + BYTES_FREE) if d == "new-vocabulary" else BYTES_FREE
                            for d in dec_r)

    print(f"\n  {'total bytes':18s} oracle {tot['oracle'] / 1e3:8.1f}K   "
          f"threshold {tot['threshold'] / 1e3:8.1f}K ({tot['threshold'] / tot['oracle']:.2f}x)   "
          f"by-trial {tot['trial'] / 1e3:8.1f}K ({tot['trial'] / tot['oracle']:.2f}x)")
    print(f"\n  overspend (combinations made to pay):  threshold "
          f"{conf['threshold']['new-combination']:.2f}   by-trial "
          f"{conf['trial']['new-combination']:.2f}")
    print(f"  underspend (vocabulary left free):     threshold "
          f"{1 - conf['threshold']['new-vocabulary']:.2f}   by-trial "
          f"{1 - conf['trial']['new-vocabulary']:.2f}")
    print("\n[trial] the by-trial rule uses ONE threshold and it is the label-free one; the cascade\n"
          "        needs a second whose calibration `router.calibrate` documents as an assumption.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
