"""Our pool is more dispersed than the real class. Where is the trade, and can it be moved?

N49 under the paper's protocol: at a matched 4-query budget our Generalization recall is 0.797
against PoE's 0.137, and our precision is **0.195 against their 0.547**. The coverage claim is solid;
"we match the class distribution" is not, and this measures the shape of that gap rather than
leaving it as a one-line caveat.

The knob is the guidance used when the MANIFOLD generates our pool (`sample_factored_pool`), which
is what makes our route different from guiding an analytic teacher.

**The premise this file was written on turned out to be wrong, and the result is better for it.**
The expectation was a frontier: higher guidance concentrates the pool on the conditional's mode, so
precision up and coverage down. Measured, there is no such trade -- raising the guidance costs
precision *and* composition accuracy while buying almost no coverage:

    guidance   G.prec   G.rec   G.F1   joint
        1.0     0.156   0.818  0.228   0.297
        1.5     0.219   0.833  0.303   0.469   <- the calibrated value
        2.5     0.078   0.887  0.133   0.094
        4.0     0.062   0.443  0.059   0.000

So the precision gap against their PoE (0.219 against 0.670) is **not a knob left in the wrong
place**; it is not reachable from this direction at all. And the label-free calibration -- which
chose 1.5 on TRAINED slots, without seeing any number in this table -- lands on the best held-out
point of the four. That is a check on the calibration procedure, not a tuning result.

**This sweep does not choose a value, and must not.** `n9`/`n49` calibrate that guidance label-free
on TRAINED slots (`backbone_gains` / the `--guidance <= 0` path), precisely so no held-out number
influences it. Picking the setting that maximizes held-out F1 would be tuning on the test set, and it
would silently convert an honest result into a fitted one. The calibrated value is marked in the
output; everything else is reported so the trade can be seen, not so a winner can be picked.

    python tests/diag_ours_frontier.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, backbone_gains, ood_queries                          # noqa: E402

from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,              # noqa: E402
                                paper_metrics, vendi_score)
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                        # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner          # noqa: E402

CALIBRATED = 1.5          # what n9/n49 use, chosen on TRAINED slots only
PAPER_POE_G = dict(precision=0.670, recall=0.698, f1=0.630)   # their Table 1, Generalization


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=160, seed=0)
    probes = fit_probes(data, seed=0, epochs=30)
    ckpt = os.path.join(CACHE, "ddpm_colormnist.pt")
    field = load_field(ckpt, device=dev)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in slots_seen]]
    gains = backbone_gains(field, C, slots_seen, probes, data, seed=0, steps=50, n=16)

    qs = ood_queries(data, n_slots=4, seed=0)
    print(f"[frontier] pool guidance sweep for `ours`; {CALIBRATED} is the CALIBRATED value "
          f"(chosen on trained slots, not here)\n")
    print(f"  {'guidance':>9s} {'G.prec':>7s} {'G.rec':>7s} {'G.F1':>7s} {'joint':>6s} "
          f"{'vendi':>6s}")
    for g in (1.0, CALIBRATED, 2.5, 4.0):
        P, A = [], []
        for qi, q in enumerate(qs):
            xq = to_model(data["X"][q["query"]]).ravel()
            L = NeuralLifelongLearner(field, device=dev, seed=0, flat=True)
            L.attach_factorization(C, slot_factors(slots_seen), gains=gains)
            L.learn_concept_factored(xq, f"c{q['slot']}", n_pool=48, epochs=10, inv_steps=400,
                                     seed=qi, guidance=g)
            S = to_data(L.sample_concept(f"c{q['slot']}", n=16, steps=50, seed=qi))
            S = np.asarray(S, np.float32)
            R = data["X"][q["rest"][:100]]
            m = paper_metrics(S, R, device=dev)
            P.append((m["precision"], m["recall"], m["f1"], vendi_score(S)))
            A.append(composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"])
        m = np.mean(np.array(P), 0)
        tag = "   <- calibrated" if g == CALIBRATED else ""
        print(f"  {g:9.1f} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f} {float(np.mean(A)):6.3f} "
              f"{m[3]:6.2f}{tag}")

    print(f"\n  {'paper PoE':>9s} {PAPER_POE_G['precision']:7.3f} {PAPER_POE_G['recall']:7.3f} "
          f"{PAPER_POE_G['f1']:7.3f}      --      --   (their Table 1, Generalization)")
    print("\n[frontier] read this as a curve, not a leaderboard: the calibrated point is the one the\n"
          "           method actually runs at, and it was chosen without seeing any of these numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
