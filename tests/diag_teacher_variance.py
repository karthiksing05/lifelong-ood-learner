"""Why does our `poe-guided` have recall ~0 when the paper reports 80.7%?

Their Table 1 (ColorMNIST, 40 OOD compositions) reports, for PoE:

    Precision 75.5 / 67.0    Recall 80.7 / 69.8    F1 75.3 / 63.0   (Faithfulness / Generalization)

Ours reports recall **0.001**, and the repository currently explains that as inherent -- "the PoE
teacher is a unimodal Gaussian, guiding toward it collapses every trajectory onto its mean" (§5e).
That explanation cannot be right: the paper's teacher is the SAME unimodal Gaussian and does not
collapse.  So the collapse is ours, and a baseline we publish at 0.001 where its authors publish
0.807 is not their method.

Reading the call chain turns up an asymmetry that the §5e explanation hides.  `sample_teacher`
multiplies the teacher covariance by `teacher_var_scale` before drawing anything:

    lifelong_neural.py:43      teacher_var_scale=30.0        <- the default OUR arm runs at
    n9_showdown.py:252         var_scale=1.0                 <- what `poe-guided` is given

So the two arms are not the same method evaluated differently; they are given teacher covariances a
factor of 30 apart, and covariance is precisely the quantity that decides sample spread.  Diversity
is the axis `poe-guided` loses on (recall 0.001 vs our 0.766) and the axis our F1 win is built from.
If the gap moves with `var_scale`, then our headline composition result is a knob the baseline was
not given, and the honest table has to either give it to them or take it from us.

This runs both arms across both settings of that one number, scored with n9's own metrics, so the
question is decided by measurement instead of by which default each call site happened to inherit.

    python tests/diag_teacher_variance.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, ood_queries                                          # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,              # noqa: E402
                                precision_recall, vendi_score)
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner                 # noqa: E402

# exactly n9's poe-guided call, minus the one number under test
PAPER_POE = dict(schedule="variance", w0=1.2, wmax=2.0, wmin=0.0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=100, seed=0)
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    probes = fit_probes(data, seed=0, epochs=30)
    L = NeuralLifelongLearner(field, device=dev)
    qs = ood_queries(data, n_slots=4, seed=0)

    print("[tvar] paper's PoE on this benchmark: recall 80.7 / F1 75.3.  We publish recall 0.001.\n")

    # ── 1. is the teacher covariance a plausible description of the class? ────
    print("  -- teacher extent vs the class it represents --")
    print(f"  {'slot':>5s} {'sd(mixture)':>12s} {'sd(poe Eq.7)':>13s} {'sd(real)':>9s} "
          f"{'mix/real':>9s} {'x30/real':>9s}")
    teachers = {}
    for q in qs:
        xq = to_model(data["X"][q["query"]]).ravel()
        t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120)
        teachers[q["slot"]] = (xq, t, q)
        R = to_model(data["X"][q["rest"][:100]])
        sr = float(np.sqrt(np.mean(np.var(R, axis=0))))
        sm = float(np.sqrt(np.mean(t["var_mixture"])))
        sp = float(np.sqrt(np.mean(t["var_poe"])))
        print(f"  {q['slot']:5d} {sm:12.4f} {sp:13.4f} {sr:9.4f} {sm/sr:9.3f} "
              f"{sm*np.sqrt(30.0)/sr:9.3f}")

    # ── 2. the asymmetry itself ───────────────────────────────────────────────
    print("\n  -- the same method at the two var_scale values the two arms actually run at --")
    print(f"  {'var_scale':>10s} {'joint acc':>10s} {'recall':>8s} {'precision':>10s} {'F1':>7s} "
          f"{'vendi':>7s}")
    for vs in (1.0, 30.0):
        acc, rec, pre, f1s, ven = [], [], [], [], []
        for slot, (xq, t, q) in teachers.items():
            S = to_data(L.sample_teacher(t, n=24, seed=slot, var_scale=vs,
                                         **PAPER_POE)).astype(np.float32)
            R = data["X"][q["rest"][:100]]
            pr = precision_recall(S, R)
            a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
            acc.append(a["joint"]); rec.append(pr["recall"]); pre.append(pr["precision"])
            f1s.append(pr["f1"]); ven.append(vendi_score(S))
        m = [float(np.mean(v)) for v in (acc, rec, pre, f1s, ven)]
        tag = "  <- poe-guided" if vs == 1.0 else "  <- ours"
        print(f"  {vs:10.1f} {m[0]:10.3f} {m[1]:8.3f} {m[2]:10.3f} {m[3]:7.3f} {m[4]:7.3f}{tag}")

    print("\n[tvar] if recall tracks var_scale, then `poe-guided`'s 0.001 is our sampling default\n"
          "       and not Wang et al.'s method, and the F1 column of the composition table is\n"
          "       reporting a knob.  Both arms then have to be run at the same value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
