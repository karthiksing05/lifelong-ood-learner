"""Why do `poe-guided` and `query-only` produce the same images?

They agree on three different backbones -- our ColorMNIST DDPM (N49: F1 0.569/0.363 vs 0.612/0.316),
the raw pretrained CelebA model, and the N47 fine-tune of the paper's own backbone (attr 0.863 vs
0.865, joint 0.500 vs 0.500).  Two explanations are dead: it is not memorization (RESULTS
5e-quinquies) and it is not the selection objective preferring query-seeded candidates (5e-quater --
forbidding them changes nothing).

What remains is that the two teachers may be different in a way the SAMPLER cannot express.  Their
means are genuinely apart: `diag_selection_bias` puts mu_T at 0.26 of a class-spread from the query,
and `query_only_teacher`'s mean IS the query.  So the question is whether that separation survives
Eq. 13.

The suspect is Eq. 14 itself.  It sets w(u) = w0 * mean(abar*Sigma_T + (1-abar)), which is ~w0 at
high noise and ~w0*Sigma_T at low noise.  With the mixture covariance measured at 0.33-0.49x the real
class spread, Sigma_T ~ 0.006, so guidance is worth ~0.008 by the time the image is being decided --
and at high noise, where the weight is large, eps_qT ~ x carries almost no information about mu.
Both arms therefore get a strong pull when there is nothing to say and none when there is, and both
land wherever the unconditional model was going.

This measures the transmission directly, sweeping w0:

    d(means)     how far apart the two TEACHERS are            -- the signal
    d(samples)   how far apart their OUTPUTS are               -- what survives
    ratio        d(samples)/d(means)                           -- the transmission
    joint        composition accuracy of each arm

A ratio near 0 means the sampler is discarding the difference: the teachers disagree and the images
do not.  If raising w0 raises the ratio, the degeneracy is a guidance-strength artifact and Eq. 14 is
under-driving on this backbone; if it does not, the teacher's EXTENT is doing the erasing.

    python tests/diag_poe_vs_queryonly.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, ood_queries                                          # noqa: E402

from lifelong_ood.compose import PAPER_COLORMNIST, query_only_teacher          # noqa: E402
from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes                # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner          # noqa: E402

PAPER_POE = dict(schedule="variance", var_scale=1.0, wmax=2.0, wmin=0.0)


def rmse(A, B):
    return float(np.sqrt(np.mean((np.asarray(A, np.float64) - np.asarray(B, np.float64)) ** 2)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=100, seed=0)
    probes = fit_probes(data, seed=0, epochs=30)
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    L = NeuralLifelongLearner(field, device=dev)
    qs = ood_queries(data, n_slots=3, seed=0)

    print("[pvq] do the two teachers differ, and does the sampler transmit the difference?\n")
    print(f"  {'slot':>5s} {'w0':>5s} {'d(means)':>9s} {'d(samples)':>11s} {'ratio':>7s} "
          f"{'joint poe':>10s} {'joint q-only':>13s}")
    agg = {}
    for q in qs:
        xq = to_model(data["X"][q["query"]]).ravel()
        t_poe = L.compose(xq, **PAPER_COLORMNIST)
        t_q = query_only_teacher(xq)
        dm = rmse(t_poe["mu"], t_q["mu"])
        for w0 in (0.5, 1.2, 3.0, 8.0):
            kw = dict(PAPER_POE, w0=w0)
            Sp = to_data(L.sample_teacher(t_poe, n=12, seed=0, u_start=1.0, **kw)).astype(np.float32)
            Sq = to_data(L.sample_teacher(t_q, n=12, seed=0, u_start=1.0, **kw)).astype(np.float32)
            ds = rmse(Sp, Sq)
            ap = composition_accuracy(probes, Sp, q["d"], q["f"], q["b"])["joint"]
            aq = composition_accuracy(probes, Sq, q["d"], q["f"], q["b"])["joint"]
            print(f"  {q['slot']:5d} {w0:5.1f} {dm:9.4f} {ds:11.4f} {ds / max(dm, 1e-9):7.3f} "
                  f"{ap:10.2f} {aq:13.2f}")
            agg.setdefault(w0, []).append((dm, ds, ap, aq))

    print(f"\n  {'w0':>5s} {'d(means)':>9s} {'d(samples)':>11s} {'ratio':>7s} {'joint poe':>10s} "
          f"{'joint q-only':>13s}   (means over slots)")
    for w0, rows in agg.items():
        m = np.mean(np.array(rows), 0)
        print(f"  {w0:5.1f} {m[0]:9.4f} {m[1]:11.4f} {m[1] / max(m[0], 1e-9):7.3f} {m[2]:10.2f} "
              f"{m[3]:13.2f}")

    print("\n[pvq] ratio << 1 at the paper's w0=1.2 means the teachers disagree and the samples do\n"
          "      not -- Eq. 14 is not transmitting the mean on this backbone. If the ratio climbs\n"
          "      with w0, that is a strength problem; if it stays flat, the covariance is erasing\n"
          "      the difference regardless of how hard the guidance pulls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
