"""Is our `poe-guided` arm the published method, or the published method mis-configured?

The case-study grids show `poe-guided` emitting the query plus speckle, and it scores recall 0.000.
Wang et al. report recall **80.7%** for the same method on the same benchmark, and reserve the
salt-and-pepper failure for their WEAKEST baseline (query-only).  Either their method genuinely does
not survive our backbone, or we are running it in a configuration they would not recognise -- and if
it is the second, every "we beat the published method" claim is against a strawman.

There is a specific, mechanical suspect.  Their Eq. 12 gives the teacher's epsilon as

    eps_qT = sqrt(1-abar) * (abar*Sigma_T + (1-abar) I)^-1 * (x_t - sqrt(abar) mu_T)

whose precision factor BLOWS UP as abar -> 1 (low noise) when Sigma_T is small.  Their Eq. 14 exists
to cancel exactly that: `w(t) = min(w0 * tr(Sigma_t)/d, wmax)` shrinks guidance as the teacher
concentrates.  Our default is `schedule="anneal"`, this repository's own schedule, which does the
OPPOSITE -- it rises to wmax as u -> 0.  Worse, the justification for annealing (recorded in
`cfg_weight`'s docstring) is about the ANALYTIC Gaussian mixture having no density at an unseen
pairing; a UNet shares weights across colours and does not have that hole, so the rationale does not
transfer to the backbone we run the baseline on.

This sweeps the baseline's configuration and scores it by THEIR metric (precision/recall/F1), so the
comparison in the paper is against the published method at its best rather than at our default.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, RESULTS, ood_queries                          # noqa: E402

from lifelong_ood.datasets import load_colormnist, rgb_img              # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry)
from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model  # noqa: E402

N_CASES = int(os.environ.get("DIAG_CASES", 4))
N_SAMP = int(os.environ.get("DIAG_N", 24))
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# (label, schedule, var_scale, w0, wmax, wmin) -- the paper's own setting first
CONFIGS = [
    ("paper Eq.14 (variance), raw Sigma", "variance", 1.0, 1.2, 2.0, 0.0),
    ("paper Eq.14, var x30",              "variance", 30.0, 1.2, 2.0, 0.0),
    ("paper Eq.14, var x100",             "variance", 100.0, 1.2, 2.0, 0.0),
    ("const w=1.0, var x30",              "const",    30.0, 1.0, 1.0, 1.0),
    ("const w=1.5, var x100",             "const",    100.0, 1.5, 1.5, 1.5),
    ("ours: anneal 0.8->2.0, var x30",    "anneal",   30.0, 1.2, 2.0, 0.8),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=100, seed=0)
    probes = fit_probes(data, seed=0, epochs=30)
    ckpt = os.path.join(CACHE, "ddpm_colormnist.pt")
    field = load_field(ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev)
    qs = ood_queries(data, n_slots=N_CASES, seed=0)
    print(f"[diag] {len(qs)} held-out slots, {N_SAMP} samples per config, device {dev}", flush=True)

    # teachers once; the configuration under test is the SAMPLER, not the discovery
    teachers, refs = [], []
    for q in qs:
        xq = to_model(data["X"][q["query"]]).ravel()
        t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120)
        teachers.append(t)
        refs.append(data["X"][q["rest"][:100]].astype(np.float32))
    v = np.concatenate([t["var"] for t in teachers])
    print(f"[diag] teacher Sigma_T: mean {v.mean():.2e}, median {np.median(v):.2e}, "
          f"fraction at the 1e-4 floor {np.mean(v <= 1.01e-4):.3f}", flush=True)
    print(f"[diag] selected experts per teacher: {[len(t['sel']) for t in teachers]}", flush=True)

    rows, panels = [], {}
    for label, sched, vs, w0, wmax, wmin in CONFIGS:
        acc, rec, f1s, div = [], [], [], []
        for i, (q, t) in enumerate(zip(qs, teachers)):
            S = to_data(L.sample_teacher(t, n=N_SAMP, seed=i, var_scale=vs, schedule=sched,
                                         w0=w0, wmax=wmax, wmin=wmin)).astype(np.float32)
            a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
            pr = precision_recall(S, refs[i])
            g = sample_geometry(S, refs[i][0], refs[i])
            acc.append(a["joint"]); rec.append(pr["recall"]); f1s.append(pr["f1"])
            div.append(g.get("content_ratio", np.nan))
            if i == 0:
                panels[label] = S[:8]
        rows.append((label, float(np.mean(acc)), float(np.mean(rec)), float(np.nanmean(f1s)),
                     float(np.nanmean(div))))
        print(f"  {label:36s} acc {rows[-1][1]:.3f}  recall {rows[-1][2]:.3f}  "
              f"F1 {rows[-1][3]:.3f}  div {rows[-1][4]:.2f}", flush=True)

    # the real-data ceiling, for reference
    r0 = refs[0]
    pr = precision_recall(r0[:N_SAMP], r0[N_SAMP:])
    print(f"\n  {'real held-out images (ceiling)':36s} recall {pr['recall']:.3f}  F1 {pr['f1']:.3f}")

    best = max(rows, key=lambda r: r[3])
    print(f"\n  BEST by their own metric (F1): {best[0]}  -> F1 {best[3]:.3f}, recall {best[2]:.3f}")
    ours = [r for r in rows if r[0].startswith("ours")][0]
    print(f"  our current default:           {ours[0]}  -> F1 {ours[3]:.3f}, recall {ours[2]:.3f}")
    if best[3] > ours[3] + 0.02:
        print("\n  => the baseline was mis-configured.  n9's `poe-guided` arm must use the best "
              "config,\n     or the comparison is against a strawman.")
    else:
        print("\n  => the baseline is not being handicapped by the sampler; the failure is real.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [r[0] for r in rows]
    fig, axes = plt.subplots(len(labs) + 1, 8, figsize=(9, 1.15 * (len(labs) + 1)))
    for c in range(8):
        axes[0][c].imshow(rgb_img(refs[0][c])); axes[0][c].set_xticks([]); axes[0][c].set_yticks([])
    axes[0][0].set_ylabel("real", fontsize=6, rotation=0, ha="right", va="center")
    for r_, lab in enumerate(labs, start=1):
        S = panels[lab]
        for c in range(8):
            if c < len(S):
                axes[r_][c].imshow(rgb_img(S[c]))
            axes[r_][c].set_xticks([]); axes[r_][c].set_yticks([])
        axes[r_][0].set_ylabel(lab, fontsize=5, rotation=0, ha="right", va="center")
    fig.suptitle("The published method's sampler, swept.  Row label = configuration; the paper's own\n"
                 "Eq. 14 is the first three.", fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "diag_poe_faithful.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
