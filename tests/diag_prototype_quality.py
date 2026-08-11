"""Why are the CelebA prototypes garbage when the paper's are faces?

Three explanations have been offered for the neon prototypes and all three are refuted by
measurement already in the repository:

  * "Adam runs away"      -- max |x0| is 1.5-2.0, not the 9 predicted.
  * "the x0 estimator"    -- un-clipped Tweedie measures 0.2975 out of range against the
                             rescaling's 0.2960.  Identical.
  * "it is only display"  -- 11% of coordinates slightly out of range clips to mild highlights.
                             It does not produce whole magenta fields.  This one was mine and it
                             was a rationalization: it let the investigation stop.

So look at the thing itself.  `candidates_ascent` runs ascent at EVERY level of the paper's grid
(CelebA: u = 0.05 ... 0.50) and pools them; submodular selection then picks K=3 from that pool.  The
pooled candidates are never inspected per level, so if the high-noise end returns junk, it enters
the pool, can be selected, and is what the panels show -- and it also becomes a prototype MEAN in
the PoE teacher, which is a correctness issue and not a rendering one.

This renders the ascent output at each noise level separately, with:
    frac |x0|>1   how far out of gamut
    eps error     the model's own error at that candidate, against the real query's as the bar
and writes a grid so the failure can be seen per level rather than after pooling.

    python tests/diag_prototype_quality.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from lifelong_ood.score import cosine_abar                                     # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    from lifelong_ood.celeba import load_celeba
    from lifelong_ood.neural import to_data, to_model
    from lifelong_ood.neural.pretrained import load_pretrained
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, img = load_pretrained("google/ddpm-celebahq-256", device=dev, batch=4)
    data = load_celeba(n_per_class=4, img=img, seed=0, min_count=25, max_rows=4000)
    xq = to_model(np.asarray(data["X"])[:1]).ravel()

    def eps_err(x0, u, seed=1):
        r = np.random.default_rng(seed)
        a = float(cosine_abar(u))
        e = r.standard_normal(np.atleast_2d(x0).shape)
        xt = np.sqrt(a) * np.atleast_2d(x0) + np.sqrt(1 - a) * e
        return float(np.mean((field.eps(xt, u) - e) ** 2))

    # HOW FAR the ascent travels is the lever the numbers point at.  Every level already BEATS the
    # query on eps error, so ascent is succeeding at its own objective -- it is just that the
    # objective does not require the point to be a natural image, and in high dimensions a diffusion
    # model assigns high density to off-manifold points (the likelihood/typicality gap).  Their
    # Table 6 says 100 iterations at step 0.09; a shorter or smaller ascent stays nearer the query,
    # which IS a natural image.  This measures the trade directly: distance travelled against
    # in-gamut fraction against eps error.
    print(f"  {'iters':>6s} {'lr':>6s} {'frac |x0|>1':>12s} {'RMSE to query':>14s} {'eps err':>9s}")
    for iters, lr in ((100, 0.09), (100, 0.02), (30, 0.09), (10, 0.09), (100, 0.005)):
        _, x0 = field.mode_ascent(xq, 0.15, n_starts=2, iters=iters, lr=lr, seed=0, init="query")
        x0 = np.asarray(x0)
        d = float(np.sqrt(np.mean((np.clip(x0, -1, 1) - xq) ** 2)))
        print(f"  {iters:6d} {lr:6.3f} {float((np.abs(x0) > 1).mean()):12.4f} {d:14.4f} "
              f"{eps_err(np.clip(x0, -1, 1), 0.15):9.4f}   (query eps {eps_err(np.clip(xq,-1,1), 0.15):.4f})")
    print()

    grid = [round(t / 1000, 3) for t in range(50, 501, 50)]      # the paper's CelebA column
    print(f"[proto] the paper's CelebA grid: u = {grid[0]} .. {grid[-1]}\n")
    print(f"  {'u':>6s} {'abar':>7s} {'frac |x0|>1':>12s} {'max |x0|':>9s} {'eps err':>9s} "
          f"{'(query)':>8s}")
    panels = []
    for u in grid:
        q = eps_err(np.clip(xq, -1, 1), u)
        _, x0 = field.mode_ascent(xq, u, n_starts=2, iters=100, lr=0.09, seed=0, init="query")
        x0 = np.asarray(x0)
        print(f"  {u:6.3f} {float(cosine_abar(u)):7.3f} {float((np.abs(x0) > 1).mean()):12.4f} "
              f"{float(np.abs(x0).max()):9.3f} {eps_err(np.clip(x0, -1, 1), u):9.4f} {q:8.4f}")
        panels.append((u, to_data(x0[:1]).astype(np.float32)))

    print("\n[proto] read down the noise levels: if the low ones are faces and the high ones are\n"
          "        fields of colour, the pool is being contaminated by its own upper end and\n"
          "        selection can pick from there.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    ncol = (n + 2) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(1.6 * ncol, 3.6))
    axes = axes.ravel()
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(np.clip(to_data(xq[None])[0].reshape(img, img, 3), 0, 1))
    axes[0].set_title("query", fontsize=8); axes[0].axis("off")
    for i, (u, S) in enumerate(panels):
        axes[i + 1].imshow(np.clip(S[0].reshape(img, img, 3), 0, 1))
        axes[i + 1].set_title(f"u={u:g}", fontsize=8); axes[i + 1].axis("off")
    fig.suptitle("Mode ascent output at each noise level of the paper's CelebA grid, before pooling "
                 "and selection.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    o = os.path.join(FIG, "f28_prototype_by_level.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[proto] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
