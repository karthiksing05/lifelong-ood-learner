"""What actually drives the CelebA colour cast -- the teacher, or the seed?

`figures/f18_celeba_paper_paperdisc_w05.png` shows two things at once, and the previous caption
asserted a mechanism for them that the figure itself refutes:

  * the PROTOTYPES are off the colour manifold (neon magenta, electric blue), and
  * within one block EVERY generated row carries the same cast and collapses to nearly one face.

The tempting story is that the bad prototypes contaminate the teacher mean and the guidance carries
the cast into the samples.  But `query-only` shows the SAME cast, and its teacher is N(x_q, sigma^2 I)
-- it never sees a prototype.  So either the cast has two independent causes, or it has one that
both arms share.  What they share is the initial noise: n44 draws every arm at `seed=args.seed`, and
their Table 8 sampler is DDIM at eta=0, which is deterministic given the teacher.

I captioned this once already with an explanation I had not measured, and it was wrong.  So this
measures it, and the design makes the answer unambiguous:

    A. one query, SIX seeds      -- if the cast tracks the seed, it is the sampler
    B. six queries, ONE seed     -- if the cast is identical across them, it is the seed, not x_q
    C. prototype vs query colour -- how far discovery itself leaves the colour manifold

Reported as mean RGB per panel plus the spread of that mean ACROSS the axis being varied.  A large
spread in A with a small spread in B locates the cast in the seed; the reverse locates it in the
query; both large means the teacher is carrying it.

    python tests/diag_celeba_cast.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from lifelong_ood.compose import PAPER_CELEBA as DISC                          # noqa: E402
from lifelong_ood.compose import PAPER_POE_CELEBA as PAPER_POE                 # noqa: E402
from lifelong_ood.compose import candidates_ascent, poe_teacher                # noqa: E402
from lifelong_ood.neural import NeuralLifelongLearner, to_data, to_model       # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained                     # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def rgb(S, img):
    """-> [n, 3] mean colour per image."""
    S = np.atleast_2d(np.asarray(S, np.float32))
    return S.reshape(len(S), img, img, 3).mean((1, 2))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    field, img = load_pretrained("google/ddpm-celebahq-256", device=dev, batch=4)
    from lifelong_ood.celeba import as_float, load_celeba_comp
    data = load_celeba_comp(img=img, max_rows=4000, min_count=20, n_holdout=8)
    L = NeuralLifelongLearner(field, steps=100, seed=0, device=dev)

    rng = np.random.default_rng(0)
    qs = rng.permutation(len(data["X8"]))[:6]
    XQ = [to_model(as_float(data["X8"][q:q + 1])).ravel() for q in qs]
    qrgb = np.concatenate([rgb(as_float(data["X8"][q:q + 1]), img) for q in qs])
    print(f"[cast] {len(qs)} real queries, mean RGB spread across them "
          f"{qrgb.std(0).mean():.4f}\n")

    def qonly(x, seed):
        return to_data(L.sample_teacher(L.baseline_teacher(x, "query_only"), n=1, seed=seed,
                                        **PAPER_POE))

    # ── A. one query, six seeds ──────────────────────────────────────────────
    A = np.concatenate([rgb(qonly(XQ[0], s), img) for s in range(6)])
    # ── B. six queries, one seed ─────────────────────────────────────────────
    B = np.concatenate([rgb(qonly(x, 0), img) for x in XQ])
    print(f"  {'test':38s} {'mean RGB':>22s} {'spread':>8s}")
    print(f"  {'A. one query, SIX seeds':38s} {str(np.round(A.mean(0), 3)):>22s} "
          f"{A.std(0).mean():8.4f}")
    print(f"  {'B. six queries, ONE seed':38s} {str(np.round(B.mean(0), 3)):>22s} "
          f"{B.std(0).mean():8.4f}")
    print(f"  {'   (the queries themselves)':38s} {str(np.round(qrgb.mean(0), 3)):>22s} "
          f"{qrgb.std(0).mean():8.4f}")

    # ── C. how far do the prototypes leave the colour manifold? ──────────────
    print(f"\n  {'case':>5s} {'query RGB':>22s} {'prototype RGB':>22s} {'|dRGB|':>8s} "
          f"{'frac |x|>1':>11s}")
    for i, x in enumerate(XQ[:3]):
        M, V, _ = candidates_ascent(field, x, seed=0, **DISC)
        t = poe_teacher(x, M, V, K=3, tau=0.5, adaptive_tau=False)
        P = to_data(np.asarray(M)[t["sel"]])
        pr, qr = rgb(P, img).mean(0), rgb(to_data(x[None]), img)[0]
        print(f"  {i:5d} {str(np.round(qr, 3)):>22s} {str(np.round(pr, 3)):>22s} "
              f"{np.abs(pr - qr).mean():8.4f} {float((np.abs(np.asarray(M)[t['sel']]) > 1).mean()):11.4f}")

    print("\n[cast] A >> B locates the cast in the SEED and the unconditional model, which would\n"
          "       mean the composition is not what is broken -- guidance at w0=0.5 is simply too\n"
          "       weak to move the sample off whatever the noise draw decodes to. B >> A locates\n"
          "       it in the query. Both large means the teacher carries it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
