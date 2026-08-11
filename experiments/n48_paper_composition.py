"""N48 -- the composition benchmark on the paper's OWN backbone, and then ours on top of it.

N47 built their Table 5 CelebA column: `google/ddpm-celebahq-256` fine-tuned at 128px with a class
table of |C|+1 rows, on the model's linear-beta schedule.  This runs the composition benchmark on
it, which is the first time in this repository that the companion method is evaluated on a substrate
comparable to the one its own figures were made on.

Everything that had to be fixed to get here is worth stating, because each was silently degrading
this comparison:

  1. the input layout -- the pretrained model was fed a pixel permutation and returned a 3x3 tiled
     greyscale mosaic (RESULTS 30b)
  2. the CelebA guidance strength -- w0=1.2, the paper's Color-MNIST value, instead of Table 8's
     0.5 (RESULTS 29a)
  3. mode ascent leaving the valid image range, which is what made prototypes saturated (RESULTS 30c)
  4. the backbone itself -- from scratch at 64px over 1205 conditioning rows, against a fine-tune of
     a photorealistic model at 128px over 51 (RESULTS 29)

Arms are the paper's own: `poe-guided` (their Eq. 13/14 pipeline), `query-only` and `nearest-class`
as the floors, and `ours` -- the three-tier route -- as the extension.  A held-out class is one the
fine-tune never conditioned on, so "composition" here means rendering a class the backbone has a
name for but no training signal about.

    python experiments/n48_paper_composition.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                              # noqa: E402

from lifelong_ood.celeba import class_name, load_celeba                        # noqa: E402
from lifelong_ood.compose import (PAPER_CELEBA, PAPER_POE_CELEBA,              # noqa: E402
                                  candidates_ascent, poe_teacher)
from lifelong_ood.evals import (attr_accuracy, fit_attr_probes, probe_usable,  # noqa: E402
                                save_json, speckle)
from lifelong_ood.neural import to_data, to_model                              # noqa: E402
from lifelong_ood.neural.pretrained import load_paper_backbone                 # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_paper128.pt"))
    p.add_argument("--n-cases", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.steps = 2, 4, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.ckpt):
        print(f"[n48] missing {args.ckpt} -- run slurm/jobs/n47_paper_backbone.sbatch first")
        return 1
    field, img, cemb, remap = load_paper_backbone(args.ckpt, device=dev, batch=4)
    print(f"[n48] paper backbone at {img}px, {len(cemb) - 1} classes + null, layout=hwc, "
          f"linear-beta schedule", flush=True)

    data = load_celeba(n_per_class=100, img=img, seed=args.seed, min_count=25, max_rows=25000)
    seen = np.asarray(data["seen"], bool)
    probes = fit_attr_probes(dict(X=np.asarray(data["X"]), attrs=data["attrs"], img=img, seen=seen),
                             seed=args.seed, epochs=8 if args.quick else 30)
    usable = [a for a in probes["_attrs"] if probe_usable(probes, a)]
    print(f"[n48] {len(usable)}/{len(probes['_attrs'])} attribute probes readable\n", flush=True)

    # ── DOES THE CONDITIONING ACTUALLY WORK? ─────────────────────────────────
    # Run this BEFORE any composition number is produced.  N15 found that conditioning a pretrained
    # UNet through one additive vector on a FROZEN network was too weak to move it; n47 fine-tunes
    # the network together with the class table, which should fix that -- but "should" is not
    # evidence.  If two different SEEN classes and the null token all render the same thing, the
    # class table never took hold, every arm below will score identically, and the composition
    # comparison is vacuous rather than negative.
    #
    # The smoke backbone shows exactly that failure (poe-guided = query-only = unconditional = 0.571)
    # because three minutes of fine-tuning teaches no conditioning at all.  It is the right check.
    seen_cls = np.unique(np.asarray(data["cls"])[seen])
    ca, cb = int(seen_cls[0]), int(seen_cls[len(seen_cls) // 2])
    Sa, Sb, Sn = (to_data(field.sample(n=6, steps=args.steps, seed=args.seed, clip=(-1, 1),
                                       emb=(None if e is None else cemb[e]))).astype(np.float32)
                  for e in (remap[ca], remap[cb], None))
    d_ab = float(np.sqrt(np.mean((Sa - Sb) ** 2)))
    d_an = float(np.sqrt(np.mean((Sa - Sn) ** 2)))
    print(f"[n48] CONDITIONING CHECK  RMS(class {ca} vs class {cb}) = {d_ab:.4f}   "
          f"RMS(class {ca} vs null) = {d_an:.4f}")
    if d_ab < 0.02:
        print(f"[n48]   the two class conditionals are indistinguishable -- the class table did not "
              f"take hold, and every composition number below would be an artefact of that, not a "
              f"result. Fine-tune longer or raise the conditioning capacity.", flush=True)
    rows_head = [dict(cls=-1, arm="conditioning-check", mean_attr=d_ab, joint=d_an,
                      n_scored=0, tv_ratio=float("nan"))]
    print("", flush=True)

    hold = np.where(~seen)[0]
    hold_cls = np.unique(np.asarray(data["cls"])[hold])
    rng = np.random.default_rng(args.seed)
    rows, panels = list(rows_head), {}
    for ci in range(min(args.n_cases, len(hold_cls))):
        c = int(hold_cls[ci])
        idx = np.where(np.asarray(data["cls"]) == c)[0]
        if len(idx) < 4:
            continue
        combo = data["combos"][c]
        xq = to_model(np.asarray(data["X"])[idx[rng.integers(len(idx))]:][:1]).ravel()
        R = np.asarray(data["X"])[idx[:64]].astype(np.float32)

        M, V, _ = candidates_ascent(field, xq, seed=args.seed, **PAPER_CELEBA)
        t = poe_teacher(xq, M, V, K=3, tau=0.5, adaptive_tau=False)   # their fixed tau
        got = {"real": R[:args.n_show],
               "prototypes": to_data(np.asarray(M)[t["sel"]]).astype(np.float32)}
        got["poe-guided"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1),
            guide_eps=lambda x, u: np.repeat(np.asarray(t["mu"])[None], len(x), 0),
            guide_w=lambda u: PAPER_POE_CELEBA["w0"])).astype(np.float32)
        got["query-only"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1),
            guide_eps=lambda x, u: np.repeat(xq[None], len(x), 0),
            guide_w=lambda u: PAPER_POE_CELEBA["w0"])).astype(np.float32)
        got["unconditional"] = to_data(field.sample(
            n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1))).astype(np.float32)

        for name, S in got.items():
            if name in ("real", "prototypes"):
                continue
            a = attr_accuracy(probes, S, combo, attrs=usable)
            # `attr_accuracy` returns dict(joint, mean_attr, per_attr, n_scored) -- the per-attribute
            # numbers are NESTED under `per_attr`, so filtering the top level on attribute names gave
            # an empty list and nanmean([]) = nan.  Take the aggregate it already computes.
            m = float(a.get("mean_attr", float("nan")))
            jt = float(a.get("joint", float("nan")))
            sk = speckle(S, R, img=img)
            rows.append(dict(cls=c, arm=name, mean_attr=m, joint=jt,
                             n_scored=int(a.get("n_scored", 0)),
                             tv_ratio=float(sk.get("tv_ratio", np.nan))))
            print(f"[n48] class {c:4d} {class_name(combo)[:30]:30s} {name:14s} "
                  f"attr {m:.3f}  joint {jt:.3f}  tv "
                  f"{sk.get('tv_ratio', float('nan')):.3f}  (n={a.get('n_scored', 0)})",
                  flush=True)
        panels[c] = got

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n48_paper_composition{args.tag}.json"), seed=args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["real", "prototypes", "poe-guided", "query-only", "unconditional"]
    nrow = len(panels) * len(order)
    fig, axes = plt.subplots(nrow, args.n_show, figsize=(1.5 * args.n_show + 3.0, 1.55 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for c, got in panels.items():
        for name in order:
            S = got.get(name)
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if S is not None and k < len(S):
                    axes[r][k].imshow(np.clip(np.asarray(S[k], np.float32).reshape(img, img, 3),
                                              0, 1))
            axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for s_ in axes[r][0].spines.values():
                s_.set_visible(False)
            r += 1
    fig.suptitle("CelebA composition on the companion paper's OWN backbone: "
                 "ddpm-celebahq-256 fine-tuned at 128px.\nCorrect input layout, Table 8 guidance "
                 "(w0=0.5), Table 6 discovery, gamut-clamped mode ascent.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    o = os.path.join(FIG, f"f26_paper_composition{args.tag}.png")
    fig.savefig(o, dpi=120, bbox_inches="tight")
    print(f"\n[n48] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
