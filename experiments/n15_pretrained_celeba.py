"""N15 — does any of this transfer to a real pretrained model?

Everything above runs on a 6.8M-parameter UNet we trained ourselves on 32x32 ColorMNIST, where the
factors are spatially separable and the class-embedding table exists to be factorized.  Two
objections follow, and this addresses both at once by moving to `google/ddpm-celebahq-256`: 113.7M
parameters, 256x256, trained by someone else on CelebA-HQ, with attributes that overlap spatially.

Two things have to be built before the method can even be attempted, and both are reported as
findings rather than as setup:

  1. The model is UNCONDITIONAL -- no class embedding, nothing for `emb_override` to override.  A
     conditioning slot is ADDED as a vector on the time embedding (`neural/pretrained.py`), which is
     where our own UNet accepts one.  Pretrained weights untouched; zero vector reproduces the model
     exactly (verified in tests/test_pretrained.py at 0.000e+00).
  2. There is no embedding table to factorize, so the basis cannot be READ -- it has to be BUILT,
     one direction per attribute value, learned from real images through the diffusion loss
     (`learn_attribute_basis`).

Arms, on held-out attribute COMBINATIONS:
  `composed`     sum the learned per-attribute directions.  Nothing was trained on the combination.
  `per-combo`    a direction learned for that exact combination from its own images.  The
                 upper reference and the analogue of one adapter per concept.
  `uncond`       the pretrained model untouched.  The floor.
  `real`         real CelebA images of the combination.  The ceiling.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                          # noqa: E402

from lifelong_ood.data.celeba import ATTRS, load_celeba                         # noqa: E402
from lifelong_ood.evals import attr_accuracy, fit_attr_probes, save_json   # noqa: E402
from lifelong_ood.neural.idealwords import (compose_attributes, learn_attribute_basis)  # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained                 # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attrs", nargs="+", default=["Smiling", "Male", "Black_Hair"])
    p.add_argument("--n-per-attr", type=int, default=64, help="images teaching one direction")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--basis-steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--probe-img", type=int, default=64, help="resolution the probes read at")
    p.add_argument("--norm", type=float, default=2.0, help="fixed magnitude per direction")
    p.add_argument("--n-eval", type=int, default=4, help="held-out combinations scored")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_per_attr, args.n_samples, args.steps = 8, 2, 10
        args.basis_steps, args.n_eval = 30, 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, img = load_pretrained(device=dev, batch=args.batch)
    print(f"[n15] pretrained {img}x{img}, emb_dim {field.emb_dim}, "
          f"{sum(q.numel() for q in field.model.parameters()) / 1e6:.1f}M params", flush=True)

    # probes read at low resolution; the generator works at 256
    dlo = load_celeba(n_per_class=200, img=args.probe_img, seed=args.seed, min_count=25,
                      max_rows=25000)
    probes = fit_attr_probes(dlo, seed=args.seed, epochs=8 if args.quick else 30)
    dhi = load_celeba(n_per_class=60, img=img, seed=args.seed, min_count=25, max_rows=12000)
    A = dhi["attrs"]
    ai = [ATTRS.index(a) for a in args.attrs]
    print(f"[n15] attributes {args.attrs} -> indices {ai}; {len(dhi['X'])} images at {img}px",
          flush=True)

    def rows_with(spec):
        """indices whose chosen attributes match `spec` (a tuple of 0/1)."""
        m = np.ones(len(A), bool)
        for k, v in zip(ai, spec):
            m &= (A[:, k] == v)
        return np.where(m)[0]

    def to_model(X01):
        return (np.asarray(X01, np.float32) * 2 - 1)

    def probe_score(S, spec_full, spec):
        """Downsample generated 256px samples to the probe resolution and read attributes off them.

        Two joint scores are reported.  `joint` is the conjunction over all 14 readable attributes,
        which RESULTS 5o showed is near-vacuous -- real data only reaches 0.17-0.25 on it.  `target`
        is the conjunction over just the attributes actually being composed, which is the number the
        experiment is about.
        """
        n = len(S)
        X = np.clip((S + 1) / 2, 0, 1).reshape(n, img, img, 3)
        f = img // args.probe_img
        Xs = X.reshape(n, args.probe_img, f, args.probe_img, f, 3).mean((2, 4))
        flat = Xs.reshape(n, -1).astype(np.float32)
        a = attr_accuracy(probes, flat, spec_full)
        hits = [probes[nm].predict(flat) == int(v) for nm, v in zip(args.attrs, spec)
                if nm in probes]
        a["target"] = float(np.stack(hits).all(0).mean()) if hits else float("nan")
        a["target_mean"] = float(np.stack(hits).mean()) if hits else float("nan")
        return a

    # ── learn ONE direction per attribute value, from real images ────────────
    groups = {}
    for k, a in zip(ai, args.attrs):
        for val in (0, 1):
            idx = np.where(A[:, k] == val)[0][:args.n_per_attr]
            if len(idx) >= 4:
                groups[f"{a}={val}"] = to_model(dhi["X"][idx])
    print(f"[n15] learning {len(groups)} attribute directions from "
          f"{args.n_per_attr} images each", flush=True)
    dirs = learn_attribute_basis(field, groups, n_steps=args.basis_steps, batch=args.batch,
                                 seed=args.seed, norm=args.norm,
                                 log_every=0 if args.quick else 100)

    # ── score held-out COMBINATIONS ──────────────────────────────────────────
    specs = [s for s in itertools.product((0, 1), repeat=len(ai))
             if len(rows_with(s)) >= args.n_samples][:args.n_eval]
    print(f"[n15] scoring {len(specs)} attribute combinations", flush=True)
    rows, panels = [], {}
    for spec in specs:
        names = [f"{a}={v}" for a, v in zip(args.attrs, spec)]
        full = np.zeros(len(ATTRS), int)
        idx = rows_with(spec)
        full[:] = np.round(A[idx].mean(0)).astype(int)      # the combination's typical attributes
        for k, v in zip(ai, spec):
            full[k] = v

        got = {}
        v = compose_attributes(dirs, names)
        got["composed"] = field.sample(n=args.n_samples, steps=args.steps, seed=args.seed, emb=v)
        Xc = to_model(dhi["X"][idx[:args.n_per_attr]])
        one = learn_attribute_basis(field, {"combo": Xc}, n_steps=args.basis_steps,
                                    batch=args.batch, seed=args.seed, norm=args.norm)["combo"]
        got["per-combo"] = field.sample(n=args.n_samples, steps=args.steps, seed=args.seed, emb=one)
        got["uncond"] = field.sample(n=args.n_samples, steps=args.steps, seed=args.seed)
        got["real"] = to_model(dhi["X"][idx[:args.n_samples]])

        for arm, S in got.items():
            a_ = probe_score(np.asarray(S), full, spec)
            rows.append(dict(arm=arm, spec="/".join(names), joint=a_["joint"],
                             target=a_["target"], target_mean=a_["target_mean"],
                             mean_attr=a_.get("mean_attr", float("nan")),
                             **{f"attr_{a}": a_["per_attr"].get(a, float("nan"))
                                for a in args.attrs if a in a_["per_attr"]}))
            panels.setdefault("/".join(names), {})[arm] = np.asarray(S)
        print(f"[n15] {'/'.join(names)}: " + "  ".join(
            f"{k} {[r for r in rows if r['arm'] == k and r['spec'] == '/'.join(names)][0]['target']:.2f}"
            for k in got), flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["real", "composed", "per-combo", "uncond"]
    nrow = len(panels) * len(order)
    ncol = min(args.n_samples, 6)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.3 * ncol, 1.35 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for spec, per in panels.items():
        for arm in order:
            S = per.get(arm)
            for c in range(ncol):
                if S is not None and c < len(S):
                    im = np.clip((S[c] + 1) / 2, 0, 1).reshape(img, img, 3)
                    axes[r, c].imshow(im)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            m = [x for x in rows if x["arm"] == arm and x["spec"] == spec]
            axes[r, 0].set_ylabel(f"{arm}\n{spec}", fontsize=5, rotation=0, ha="right", va="center")
            if m:
                axes[r, ncol - 1].set_title(f"target {m[0]['target']:.2f}", fontsize=5, loc="right")
            r += 1
    fig.suptitle("N15 — google/ddpm-celebahq-256 (pretrained, unconditional, 256px)\n"
                 "`composed` sums per-attribute directions for a combination nobody trained",
                 fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "n15_pretrained_celeba.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary = {a: {k: float(np.nanmean([x[k] for x in rows if x["arm"] == a]))
                   for k in ("joint", "target", "target_mean", "mean_attr")}
               for a in order if any(x["arm"] == a for x in rows)}
    save_json(dict(rows=rows, summary=summary, attrs=args.attrs, img=img,
                   model="google/ddpm-celebahq-256"),
              os.path.join(RESULTS, "n15_pretrained_celeba.json"), seed=args.seed)
    print(f"\n[n15] pretrained CelebA-HQ, {len(specs)} combinations x {args.n_samples} samples")
    print(f"    {'arm':14s}{'TARGET joint':>14s}{'target per-attr':>18s}"
          f"{'all-14 joint':>14s}{'all-14 per-attr':>18s}")
    for a in order:
        if a in summary:
            s_ = summary[a]
            print(f"    {a:14s}{s_['target']:14.3f}{s_['target_mean']:18.3f}"
                  f"{s_['joint']:14.3f}{s_['mean_attr']:18.3f}")
    print("\n  TARGET is the conjunction over the attributes actually being composed -- the number")
    print("  this experiment is about.  all-14 is reported only because 5o showed it is the metric")
    print("  n1 prints, and that real data itself only reaches 0.17-0.25 on it.")
    print(f"[n15] wrote figures/n15_pretrained_celeba.png")


if __name__ == "__main__":
    main()
