"""N10 — CelebA: is the backbone good enough to compose on, measured against the real-data ceiling?

`n1_train` reported conditional joint accuracy 0.0625 for the CelebA backbone, which looks like a
failure and is uninterpretable on its own: the joint score is a conjunction over 12 attributes, so
even a perfect generator scored by imperfect probes cannot reach 1.0.  The number that makes it
readable is what REAL held-out images of the same class score on exactly the same metric.

Reports, for the seen classes:
  real     held-out real images of the class            -- the ceiling
  cond     the model conditioned on that class          -- what we can actually generate
  uncond   the model with no conditioning               -- the floor, i.e. chance for this metric

CelebA matters here because its attributes OVERLAP spatially, unlike ColorMNIST's digit/foreground/
background, and several of this repo's conclusions lean on that separability.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                     # noqa: E402

from lifelong_ood.celeba import load_celeba                           # noqa: E402
from lifelong_ood.evals import attr_accuracy, fit_attr_probes, save_grid, save_json  # noqa: E402
from lifelong_ood.neural import load_field, to_data                   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba.pt"))
    p.add_argument("--n-classes", type=int, default=12, help="seen classes to score")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img = ck.get("img", 64)
    data = load_celeba(n_per_class=100, img=img, seed=args.seed, min_count=25, max_rows=25000)
    probes = fit_attr_probes(data, seed=args.seed, epochs=30)
    field = load_field(args.ckpt, device=dev)
    remap = ck["remap"]

    seen = data["seen"]
    classes = [c for c in sorted(remap) if (data["cls"][seen] == c).sum() >= args.n_samples]
    classes = classes[:args.n_classes]
    print(f"[n10] {len(classes)} seen classes, {img}x{img}, device {dev}", flush=True)

    rows = []
    for c in classes:
        combo = data["combo"][data["cls"] == c][0]
        R = data["X"][seen][data["cls"][seen] == c][:args.n_samples]
        S = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                 clip=(-1, 1), branch=int(remap[int(c)]))).astype(np.float32)
        U = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed + 1,
                                 clip=(-1, 1))).astype(np.float32)
        a_r, a_c, a_u = (attr_accuracy(probes, X, combo) for X in (R, S, U))
        rows.append(dict(cls=int(c), n_scored=a_r["n_scored"],
                         real=a_r["joint"], cond=a_c["joint"], uncond=a_u["joint"],
                         real_mean=a_r["mean_attr"], cond_mean=a_c["mean_attr"],
                         uncond_mean=a_u["mean_attr"]))
        print(f"[n10] class {c}: real {a_r['joint']:.2f} | cond {a_c['joint']:.2f} | "
              f"uncond {a_u['joint']:.2f}  (per-attr {a_r['mean_attr']:.2f} / "
              f"{a_c['mean_attr']:.2f} / {a_u['mean_attr']:.2f})", flush=True)
    save_grid(S[:16], os.path.join(RESULTS, "n10_celeba_cond.png"), ncol=8,
              title=f"CelebA, conditioned on a seen class ({img}x{img})")

    m = {k: float(np.nanmean([r[k] for r in rows]))
         for k in ("real", "cond", "uncond", "real_mean", "cond_mean", "uncond_mean")}
    save_json(dict(rows=rows, mean=m, img=img), os.path.join(RESULTS, "n10_celeba.json"),
              seed=args.seed)
    print(f"\n[n10] CelebA, {len(classes)} seen classes x {args.n_samples} samples, "
          f"conjunction over {rows[0]['n_scored']} readable attributes")
    print(f"    {'source':10s}{'joint':>8s}{'per-attribute':>16s}")
    for k, lab in (("real", "real"), ("cond", "conditional"), ("uncond", "unconditional")):
        print(f"    {lab:10s}{m[k]:8.3f}{m[k + '_mean']:16.3f}")
    gap = m["cond"] / m["real"] if m["real"] > 0 else float("nan")
    print(f"\n  conditional reaches {gap:.0%} of the real-data ceiling on the joint metric.")
    print("  The joint score is a conjunction over every readable attribute, so real data does not")
    print("  reach 1.0 either -- that is the point of reporting the ceiling next to it.")
    print(f"[n10] wrote {RESULTS}/n10_celeba.json")


if __name__ == "__main__":
    main()
