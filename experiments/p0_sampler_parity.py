"""GPU probe p0: the one sampler through the Backbone handle reproduces the legacy `ddim_sample`
path the c1/c5 fixtures used, bit for bit (or to 1e-5), on both checkpoints on disk.

    sbatch --export=ALL,SCRIPT="experiments/p0_sampler_parity.py" slurm/jobs/c_gpu.sbatch

Writes one ledger row per backbone (bytes all zero: nothing is stored).  Exits 1 on any mismatch.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.backbones.loader import load_backbone                # noqa: E402
from lifelong_ood.backbones.sampler import ddim_sample, sample         # noqa: E402

ZERO = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)


def legacy_field(path, device):
    """The archived loaders, exactly as the fixtures call them."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "ema_cemb" in ck:
        from lifelong_ood.neural.ve_bridge import load_ve_field
        f, _ = load_ve_field(path, device=device)
        return f.model.eval(), "hwc", int(ck["img"]), (0.0, 1.0)
    from lifelong_ood.neural.ddpm import load_field
    f = load_field(path, device=device)
    return f.model.eval(), f.layout, f.img, (-1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ckpts", default="results/cache/ddpm_colormnist_nobg3.pt:results/cache/ddpm_s3d_full.pt")
    ap.add_argument("--hf", default="", help="optional diffusers name to smoke through the third loader")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rc = 0
    for path in args.ckpts.split(":"):
        path = os.path.join(ROOT, path) if not os.path.isabs(path) else path
        bench = os.path.basename(path).replace("ddpm_", "").replace(".pt", "")
        t0 = time.time()
        bk = load_backbone(path, device=dev)
        t_load = time.time() - t0
        print(f"[p0] {bench}: {bk.describe()} (load {t_load:.1f}s)", flush=True)
        model_l, layout_l, img_l, clip_l = legacy_field(path, dev)
        assert (layout_l, img_l, clip_l) == (bk.layout, bk.img, bk.clip), (layout_l, img_l, clip_l, bk)
        cls = None if bk.class_table() is None else 0
        for name, c in (("null", None), ("class0", cls)):
            t0 = time.time()
            A = sample(bk, args.n, steps=args.steps, seed=0, cls=c)
            tA = time.time() - t0
            t0 = time.time()
            B = ddim_sample(model_l, args.n, steps=args.steps, seed=0, cls=c, layout=layout_l, img=img_l,
                            device=dev, clip=clip_l)
            tB = time.time() - t0
            mx, mean = float(np.abs(A - B).max()), float(np.abs(A - B).mean())
            ok = mx < 1e-5
            rc |= (0 if ok else 1)
            print(f"[p0] {bench} cond={name}: max|diff| {mx:.2e} mean {mean:.2e} "
                  f"({'OK' if ok else 'MISMATCH'}); new {tA:.1f}s legacy {tB:.1f}s; "
                  f"rows in [{A.min():.3f}, {A.max():.3f}]", flush=True)
            ledger.record("p0_sampler_parity", bench=bench, tag=args.tag, seed=0, keys={"cond": name},
                          metrics=dict(max_abs_diff=mx, mean_abs_diff=mean, ok=bool(ok), secs_new=tA,
                                       secs_legacy=tB, n=args.n, steps=args.steps),
                          bytes=ZERO, config=vars(args), script="experiments/p0_sampler_parity.py",
                          command=" ".join(sys.argv), backbone=dict(path=os.path.relpath(path, ROOT),
                          sha256=bk.sha256, kind=bk.kind, layout=bk.layout, img=bk.img, n_params=bk.n_params))
    if args.hf:
        t0 = time.time()
        bk = load_backbone(args.hf, device=dev)
        S = sample(bk, 8, steps=20, seed=0)
        print(f"[p0] {args.hf}: {bk.describe()}; 8 samples at 20 steps in {time.time() - t0:.1f}s; "
              f"rows mean {S.mean():.3f} std {S.std():.3f} in [{S.min():.3f}, {S.max():.3f}]", flush=True)
        ledger.record("p0_sampler_parity", bench=args.hf.replace("/", "_"), tag=args.tag, seed=0,
                      keys={"cond": "null"}, metrics=dict(smoke=True, row_mean=float(S.mean()),
                      row_std=float(S.std()), secs=time.time() - t0), bytes=ZERO, config=vars(args),
                      script="experiments/p0_sampler_parity.py", command=" ".join(sys.argv),
                      backbone=dict(path=args.hf, kind=bk.kind, layout=bk.layout, img=bk.img, n_params=bk.n_params))
    print("[p0] PASS" if rc == 0 else "[p0] FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
