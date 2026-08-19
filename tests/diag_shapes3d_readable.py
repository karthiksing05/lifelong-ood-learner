"""Is there a readable 3-factor Shapes3D split, and at what table size? Measure before training.

Phase 2 of the pretrained-backbone plan wants Shapes3D as the flagship, and it has two properties
CelebA never could: a pretrained backbone (`eurecom-ds/scoresdeve-ema-shapes3d-64`) and a table big
enough for the basis claim. What it does NOT yet have is a demonstrated three-factor readout -- the
last measurement on the pretrained split (N70) reads

    floor_hue 0.997    object_hue 0.239    shape 0.714

on REAL images, so `joint` there is a two-factor joint and the recorded cause is that pinning
`wall_hue` shrank the set to 9000 images. Establishing readability is therefore step 1, and it comes
before a training run rather than after -- that ordering is the whole lesson of RESULTS 7ee.

THE TENSION BEING MEASURED, which is real and is not housekeeping. Two knobs trade against each
other:

  * `value_stride` sets the table size. Stride 1 gives 10 x 10 x 4 = 400 classes against 25 free
    vectors (ratio 16); stride 2 gives 5 x 5 x 4 = 100 against 15 (ratio 6.7, essentially
    ColorMNIST's 6.3, where in-sample and cross-validated R^2 agree).
  * `n_per_slot` sets how DENSE each class is. `shapes3d.py` records why stride 2 exists: at 400
    classes and 40 images each "every class was too rare for conditioning to take hold against an
    easy marginal -- the model generated excellent Shapes3D and ignored its label, embedding
    pairwise cosine 0.007".

So a bigger table is not automatically better, and the sweep is the point. **This is the same design
constraint a VLA faces as tasks-versus-demonstrations-per-task**, which makes measuring it here
transferable rather than incidental.

With `wall_hue` pinned there are 480k/10 = 48k images available, i.e. ~120 per class at stride 1 and
~480 at stride 2 -- against the ~90 the earlier run used.

    python tests/diag_shapes3d_readable.py
    python tests/diag_shapes3d_readable.py --strides 1 --per-slot 120
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from lifelong_ood.data.factored import holdout_mask                  # noqa: E402
from lifelong_ood.data.shapes3d import load_shapes3d                 # noqa: E402
from lifelong_ood.evals import fit_factor_probes, probe_readable     # noqa: E402
from lifelong_ood.instrument import MARGIN, check_instrument         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strides", type=int, nargs="*", default=[2, 1])
    p.add_argument("--per-slot", type=int, nargs="*", default=[40, 120, 480])
    p.add_argument("--n-holdout", type=int, default=12)
    p.add_argument("--n-arm", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[s3d] sweeping value_stride {args.strides} x n_per_slot {args.per_slot}, "
          f"wall_hue pinned, {args.n_holdout} held-out combinations", flush=True)
    print(f"\n[s3d] {'stride':>7s}{'per-slot':>10s}{'classes':>9s}{'images':>9s}"
          f"{'floor_hue':>11s}{'object_hue':>12s}{'shape':>8s}{'ceiling':>9s}"
          f"{'readable':>10s}{'ratio':>7s}")

    rows = []
    for stride in args.strides:
        for nps in args.per_slot:
            try:
                d = load_shapes3d(n_per_slot=nps, seed=args.seed, value_stride=stride,
                                  fix_nuisance=["wall_hue"], n_holdout=0)
            except Exception as e:                                    # noqa: BLE001
                print(f"[s3d] {stride:>7d}{nps:>10d}  load failed: {type(e).__name__}: {e}")
                continue
            sizes = [int(s) for s in d["sizes"]]
            n_cls = int(np.prod(sizes))
            # hold out COMBINATIONS while guaranteeing every factor VALUE survives -- the ADDRESS
            # vs EXTEND distinction, and the reason this helper exists rather than a random split
            seen = holdout_mask(np.asarray(d["factors"]), sizes, args.n_holdout, seed=args.seed)
            d["seen"] = seen
            probes = fit_factor_probes(d, seed=args.seed)
            per = {n: float(probes.get(f"{n}_val", float("nan"))) for n in d["names"]}
            base = {n: float(probes.get(f"{n}_base", float("nan"))) for n in d["names"]}
            n_read = sum(1 for n in d["names"] if probe_readable(probes, n))

            held_slots = np.unique(np.asarray(d["slot"])[~seen])[:args.n_holdout]
            F, slot = np.asarray(d["factors"]), np.asarray(d["slot"])
            rep = check_instrument(
                tag="s3d", real=d, probes=probes, n_arm=args.n_arm,
                targets={int(h): tuple(int(v) for v in F[slot == int(h)][0]) for h in held_slots},
                seed=args.seed)
            ceil = float(np.mean([g.ceiling for g in rep.targets.values()])) if rep.targets else 0.0
            n_ok = len(rep.scored(strict=True))
            n_free = 1 + sum(s - 1 for s in sizes)

            print(f"[s3d] {stride:>7d}{nps:>10d}{n_cls:>9d}{len(d['X']):>9d}"
                  f"{per.get('floor_hue', float('nan')):>11.3f}"
                  f"{per.get('object_hue', float('nan')):>12.3f}"
                  f"{per.get('shape', float('nan')):>8.3f}{ceil:>9.3f}"
                  f"{n_ok:>6d}/{len(rep.targets):<3d}{n_cls / n_free:>7.1f}", flush=True)
            rows.append(dict(stride=stride, per_slot=nps, n_classes=n_cls, n_images=len(d["X"]),
                             per_factor=per, base=base, n_probes_readable=n_read, ceiling=ceil,
                             n_readable=n_ok, n_targets=len(rep.targets),
                             ratio=n_cls / n_free, sizes=sizes))

    print(f"\n[s3d] {'=' * 78}")
    if not rows:
        print("[s3d] nothing loaded -- is the Shapes3D h5 present?")
        return 0
    three = [r for r in rows if r["n_probes_readable"] == 3]
    print(f"[s3d] per-factor accuracies are on REAL images; `ceiling` is the mean per-target gate")
    print(f"[s3d] over held-out combinations, and `readable` counts targets clearing the "
          f"+{MARGIN:.2f} bar.")
    if three:
        best = max(three, key=lambda r: (r["n_readable"], r["ceiling"]))
        print(f"[s3d] A THREE-FACTOR READABLE SPLIT EXISTS: stride {best['stride']}, "
              f"{best['per_slot']} per slot -> {best['n_classes']} classes at ratio "
              f"{best['ratio']:.1f}, ceiling {best['ceiling']:.3f}, "
              f"{best['n_readable']}/{best['n_targets']} targets readable.")
        print("[s3d]   Train the class table at this setting (the n57 pattern: freeze the backbone,")
        print("[s3d]   bolt on cemb, give it its own optimiser group at 50x LR).")
    else:
        best = max(rows, key=lambda r: r["ceiling"])
        print("[s3d] NO setting makes all three factors readable. The honest move is to scope the")
        print(f"[s3d]   claim to the factors that ARE readable -- best was stride {best['stride']} / "
              f"{best['per_slot']} per slot at ceiling {best['ceiling']:.3f} with "
              f"{best['n_probes_readable']}/3 probes -- and to say so, rather than report a joint")
        print("[s3d]   over a factor the instrument cannot see. That is 7ee's lesson applied in")
        print("[s3d]   advance instead of in retrospect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
