"""Is the factored conditioning WEAK, or ABSENT?  A decisive cheap test on a partial checkpoint.

The 15-epoch de-risk backbone conditions at +0.011 per-attribute over unconditional, against the
+0.048 that RESULTS 5o's per-class backbone managed.  Two things understate it -- it has had a
quarter of the training, and it was evaluated with plain conditional sampling -- but the number is
small enough to be worth resolving before three 60-epoch runs finish.

The two hypotheses make opposite predictions under classifier-free guidance:

  * conditioning is LEARNED BUT WEAK  -> the direction is right and CFG amplifies it, so the margin
    grows with the guidance scale before diversity collapses.
  * conditioning is NOT LEARNED       -> there is no direction to amplify; the margin stays flat
    while the samples degrade, because CFG extrapolates along a vector carrying no attribute signal.

Sweeps guidance on TRAINED combinations, so it uses no held-out label, and reports the margin over
the unconditional model at each scale alongside a diversity proxy so a "win" that is really mode
collapse is visible as one.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                # noqa: E402
from n19_celeba_train import build_probes                               # noqa: E402

from lifelong_ood.data.celeba import IMG_CELEBA, load_celeba_comp            # noqa: E402
from lifelong_ood.evals import attr_accuracy                            # noqa: E402
from lifelong_ood.neural import load_field, sample_embedding, to_data   # noqa: E402

CKPT = os.environ.get("DIAG_CKPT", os.path.join(CACHE, "ddpm_celeba_factored_e15.pt"))
GRID = [float(x) for x in os.environ.get("DIAG_GUIDANCE", "1,1.5,2,3,4,6").split(",")]
N_COMBO = int(os.environ.get("DIAG_NCOMBO", 6))
N = int(os.environ.get("DIAG_N", 16))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    img = ck["img"]
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = img, 3 * img * img
    data = load_celeba_comp(img=img, min_count=ck.get("min_count", 60),
                            n_holdout=ck.get("n_holdout", len(ck["hold_ids"])),
                            holdout_pair=ck.get("holdout_pair"),
                            holdout_attr=ck.get("holdout_attr"),
                            seed=ck.get("seed", 0), cache_dir=CACHE)
    sp = data["split"]
    probes = build_probes(data, 0, 30)
    field = load_field(CKPT, device=dev, batch=64)
    model = field.model
    print(f"[diag] {os.path.basename(CKPT)} | {len(ck['remap'])} trained combos", flush=True)

    rng = np.random.default_rng(0)
    tr = np.array(sorted(ck["remap"], key=lambda z: ck["remap"][z]))
    picks = rng.permutation(tr)[:N_COMBO]

    # unconditional reference, scored against each target combination
    S0 = to_data(field.sample(n=N, steps=50, seed=0, clip=(-1, 1))).astype(np.float32)
    unc = float(np.nanmean([attr_accuracy(probes, S0, sp["all_combos"][int(j)])
                            .get("mean_attr", np.nan) for j in picks]))
    real = float(np.nanmean([
        attr_accuracy(probes, np.asarray(data["X8"][np.where(sp["combo_of"] == int(j))[0][:N]],
                                         np.float32) / 255.0,
                      sp["all_combos"][int(j)]).get("mean_attr", np.nan) for j in picks]))
    print(f"[diag] real images {real:.3f} | unconditional {unc:.3f}", flush=True)
    print(f"\n  {'guidance':>9s}{'mean-attr':>11s}{'margin':>9s}{'% of headroom':>15s}"
          f"{'pixel sd':>10s}")
    for g in GRID:
        accs, sds = [], []
        for j in picks:
            cb = sp["all_combos"][int(j)]
            c = model.attr_embedding(torch.as_tensor(cb, dtype=torch.long))
            S = to_data(sample_embedding(field, c.detach(), n=N, steps=50, seed=0,
                                         guidance=g)).astype(np.float32)
            accs.append(attr_accuracy(probes, S, cb).get("mean_attr", np.nan))
            sds.append(float(S.std(0).mean()))          # collapse shows up as this falling
        m = float(np.nanmean(accs))
        head = (m - unc) / max(real - unc, 1e-9)
        print(f"  {g:9.1f}{m:11.3f}{m - unc:+9.3f}{100 * head:14.0f}%{np.mean(sds):10.4f}",
              flush=True)
    print("\n  Reading: a margin that GROWS with guidance means the conditioning is learned but")
    print("  weak (amplifiable).  A margin that stays flat while pixel sd falls means there is no")
    print("  attribute direction to amplify, and more training will not fix the architecture.")


if __name__ == "__main__":
    main()
