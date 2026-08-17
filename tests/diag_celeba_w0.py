"""Did guiding CelebA at the wrong strength actually matter?  Measure it instead of assuming.

docs/SETTINGS.md found that every CelebA run in this repository guided at w0=1.2 -- the companion
paper's **Color-MNIST** value -- where its Table 8 gives CelebA **w0=0.5**.  That is 2.4x the
published guidance, applied to `poe-guided`, the paper's own method, in the experiment whose output
prompted "the picture quality seems significantly worse".

Over-guidance has a known signature: it sharpens and saturates, pushing samples off the data
manifold.  So the prediction is specific and falsifiable -- at w0=1.2 the samples should be
*rougher* than real faces (tv_ratio > 1) and no better on attributes, and dropping to 0.5 should
move tv_ratio toward 1.0.  If tv_ratio is already <= 1 at 1.2, over-guidance is NOT what is wrong
with our CelebA pictures and the backbone remains the whole story.

Same queries, same seed, same teacher -- the ONLY thing that varies is w0, so whatever moves is
attributable to it.

    python tests/diag_celeba_w0.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, RESULTS                                              # noqa: E402

from lifelong_ood.data.celeba import as_float, load_celeba_comp                     # noqa: E402
from lifelong_ood.compose import PAPER_POE_CELEBA, PAPER_POE_COLORMNIST        # noqa: E402
from lifelong_ood.evals import (attr_accuracy, fit_attr_probes, probe_usable,  # noqa: E402
                                save_json, speckle)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, to_data,   # noqa: E402
                                 to_model)

CK = os.path.join(CACHE, "ddpm_celeba_factored.pt")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "initial")
ARMS = [("paper CelebA  w0=0.5", PAPER_POE_CELEBA),
        ("what we ran   w0=1.2", PAPER_POE_COLORMNIST)]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    img = int(ck.get("img", 64))
    data = load_celeba_comp(img=img, max_rows=25000, min_count=ck.get("min_count", 60),
                            n_holdout=ck.get("n_holdout", 24))
    field = load_field(CK, device=dev)
    rng = np.random.default_rng(0)

    idx = rng.permutation(len(data["X8"]))[:6000]
    probes = fit_attr_probes(dict(X=as_float(data["X8"][idx]), attrs=data["attrs"][idx], img=img,
                                  seen=np.ones(len(idx), bool)), seed=0, epochs=30)
    usable = [a for a in probes["_attrs"] if probe_usable(probes, a)]
    print(f"[w0] {len(usable)}/{len(probes['_attrs'])} probes readable; guiding at "
          f"{[a for a, _ in ARMS]}\n", flush=True)

    # hold_ids / combo_of / all_combos live under `split`, not at top level -- the same trap n44 hit
    sp = data["split"]
    hold = np.asarray(sp["hold_ids"])
    rows, panels = [], {}
    for ci in range(4):
        h = int(hold[ci % len(hold)])
        qrows = np.where(sp["combo_of"] == h)[0]
        if len(qrows) < 8:
            continue
        xq = to_model(as_float(data["X8"][qrows[:1]])).ravel()
        L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev)
        t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=12, iters=80)
        R = as_float(data["X8"][qrows[:64]]).astype(np.float32)
        got = {"real": R[:6]}
        for name, poe in ARMS:
            S = to_data(L.sample_teacher(t, n=16, seed=0, **poe)).astype(np.float32)
            sk = speckle(S, R, img=img)
            acc = attr_accuracy(probes, S, sp["all_combos"][h], attrs=usable)
            m = float(np.mean([v for k, v in acc.items() if k in usable]))
            rows.append(dict(case=h, arm=name, w0=poe["w0"],
                             tv_ratio=float(sk.get("tv_ratio", np.nan)), attr=m))
            print(f"[w0] case {h}  {name}   tv_ratio {sk.get('tv_ratio', float('nan')):.3f}"
                  f"   mean-attr {m:.3f}", flush=True)
            got[name] = S[:6]
        panels[h] = got
        del L
        torch.cuda.empty_cache()

    save_json(dict(rows=rows), os.path.join(RESULTS, "diag_celeba_w0.json"), seed=0)
    for w in (0.5, 1.2):
        sel = [r for r in rows if r["w0"] == w]
        print(f"\n  w0={w}:  tv_ratio {np.mean([r['tv_ratio'] for r in sel]):.3f}"
              f"   mean-attr {np.mean([r['attr'] for r in sel]):.3f}")
    print("\n[w0] tv_ratio > 1 means ROUGHER than real -- the over-guidance signature.  If both "
          "arms sit well below 1, our CelebA samples are too SMOOTH and guidance strength is not "
          "the defect; the from-scratch backbone is.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["real"] + [a for a, _ in ARMS]
    nrow = len(panels) * len(order)
    fig, axes = plt.subplots(nrow, 6, figsize=(9.5, 1.5 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for h, got in panels.items():
        for name in order:
            S = got.get(name)
            for c in range(6):
                axes[r][c].axis("off")
                if S is not None and c < len(S):
                    axes[r][c].imshow(np.clip(np.asarray(S[c], np.float32).reshape(img, img, 3),
                                              0, 1))
            axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for s_ in axes[r][0].spines.values():
                s_.set_visible(False)
            r += 1
    fig.suptitle("CelebA guidance strength: the paper's w0=0.5 vs the w0=1.2 this repo ran.\n"
                 "Same queries, same teacher, same seed -- only w0 differs.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(FIG, "extra", "f19_celeba_w0.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[w0] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
