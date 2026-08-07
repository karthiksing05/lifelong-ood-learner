"""N20 — the N9 showdown on CelebA: every method, on held-out attribute COMBINATIONS.

This is the experiment RESULTS 7's separability threat has been waiting for.  Every composition
result in this repository was measured on ColorMNIST, whose digit / foreground / background occupy
DISJOINT pixels, and that disjointness is load-bearing: a per-dimension product-of-experts can
assemble an unseen (fg, bg) pair because a pixel belongs to exactly one factor.  CelebA attributes
overlap -- `Smiling`, `Big_Lips` and `Oval_Face` all write to the same mouth pixels -- so this
either reproduces the ordering or bounds where it holds.

Methods, matched to N9 so the two tables read side by side:

  `real`           held-out images of the target combination.  The ceiling, not a method.
  `ours`           condition on the composed factor vector `sum_k attr[k, v_k]` for the held-out
                   combination.  ZERO new parameters; the backbone renders it.
  `poe-guided`     the published method -- mode ascent on the score field, submodular selection,
                   per-dimension PoE, analytic-teacher CFG.
  `query-only`     the paper's own weakest baseline: a Gaussian around the single query.
  `nearest-class`  condition on the closest TRAINED combination.  The retrieval control, and on this
                   benchmark it is a genuinely strong one -- held-out combinations differ from a
                   trained one on a single attribute, so copying the neighbour gets 13 of 14 right.

THE METRIC THAT MATTERS HERE IS `target`.  A 14-way conjunction has a real-data ceiling near 0.25
and is dominated by attributes every method inherits from the neighbour.  Because each held-out
combination differs from its nearest trained one on exactly ONE attribute, the discriminative
question is whether a method flips THAT attribute while holding the other thirteen.  `nearest-class`
scores ~0 on it by construction and near-ceiling on everything else, which is exactly the
copying-vs-composing distinction N9 made with coverage.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                       # noqa: E402
from n19_celeba_train import build_probes                               # noqa: E402

from lifelong_ood.celeba import (ATTRS, as_float, class_name, load_celeba_comp,  # noqa: E402
                                 nearest_seen)
from lifelong_ood.evals import (attr_accuracy, precision_recall, sample_geometry,  # noqa: E402
                                save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, sample_embedding,  # noqa: E402
                                 to_data, to_model)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def score(probes, S01, combo, R01, target_idx):
    """-> joint / mean-attribute / TARGET-attribute accuracy + coverage geometry."""
    a = attr_accuracy(probes, np.asarray(S01, np.float32), combo)
    pr = precision_recall(np.asarray(S01, np.float32), np.asarray(R01, np.float32))
    g = sample_geometry(np.asarray(S01, np.float32), np.asarray(R01, np.float32)[0],
                        np.asarray(R01, np.float32))
    tgt = float("nan")
    if len(target_idx):
        hits = []
        for k in target_idx:
            a_k = ATTRS[k]
            if probes.get(f"{a_k}_val_acc", 0.0) >= 0.65:
                hits.append(probes[a_k].predict(np.asarray(S01, np.float32)) == int(combo[k]))
        if hits:
            tgt = float(np.stack(hits).all(0).mean())
    return dict(joint=a["joint"], mean_attr=a.get("mean_attr", float("nan")), target=tgt,
                precision=pr["precision"], recall=pr["recall"], f1=pr["f1"],
                content_ratio=g.get("content_ratio", float("nan")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_factored.pt"))
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-poe", action="store_true", help="mode ascent is the expensive arm")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_cases, args.n_samples, args.steps = 2, 8, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img = ck["img"]
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = img, 3 * img * img
    data = load_celeba_comp(img=img, min_count=60, n_holdout=len(ck["hold_ids"]),
                            seed=ck.get("seed", 0), cache_dir=CACHE,
                            verify_hold_ids=ck["hold_ids"])
    sp = data["split"]
    probes = build_probes(data, args.seed, 8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev, batch=64)
    model = field.model
    print(f"[n20] backbone {os.path.basename(args.ckpt)} | {len(sp['hold_ids'])} held-out combos "
          f"| device {dev}", flush=True)

    rng = np.random.default_rng(args.seed)
    cases = list(rng.permutation(sp["hold_ids"])[:args.n_cases])
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)

    rows, panels = [], {}
    for ci, j in enumerate(cases):
        combo = sp["all_combos"][j]
        idx = np.where(sp["combo_of"] == j)[0]
        R = as_float(data["X8"][idx])                      # real held-out images of this combo
        xq = to_model(R[:1]).ravel()                       # ONE out-of-distribution query
        nn_j, diff = nearest_seen(combo, sp)
        got = {"real": R[1:1 + args.n_samples].astype(np.float32)}

        # ── ours: the composed factor vector, zero new parameters ───────────
        c_new = model.attr_embedding(torch.as_tensor(np.asarray(combo), dtype=torch.long))
        got["ours"] = to_data(sample_embedding(field, c_new.detach(), n=args.n_samples,
                                               steps=args.steps,
                                               seed=args.seed + ci)).astype(np.float32)

        # ── nearest trained combination (retrieval control) ─────────────────
        nn_cls = ck["remap"].get(int(nn_j))
        if nn_cls is None:                                 # neighbour not in the class table
            c_nn = model.attr_embedding(torch.as_tensor(sp["all_combos"][nn_j], dtype=torch.long))
            got["nearest-class"] = to_data(sample_embedding(field, c_nn.detach(),
                                                            n=args.n_samples, steps=args.steps,
                                                            seed=args.seed + ci)).astype(np.float32)
        else:
            got["nearest-class"] = to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                        seed=args.seed + ci, clip=(-1, 1),
                                                        branch=int(nn_cls))).astype(np.float32)

        # ── the published method, and its own weakest baseline ──────────────
        if not args.skip_poe:
            t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4),
                          n_starts=4 if args.quick else 12, iters=20 if args.quick else 80)
            got["poe-guided"] = to_data(L.sample_teacher(t, n=args.n_samples,
                                                         seed=args.seed + ci)).astype(np.float32)
            tq = L.baseline_teacher(xq, kind="query_only")
            got["query-only"] = to_data(L.sample_teacher(tq, n=args.n_samples,
                                                         seed=args.seed + ci)).astype(np.float32)

        for arm, S in got.items():
            ref = R[1:] if arm == "real" else R
            rows.append(dict(arm=arm, combo=int(j), n_diff=int(len(diff)),
                             **score(probes, S, combo, ref, diff)))
        panels[int(j)] = got
        m = [x for x in rows if x["combo"] == int(j) and x["arm"] == "ours"][0]
        print(f"[n20] case {ci + 1}/{len(cases)} combo {int(j)} "
              f"[{class_name(combo)[:48]}] diff={[ATTRS[k] for k in diff]} "
              f"ours target {m['target']:.2f} F1 {m['f1']:.2f}", flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    order = [a for a in ("real", "ours", "poe-guided", "query-only", "nearest-class")
             if any(x["arm"] == a for x in rows)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = cases[:5]
    nrow = len(show) * len(order)
    fig, axes = plt.subplots(nrow, args.n_show + 1, figsize=(1.05 * (args.n_show + 1), 1.1 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for j in show:
        combo = sp["all_combos"][j]
        q = as_float(data["X8"][np.where(sp["combo_of"] == j)[0][:1]])[0]
        for arm in order:
            S = panels[int(j)].get(arm)
            axes[r, 0].imshow(q.reshape(img, img, 3))
            axes[r, 0].set_ylabel(arm, fontsize=5.5, rotation=0, ha="right", va="center")
            for c in range(args.n_show):
                if S is not None and c < len(S):
                    axes[r, 1 + c].imshow(np.clip(S[c].reshape(img, img, 3), 0, 1))
            m = [x for x in rows if x["combo"] == int(j) and x["arm"] == arm]
            if m:
                axes[r, args.n_show].set_title(
                    f"tgt {m[0]['target']:.2f} | rec {m[0]['recall']:.2f} | "
                    f"F1 {m[0]['f1']:.2f}", fontsize=5, loc="right")
            for c in range(args.n_show + 1):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            r += 1
    axes[0, 0].set_title("query", fontsize=6)
    fig.suptitle("N20 — held-out CelebA attribute combinations, every method.\n`target` is the ONE "
                 "attribute separating this combination from the nearest trained one: the test is "
                 "flipping it while holding the other thirteen.", fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "n20_celeba_composition.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    keys = ("target", "joint", "mean_attr", "precision", "recall", "f1", "content_ratio")
    summary = {a: {k: float(np.nanmean([x[k] for x in rows if x["arm"] == a])) for k in keys}
               for a in order}
    save_json(dict(rows=rows, summary=summary, n_cases=len(cases),
                   ckpt=os.path.basename(args.ckpt)),
              os.path.join(RESULTS, "n20_celeba_composition.json"), seed=args.seed)
    print(f"\n[n20] {len(cases)} held-out CelebA combinations, {args.n_samples} samples each")
    print(f"    {'method':16s}{'target':>8s}{'joint':>7s}{'mean_at':>9s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'div':>7s}")
    for a in order:
        v = summary[a]
        print(f"    {a:16s}{v['target']:8.2f}{v['joint']:7.2f}{v['mean_attr']:9.2f}"
              f"{v['precision']:7.2f}{v['recall']:8.2f}{v['f1']:7.2f}{v['content_ratio']:7.2f}")
    print(f"[n20] wrote figures/n20_celeba_composition.png")


if __name__ == "__main__":
    main()
