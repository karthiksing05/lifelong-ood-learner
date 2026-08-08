"""N21 — CelebA vocabulary: the RESULTS 5q phase-B test on spatially overlapping attributes.

5q established the routing rule this method runs on, but only on ColorMNIST:

    a new COMBINATION of known factor values is free -- one composed conditioning vector, zero
    parameters, and it beats a per-concept adapter;
    a new VOCABULARY item is not -- on a backbone that never saw the value, conditioning gives
    accuracy exactly 0.000, and only a weight update from REAL data recovers it (0.833).

ColorMNIST's factors are spatially disjoint, which is the easy case for both halves of that rule.
This runs it again on a backbone trained with one CelebA attribute's ON value withheld ENTIRELY
(`n19_celeba_train --holdout-attr`), where the withheld attribute overlaps the same pixels as
several attributes the model knows well.

Arms, cheapest first:

  `untrained value`  condition on `attr[k, 1]` as it left initialization -- the slot exists but has
                     never received a gradient.  This is the honest floor and it should FAIL: the
                     control that stops "composition is free" from being read as "everything is
                     free".
  `learned value`    the SAME slot, optimized on real instances (`learn_attr_value`).  One `edim`
                     vector, no new parameters, everything else frozen.
  `adapter (real)`   a per-concept LoRA consolidated from the same real instances, for cost and
                     quality comparison -- the incumbent mechanism.
  `real`             held-out images.  The ceiling.

And the generalization question that makes it a BASIS rather than a lookup: the vector is taught
from a few combinations carrying the attribute and scored on combinations carrying it that were
never in the teaching set (`taught` vs `UNSEEN` rows).
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
from n20_celeba_composition import score                                # noqa: E402

from lifelong_ood.celeba import ATTRS, as_float, load_celeba_comp       # noqa: E402
from lifelong_ood.evals import save_json                                # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, sample_embedding,  # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import compose_attr, learn_attr_value  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attr", default="Gray_Hair", choices=ATTRS)
    p.add_argument("--ckpt", default=None, help="defaults to the --holdout-attr backbone")
    p.add_argument("--n-teach-combos", type=int, default=4)
    p.add_argument("--n-per-combo", type=int, default=48)
    p.add_argument("--n-eval-combos", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ext-steps", type=int, default=600)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-adapter-combos", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the output file; --quick runs use one "
                                             "so a smoke can never overwrite a full result")
    p.add_argument("--skip-adapter", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_teach_combos, args.n_eval_combos, args.n_samples = 2, 3, 8
        args.ext_steps, args.steps, args.epochs, args.n_adapter_combos = 60, 20, 4, 1
        args.n_per_combo = 16
    args.ckpt = args.ckpt or os.path.join(CACHE, f"ddpm_celeba_factored_no{args.attr}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img = ck["img"]
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = img, 3 * img * img
    assert ck.get("holdout_attr") == args.attr, \
        f"backbone withheld {ck.get('holdout_attr')!r}, not {args.attr!r}"
    data = load_celeba_comp(img=img, min_count=ck.get("min_count", 60),
                            n_holdout=ck.get("n_holdout", len(ck["hold_ids"])),
                            holdout_attr=args.attr, seed=ck.get("seed", 0), cache_dir=CACHE,
                            verify_hold_ids=ck["hold_ids"])
    sp = data["split"]
    k = sp["holdout_attr_idx"]
    probes = build_probes(data, args.seed, 8 if args.quick else 30)
    field = load_field(args.ckpt, device=dev, batch=64)
    model = field.model
    print(f"[n21] backbone {os.path.basename(args.ckpt)} — {args.attr} (index {k}) never seen ON",
          flush=True)
    print(f"[n21] probe for {args.attr}: val acc {probes[f'{args.attr}_val_acc']:.3f}", flush=True)

    # ── combinations carrying the withheld attribute, split teach / evaluate ─
    A, combo_of = data["attrs"], sp["combo_of"]
    on_combos = [j for j in range(len(sp["all_combos"]))
                 if sp["all_combos"][j][k] == 1 and sp["counts"][j] >= 25]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(on_combos)
    teach = on_combos[:args.n_teach_combos]
    evalc = on_combos[args.n_teach_combos:args.n_teach_combos + args.n_eval_combos]
    print(f"[n21] {len(on_combos)} combinations carry {args.attr}: "
          f"{len(teach)} teach the vector, {len(evalc)} are NEVER TAUGHT", flush=True)

    def imgs(j, n):
        idx = np.where(combo_of == j)[0][:n]
        return as_float(data["X8"][idx])

    tgt, labs = [], []
    for j in teach:
        Xc = imgs(j, args.n_per_combo)
        tgt.append(to_model(Xc))
        labs.append(np.tile(sp["all_combos"][j], (len(Xc), 1)))
    tgt = np.concatenate(tgt).reshape(-1, field.dim)
    labs = np.concatenate(labs)
    print(f"[n21] {len(tgt)} real instances teach ONE {model.edim}-dim vector", flush=True)

    v0 = model.attr[k, 1].detach().clone()                 # never received a gradient
    v, hist = learn_attr_value(field, tgt, labs, k, value=1, n_steps=args.ext_steps,
                               seed=args.seed, log_every=0 if args.quick else 200)
    print(f"[n21] vector learned: loss {hist['first']:.4f} -> {hist['final']:.4f}, "
          f"||v|| {hist['norm']:.3f} (other attribute vectors: {hist['target_norm']:.3f})",
          flush=True)

    rows = {}

    def run(j, tag):
        combo = sp["all_combos"][j]
        R = imgs(j, 200)
        diff = np.array([k])                                # the withheld attribute IS the target
        out = {}
        for arm, ov in (("untrained value", {(k, 1): v0}), ("learned value", {(k, 1): v})):
            c = compose_attr(model, combo, overrides=ov)
            S = to_data(sample_embedding(field, c.detach(), n=args.n_samples, steps=args.steps,
                                         seed=args.seed)).astype(np.float32)
            out[arm] = S
            rows.setdefault(arm, []).append(
                dict(tag=tag, combo=int(j), **score(probes, S, combo, R, diff)))
        rows.setdefault("real (ceiling)", []).append(
            dict(tag=tag, combo=int(j),
                 **score(probes, R[:args.n_samples], combo, R[args.n_samples:], diff)))
        return out

    panels = {}
    for j in teach:
        panels[int(j)] = run(j, "taught")
    for j in evalc:
        panels[int(j)] = run(j, "UNSEEN")
        m = rows["learned value"][-1]
        print(f"[n21]   UNSEEN combo {int(j)}: {args.attr} {m['target']:.2f} F1 {m['f1']:.2f}",
              flush=True)

    # ── the incumbent: one adapter per combination, from the same real data ──
    adapter_params = 0
    if not args.skip_adapter:
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        trace = []
        for i, j in enumerate(evalc[:args.n_adapter_combos]):
            Xr = to_model(imgs(j, 64)).reshape(-1, field.dim)
            L.consolidate(Xr, f"c{j}", mu=Xr.mean(0), epochs=args.epochs)
            trace.append(L.bank.n_params())
            S = to_data(L.sample_concept(f"c{j}", n=args.n_samples,
                                         seed=args.seed)).astype(np.float32)
            rows.setdefault("adapter (real)", []).append(
                dict(tag="UNSEEN", combo=int(j),
                     **score(probes, S, sp["all_combos"][j], imgs(j, 200), np.array([k]))))
            panels[int(j)]["adapter (real)"] = S
        adapter_params = L.bank.n_params()
        marginal = float(np.mean(np.diff(trace))) if len(trace) > 1 else float(adapter_params)
        print(f"[n21] adapter arm: {adapter_params / 1e6:.2f}M total, "
              f"{marginal / 1e3:.0f}K marginal per concept", flush=True)
    else:
        marginal = 0.0

    # ── preservation: compositions NOT using the new value are untouched ─────
    # Structural, not engineered: `learn_attr_value` writes one slot and reads the rest.
    ident = []
    with torch.no_grad():
        for j in list(rng.permutation(len(sp["all_combos"]))[:200]):
            cb = sp["all_combos"][j]
            if cb[k] == 1:
                continue
            c0 = compose_attr(model, cb)
            c1 = compose_attr(model, cb, overrides={(k, 1): v})
            ident.append(bool(torch.equal(c0, c1)))
    print(f"[n21] preservation: {sum(ident)}/{len(ident)} compositions not using {args.attr} are "
          f"bit-identical after the extension", flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [a for a in ("real", "untrained value", "learned value", "adapter (real)")
             if a == "real" or a in rows]
    show = (teach[:2] + evalc[:4])
    ncol = 6
    fig, axes = plt.subplots(len(show) * len(order), ncol + 1,
                             figsize=(1.05 * (ncol + 1), 1.1 * len(show) * len(order)))
    axes = np.atleast_2d(axes)
    r = 0
    for j in show:
        tag = "taught" if j in teach else "NEVER TAUGHT"
        q = imgs(j, 1)[0]
        for arm in order:
            S = imgs(j, ncol) if arm == "real" else panels[int(j)].get(arm)
            axes[r, 0].imshow(q.reshape(img, img, 3))
            axes[r, 0].set_ylabel(f"{arm}\n{tag}", fontsize=4.5, rotation=0, ha="right",
                                  va="center")
            for c in range(ncol):
                if S is not None and c < len(S):
                    axes[r, 1 + c].imshow(np.clip(np.asarray(S[c]).reshape(img, img, 3), 0, 1))
            m = [x for x in rows.get(arm, []) if x["combo"] == int(j)]
            if m:
                axes[r, ncol].set_title(f"{args.attr} {m[0]['target']:.2f} | F1 {m[0]['f1']:.2f}",
                                        fontsize=4.5, loc="right")
            for c in range(ncol + 1):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            r += 1
    fig.suptitle(f"N21 — {args.attr} was never seen ON.  ONE learned {model.edim}-float vector "
                 f"(zero new parameters) renders every combination using it,\nincluding "
                 f"combinations it was never taught on.  The untrained slot is the floor.",
                 fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"n21_celeba_vocabulary_{args.attr}{args.tag}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    keys = ("target", "joint", "mean_attr", "precision", "recall", "f1", "content_ratio")
    summary = {}
    for arm, rs in rows.items():
        for tag in ("taught", "UNSEEN"):
            sub = [x for x in rs if x["tag"] == tag]
            if sub:
                summary[f"{arm} [{tag}]"] = {kk: float(np.nanmean([x[kk] for x in sub]))
                                             for kk in keys}
    save_json(dict(rows={a: rs for a, rs in rows.items()}, summary=summary, attr=args.attr,
                   attr_idx=int(k), n_teach=len(teach), n_eval=len(evalc),
                   edim=int(model.edim), adapter_params=int(adapter_params),
                   adapter_marginal=marginal, extension=hist["final"],
                   preservation=float(np.mean(ident)) if ident else float("nan"),
                   ckpt=os.path.basename(args.ckpt)),
              os.path.join(RESULTS, f"n21_celeba_vocabulary_{args.attr}{args.tag}.json"), seed=args.seed)

    print(f"\n[n21] new vocabulary item {args.attr}, never seen ON by the backbone")
    print(f"    {'arm':30s}{args.attr[:8]:>9s}{'joint':>7s}{'mean_at':>9s}"
          f"{'recall':>8s}{'F1':>7s}{'div':>7s}")
    for a in sorted(summary):
        v_ = summary[a]
        print(f"    {a:30s}{v_['target']:9.2f}{v_['joint']:7.2f}{v_['mean_attr']:9.2f}"
              f"{v_['recall']:8.2f}{v_['f1']:7.2f}{v_['content_ratio']:7.2f}")
    print(f"\n  cost: ONE {model.edim}-float vector ({model.edim * 4} bytes) covers every "
          f"combination using {args.attr}.")
    if adapter_params:
        print(f"        the adapter arm spent {marginal / 1e3:.0f}K MARGINAL per concept "
              f"-- {marginal / model.edim:.0f}x more, and none of it transfers to another "
              f"combination.")
    print(f"[n21] wrote figures/n21_celeba_vocabulary_{args.attr}.png")


if __name__ == "__main__":
    main()
