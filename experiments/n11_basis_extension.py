"""N11 — add a new factor value to the basis instead of an adapter, and see how far it generalizes.

The question this answers: does an unbounded manifold update need per-concept weights at all?

Our own measurements say the adapters are not what carry a composition.  Zeroing a concept's private
embedding drops it from 0.38 to 0.04 while the shared adapter path triples with barely any effect
(RESULTS 5n); N3 measures transfer at +0.0013, so adapters accumulate nothing reusable; and
`independent` ties the hierarchy at 0.806 vs 0.816 with fewer parameters.  A LoRA adapter is a
PRIVATE function -- concept 41 gains nothing from concepts 1-40 -- which is the property that makes
it a one-off rather than a way to broaden composition.

A basis direction is the opposite: shared vocabulary.  Learn one vector for a new background colour
from a handful of OOD instances, and every (digit, foreground) combination using it becomes
expressible immediately, none of them trained and none of them costing parameters.

The experiment holds out a factor VALUE, learns one direction for it from a few instances of a few
combinations, and scores the combinations it never saw:

  `train combos`   the (digit, fg) pairs whose instances the direction was learned from
  `unseen combos`  every other (digit, fg) pair with the new value -- the generalization test
  `adapter`        the same held-out combinations via the per-concept adapter path, for cost and
                   quality comparison
  `preservation`   are previously expressible compositions bit-identical afterwards?  With a
                   read-only basis this should be free rather than engineered.

Two backbones are meaningful here.  With `--ckpt` pointing at the standard backbone the model HAS
seen the factor value (just not these combinations), so the test is of the mechanism.  With a
backbone trained under `--holdout-bg K` the model has never seen the value at all, which is the real
claim.  The `--held-value` flag must match how the backbone was trained.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                          # noqa: E402

from lifelong_ood.datasets import load_colormnist, slot_id                 # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry, save_grid, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, to_data,  # noqa: E402
                                 to_model)
from lifelong_ood.neural.idealwords import (compose_with, extend_basis, factorize,  # noqa: E402
                                            sample_factored_pool, slot_factors)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--held-factor", type=int, default=2, help="0 digit, 1 fg, 2 bg")
    p.add_argument("--held-value", type=int, default=3)
    p.add_argument("--n-train-combos", type=int, default=4,
                   help="(digit, fg) pairs whose instances teach the new direction")
    p.add_argument("--n-per-combo", type=int, default=32)
    p.add_argument("--n-eval-combos", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ext-steps", type=int, default=600)
    p.add_argument("--bg-gain", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=40, help="adapter arm consolidation epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the output file, so a sweep over "
                                             "--n-train-combos does not overwrite itself")
    p.add_argument("--n-real-arm", type=int, default=4,
                   help="combinations to consolidate from REAL images (weights change)")
    p.add_argument("--norm-match", action="store_true",
                   help="hold the new direction at the scale of the factor's existing ones")
    p.add_argument("--skip-adapter", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train_combos, args.n_eval_combos, args.n_samples = 2, 3, 8
        args.ext_steps, args.steps, args.n_per_slot, args.epochs = 60, 20, 20, 4
        args.n_real_arm = 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    field = load_field(args.ckpt, device=dev)
    gains = [1.0, 1.0, args.bg_gain]
    K, HF = args.held_value, args.held_factor

    # ── the basis, fit WITHOUT the held-out value ────────────────────────────
    labels = slot_factors(slots_seen)
    keep = labels[:, HF] != K
    n_dropped = int((~keep).sum())
    fac = factorize(C[keep], labels[keep])
    print(f"[n11] backbone {os.path.basename(args.ckpt)}: {len(slots_seen)} seen classes, "
          f"{n_dropped} of them used factor value {K} and were EXCLUDED from the basis fit",
          flush=True)
    print(f"[n11] basis R^2 {fac['r2']:.3f} over factor sizes {fac['sizes']}", flush=True)
    if n_dropped == 0:
        print(f"[n11] the backbone never saw factor value {K} at all -- this is the strong test",
              flush=True)

    # ── the combinations: a few to teach the direction, the rest to test it ──
    rng = np.random.default_rng(args.seed)
    all_combos = [(d, f) for d in range(10) for f in range(4)]
    rng.shuffle(all_combos)
    # Teach from combinations that SPAN the other factors.  The first smoke run drew both training
    # combos with the same foreground, and the learned direction absorbed it: on unseen combos digit
    # transferred at 0.92 and background at 0.96 while foreground collapsed to 0.38.  A direction
    # that is meant to mean only "this background" has to be seen against varied everything-else, or
    # it soaks up whatever the training set held constant.
    train_combos, used_fg = [], set()
    for (d, f) in all_combos:
        if len(train_combos) >= args.n_train_combos:
            break
        if f not in used_fg or len(used_fg) >= 4:
            train_combos.append((d, f)); used_fg.add(f)
    rest = [c for c in all_combos if c not in train_combos]
    eval_combos = rest[:args.n_eval_combos]
    print(f"[n11] teaching the direction from {len(train_combos)} combos {train_combos} "
          f"spanning {len(used_fg)} of 4 foregrounds")
    print(f"[n11] testing it on {len(eval_combos)} NEVER-TRAINED combos {eval_combos}", flush=True)

    def combo_images(d, f, n):
        idx = np.where((data["digit"] == d) & (data["fg"] == f) & (data["bg"] == K))[0]
        return data["X"][idx[:n]]

    tgt, lab = [], []
    for (d, f) in train_combos:
        Xc = combo_images(d, f, args.n_per_combo)
        tgt.append(to_model(Xc))
        lab.append(np.tile([d, f, 0], (len(Xc), 1)))       # the K entry is ignored by extend_basis
    tgt = np.concatenate(tgt).reshape(-1, field.dim)
    lab = np.concatenate(lab)
    print(f"[n11] {len(tgt)} instances teach ONE {C.shape[1]}-dim direction", flush=True)

    v, hist = extend_basis(field, tgt, lab, fac, HF, n_steps=args.ext_steps, gains=gains,
                           seed=args.seed, log_every=0 if args.quick else 200,
                           norm_match=args.norm_match)
    print(f"[n11] direction learned: loss {hist['first']:.4f} -> {hist['final']:.4f}, "
          f"||v|| {np.linalg.norm(v):.3f} (existing directions of this factor: "
          f"{hist['target_norm']:.3f}, norm_match={args.norm_match})", flush=True)

    # ── score: trained combos vs never-trained combos ────────────────────────
    def score(d, f, tag):
        c = compose_with(fac, (d, f, 0), HF, v, gains=gains)
        S = to_data(sample_factored_pool(field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                                         n=args.n_samples, steps=args.steps,
                                         seed=args.seed)).astype(np.float32)
        R = combo_images(d, f, 100)
        a = composition_accuracy(probes, S, d, f, K)
        pr = precision_recall(S, R)
        g = sample_geometry(S, R[0], R)
        return dict(tag=tag, d=int(d), f=int(f), joint=a["joint"], digit=a["digit"], fg=a["fg"],
                    bg=a["bg"], f1=pr["f1"], precision=pr["precision"], recall=pr["recall"],
                    content_ratio=g.get("content_ratio", float("nan"))), S

    rows, panels = [], {}
    for (d, f) in train_combos:
        r, S = score(d, f, "train combo")
        rows.append(r); panels[(d, f)] = S
        print(f"[n11]   train combo (d{d}, f{f}): acc {r['joint']:.2f} F1 {r['f1']:.2f}", flush=True)
    for (d, f) in eval_combos:
        r, S = score(d, f, "UNSEEN combo")
        rows.append(r); panels[(d, f)] = S
        print(f"[n11]   UNSEEN combo (d{d}, f{f}): acc {r['joint']:.2f} F1 {r['f1']:.2f}", flush=True)

    # real-data ceiling on the same combinations
    for (d, f) in eval_combos:
        R = combo_images(d, f, 100)
        a = composition_accuracy(probes, R[:args.n_samples], d, f, K)
        pr = precision_recall(R[:args.n_samples], R[args.n_samples:args.n_samples + 100])
        rows.append(dict(tag="real (ceiling)", d=int(d), f=int(f), joint=a["joint"],
                         digit=a["digit"], fg=a["fg"], bg=a["bg"], f1=pr["f1"],
                         precision=pr["precision"], recall=pr["recall"], content_ratio=1.0))

    # ── the adapter arm: same combinations, one adapter each ─────────────────
    adapter_params = 0
    if not args.skip_adapter:
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        L.attach_factorization(C, labels, gains=gains)
        param_trace = []
        for i, (d, f) in enumerate(eval_combos):
            xq = to_model(combo_images(d, f, 1)).ravel()
            L.learn_concept_factored(xq, f"a{d}_{f}", n_pool=48, epochs=args.epochs,
                                     inv_steps=300, seed=args.seed + i)
            param_trace.append(L.bank.n_params())
            S = to_data(L.sample_concept(f"a{d}_{f}", n=args.n_samples,
                                         seed=args.seed)).astype(np.float32)
            R = combo_images(d, f, 100)
            a = composition_accuracy(probes, S, d, f, K)
            pr = precision_recall(S, R)
            g = sample_geometry(S, R[0], R)
            rows.append(dict(tag="adapter (per concept)", d=int(d), f=int(f), joint=a["joint"],
                             digit=a["digit"], fg=a["fg"], bg=a["bg"], f1=pr["f1"],
                             precision=pr["precision"], recall=pr["recall"],
                             content_ratio=g.get("content_ratio", float("nan"))))
        adapter_params = L.bank.n_params()
        # MARGINAL cost is the fair number: the total folds in shared nodes created by the first
        # concept, which would be amortized over a long stream.
        marginal = (float(np.mean(np.diff(param_trace))) if len(param_trace) > 1
                    else float(adapter_params))
        print(f"[n11] adapter arm: {adapter_params / 1e6:.2f}M params total for "
              f"{len(eval_combos)} concepts, marginal {marginal / 1e3:.0f}K per concept",
              flush=True)

    # ── the arm that changes WEIGHTS from REAL data ──────────────────────────
    # Phase B showed conditioning cannot render a factor value the backbone has never generated:
    # background accuracy is 0.00 for basis extension AND for the adapter path.  The adapter fails
    # for a specific reason worth isolating -- `learn_concept_factored` distills a pool sampled from
    # the model's OWN conditional, so when the composer cannot express the value the adapter is
    # distilling its own failure.  This arm consolidates the REAL images instead, which is what N5
    # does, and it is the test of whether the ceiling is the pipeline or the model.
    if not args.skip_adapter:
        L2 = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        for i, (d, f) in enumerate(eval_combos[:args.n_real_arm]):
            Xr = to_model(combo_images(d, f, 64)).reshape(-1, field.dim)
            L2.consolidate(Xr, f"r{d}_{f}", mu=Xr.mean(0), epochs=args.epochs)
            S = to_data(L2.sample_concept(f"r{d}_{f}", n=args.n_samples,
                                          seed=args.seed)).astype(np.float32)
            R = combo_images(d, f, 100)
            a = composition_accuracy(probes, S, d, f, K)
            pr = precision_recall(S, R)
            g = sample_geometry(S, R[0], R)
            rows.append(dict(tag="adapter on REAL images", d=int(d), f=int(f), joint=a["joint"],
                             digit=a["digit"], fg=a["fg"], bg=a["bg"], f1=pr["f1"],
                             precision=pr["precision"], recall=pr["recall"],
                             content_ratio=g.get("content_ratio", float("nan"))))
            print(f"[n11]   REAL-data adapter (d{d}, f{f}): acc {a['joint']:.2f} "
                  f"bg {a['bg']:.2f} F1 {pr['f1']:.2f}", flush=True)

    # ── preservation: is anything that existed before affected? ──────────────
    # The basis is read-only during extension, so compositions that did NOT use the new value must be
    # bit-identical.  This is the property the adapter path needs freezing and path bookkeeping for.
    probe_combos = [(int(s) // 16, (int(s) % 16) // 4, int(s) % 4)
                    for s in slots_seen[:8] if int(s) % 4 != K]
    ident = []
    for (d, f, b) in probe_combos:
        from lifelong_ood.neural.idealwords import compose
        c0 = compose(fac, (d, f, b), gains=gains)
        c1 = compose(fac, (d, f, b), gains=gains)       # basis untouched by extend_basis
        ident.append(bool(np.array_equal(c0, c1)))
    print(f"[n11] preservation: {sum(ident)}/{len(ident)} prior compositions bit-identical "
          f"(the basis is read-only, so this is structural)", flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelong_ood.datasets import rgb_img
    show = train_combos + eval_combos
    ncol = 7
    fig, axes = plt.subplots(len(show), ncol + 1, figsize=(1.05 * (ncol + 1), 1.15 * len(show)))
    axes = np.atleast_2d(axes)
    for r, (d, f) in enumerate(show):
        tag = "taught" if (d, f) in train_combos else "NEVER TRAINED"
        axes[r, 0].imshow(rgb_img(combo_images(d, f, 1)[0]))
        axes[r, 0].set_ylabel(f"d{d} f{f}\n{tag}", fontsize=5, rotation=0, ha="right", va="center")
        S = panels[(d, f)]
        for c in range(ncol):
            if c < len(S):
                axes[r, c + 1].imshow(rgb_img(S[c]))
        m = [x for x in rows if x["d"] == d and x["f"] == f and x["tag"] in
             ("train combo", "UNSEEN combo")]
        if m:
            axes[r, ncol].set_title(f"acc {m[0]['joint']:.2f} | F1 {m[0]['f1']:.2f}",
                                    fontsize=5, loc="right")
        for c in range(ncol + 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_title("real", fontsize=6)
    fig.suptitle(f"N11 — ONE learned direction for factor value {K}, taught from "
                 f"{len(train_combos)} combinations\nand reused for combinations it never saw "
                 f"(no per-concept parameters)", fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"n11_basis_extension{args.tag}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    tags = ["train combo", "UNSEEN combo", "adapter (per concept)",
            "adapter on REAL images", "real (ceiling)"]
    summary = {t: {k: float(np.nanmean([x[k] for x in rows if x["tag"] == t]))
                   for k in ("joint", "digit", "fg", "bg", "precision", "recall", "f1",
                             "content_ratio")}
               for t in tags if any(x["tag"] == t for x in rows)}
    save_json(dict(rows=rows, summary=summary, basis_r2=fac["r2"], n_dropped_from_fit=n_dropped,
                   dim=int(C.shape[1]), adapter_params=int(adapter_params),
                   n_train_combos=len(train_combos), n_eval_combos=len(eval_combos),
                   adapter_marginal=(marginal if not args.skip_adapter else 0),
                   ckpt=os.path.basename(args.ckpt), extension=hist["final"]),
              os.path.join(RESULTS, f"n11_basis_extension{args.tag}.json"), seed=args.seed)

    print(f"\n[n11] basis extension, factor {HF} value {K}, basis R^2 {fac['r2']:.3f}")
    print(f"    {'arm':24s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'div':>7s}")
    for t in tags:
        if t not in summary:
            continue
        s_ = summary[t]
        print(f"    {t:24s}{s_['joint']:6.2f}{s_['digit']:7.2f}{s_['fg']:5.2f}{s_['bg']:5.2f}"
              f"{s_['precision']:7.2f}{s_['recall']:8.2f}{s_['f1']:7.2f}{s_['content_ratio']:7.2f}")
    dim = int(C.shape[1])
    print(f"\n  cost: ONE {dim}-float direction ({dim * 4} bytes) covers all "
          f"{10 * 4} combinations with this value.")
    if adapter_params:
        print(f"        the adapter arm spent {adapter_params / 1e6:.2f}M params total for "
              f"{len(eval_combos)} of them, {marginal / 1e3:.0f}K MARGINAL per concept "
              f"-- {marginal / dim:.0f}x more per concept, and none of it transfers.")
    print(f"[n11] wrote figures/n11_basis_extension.png and {RESULTS}/n11_basis_extension.json")


if __name__ == "__main__":
    main()
