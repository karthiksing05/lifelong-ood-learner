"""N27 — is score addition's failure on ColorMNIST about the OPERATOR or about the CONDITIONALS?

Across this repository, score composition behaves in two opposite ways and the split is suspicious:

| backbone | conditioning channel | embedding composition | score addition |
|---|---|---|---|
| ColorMNIST (ours) | one entangled vector per SLOT | works, F1 0.605 | **0.000** |
| CelebA (ours) | one vector per ATTRIBUTE VALUE | works, pair 0.903 | works, 0.934-0.965 |
| CelebA-HQ (pretrained) | none at all | path union 0.17 | works, 0.50 |

The obvious reading is that ColorMNIST is simply harder.  The alternative -- and this experiment
exists to decide between them -- is that score addition never failed as an operator, and what it
lacked was **per-factor conditionals**.  ColorMNIST's backbone has 120 conditionals, one per (digit,
fg, bg) SLOT, and not one of them means "cyan foreground".  Adding entangled conditionals cannot
isolate a factor no conditional represents, so the arm is at 0.000 for a reason that has nothing to
do with the arithmetic.

The test is to GIVE the model per-factor conditionals it does not have, on the same backbone and the
same data, and re-run the same operator.  A factor conditional is built as an adapter concept:

    concept `digit=d`  trained on every SEEN image with that digit, any colours
    concept `fg=f`     trained on every SEEN image with that foreground, any digit or background
    concept `bg=b`     trained on every SEEN image with that background

Each is in-distribution -- every factor VALUE appears in the seen slots, only the COMBINATION is
held out -- so nothing about the held-out composition leaks.  A registered concept's identity travels
through its adapter path and the null row (`ddpm._cond`), which means each of these IS a conditional
on the unconditional branch, and score addition over them is composable diffusion in its intended
setting:

    eps_hat = eps_null + sum_i w_i ( eps_factor_i - eps_null )

Arms:
  `real`                held-out images of the target.
  `factor-score`        score addition over the three FACTOR adapters.        <- the test
  `factor-union`        the same three adapters, activated together instead.
  `slot-score`          score addition over the 3 nearest SLOT conditionals.  <- the old 0.000
  `ours`                our embedding-space composition, for reference.
  `factor-<k> alone`    each factor adapter by itself: did it learn its factor?

If `factor-score` works where `slot-score` does not, the failure was the conditioning granularity and
not the operator -- and the property a composition inherits is the property of the conditionals it
composes.

    python experiments/n27_factor_conditionals.py --n-cases 3
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                          # noqa: E402

from lifelong_ood.data.datasets import load_colormnist, rgb_img              # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,       # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,     # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                 # noqa: E402
from lifelong_ood.neural.inversion import teacher_init_alpha            # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def train_factor(L, name, X01, epochs=40, batch=16, lr=1e-3, seed=0):
    """Consolidate one FACTOR concept from real seen images sharing that factor value."""
    X = to_model(X01)
    L.bank.add_concept(name, X.mean(0))
    L.consolidate(X, name, mu=X.mean(0), epochs=epochs, batch=batch, lr=lr, seed=seed,
                  already_registered=True)


def score_add(field, names, n, steps, seed, scale=1.0, clip=(-1, 1)):
    """eps_null + sum_i scale * (eps_i - eps_null), each eps_i on its own adapter path.

    `field.eps(x, u, name)` already activates that concept's path AND writes its learned embedding
    into the null row, so each term is exactly the conditional that concept defines.  `guide_w` is
    1.0 so the composed epsilon is used as-is: the outer CFG would be a SECOND amplification against
    the same frozen base, and on this backbone pushing away from it is measured to destroy the
    composition (sample_concept: joint 0.38 at w=1.0, 0.00 at w>=2.0).
    """
    def eps(x, u):
        e0 = field.eps(x, u, None)
        out = e0.copy()
        for nm in names:
            out = out + float(scale) * (field.eps(x, u, nm) - e0)
        return out
    return field.sample(n=n, steps=steps, seed=seed, clip=clip,
                        guide_eps=eps, guide_w=lambda u: 1.0)


def path_union(field, bank, names, n, steps, seed, clip=(-1, 1)):
    """Activate every concept's path at once -- the naive bank composition (RESULTS 12e)."""
    prev = list(bank.state["path"])
    path = sorted({p for nm in names for p in bank.concepts[nm]["path"]})
    emb = torch.stack([bank._emb_extra[nm].detach() for nm in names]).mean(0)

    def eps(x, u):
        bank.state["path"] = path
        try:
            return field.eps(x, u, None, emb=emb)
        finally:
            bank.state["path"] = prev
    try:
        return field.sample(n=n, steps=steps, seed=seed, clip=clip,
                            guide_eps=eps, guide_w=lambda u: 1.0)
    finally:
        bank.state["path"] = prev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--n-train", type=int, default=96, help="real images per FACTOR concept")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                   help="score-addition weight; swept so the arm is reported at its best")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.n_per_slot = 1, 8, 20
        args.n_train, args.epochs, args.steps, args.inv_steps = 24, 4, 20, 60
        args.scales = [1.0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    seen = data["seen"]
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    print(f"[n27] {len(qs)} held-out compositions; factor concepts built from SEEN images only",
          flush=True)

    rows, panels = [], {}
    for qi, q in enumerate(qs):
        d, f, b = int(q["d"]), int(q["f"]), int(q["b"])
        R = data["X"][q["rest"][:100]].astype(np.float32)
        got = {"real": R[:args.n_samples]}

        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev, flat=True)
        L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 1.0])

        # ── build one conditional per FACTOR VALUE, from seen images only ────
        rng = np.random.default_rng(args.seed + qi)
        specs = [(f"digit{d}", seen & (data["digit"] == d)),
                 (f"fg{f}", seen & (data["fg"] == f)),
                 (f"bg{b}", seen & (data["bg"] == b))]
        names = []
        for nm, mask in specs:
            idx = np.where(mask)[0]
            rng.shuffle(idx)
            idx = idx[:args.n_train]
            t0 = time.time()
            train_factor(L, nm, data["X"][idx], epochs=args.epochs, seed=args.seed)
            names.append(nm)
            print(f"[n27] case {q['slot']}: `{nm}` from {len(idx)} seen images "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # each factor alone -- did the adapter learn its own factor?
        for nm in names:
            got[f"{nm} alone"] = to_data(L.sample_concept(
                nm, n=args.n_samples, steps=args.steps, seed=args.seed)).astype(np.float32)

        # ── THE TEST: score addition over factor conditionals, swept ────────
        best, best_j = None, -1.0
        for sc in args.scales:
            S = to_data(L.sample_composed(names, n=args.n_samples, steps=args.steps,
                                          seed=args.seed, scale=sc)).astype(np.float32)
            j = composition_accuracy(probes, S, d, f, b)["joint"]
            print(f"[n27]   factor-score scale {sc}: joint {j:.3f}", flush=True)
            if j > best_j:
                best, best_j = S, j
        got["factor-score"] = best

        got["factor-union"] = to_data(path_union(field, L.bank, names, args.n_samples,
                                                 args.steps, args.seed)).astype(np.float32)

        # ── the old arm: score addition over ENTANGLED slot conditionals ────
        xq = to_model(data["X"][q["query"]]).ravel()
        t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=8 if args.quick else 16,
                      iters=30 if args.quick else 120)
        a0 = teacher_init_alpha(field, t, xq, seed=args.seed)
        topk = np.argsort(-a0)[:3]
        null_row = ck["ema"]["cemb.weight"].numpy()[field.model.null_id]
        from lifelong_ood.neural.composable import compose_classes
        got["slot-score"] = to_data(compose_classes(
            field, C, topk, n=args.n_samples, steps=args.steps, seed=args.seed,
            null_emb=null_row, scale=1.0)).astype(np.float32)

        # ── our embedding-space composition, same backbone ──────────────────
        f2 = load_field(args.ckpt, device=dev)
        L2 = NeuralLifelongLearner(f2, steps=args.steps, seed=args.seed, device=dev)
        L2.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 1.0])
        L2.learn_concept_factored(xq, f"c{q['slot']}", n_pool=48, epochs=args.epochs,
                                  inv_steps=args.inv_steps, seed=args.seed,
                                  guidance=args.guidance)
        got["ours"] = to_data(L2.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                                seed=args.seed)).astype(np.float32)

        for arm, S in got.items():
            a = composition_accuracy(probes, S, d, f, b)
            pr = precision_recall(S, R)
            rows.append(dict(arm=arm, slot=int(q["slot"]), joint=a["joint"], digit=a["digit"],
                             fg=a["fg"], bg=a["bg"], precision=pr["precision"],
                             recall=pr["recall"], f1=pr["f1"], vendi=vendi_score(S)))
        panels[int(q["slot"])] = got
        print(f"[n27] case {q['slot']} done", flush=True)
        del field, L, f2, L2
        torch.cuda.empty_cache()

    order = ["real", "ours", "factor-score", "factor-union", "slot-score"]
    order += sorted({r["arm"] for r in rows if r["arm"].endswith("alone")})
    summary = {a: {k: float(np.nanmean([r[k] for r in rows if r["arm"] == a]))
                   for k in ("joint", "digit", "fg", "bg", "precision", "recall", "f1", "vendi")}
               for a in order if any(r["arm"] == a for r in rows)}
    save_json(dict(rows=rows, summary=summary, config=vars(args)),
              os.path.join(RESULTS, f"n27_factor_conditionals{args.tag}.json"), seed=args.seed)

    print(f"\n[n27] {len(qs)} held-out compositions, {args.n_samples} samples")
    print(f"    {'arm':18s}{'joint':>8s}{'digit':>7s}{'fg':>6s}{'bg':>6s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'vendi':>7s}")
    for a in order:
        if a not in summary:
            continue
        s = summary[a]
        print(f"    {a:18s}{s['joint']:8.3f}{s['digit']:7.3f}{s['fg']:6.2f}{s['bg']:6.2f}"
              f"{s['precision']:7.3f}{s['recall']:8.3f}{s['f1']:7.3f}{s['vendi']:7.2f}")
    if "factor-score" in summary and "slot-score" in summary:
        fs, ss = summary["factor-score"]["joint"], summary["slot-score"]["joint"]
        print(f"\n[n27] factor conditionals {fs:.3f} vs slot conditionals {ss:.3f} "
              f"-- {'the OPERATOR was fine; the conditionals were not' if fs > ss + 0.05 else 'no separation: the operator itself fails here'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [a for a in order if a in summary]
    nrow = len(panels) * len(show)
    ncol = min(6, args.n_samples)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.15 * ncol + 3.0, 1.2 * nrow))
    axes = np.atleast_2d(axes)
    r = 0
    for slot, got in panels.items():
        for a in show:
            S = got.get(a)
            for c in range(ncol):
                ax = axes[r][c]; ax.axis("off")
                if S is not None and c < len(S):
                    ax.imshow(rgb_img(S[c]))
            axes[r][0].set_ylabel(f"slot {slot}\n{a}", rotation=0, ha="right", va="center",
                                  fontsize=7)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp in axes[r][0].spines.values():
                sp.set_visible(False)
            r += 1
    fig.suptitle("ColorMNIST: score addition over FACTOR conditionals vs over SLOT conditionals",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, f"n27_factor_conditionals{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[n27] wrote {out}")


if __name__ == "__main__":
    main()
