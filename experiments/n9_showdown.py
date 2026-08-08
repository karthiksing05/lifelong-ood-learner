"""N9 — the showdown: our composer against every baseline, on the same held-out compositions.

This is the figure the project is for.  One out-of-distribution composition per row; each method
gets the same query, the same backbone and the same sampling budget; each panel is annotated with
the accuracy, recall and content diversity measured on exactly the samples shown.

Methods, and what each is a fair test of:

  `real`             held-out images of the target composition.  Not a method -- the ceiling, and
                     the thing every row above is trying to look like.
  `ours`             teacher discovers -> ideal-words factor basis reads the composition -> per-factor
                     gain corrects the model's under-response -> the MANIFOLD generates -> consolidated.
  `poe-guided`       the published method: CFG toward the analytic PoE Gaussian.
  `inversion`        conditioning learned in the entangled class basis (our previous best).
  `nearest-class`    condition on the single closest trained class.  The retrieval control.
  `finetune-stream`  the same concept, but with the shared adapter left unprotected while REAL
                     held-out images of other concepts stream past it.  Real images are used
                     deliberately: a composed pool delivers no gradient to the shared path
                     (RESULTS 5m), so a composed stream would leave every arm untouched and test
                     nothing.  With real images the shared path genuinely moves -- and this row
                     STILL matches ours, which is the measurement behind RESULTS 5n: a factored
                     concept's semantics live in its private embedding, which no later concept can
                     write to, so the composition is preserved by isolation of the conditioning
                     rather than by protecting the adapter.  Read this row as evidence for that,
                     not as a forgetting demonstration -- forgetting is measured in n5_classinc,
                     where the adapter carries the content and protection is the whole story.
  `ttt-stream`       test-time training on the same stream: re-fits from stored data on demand.

The last two are the point of putting baselines in a CASE-STUDY figure rather than only in a table.
A table says finetune loses 0.2 accuracy; the figure says what that looks like -- the composition
drifting to a different colour or digit while the row above holds it.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries         # noqa: E402

from lifelong_ood.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, learn_embedding,  # noqa: E402
                                 load_field, sample_embedding, to_data, to_model)
from lifelong_ood.neural.idealwords import slot_factors                # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def score(probes, S01, q, R):
    pr = precision_recall(S01, R)
    g = sample_geometry(S01, R[0], R)
    a = composition_accuracy(probes, S01, q["d"], q["f"], q["b"])
    return dict(joint=a["joint"], digit=a["digit"], fg=a["fg"], bg=a["bg"],
                precision=pr["precision"], recall=pr["recall"], f1=pr["f1"],
                content_ratio=g.get("content_ratio", float("nan")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=5)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bg-gain", type=float, default=-1.0,
                   help="fixed background gain; <=0 calibrates it on this backbone")
    p.add_argument("--stream-len", type=int, default=6,
                   help="concepts learned AFTER the case, so the forgetting arms have forgotten")
    p.add_argument("--stream-source", choices=("real", "factored"), default="real",
                   help="real held-out images actually move the shared path; composed pools do not")
    p.add_argument("--stream-train", type=int, default=64, help="real images per follow-on concept")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_cases, args.n_samples, args.n_per_slot = 2, 12, 20
        args.steps, args.inv_steps, args.epochs, args.stream_len = 20, 60, 4, 2
        args.stream_train = 16
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    # calibrated on THIS backbone from seen slots, not hardcoded: the constant 2.0 was tuned
    # against one checkpoint and does not transfer to a retrained one (RESULTS 5p).
    field0 = load_field(args.ckpt, device=dev)
    gains = ([1.0, 1.0, args.bg_gain] if args.bg_gain > 0 else
             backbone_gains(field0, C, slots_seen, probes, data, seed=args.seed,
                            steps=args.steps, n=8 if args.quick else 16))

    qs = ood_queries(data, n_slots=args.n_cases + args.stream_len, seed=args.seed)
    cases, stream = qs[:args.n_cases], qs[args.n_cases:]
    print(f"[n9] {len(cases)} cases, {len(stream)} follow-on concepts, device {dev}", flush=True)

    rows, panels = [], {}

    def run_stream(L, tag):
        """The interference every retention claim has to survive.

        With `--stream-source real` this consolidates real held-out images, which is what actually
        moves the shared adapter path (RESULTS 5m).  The earlier composed-pool version of this loop
        left the shared path bit-identical, so `finetune-stream` was never under any threat and the
        row could not show what it exists to show.
        """
        for si, q in enumerate(stream):
            nm = f"s{q['slot']}"
            if args.stream_source == "real":
                idx = np.concatenate([[q["query"]], q["rest"]])[:args.stream_train]
                X = to_model(data["X"][idx]).reshape(len(idx), -1)
                L.consolidate(X, nm, mu=X.mean(0), epochs=args.epochs)
            else:
                L.learn_concept_factored(to_model(data["X"][q["query"]]).ravel(), nm,
                                         n_pool=48, epochs=args.epochs, inv_steps=args.inv_steps,
                                         seed=args.seed + 100 + si)
        print(f"[n9] {tag}: absorbed {len(stream)} follow-on concepts "
              f"({args.stream_source})", flush=True)

    def fresh():
        f = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(f, steps=args.steps, seed=args.seed, device=dev)
        L.attach_factorization(C, slot_factors(slots_seen), gains=gains)
        return f, L

    # ── OURS: learn each case, then keep learning; the case must survive ─────
    field, L = fresh()
    for qi, q in enumerate(cases):
        L.learn_concept_factored(to_model(data["X"][q["query"]]).ravel(), f"c{q['slot']}",
                                 n_pool=48, epochs=args.epochs, inv_steps=args.inv_steps,
                                 seed=args.seed + qi)
    run_stream(L, "ours")                     # the interference the method must survive
    for q in cases:
        S = to_data(L.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                     seed=args.seed)).astype(np.float32)
        panels.setdefault(q["slot"], {})["ours"] = S
        rows.append(dict(arm="ours", slot=q["slot"],
                         **score(probes, S, q, data["X"][q["rest"][:100]])))

    # ── FINETUNE / TTT streams: same concepts, unprotected shared adapter ────
    for arm, freeze in (("finetune-stream", False),):
        f2 = load_field(args.ckpt, device=dev)
        L2 = NeuralLifelongLearner(f2, steps=args.steps, seed=args.seed, device=dev,
                                   flat=True, freeze_ancestors=freeze)
        L2.attach_factorization(C, slot_factors(slots_seen), gains=gains)
        for qi, q in enumerate(cases):
            L2.learn_concept_factored(to_model(data["X"][q["query"]]).ravel(), f"c{q['slot']}",
                                      n_pool=48, epochs=args.epochs, inv_steps=args.inv_steps,
                                      seed=args.seed + qi)
        run_stream(L2, arm)
        for q in cases:
            S = to_data(L2.sample_concept(f"c{q['slot']}", n=args.n_samples,
                                          seed=args.seed)).astype(np.float32)
            panels[q["slot"]][arm] = S
            rows.append(dict(arm=arm, slot=q["slot"],
                             **score(probes, S, q, data["X"][q["rest"][:100]])))

    # ── the composition baselines, no stream (they are one-shot methods) ─────
    f3, L3 = fresh()
    for qi, q in enumerate(cases):
        xq = to_model(data["X"][q["query"]])
        R = data["X"][q["rest"][:100]]
        t = L3.compose(xq.ravel(), u_grid=(0.1, 0.2, 0.3, 0.4),
                       n_starts=8 if args.quick else 16, iters=30 if args.quick else 120)
        # THE PUBLISHED METHOD IN ITS PUBLISHED CONFIGURATION.  This arm used to inherit our
        # `schedule="anneal"` default, which rises to wmax as u -> 0 and, against a teacher whose
        # Eq. 12 precision blows up at low noise, injected the speckle the case-study figure showed.
        # Their Eq. 14 exists to cancel exactly that (`w = min(w0*tr(Sigma_t)/d, wmax)`), and with it
        # the same teacher produces CLEAN, correct digits (tests/diag_poe_faithful.py: accuracy
        # 0.865 against our default's 0.521).  Their method must be run their way, or the comparison
        # is against a strawman -- and the honest result is stronger, because what remains is not bad
        # images but eight identical ones (recall 0.000, content diversity 0.24).
        got = {"poe-guided": to_data(L3.sample_teacher(t, n=args.n_samples, seed=args.seed + qi,
                                                       schedule="variance", var_scale=1.0,
                                                       w0=1.2, wmax=2.0,
                                                       wmin=0.0)).astype(np.float32)}
        c_cls, _, _ = learn_embedding(f3, xq, n_steps=args.inv_steps, seed=args.seed)
        got["inversion"] = to_data(sample_embedding(f3, c_cls, n=args.n_samples, steps=args.steps,
                                                    seed=args.seed + qi)).astype(np.float32)
        from lifelong_ood.neural import teacher_init_alpha
        a0 = teacher_init_alpha(f3, t, xq, seed=args.seed)
        got["nearest-class"] = to_data(f3.sample(n=args.n_samples, steps=args.steps,
                                                 seed=args.seed + qi, clip=(-1, 1),
                                                 branch=int(np.argmax(a0)))).astype(np.float32)
        got["real"] = R[:args.n_samples].astype(np.float32)
        for arm, S in got.items():
            panels[q["slot"]][arm] = S
            rows.append(dict(arm=arm, slot=q["slot"], **score(probes, S, q, R)))
        print(f"[n9] case {q['slot']} done", flush=True)

    # ── the figure ───────────────────────────────────────────────────────────
    order = ["real", "ours", "poe-guided", "inversion", "nearest-class", "finetune-stream"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelong_ood.datasets import rgb_img
    nrow = len(cases) * len(order)
    fig, axes = plt.subplots(nrow, args.n_show + 1, figsize=(1.05 * (args.n_show + 1), 1.1 * nrow))
    r = 0
    for q in cases:
        for arm in order:
            S = panels[q["slot"]].get(arm)
            axes[r, 0].imshow(rgb_img(data["X"][q["query"]]))
            axes[r, 0].set_ylabel(f"{arm}", fontsize=5.5, rotation=0, ha="right", va="center")
            for c in range(args.n_show):
                if S is not None and c < len(S):
                    axes[r, 1 + c].imshow(rgb_img(S[c]))
            m = [x for x in rows if x["slot"] == q["slot"] and x["arm"] == arm]
            if m:
                axes[r, args.n_show].set_title(
                    f"acc {m[0]['joint']:.2f} | rec {m[0]['recall']:.2f} | "
                    f"div {m[0]['content_ratio']:.2f}", fontsize=5, loc="right")
            for c in range(args.n_show + 1):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            r += 1
        if r < nrow:
            for c in range(args.n_show + 1):
                axes[r - 1, c].spines[:].set_color("red")
    axes[0, 0].set_title("query", fontsize=6)
    fig.suptitle("N9 — the same OOD composition, every method, annotated with what was measured on "
                 "exactly these samples\n(`real` is the ceiling; `finetune-stream` is the same "
                 "concept after a real-image stream ran over an unprotected shared adapter — it "
                 "holds, because the composition lives in a private embedding: RESULTS §5n)",
                 fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "n9_showdown.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    arms = order
    summary = {a: {k: float(np.nanmean([x[k] for x in rows if x["arm"] == a]))
                   for k in ("joint", "digit", "fg", "bg", "precision", "recall", "f1",
                             "content_ratio")} for a in arms if any(x["arm"] == a for x in rows)}
    save_json(dict(rows=rows, summary=summary), os.path.join(RESULTS, "n9_showdown.json"),
              seed=args.seed)
    print(f"\n[n9] {len(cases)} held-out compositions, {args.n_samples} samples each")
    print(f"    {'method':18s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'content div':>13s}")
    for a in arms:
        if a not in summary:
            continue
        v = summary[a]
        print(f"    {a:18s}{v['joint']:6.2f}{v['digit']:7.2f}{v['fg']:5.2f}{v['bg']:5.2f}"
              f"{v['precision']:7.2f}{v['recall']:8.2f}{v['f1']:7.2f}{v['content_ratio']:13.2f}")
    print(f"[n9] wrote figures/n9_showdown.png and {RESULTS}/n9_showdown.json")


if __name__ == "__main__":
    main()
