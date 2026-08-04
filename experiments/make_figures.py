"""Build the figures used by RESULTS.md into `figures/` (tracked, unlike results/).

Two kinds:

  CURVES, drawn from the result JSONs -- no model needed, runs anywhere in seconds.
  CASE STUDIES, which need the trained checkpoint and a GPU: for a handful of held-out
  compositions, the query beside the composed prototype and the samples drawn from it, annotated
  with the accuracy and the diversity actually measured on those exact samples.

The case-study panels annotate `div/ref`: sample diversity divided by the real class's own internal
spread. 1.0 is the target. Below ~0.5 the sampler is emitting near-copies; above ~1.5 it is off the
manifold. Printing it on the figure keeps the picture honest -- a grid of pretty digits that are all
the same digit should not read as success.

    python experiments/make_figures.py --curves-only     # no GPU
    python experiments/make_figures.py                   # + case studies (needs the checkpoint)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# `hier` and `flat` coincide exactly on the preservation panels -- both are perfect -- so the
# frozen arms are drawn with different linestyles and marker sizes, otherwise one hides the other
# and the figure silently shows two curves as one.
ARM_STYLE = {
    "hier": dict(color="#1f77b4", marker="o", ms=7, lw=2.4,
                 label="ours (hierarchical, frozen)"),
    "ours": dict(color="#1f77b4", marker="o", ms=7, lw=2.4,
                 label="ours (hierarchical, frozen)"),
    "flat": dict(color="#2ca02c", marker="s", ms=4, lw=1.4, ls="--",
                 label="flat (isolated adapters)"),
    "independent": dict(color="#2ca02c", marker="s", ms=4, lw=1.4, ls="--",
                        label="independent (isolation)"),
    "unfrozen": dict(color="#d62728", marker="v", ms=6, lw=2.0,
                     label="unfrozen (shared, trainable)"),
    "finetune": dict(color="#d62728", marker="v", ms=6, lw=2.0,
                     label="finetune (shared, sequential)"),
    "replay": dict(color="#ff7f0e", marker="^", ms=6, lw=2.0, ls=":",
                   label="+ generative replay"),
}


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def style(arm):
    return ARM_STYLE.get(arm, dict(color="grey", marker="x", label=arm))


# ── curves ───────────────────────────────────────────────────────────────────
def fig_lifelong_curves():
    """N3: what happens to the manifold as concepts are added, one panel per thing at risk."""
    r = load("n3_lifelong")
    if not r:
        return None
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for arm, blob in r.items():
        st = blob["steps"]
        ks = [s["step"] for s in st]
        s_ = style(arm)
        ax[0].plot(ks, [s["base_drift"] for s in st], **s_)
        ax[1].plot(ks, [s["retention_exact"] for s in st], **s_)
        ka = [s["step"] for s in st if s.get("retention_acc") is not None]
        va = [s["retention_acc"] for s in st if s.get("retention_acc") is not None]
        ax[2].plot(ka, va, **s_)
        kn = [s["step"] for s in st if "new_acc" in s]
        vn = [s["new_acc"]["joint"] for s in st if "new_acc" in s]
        ax[3].plot(kn, vn, **s_)
    for a, t, yl in zip(ax,
                        ("frozen base: drift (want 0)",
                         "retention: fraction bit-identical",
                         "retention: composition accuracy",
                         "plasticity: accuracy of the new concept"),
                        ("Δ ε-error", "fraction", "joint accuracy", "joint accuracy")):
        a.set_title(t, fontsize=10)
        a.set_xlabel("concepts consolidated")
        a.set_ylabel(yl, fontsize=9)
        a.grid(alpha=.25)
    ax[1].set_ylim(-.05, 1.05)
    ax[0].legend(fontsize=8)
    # Panels 3 and 4 are the two that are easy to misread, so each carries its caveat -- placed in
    # reserved space BELOW the data rather than over it, which the first version of this figure did.
    for a, note in ((ax[2], "wanders because the probed SUBSET changes each step,\n"
                            "not because any concept moved — see panel 2"),
                    (ax[3], "dips to 0 where the COMPOSITION failed (bad pool),\n"
                            "not where consolidation failed")):
        a.set_ylim(-.34, 1.05)
        a.axhline(0, color="k", lw=.6, alpha=.4)
        a.text(.5, .02, note, transform=a.transAxes, fontsize=6.5, ha="center", alpha=.8)
    fig.suptitle("N3 — adding 10 out-of-distribution concepts one at a time", fontsize=11)
    fig.tight_layout()
    return save(fig, "n3_curves.png")


def fig_classinc():
    """N5: the standard class-incremental picture -- average accuracy over the seen tasks."""
    r = load("n5_classinc")
    if not r:
        return None
    rows = r["rows"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for row in rows:
        acc = np.array(row["acc_matrix"], dtype=object)
        T = max(len(a) for a in acc)
        # average accuracy over tasks learned so far, after each step
        avg = []
        for t in range(T):
            vals = [a[t] for a in acc if len(a) > t]
            avg.append(np.mean(vals))
        s_ = style(row["arm"])
        ax[0].plot(range(1, T + 1), avg, **s_)
        first = [a[0] for a in acc]
        last = [a[-1] for a in acc]
        ax[1].plot(range(1, len(acc) + 1), np.array(last) - np.array(first), **s_)
    ax[0].set_title("average accuracy over all tasks seen so far", fontsize=10)
    ax[0].set_xlabel("evaluation step"); ax[0].set_ylabel("mean joint accuracy", fontsize=9)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25); ax[0].set_ylim(-.05, 1.05)
    ax[1].set_title("per-task change since it was learned\n(0 = no forgetting)", fontsize=10)
    ax[1].set_xlabel("task index"); ax[1].set_ylabel("Δ accuracy", fontsize=9)
    ax[1].axhline(0, color="k", lw=.8); ax[1].grid(alpha=.25); ax[1].legend(fontsize=8)
    names = [r_["arm"] for r_ in rows]
    xs = np.arange(len(names))
    ax[2].bar(xs - .2, [r_["ACC"] for r_ in rows], .38, label="ACC (higher better)",
              color="#1f77b4")
    ax[2].bar(xs + .2, [-r_["BWT"] for r_ in rows], .38, label="forgetting (−BWT, lower better)",
              color="#d62728")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(names, fontsize=8, rotation=15)
    ax[2].set_title("final scores", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=.25)
    fig.suptitle("N5 — Split-ColorMNIST class-incremental on REAL held-out images", fontsize=11)
    fig.tight_layout()
    return save(fig, "n5_classinc.png")


def fig_forgetting_matrix():
    """The acc[i][t] matrix: the standard continual-learning heatmap, ours vs the control."""
    r = load("n5_classinc")
    if not r:
        return None
    show = [x for x in r["rows"] if x["arm"] in ("ours", "finetune")]
    if not show:
        return None
    fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 4.2))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, show):
        acc = row["acc_matrix"]
        T = max(len(a) for a in acc)
        M = np.full((len(acc), T), np.nan)
        for i, a in enumerate(acc):
            M[i, T - len(a):] = a          # task i is first evaluated when it is learned
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"{row['arm']}   ACC {row['ACC']:.3f}, BWT {row['BWT']:+.3f}", fontsize=10)
        ax.set_xlabel("evaluation step"); ax.set_ylabel("task")
        plt.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("accuracy of every task after every step — a row that fades to the right "
                 "is a task being forgotten", fontsize=10)
    fig.tight_layout()
    return save(fig, "n5_forgetting_matrix.png")


def fig_batch():
    """N4: what batching costs and what it buys."""
    r = load("n4_batch")
    if not r:
        return None
    rows = r["rows"]
    arms = sorted({x["arm"] for x in rows})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    w = .38
    xs = np.arange(len(arms))
    for k, mode in enumerate(("sequential", "batch")):
        sel = {x["arm"]: x for x in rows if x["mode"] == mode}
        off = (k - .5) * w
        ax[0].bar(xs + off, [sel[a]["wall_seconds"] if a in sel else 0 for a in arms], w,
                  label=mode)
        ax[1].bar(xs + off, [sel[a]["ACC"] if a in sel else 0 for a in arms], w, label=mode)
        ax[2].bar(xs + off, [-sel[a]["BWT"] if a in sel else 0 for a in arms], w, label=mode)
    for a, t, yl in zip(ax, ("wall clock", "final accuracy (ACC)", "forgetting (−BWT)"),
                        ("seconds", "joint accuracy", "lower is better")):
        a.set_xticks(xs); a.set_xticklabels(arms, fontsize=9)
        a.set_title(t, fontsize=10); a.set_ylabel(yl, fontsize=9)
        a.legend(fontsize=8); a.grid(alpha=.25, axis="y")
    fig.suptitle("N4 — batch addition vs one concept at a time (8 concepts, batches of 4)",
                 fontsize=11)
    fig.tight_layout()
    return save(fig, "n4_batch.png")


def fig_mode_hierarchy():
    """E1: the claim that the tree's own nodes ARE the modes of the noisy marginals."""
    r = load("e1_manifold")
    if not r or "mode_hierarchy" not in r:
        return None
    h = r["mode_hierarchy"]
    u = [x["u"] for x in h]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(u, [x["matched_depth"] for x in h], marker="o", color="#1f77b4")
    ax[0].set_title("depth of the matched tree node", fontsize=10); ax[0].set_ylabel("depth")
    ax[1].semilogy(u, [max(x["match_rmse"], 1e-6) for x in h], marker="o", color="#1f77b4")
    ax[1].set_title("distance from the mode to that node", fontsize=10); ax[1].set_ylabel("RMSE")
    ax[2].plot(u, [x["matched_leaf_frac"] for x in h], marker="o", color="#1f77b4")
    ax[2].set_title("fraction matching a LEAF", fontsize=10); ax[2].set_ylabel("fraction")
    ax[2].set_ylim(-.05, 1.05)
    for a in ax:
        a.set_xlabel("noise level u"); a.grid(alpha=.25)
    fig.suptitle("E1 — modes of p_t coarsen from leaves to internal nodes as noise grows",
                 fontsize=11)
    fig.tight_layout()
    return save(fig, "e1_mode_hierarchy_curve.png")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return p


# ── case studies ─────────────────────────────────────────────────────────────
def case_studies(args):
    """Query -> composed prototype -> samples, for several held-out compositions, annotated with
    the accuracy and diversity measured on exactly those samples."""
    import torch
    from common import CACHE, ood_queries
    from lifelong_ood.datasets import FG_NAMES, BG_NAMES, load_colormnist, rgb_img
    from lifelong_ood.evals import composition_accuracy, fit_probes, sample_geometry
    from lifelong_ood.neural import NeuralLifelongLearner, load_field, to_data, to_model

    from lifelong_ood.evals import precision_recall
    from lifelong_ood.neural import learn_embedding, sample_embedding
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)

    xqs = [to_model(data["X"][q["query"]]).ravel() for q in qs]
    teachers = L.compose_batch(xqs, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120)
    if args.mode == "teacher":
        pools = L.sample_teacher_batch(teachers, n=args.n_show, seed=args.seed)
    else:
        # the shipped path: a concept learned in the model's conditioning space, then sampled from
        # the model's own conditional.  Each panel below carries the coverage measured on exactly
        # the samples shown, so the figure certifies itself rather than relying on its caption.
        pools = []
        for xq in xqs:
            c_new, _, _ = learn_embedding(field, np.atleast_2d(xq), n_steps=args.inv_steps,
                                          lr=0.05, seed=args.seed)
            pools.append(sample_embedding(field, c_new, n=args.n_show, steps=args.steps,
                                          seed=args.seed))

    ncol = args.n_show + 2
    fig, axes = plt.subplots(len(qs), ncol, figsize=(1.05 * ncol, 1.25 * len(qs)))
    axes = np.atleast_2d(axes)
    for r, (q, t, P) in enumerate(zip(qs, teachers, pools)):
        xq01 = data["X"][q["query"]]
        S = to_data(P).astype(np.float32)
        ref = data["X"][q["rest"][:100]]
        acc = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
        geo = sample_geometry(S, xq01, ref)
        ratio = geo["diversity"] / max(geo["ref_spread"], 1e-9)
        axes[r, 0].imshow(rgb_img(xq01)); axes[r, 0].set_ylabel(
            f"{q['d']} {FG_NAMES[q['f']][:4]}/{BG_NAMES[q['b']][:5]}", fontsize=6, rotation=0,
            ha="right", va="center")
        axes[r, 1].imshow(rgb_img(np.clip((t["mu"] + 1) / 2, 0, 1)))
        for c in range(args.n_show):
            axes[r, 2 + c].imshow(rgb_img(S[c]))
        pr = precision_recall(S, ref)
        axes[r, ncol - 1].set_title(
            f"acc {acc['joint']:.2f} | recall {pr['recall']:.2f} | div/ref {ratio:.2f}",
            fontsize=5.5, loc="right")
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, lab in enumerate(["query", "prototype μ_T"] + [f"sample {i+1}" for i in range(args.n_show)]):
        axes[0, c].set_title(lab, fontsize=7)
    fig.suptitle(f"Case studies ({args.mode}) — one OOD query, the composed prototype, and samples\n"
                 "each row annotated with the accuracy, RECALL and diversity measured on exactly "
                 "those samples (recall/div-ratio target 1.0)", fontsize=8.5)
    fig.tight_layout()
    p = save(fig, f"case_studies_{args.mode}.png")

    def content_diversity(X):
        """Pairwise RMSE after a 2x downsample -- a crude low-pass.

        Raw pixel diversity cannot tell "a different instance of the class" from "the same image
        plus salt-and-pepper", and the case-study panels look far more uniform than their diversity
        numbers suggest.  Averaging 2x2 blocks destroys per-pixel noise while leaving stroke shape
        and colour intact, so the ratio of smoothed to raw diversity says how much of the measured
        variety is CONTENT rather than speckle.
        """
        A = np.asarray(X, np.float64).reshape(len(X), 32, 32, 3)
        A = A.reshape(len(X), 16, 2, 16, 2, 3).mean((2, 4)).reshape(len(X), -1)
        d = np.sqrt(((A[:, None] - A[None]) ** 2).mean(2))
        return float(d[np.triu_indices(len(A), 1)].mean())

    stats = []
    for q, t, P in zip(qs, teachers, pools):
        S = to_data(P).astype(np.float32)
        ref = data["X"][q["rest"][:100]]
        geo = sample_geometry(S, data["X"][q["query"]], ref)
        acc = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
        cs, cr = content_diversity(S), content_diversity(ref)
        stats.append(dict(slot=q["slot"], **precision_recall(S, data["X"][q["rest"][:100]]),
                          joint=acc["joint"], diversity=geo["diversity"],
                          ref_spread=geo["ref_spread"], novelty=geo["novelty"],
                          div_ratio=geo["diversity"] / max(geo["ref_spread"], 1e-9),
                          content_div=cs, content_ref=cr,
                          content_ratio=cs / max(cr, 1e-9),
                          div_over_nov=geo["diversity"] / max(geo["novelty"], 1e-9)))
    with open(os.path.join(FIG, f"case_studies_{args.mode}.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print("  case-study stats:")
    print(f"    {'slot':>5s}{'acc':>6s}{'div':>7s}{'ref':>7s}{'div/ref':>9s}"
          f"{'nov':>7s}{'div/nov':>9s}{'CONTENT div':>13s}{'ref':>7s}{'ratio':>7s}")
    for s in stats:
        print(f"    {s['slot']:5d}{s['joint']:6.2f}{s['diversity']:7.3f}{s['ref_spread']:7.3f}"
              f"{s['div_ratio']:9.2f}{s['novelty']:7.3f}{s['div_over_nov']:9.2f}"
              f"{s['content_div']:13.3f}{s['content_ref']:7.3f}{s['content_ratio']:7.2f}")
    m = lambda k: float(np.mean([x[k] for x in stats]))                 # noqa: E731
    print(f"    MEAN  acc {m('joint'):.2f} | raw div/ref {m('div_ratio'):.2f} | "
          f"CONTENT div/ref {m('content_ratio'):.2f} | div/nov {m('div_over_nov'):.2f}")
    print("    div/nov near 1.0 (not sqrt2) means samples share a common offset from the query,")
    print("    i.e. they are all the same rendering of it rather than independent draws.")
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves-only", action="store_true")
    p.add_argument("--ckpt", default=os.path.join(RES, "cache", "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=6)
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["inversion", "teacher"], default="inversion")
    p.add_argument("--inv-steps", type=int, default=400)
    args = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    print("curves:")
    for fn in (fig_lifelong_curves, fig_classinc, fig_forgetting_matrix, fig_batch,
               fig_mode_hierarchy):
        try:
            fn()
        except Exception as e:                                          # noqa: BLE001
            print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")
    if not args.curves_only:
        print("case studies:")
        case_studies(args)


if __name__ == "__main__":
    main()
