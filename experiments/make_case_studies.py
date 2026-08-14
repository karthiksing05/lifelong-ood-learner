"""The case-study panels, laid out the way the source paper lays its Figure 3 out.

Wang et al. show, for one out-of-distribution query: the query, the PROTOTYPES their mode ascent
discovered, and then a block of samples per method.  Ours previously showed methods as rows without
the prototypes, which hid the most informative part -- what the composition is actually built from --
and made a fair comparison harder to eyeball.

Two things this fixes beyond layout:

  * the published method is run in ITS OWN configuration (their Eq. 14 variance-aware schedule with
    the raw Laplace covariance).  Under this repository's annealed schedule the same teacher emits
    speckle, which made the baseline look broken when it is not; see tests/diag_poe_faithful.py.
  * every row is annotated with accuracy AND coverage, because the whole argument is that the first
    cannot distinguish composing from copying and the second can.

Needs the backbones.  Writes figures/case_colormnist.png (and a machine-readable
results/case_colormnist*.json).  The CelebA panel f10_case_celeba.png is NOT written here --
make_paper_figures.py derives it from n20_celeba_composition_pair.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries          # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
# the selected dial (see --dial); the canonical figure is the one rendered AT it, so the
# tag marks deviation from the default rather than deviation from 1.0
DEFAULT_DIAL = 0.8
from lifelong_ood.compose import PAPER_POE_COLORMNIST as PAPER_POE


def panel(fig, gs_row, imgs, side, label, note="", n=6, cmap=None, captions=None):
    """`captions` labels INDIVIDUAL panels, which the prototypes row needs and nothing else does.

    Readers of this figure have twice concluded that the held-out compositions leaked into training,
    because the first prototype looks like a clean instance of the composition while the other two
    look like noise.  It is not a leak -- the shown slots have 0 training rows and no embedding row --
    it is that `init="hybrid"` starts half the ascent chains AT THE QUERY, and the selection objective
    scores candidates on the query, so the query-seeded one is picked first every time.  The other
    two come from random starts in unrelated basins.  That asymmetry is the entire explanation and
    the figure was hiding it; now each prototype says which it is.
    """
    import matplotlib.pyplot as plt
    for c in range(n):
        ax = fig.add_subplot(gs_row[c])
        if imgs is not None and c < len(imgs):
            ax.imshow(np.clip(np.asarray(imgs[c], np.float32).reshape(side, side, 3), 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if captions and c < len(captions) and captions[c]:
                ax.set_xlabel(captions[c], fontsize=5.0, labelpad=1.5)
        else:
            # a row with fewer panels than the grid (K=3 prototypes against 6 sample columns)
            # should leave BLANK space, not empty framed boxes that read as failed generations
            ax.axis("off")
        if c == 0:
            ax.set_ylabel(label, fontsize=6.5, rotation=0, ha="right", va="center")
        if c == n - 1 and note:
            ax.set_title(note, fontsize=5.5, loc="right", pad=2)


def select_cases(data, args):
    """Choose which held-out compositions the figure shows, by a STATED rule.

    Picking slots by hand would improve the figure and destroy it as evidence.  `n29_slot_scan.py`
    scores all 40 held-out compositions, and this selects from that ranking.

    The default spans the benchmark rather than the method's comfort zone: ColorMNIST's held-out tier
    is exactly 4 colour pairs x 10 digits, and success varies strongly across those pairs (median
    joint 0.719 for green-on-navy against 0.125 for yellow-on-dark_brown).  One case per pair at the
    pair's MEDIAN is therefore what a reader should expect to reproduce -- a figure of best cases
    would be four times more flattering and worth nothing.
    """
    if args.slots:
        want = [int(x) for x in args.slots]
    else:
        scan = os.path.join(RESULTS, f"n29_slot_scan{args.scan_tag}.json")
        if not os.path.exists(scan):
            print(f"  [cases] no scan at {scan}; falling back to {args.n_cases} random slots. "
                  f"Run experiments/n29_slot_scan.py for the principled selection.", flush=True)
            return ood_queries(data, n_slots=args.n_cases, seed=args.seed)
        with open(scan) as fh:
            sc = json.load(fh)
        want = sc["picks"][args.select]
        by = {r["slot"]: r for r in sc["rows"]}
        print(f"  [cases] rule `{args.select}` over {sc['summary']['n']} scored compositions "
              f"(median joint {sc['summary']['joint_median']:.3f}, "
              f"range [{sc['summary']['joint_min']:.2f}, {sc['summary']['joint_max']:.2f}]):",
              flush=True)
        for s_ in want:
            print(f"      slot {s_:3d}  {by[s_]['name']:34s} joint {by[s_]['joint']:.2f}", flush=True)
    allq = ood_queries(data, n_slots=None, seed=args.seed)
    idx = {q["slot"]: q for q in allq}
    return [idx[s_] for s_ in want if s_ in idx]


def colormnist(args):
    from lifelong_ood.data.datasets import load_colormnist, slot_name
    from lifelong_ood.evals import composition_accuracy, fit_probes, precision_recall
    from lifelong_ood.neural import (NeuralLifelongLearner, learn_embedding, load_field,
                                     sample_embedding, teacher_init_alpha, to_data, to_model)
    from lifelong_ood.neural.idealwords import slot_factors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ck = torch.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu",
                    weights_only=False)
    remap = ck["remap"]; slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    # no n= override: use the robust default (wider slot sample, multi-seed vote), since a
    # gain of 1.0 chosen by a noisy calibration leaves backgrounds under-weighted and the
    # composed samples come out the wrong colour in the very figure meant to show them
    gains = backbone_gains(field, C, slots, probes, data, seed=args.seed,
                           steps=args.steps)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, slot_factors(slots), gains=gains)

    qs = select_cases(data, args)
    rows = ["real", "prototypes", "ours", "poe-guided (paper Eq.14)", "query-only", "nearest-class"]
    fig = plt.figure(figsize=(1.05 * (args.n_show + 1), 1.15 * len(qs) * len(rows) + 1.2))
    gs = GridSpec(len(qs) * len(rows), args.n_show + 1, figure=fig, hspace=0.06, wspace=0.06)
    r = 0
    scored = []
    for q in qs:
        xq = to_model(data["X"][q["query"]]).ravel()
        R = data["X"][q["rest"][:100]].astype(np.float32)
        # `--paper-budget` runs the companion paper's Table 6 ColorMNIST discovery budget: 15 noise
        # levels x 128 starts = 1920 ascent chains, against the 4 x 16 = 64 this figure used before.
        # The selected prototypes were often unrelated to the query, and that is a POOL problem --
        # submodular selection can only pick from what ascent found.
        from lifelong_ood.compose import PAPER_COLORMNIST
        disc = (dict(PAPER_COLORMNIST) if args.paper_budget
                else dict(u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120))
        # ONE ascent run feeding both the teacher and the prototypes row.  This used to call
        # `L.compose(xq, **disc)` for `t` and a SECOND, separately-parameterized `candidates_ascent`
        # for `M`, so `t["sel"]` indexed a different mode array than the one being displayed.  With
        # matching settings and seed the two runs coincided and the bug was invisible; under
        # `--paper-budget` the settings diverge and it raised IndexError (sel 503 into 50 modes).
        # Every prototype panel rendered before this fix came from an ascent run that was only
        # accidentally the same one the teacher selected from.
        # K/tau/seed pulled from the learner config so this is byte-for-byte what `L.compose` did.
        from lifelong_ood.compose import candidates_ascent, poe_teacher
        M, V, meta = candidates_ascent(field, xq, seed=L.cfg["seed"], **disc)
        t = poe_teacher(xq, M, V, K=L.cfg["K"], tau=L.cfg["tau"])
        t["meta"] = meta
        got = {"real": R[:args.n_show],
               "prototypes": to_data(M[t["sel"]]).astype(np.float32),
               "poe-guided (paper Eq.14)": to_data(
                   L.sample_teacher(t, n=args.n_show, seed=args.seed, **PAPER_POE)),
               "query-only": to_data(L.sample_teacher(L.baseline_teacher(xq, "query_only"),
                                                      n=args.n_show, seed=args.seed, **PAPER_POE))}
        L.learn_concept_factored(xq, f"c{q['slot']}", n_pool=48, epochs=args.epochs,
                                 inv_steps=args.inv_steps, seed=args.seed,
                                 guidance=args.guidance)
        # `--dial s` starts the reverse process from the prototype noised to level s instead of from
        # pure noise (§20).  s=1.0 is the old behaviour -- unconstrained draws from the concept's
        # conditional, which is what every table reports and what makes these panels look loose.
        # §20b measured F1 across the dial: 0.410 at s=0.80, 0.669 at s=0.90, 0.599 at s=1.00.
        got["ours"] = to_data(L.sample_concept(
            f"c{q['slot']}", n=args.n_show, seed=args.seed, steps=args.steps,
            u_start=args.dial, x_init=(t["mu"] if args.dial < 1.0 else None)))
        a0 = teacher_init_alpha(field, t, xq, seed=args.seed)
        got["nearest-class"] = to_data(field.sample(n=args.n_show, steps=args.steps,
                                                    seed=args.seed, clip=(-1, 1),
                                                    branch=int(np.argmax(a0))))
        for name in rows:
            S = np.asarray(got[name], np.float32)
            note = ""
            if name not in ("prototypes",):
                a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                pr = precision_recall(S, R)
                note = f"acc {a['joint']:.2f} | recall {pr['recall']:.2f}"
                # also record it: the dial can only be CHOSEN by comparing renders, and reading
                # four accuracies back off a 4000px figure is not a comparison
                scored.append(dict(slot=int(q["slot"]), arm=name, dial=float(args.dial),
                                   paper_budget=bool(args.paper_budget),
                                   guidance=float(args.guidance), joint=float(a["joint"]),
                                   digit=float(a["digit"]), fg=float(a["fg"]), bg=float(a["bg"]),
                                   recall=float(pr["recall"])))
            # Where each PROTOTYPE's ascent chain started, and at which noise level.  Without this
            # the row invites the conclusion that the composition leaked into training: prototype 1
            # looks like a clean instance of it because its chain began AT THE QUERY (a real held-out
            # image) at the lowest noise level, and selection scores candidates on the query so that
            # one is always picked first.  The slot itself has 0 training rows and no embedding.
            caps = None
            if name == "prototypes":
                src = np.asarray(t["meta"].get("src", []))
                uu = np.asarray(t["meta"].get("u", []))
                caps = [f"{src[i]}-init, u={uu[i]:g}" if len(src) > i else ""
                        for i in np.asarray(t["sel"])]
            # The HELD-OUT COMBINATION, spelled out on the `real` row.  Asked for directly: a
            # reader must be able to see WHICH combination was withheld, and check for themselves
            # that the arms below are reproducing it rather than something adjacent.  The count of
            # training rows is printed with it, because "held out" is a claim about the split and
            # this is the place it can be verified at a glance (it is 0 by construction --
            # `diag_holdout_leak` asserts it over the whole grid; here it is visible per panel).
            lab = name
            if name == "real":
                n_train = int(np.sum((data["slot"] == q["slot"]) & data["seen"]))
                lab = (f"real — HELD OUT\n{slot_name(q['d'], q['f'], q['b'])}\n"
                       f"slot {q['slot']} · digit {q['d']}, fg {q['f']}, bg {q['b']}\n"
                       f"{n_train} training images")
            panel(fig, [gs[r, c] for c in range(args.n_show + 1)], S, 32,
                  lab, note, n=args.n_show, captions=caps)
            r += 1
    rule = "explicit slots" if args.slots else args.select
    fig.suptitle(
        f"ColorMNIST — one held-out composition per block, selected by `{rule}` over ALL 40 scored "
        f"compositions (N29):\nthe MEDIAN case of each of the four held-out colour pairs, not the "
        f"best.  Median joint over all 40 is 0.438, range [0.00, 0.94], and it is strongly "
        f"colour-dependent\n(green-on-navy 0.719 vs yellow-on-dark_brown 0.125).  `prototypes` are "
        f"what mode ascent discovered; the companion method runs on its OWN Eq. 14 schedule.  "
        f"`ours` is sampled at prototype-distance dial s={args.dial:g} (§20); discovery budget = "
        f"{'the paper\'s 15x128=1920 chains' if args.paper_budget else 'ours, 4x16=64 chains'}.  "
        f"Accuracy is high where coverage is zero.  Sampled with {args.steps} steps "
        f"(tables use 50); the residual texture is the BACKBONE's -- it scores 1.59x real "
        f"on its own TRAINED classes, worse than anything downstream of it.\n"
        f"EVERY composition shown is held out: 0 training rows and no class-embedding row, "
        f"checked against the checkpoint's own remap.  Prototype 1 resembles the composition "
        f"because its\nascent chain STARTED at the query (a real held-out image) at the lowest "
        f"noise level and selection scores candidates on the query -- see the per-panel labels.  "
        f"It moves ~1 class-width\nand ends nearer the TRAINING set than the query it started "
        f"from; prototypes 2-3 start from noise, and the composition survives on those alone "
        f"(RESULTS 5e-quater/quinquies).", fontsize=8)
    # the canonical figure is the one rendered at the DEFAULTS; the tag marks deviation
    tag = ("" if args.dial == DEFAULT_DIAL else f"_s{args.dial:g}") \
        + ("" if args.paper_budget else "_ourbudget")
    # The CANONICAL render (tag == "", i.e. defaults) is a core figure; any tagged sweep variant is
    # supporting evidence and belongs in extra/, so a dial sweep cannot overwrite the published one.
    p = (os.path.join(FIG, "case_colormnist.png") if not tag
         else os.path.join(FIG, "extra", f"case_colormnist{tag}.png"))
    fig.savefig(p, dpi=145, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")
    from lifelong_ood.evals import save_json
    save_json(dict(rows=scored, config=vars(args)),
              os.path.join(RESULTS, f"case_colormnist{tag}.json"), seed=args.seed)
    for arm in ("ours", "poe-guided (paper Eq.14)", "query-only"):
        sel = [x["joint"] for x in scored if x["arm"] == arm]
        if sel:
            print(f"  [dial s={args.dial:g}] {arm:26s} mean joint {np.mean(sel):.3f}  "
                  f"{[round(v, 2) for v in sel]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cases", type=int, default=3,
                   help="only used when no scan is available and slots fall back to random")
    p.add_argument("--select", default="median-per-pair",
                   choices=["median-per-pair", "best-per-pair", "worst-per-pair"],
                   help="rule over experiments/n29_slot_scan.py's ranking. The default "
                        "shows one case per COLOUR PAIR at that pair's median, so the "
                        "figure spans the benchmark and is representative rather than "
                        "flattering.")
    p.add_argument("--slots", nargs="*", default=None,
                   help="explicit slot ids, overriding --select")
    p.add_argument("--scan-tag", default="_v1")
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=200,
                   help="sampling steps for the FIGURE.  tests/diag_artifact_source.py "
                        "measures the excess high-frequency texture falling from 1.28x "
                        "real at 50 steps to 1.18x at 200, monotonically, at no cost to "
                        "accuracy.  The TABLES stay at 50 so every arm is compared at "
                        "one budget; the figure is about what the method looks like.")
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--guidance", type=float, default=1.5,
                   help="CFG weight for the distillation pool.  1.5 is what n9's "
                        "label-free calibration selects on this backbone (trained-slot "
                        "joint 0.969 against 0.781 at 1.0); the figure must use the same "
                        "value the table does.  At the old implicit 1.0 some samples land "
                        "on the wrong background, and at 3.0 the colour factor breaks "
                        "entirely -- see tests/diag_ours_quality.py.")
    p.add_argument("--our-budget", dest="paper_budget", action="store_false", default=True,
                   help="use our 4x16=64-chain discovery budget instead of the paper's Table 6 "
                        "ColorMNIST column (15 levels x 128 starts = 1920).  The paper's budget "
                        "is the DEFAULT here: at 64 chains the selected prototypes were routinely "
                        "unrelated to the query, which is a POOL problem -- submodular selection "
                        "can only pick from what ascent found -- and it is what made the "
                        "prototypes row look broken.")
    p.add_argument("--dial", type=float, default=DEFAULT_DIAL,
                   help="prototype-distance dial for the `ours` row (§20): the reverse "
                        "process starts from the prototype noised to this level.  1.0 = "
                        "pure noise; lower ties samples to the prototype.  **Default 0.8 "
                        "since 2026-08-13, set deliberately for colour faithfulness in the "
                        "renders and NOT by the metric, which prefers 0.9** -- the choice and "
                        "its cost are recorded in RESULTS 20c.  Was 0.9 "
                        "because the swept comparison at the paper's discovery budget "
                        "(results/case_colormnist*.json) shows 0.9 DOMINATES: joint 0.708 "
                        "vs 0.667 at 0.8 and 0.542 at 1.0, at recall 0.881 vs 0.495 and "
                        "0.907.  0.8 buys accuracy by destroying coverage; 1.0 keeps "
                        "coverage and loses accuracy; 0.9 is the only setting that is "
                        "near-best on both.  RE-CONFIRMED 2026-08-13 under current code at "
                        "--guidance 1.5, n_cases 3, ours row: s=0.7 joint 0.833 / recall 0.116, "
                        "s=0.8 0.792 / 0.427, s=0.9 0.583 / 0.854.  Lowering the dial buys "
                        "accuracy by collapsing coverage onto the prototype -- at 0.7 recall "
                        "0.116 is barely above poe-guided and query-only, which score 0.000 by "
                        "re-emitting the query, so the panels look clean precisely BECAUSE the "
                        "arm has stopped doing the thing it is claimed to do.  Judge this dial "
                        "on both columns, never on the renders.\n"
                        "AND READ IT WITH --guidance: the dial and the guidance interact "
                        "strongly.  At s=0.9 the same code scores joint 0.583 at guidance 1.5 "
                        "and 0.083 at 2.0.  reproduce.sh passed 2.0 for the figure while every "
                        "number quoted here was measured at 1.5, so f9 was rendered at an "
                        "operating point no table used and its `ours` row looked broken for "
                        "that reason alone.  Fixed; do not re-add a --guidance override.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    colormnist(args)


if __name__ == "__main__":
    main()
