"""N73 -- the ColorMNIST test, on a CelebA backbone with a DENSITY holdout.

Every previous CelebA result used a backbone fine-tuned from `google/ddpm-celebahq-256`, so a
"held-out" combination was held out from the CONDITIONING while its images were already in the
pretrained density.  That is why mode ascent handed the query back at every noise level there, and
why `nearest-class` scores 0.390 on the (also pretrained) Shapes3D backbone against 0.000 on
ColorMNIST.  `ddpm_celeba_grid.pt` is trained FROM SCRATCH on the nine seen cells only: the three
held-out combinations have zero training images, so they are absent from the model.

The grid is hair(3) x Male(2) x Smiling(2) = 12 cells, its size fixed by measurement rather than
preference -- only four attributes are readable at 64px and two of them are mutually exclusive hair
colours (see `load_celeba_grid`).  Held out: Blond-Male-notSmiling, Brown-Male-Smiling,
other-Female-Smiling.

Arms are the ones that make the claim falsifiable, matching N72:

  address        compose(fac, TRUE factors) and sample the model's conditional
  nearest-class  the closest SEEN class embedding -- if this matches ADDRESS the model already had it
  unconditional  the floor

and two COPYING tests, which is what makes the ADDRESS number evidence of a donation rather than of
retrieval: `memorized` (MOSAIC's d_NN1/d_NN2 against the TRAINING set) and `d_query` (distance to a
real held-out image, read against the real-vs-real spread of that cell).  Neither can see the
other's failure -- the NN test is structurally blind to a sample copied from the query, because the
query was never trained on.

Read every number against the PROBE CEILING, not against 1.0: the factor probes score around
hair 0.57 / Male 0.85 / Smiling 0.86 on REAL images, so a perfect generator caps near 0.41 joint
against a chance of 1/n_seen.  The exact ceiling is refit per seed and moves a long way -- 0.414 to
0.570 across three -- so it is printed by every run rather than quoted from one.  Reporting raw
joint without it would understate a working model and overstate a broken one.

The grid and the holdout are DERIVED from the checkpoint, so this runs unchanged on any
`celeba_grid` backbone.

    python experiments/n73_celeba_grid_compose.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.celeba import load_celeba_grid                  # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,    # noqa: E402
                                memorization, precision_recall, save_json, vendi_score)
from lifelong_ood.instrument import check_instrument                  # noqa: E402
from lifelong_ood.neural import (NeuralScoreField, UNet,               # noqa: E402
                                 sample_embedding, to_data)
from lifelong_ood.neural.idealwords import compose, cv_r2, factorize   # noqa: E402

ARMS = ("address", "nearest-class", "unconditional")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid.pt"))
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-gate", type=int, default=200,
                   help="real images used for the per-cell READABILITY gate. Separate from "
                        "--n-samples because the gate decides whether a cell is scored at all and "
                        "24 images give it a standard error near 0.06.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="",
                   help="suffix for the result filename. REQUIRED when pointing --ckpt at a "
                        "different backbone: without it every backbone writes the same "
                        "`n73_celeba_grid_compose.json` and the last run silently replaces the others, which pools "
                        "as though it were one experiment.")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})

    def grid(n_holdout):
        return load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                                n_per_cell=int(sp.get("n_per_slot", 3000)),
                                n_holdout=int(n_holdout), max_rows=sp.get("max_rows"))

    # DERIVE the holdout from the checkpoint; never assume it.  This used to be a hardcoded 3, which
    # is right for the first backbone and silently wrong for any other -- every arm still runs, so
    # the failure is a plausible number scored against the wrong cells.  Checkpoints written after
    # 2026-08-19 record `n_holdout`; older ones are recovered from the seen-class count, and either
    # way the seen set is ASSERTED against `remap` before anything is measured.
    d = grid(sp.get("n_holdout") or 3)
    want = sorted(int(k) for k in ck["remap"])
    if sorted(set(np.asarray(d["slot"])[np.asarray(d["seen"])].tolist())) != want:
        d = grid(len(d["live"]) - len(want))
    seen_cells = sorted(set(np.asarray(d["slot"])[np.asarray(d["seen"])].tolist()))
    if seen_cells != want:
        raise SystemExit(f"[n73] split mismatch: checkpoint trained on cells {want}, this loader "
                         f"gives {seen_cells}. The backbone and the evaluation disagree about "
                         f"which combinations were held out -- refusing to report a number.")
    print(f"[n73] {len(want)} seen cells, {len(d['held'])} held out: {d['held']}", flush=True)
    X, F, slot = (np.asarray(d["X"], np.float32), np.asarray(d["factors"]),
                  np.asarray(d["slot"]))
    names, sizes = d["names"], d["sizes"]
    probes = fit_factor_probes(d, seed=args.seed)
    ceil = float(np.prod([probes[f"{n}_val"] for n in names if f"{n}_val" in probes]))
    print(f"[n73] probe ceiling on REAL images ~ {ceil:.3f}; chance = "
          f"{1.0 / len(ck['remap']):.3f}", flush=True)

    remap = ck["remap"]
    seen = np.array(sorted(int(k) for k in remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen]]
    lab = np.stack([F[slot == int(s)][0] for s in seen])
    fac = factorize(C, lab)
    # BOTH R^2s.  In-sample is a real statement about a 120-row ColorMNIST table and close to
    # arithmetic on a 7-row one -- which is what a bigger holdout leaves -- so the leave-one-cell-out
    # number is the one that says whether the table is compositional rather than merely fitted.
    r2cv, n_cv = cv_r2(C, lab)
    print(f"[n73] ideal-words basis R^2 {fac['r2']:.3f} in-sample over {len(seen)} seen cells | "
          f"{r2cv:.3f} leave-one-cell-out over {n_cv} of them", flush=True)

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    # ── THE INSTRUMENT GATE, before anything is sampled ──────────────────────
    # This experiment is where the gate's design came from, so it is also the run that validates
    # it: at the legacy bar it must reproduce the readable set n75 already pooled (16 of 36 across
    # three backbones x three seeds), and at the strict +0.10 bar it should report a SMALLER one --
    # which is the honest number and has never been quoted.
    gate = check_instrument(
        tag="n73", real=d, probes=probes, field=field,
        targets={int(h): tuple(int(v) for v in F[slot == int(h)][0]) for h in d["held"]},
        n_arm=args.n_samples, n_gate=args.n_gate, table=C, labels=lab, seed=args.seed)
    print(gate.summary("n73"), flush=True)
    gate.require("n73")
    # THE EXPERIMENT CHECKS ITS OWN VERDICT BAR against what its sample count can resolve. n73
    # declares margins at +0.10 and `min_detectable` at n_samples=24 is ~0.11-0.13, so the bar sits
    # BELOW the noise floor: the large ADDRESS-over-floor effect (+0.180) is resolvable, the small
    # ADDRESS-over-retrieval one (+0.018) is not, and 7ee's conclusion rests on the first.
    _md = float(np.mean([g.min_detectable for g in gate.targets.values()])) if gate.targets else 0.0
    if _md > 0.10:
        print(f"[n73] NOTE: at n={args.n_samples} the smallest resolvable margin is {_md:.3f}, above "
              f"this experiment's +0.100 verdict bar. Margins below {_md:.3f} are NOT supported by "
              f"this sample count -- raise --n-samples to about {int(4 * 0.21 / 0.01):d} to claim "
              f"one.", flush=True)

    rows = []
    for ci, h in enumerate(d["held"]):
        idx = np.where(slot == int(h))[0]
        if not len(idx):
            continue
        tgt = tuple(int(v) for v in F[idx[0]])
        R = X[idx[:60]]
        c_new = compose(fac, tgt)
        j_near = int(np.argmin(((C - c_new[None]) ** 2).sum(1)))
        # RANGE AND CLIP, both of which this block had wrong and both of which are silent.
        #
        # The field works in MODEL space [-1,1]; the probes, `X`, and every `real` arm live in DATA
        # space [0,1]. `sample_embedding` returns model space (its `clip` defaults to (-1,1)), so it
        # needs `to_data` -- without it the probes saw a distribution shifted by -1 and scaled by 2,
        # and 49.3% of sample pixels were negative.
        #
        # `field.sample` is worse, because its `clip` defaults to (-0.2, 1.2) -- a DATA-range clamp,
        # applied to a MODEL-space trajectory at every reverse step. That is `VLA_PATH` lesson
        # 2-bis, which already cost a 109-minute retrain on the multi-object backbone: it destroys
        # the negative half of the range as the trajectory runs, and the result still looks like
        # plausible images. `clip=(-1,1)` then `to_data`, exactly as n72 does.
        got = {
            "address": to_data(sample_embedding(field, c_new, n=args.n_samples, steps=args.steps,
                                                seed=args.seed + ci)),
            "nearest-class": to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                  seed=args.seed + ci, clip=(-1, 1),
                                                  branch=remap[int(seen[j_near])])),
            "unconditional": to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                  seed=args.seed + 37 * (ci + 1), clip=(-1, 1),
                                                  branch=None)),
        }
        # THE INSTRUMENT'S OWN READING, on a DISJOINT set of real images of this same cell.  Every
        # column here needs a ceiling, not just `joint`: the first run scored precision 0.000 for
        # `address` against 0.083 for the other two arms, which reads as "our samples are
        # off-manifold" -- and is unreadable until real images of the cell are scored the same way
        # and turn out to be near the same floor.  Four verdicts in this repository have already
        # fired on a metric that could not discriminate; this arm is what makes that impossible
        # here.  It is NOT a competitor: it is real data, and it is excluded from the arm table.
        # TWO copying tests, because neither can see the other's failure -- the pair N72 settled on.
        # MOSAIC's d_NN1/d_NN2 compares against the TRAINING set, so it can catch a sample copied
        # from a seen cell; it is structurally BLIND to a sample copied from the query, because the
        # query is held out and was never trained on. `d_query` covers that, read against the
        # real-vs-real spread of the held-out cell rather than against zero.
        train_ref = X[np.asarray(d["seen"])][:600]
        xq = X[idx[0]]
        got["real"] = X[idx[60:60 + args.n_samples]] if len(idx) >= 60 + args.n_samples \
            else X[idx[-args.n_samples:]]

        # THE READABILITY GATE NEEDS MORE SAMPLES THAN THE ARM COMPARISON DOES.  The `real` arm is
        # held to `n_samples` so its vendi and precision are comparable to the generated arms, but
        # 24 images give a joint accuracy with a standard error near 0.06 -- and this number decides
        # whether a cell is scored at all.  So the gate reads a larger disjoint set.  It is worth
        # separating: on the 9-seen backbone the readable cell is the BROWN one while blond and
        # `other` read blind, and on the 7-seen backbone it was the reverse, so per-cell readability
        # is a property of the fitted probe and this particular cell -- not of a hair value that
        # could be excluded once and for all.
        big = X[idx[60:60 + args.n_gate]] if len(idx) >= 60 + args.n_gate else X[idx[-args.n_gate:]]
        gate_acc = factor_accuracy(probes, np.asarray(big, np.float32), tgt)
        gate_j = float(gate_acc["joint"]) if gate_acc["joint"] == gate_acc["joint"] else 0.0
        rows.append(dict(arm="real-gate", cell=int(h), target=list(tgt), joint=gate_j,
                         n=int(len(big)), recall=float("nan"), precision=float("nan"),
                         vendi=float("nan"), memorized=float("nan"), d_query=float("nan")))
        for arm in ARMS + ("real",):
            S = np.asarray(got[arm], np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, R)
            mem = memorization(S, train_ref)
            rows.append(dict(arm=arm, cell=int(h), target=list(tgt),
                             joint=float(a["joint"]) if a["joint"] == a["joint"] else 0.0,
                             recall=pr["recall"], precision=pr["precision"],
                             vendi=float(vendi_score(S)),
                             memorized=float(mem["frac_memorized"]),
                             d_query=float(np.mean([np.linalg.norm(np.asarray(s, np.float32)
                                                                   - xq) for s in S]))))
        r = {x["arm"]: x for x in rows if x["cell"] == int(h)}
        print(f"[n73] cell {h} {tgt}: address {r['address']['joint']:.3f} | "
              f"nearest-class {r['nearest-class']['joint']:.3f} | "
              f"uncond {r['unconditional']['joint']:.3f} | "
              f"REAL {r['real-gate']['joint']:.3f} (n={r['real-gate']['n']})"
              f"{'  <- INSTRUMENT BLIND ON THIS CELL' if r['real-gate']['joint'] <= 1.0 / len(seen) + 0.02 else ''}",
              flush=True)

    def m(arm, k):
        v = [x[k] for x in rows if x["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    # THE SCALE BAR for d_query: how far two REAL images of a held-out cell sit from each other.
    # Without it a distance is uninterpretable -- "the sample is 34 from the query" says nothing
    # until you know real images of the same class are 34 apart too.
    rr = []
    for h in d["held"]:
        ix = np.where(slot == int(h))[0][:12]
        rr += [float(np.linalg.norm(X[a_] - X[b_])) for i_, a_ in enumerate(ix) for b_ in ix[i_ + 1:]]
    real_spread = float(np.mean(rr)) if rr else float("nan")

    print(f"\n[n73] {'=' * 78}")
    print(f"{'arm':16s}{'joint':>9s}{'/ceiling':>10s}{'recall':>9s}{'vendi':>8s}"
          f"{'memorized':>11s}{'d(query)':>10s}")
    for arm in ARMS:
        print(f"{arm:16s}{m(arm, 'joint'):>9.3f}{m(arm, 'joint') / max(ceil, 1e-9):>10.2f}"
              f"{m(arm, 'recall'):>9.3f}{m(arm, 'vendi'):>8.2f}"
              f"{m(arm, 'memorized'):>11.2f}{m(arm, 'd_query'):>10.2f}")
    print(f"{'--- REAL images':16s}{m('real', 'joint'):>9.3f}{m('real', 'joint') / max(ceil, 1e-9):>10.2f}"
          f"{m('real', 'recall'):>9.3f}{m('real', 'vendi'):>8.2f}"
          f"{m('real', 'memorized'):>11.2f}{m('real', 'd_query'):>10.2f}")
    print(f"{'  precision':16s}" + "".join(f"{m(a, 'precision'):>9.3f}"
                                          for a in ARMS + ("real",))
          + "   <- arms in order, then REAL")
    a_, n_, u_ = m("address", "joint"), m("nearest-class", "joint"), m("unconditional", "joint")
    # the GATE arm, not the display arm -- the verdict must use the larger sample
    r_ = m("real-gate", "joint")
    chance = 1.0 / len(seen)
    print(f"[n73] readability gate: REAL joint {r_:.3f} on {args.n_gate} images per cell, "
          f"chance {chance:.3f}")
    print("[n73] VERDICT")

    # GATE ON THE REAL ARM, NOT ON THE GLOBAL CEILING.  `ceil` is the product of per-factor probe
    # accuracies over ALL cells, and held-out cells are the RAREST combinations in CelebA -- the
    # ones the probes are worst at.  On the 7-seen backbone the global ceiling reads 0.524 while
    # REAL images of the held-out cells score 0.142 against a chance of 0.143, i.e. the instrument
    # cannot read those cells at all and no arm's number there means anything.  The right
    # denominator for judging composition on a cell is what real images of THAT cell score.
    if r_ <= chance + 0.02:
        print(f"[n73]   THE INSTRUMENT IS BLIND ON THESE CELLS: REAL images of the held-out cells")
        print(f"[n73]   score joint {r_:.3f} against a chance of {chance:.3f} (the global probe")
        print(f"[n73]   ceiling of {ceil:.3f} is measured over ALL cells and does not apply here --")
        print(f"[n73]   held-out cells are the rarest combinations, so they are the hardest to")
        print(f"[n73]   read). NO arm can be judged from this run, ADDRESS's {a_:.3f} included.")
    elif a_ > max(n_, u_) + 0.10:
        print("[n73]   COMPOSES A COMBINATION THE MODEL NEVER SAW. The held-out cells have zero")
        print("[n73]   training images, so this is a DENSITY holdout -- the first CelebA result")
        print("[n73]   here that is not merely a missing embedding row.")
        print(f"[n73]   ADDRESS {a_:.3f} is {a_ / max(r_, 1e-9):.0%} of what REAL images of these")
        print(f"[n73]   cells score ({r_:.3f}), against retrieval's {n_:.3f}.")
    elif a_ > u_ + 0.10:
        print("[n73]   Beats the floor but not nearest-class: the composition is close enough to a")
        print("[n73]   seen cell that retrieval explains it. Report as such.")
    else:
        print(f"[n73]   Does not clear the floor. With {len(seen)} seen cells, real images of the")
        print(f"[n73]   held-out cells at {r_:.3f} and chance at {chance:.3f}, this backbone may")
        print("[n73]   simply be too weak -- judge against the real-image arm, not against 1.0.")
    save_json(dict(rows=rows, instrument=gate.as_dict(), ceiling=ceil, real_joint=r_, chance=chance,
                   basis_r2=float(fac["r2"]),
                   basis_r2_cv=float(r2cv), n_cv=int(n_cv), n_seen=len(seen), held=d["held"],
                   real_spread=real_spread, config=vars(args)),
              os.path.join(RESULTS, f"n73_celeba_grid_compose{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
