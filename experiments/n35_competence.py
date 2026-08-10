"""N35 -- how much backbone does the method need, and WHAT about the backbone is it needing?

Every result in this repository sits on one backbone trained to one budget, so "composition is the
substrate for lifelong OOD learning" has been asserted at a single point on the axis that matters
most to it.  If composition is really the mechanism, the method's quality should be a function of how
COMPOSITIONAL the backbone is -- and that is a different quantity from how GOOD it is.  This sweeps
the axis and tries to separate the two.

**The ladder.**  `n1_train.py --snapshot-epochs` writes the EMA at a geometric ladder of epochs from
ONE run, so every rung is a prefix of the same trajectory.  That matters: RESULTS 5p measured up to
0.07 F1 between two runs of the SAME command, so training one backbone per rung would put
between-run variance on exactly the axis under study.  Here the rungs differ only in training length.

**Two families of measurement at each rung.**

*Backbone properties* (the independent variables, none of which involve our method):

  `seen_joint`   conditional accuracy on slots it was TRAINED on -- raw competence, the ceiling
  `seen_f1`      coverage on trained slots, so "competence" is not just accuracy at recall 0
  `uncond_tv`    sample roughness vs real, the artifact axis (§19)
  `basis_r2`     ideal-words R^2: does the embedding space FACTORIZE additively?
  `memorized`    fraction of samples that are near-copies of training images (§overfit check)

*Downstream* (every arm, on the SAME held-out queries, at every rung):

  `ours`, `poe-guided` (the companion method in its own Eq. 14 configuration), `inversion`,
  `nearest-class` as a floor.  Plus `retention_exact` for ours.

**Conjectures this is designed to falsify, stated before the run:**

  C1  There is a COMPETENCE THRESHOLD.  Below it, mode ascent has no modes to find and every
      composition method is at chance; above it they separate.  So the curve is a step or an S,
      not a line.
  C2  Our method needs MORE backbone than the companion method does, because we distil a pool from
      the model's own conditional -- a bad conditional gives a bad pool and we consolidate that --
      whereas guiding toward an analytic Gaussian leans on the backbone less.  If true, our curve
      starts lower and crosses theirs.
  C3  PRESERVATION IS INVARIANT.  `retention_exact` = 1.000 at every rung including the worst,
      because it is structural and has nothing to do with what the backbone knows.  A single
      non-1.000 here would falsify the central claim of the repository.
  C4  R^2 IS THE MEDIATOR, not raw quality.  Our composition score should track `basis_r2` more
      tightly than it tracks `seen_joint`.  This is the thesis in correlational form: what the
      method needs is not a good model, it is a COMPOSITIONALLY STRUCTURED one.
  C5  There may be a non-monotone tail: long training can sharpen per-class embeddings into
      memorized, less additive codes, so `basis_r2` (and with it our score) could PEAK and decline
      while `seen_joint` keeps rising.  If seen, C4 and C5 together are the interesting result.

Reported per rung so the conjectures are checkable rather than narrated, plus each arm normalized by
that rung's own `seen_joint` ceiling -- which is what separates "the backbone is bad" from "the
method needs more backbone than the backbone being bad explains".

    python experiments/n35_competence.py --ckpts results/cache/ddpm_colormnist_ep{2,5,...}.pt
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries                # noqa: E402

from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                memorization, precision_recall, save_json,
                                speckle, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.adapters import fingerprint                          # noqa: E402
from lifelong_ood.neural.idealwords import factorize, slot_factors            # noqa: E402
from lifelong_ood.neural.inversion import learn_embedding, sample_embedding   # noqa: E402

PAPER_POE = dict(schedule="variance", var_scale=1.0, w0=1.2, wmax=2.0, wmin=0.0)


def epochs_of(path):
    m = re.search(r"_ep(\d+)\.pt$", path)
    if m:
        return int(m.group(1))
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("epochs_done", -1))
    except Exception:
        return -1


def backbone_props(field, C, slots, probes, data, args, seed):
    """Everything measurable about the BACKBONE, with no reference to our method."""
    rng = np.random.default_rng(seed)
    seen_slots = np.unique(data["slot"][data["seen"]])
    picks = rng.choice(seen_slots, min(args.n_seen_slots, len(seen_slots)), replace=False)
    remap = {int(s): i for i, s in enumerate(slots)}
    J, F1, S_all = [], [], []
    for s in picks:
        i = remap.get(int(s))
        if i is None:
            continue
        S = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=seed,
                                 branch=int(i), clip=(-1, 1))).astype(np.float32)
        idx = np.where(data["slot"] == s)[0]
        R = data["X"][idx[:100]].astype(np.float32)
        d, f, b = (int(data[k][idx[0]]) for k in ("digit", "fg", "bg"))
        J.append(composition_accuracy(probes, S, d, f, b)["joint"])
        F1.append(precision_recall(S, R)["f1"])
        S_all.append(S)
    S_all = np.concatenate(S_all) if S_all else np.zeros((0, field.dim), np.float32)
    Rall = data["X"][data["seen"]][:400].astype(np.float32)
    sp = speckle(S_all, Rall) if len(S_all) else {}
    fac = factorize(C, slot_factors(slots))          # [N, 3] label rows
    return dict(seen_joint=float(np.mean(J)) if J else float("nan"),
                seen_f1=float(np.mean(F1)) if F1 else float("nan"),
                uncond_tv=float(sp.get("tv_ratio", float("nan"))),
                basis_r2=float(fac["r2"]),
                memorized=float(memorization(S_all, Rall)["frac_memorized"]) if len(S_all) else
                float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=None,
                   help="checkpoints forming the ladder; default globs "
                        "results/cache/ddpm_colormnist_ep*.pt plus the final one")
    p.add_argument("--arms", nargs="+",
                   default=["ours", "poe-guided", "inversion", "nearest-class"])
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-seen-slots", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.n_per_slot, args.n_seen_slots = 1, 8, 20, 2
        args.epochs, args.steps, args.inv_steps = 4, 20, 60
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck_paths = args.ckpts or (sorted(glob.glob(os.path.join(CACHE, "ddpm_colormnist_ep*.pt")),
                                     key=epochs_of) + [os.path.join(CACHE, "ddpm_colormnist.pt")])
    ck_paths = [c for c in ck_paths if os.path.exists(c)]
    if not ck_paths:
        print("[n35] no ladder found -- train with "
              "`n1_train.py --epochs 200 --snapshot-epochs 2 5 10 20 40 80 130`")
        return 1

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    print(f"[n35] ladder of {len(ck_paths)} rungs x {len(qs)} held-out compositions x "
          f"{len(args.arms)} arms\n", flush=True)

    rows, props = [], []
    for ck in ck_paths:
        ep = epochs_of(ck)
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        slots = np.array(sorted(meta["remap"]))
        C = meta["ema"]["cemb.weight"].numpy()[[meta["remap"][int(x)] for x in slots]]
        field0 = load_field(ck, device=dev)
        bp = backbone_props(field0, C, slots, probes, data, args, args.seed)
        bp.update(epochs=ep, ckpt=os.path.basename(ck))
        props.append(bp)
        print(f"[n35] ep {ep:>4}: seen_joint {bp['seen_joint']:.3f}  seen_f1 {bp['seen_f1']:.3f}  "
              f"basis_r2 {bp['basis_r2']:.3f}  tv {bp['uncond_tv']:.3f}  "
              f"memorized {bp['memorized']:.3f}", flush=True)
        gains = backbone_gains(field0, C, slots, probes, data, seed=args.seed, steps=args.steps,
                               n=8 if args.quick else 16)
        del field0
        torch.cuda.empty_cache()

        for q in qs:
            R = data["X"][q["rest"][:100]].astype(np.float32)
            xq = to_model(data["X"][q["query"]]).ravel()
            for arm in args.arms:
              # 8 rungs x 4 arms is a long sweep, and an arm that fails on ONE rung should cost that
              # cell rather than every cell computed so far.  Recorded as nan and reported, never
              # silently skipped.
              try:
                  field = load_field(ck, device=dev)
                  L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
                  L.attach_factorization(C, slot_factors(slots), gains=gains)
                  exact = float("nan")
                  if arm == "ours":
                      # a second concept first, so retention has something to preserve at every rung
                      other = to_model(data["X"][qs[(qs.index(q) + 1) % len(qs)]["query"]]).ravel()
                      L.learn_concept_factored(other, "prev", n_pool=32, epochs=args.epochs,
                                               inv_steps=args.inv_steps, seed=args.seed,
                                               guidance=args.guidance)
                      fp = fingerprint(field, "prev", seed=args.seed)
                      L.learn_concept_factored(xq, "cur", n_pool=48, epochs=args.epochs,
                                               inv_steps=args.inv_steps, seed=args.seed,
                                               guidance=args.guidance)
                      exact = float(np.max(np.abs(fingerprint(field, "prev", seed=args.seed)
                                                  - fp)) == 0.0)
                      S = to_data(L.sample_concept("cur", n=args.n_samples, steps=args.steps,
                                                   seed=args.seed)).astype(np.float32)
                  elif arm == "poe-guided":
                      t = L.compose(xq, u_grid=(0.1, 0.2, 0.3, 0.4),
                                    n_starts=8 if args.quick else 16,
                                    iters=30 if args.quick else 120)
                      S = to_data(L.sample_teacher(t, n=args.n_samples, seed=args.seed,
                                                   **PAPER_POE)).astype(np.float32)
                  elif arm == "inversion":
                      emb, _, _ = learn_embedding(field, xq[None], n_steps=args.inv_steps,
                                                  seed=args.seed)
                      S = to_data(sample_embedding(field, emb, n=args.n_samples, steps=args.steps,
                                                   seed=args.seed)).astype(np.float32)
                  else:
                      j = int(np.argmin(np.linalg.norm(C - C.mean(0), axis=1)))
                      S = to_data(field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                               branch=j, clip=(-1, 1))).astype(np.float32)
                  a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
                  pr = precision_recall(S, R)
                  rows.append(dict(arm=arm, epochs=ep, slot=int(q["slot"]), joint=a["joint"],
                                   digit=a["digit"], fg=a["fg"], bg=a["bg"], recall=pr["recall"],
                                   f1=pr["f1"], vendi=vendi_score(S), retention_exact=exact))
                  del field, L
                  torch.cuda.empty_cache()
              except Exception as exc:
                nan = float("nan")
                rows.append(dict(arm=arm, epochs=ep, slot=int(q["slot"]), joint=nan, digit=nan,
                                 fg=nan, bg=nan, recall=nan, f1=nan, vendi=nan,
                                 retention_exact=nan, error=f"{type(exc).__name__}: {exc}"))
                print(f"[n35] ep {ep} slot {q['slot']} {arm}: FAILED -- "
                      f"{type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
            print(f"[n35] ep {ep:>4} slot {q['slot']}: " + "  ".join(
                f"{r['arm']} {r['f1']:.2f}" if r["f1"] == r["f1"] else f"{r['arm']} --"
                for r in rows[-len(args.arms):]), flush=True)

    # ── summarize, and check each conjecture against the numbers ─────────────
    eps = [p_["epochs"] for p_ in props]
    summary = {}
    for arm in args.arms:
        for e in eps:
            sel = [r for r in rows if r["arm"] == arm and r["epochs"] == e]
            if sel:
                summary[f"{arm}@{e}"] = {k: float(np.nanmean([r[k] for r in sel]))
                                         for k in ("joint", "recall", "f1", "vendi",
                                                   "retention_exact")}
    save_json(dict(rows=rows, backbone=props, summary=summary, config=vars(args)),
              os.path.join(RESULTS, f"n35_competence{args.tag}.json"), seed=args.seed)

    print(f"\n[n35] the competence ladder ({len(eps)} rungs)")
    print(f"    {'epochs':>7s}{'seen_joint':>11s}{'basis_r2':>10s}" +
          "".join(f"{a[:11]:>12s}" for a in args.arms))
    for e, p_ in zip(eps, props):
        print(f"    {e:7d}{p_['seen_joint']:11.3f}{p_['basis_r2']:10.3f}" +
              "".join(f"{summary.get(f'{a}@{e}', {}).get('f1', float('nan')):12.3f}"
                      for a in args.arms))

    def col(arm, k="f1"):
        return np.array([summary.get(f"{arm}@{e}", {}).get(k, np.nan) for e in eps], float)

    print("\n[n35] conjectures")
    ours, poe = col("ours"), col("poe-guided")
    r2 = np.array([p_["basis_r2"] for p_ in props], float)
    sj = np.array([p_["seen_joint"] for p_ in props], float)
    ret = col("ours", "retention_exact")

    # C3 first: it is the falsifier for the whole repository
    bad = [(e, v) for e, v in zip(eps, ret) if v == v and v != 1.0]
    print(f"  C3 preservation invariant: retention_exact = "
          f"{'1.000 at EVERY rung' if not bad else 'VIOLATED at ' + str(bad)}")
    # C1 threshold: is the jump between consecutive rungs much larger than the rest?
    d = np.diff(ours[~np.isnan(ours)])
    if len(d) > 1:
        print(f"  C1 threshold: largest single-rung gain {np.nanmax(d):+.3f} vs median "
              f"{np.nanmedian(d):+.3f} -- {'step-like' if np.nanmax(d) > 3 * abs(np.nanmedian(d)) else 'gradual'}")
    # C2 crossing
    both = ~np.isnan(ours) & ~np.isnan(poe)
    if both.sum() > 1:
        lead = np.sign(ours[both] - poe[both])
        print(f"  C2 ours-needs-more: ours leads at {int((lead > 0).sum())}/{int(both.sum())} rungs"
              f"{'; CROSSES' if len(set(lead.tolist())) > 1 else '; no crossing'}")

    def corr(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")

    c_r2, c_sj = corr(ours, r2), corr(ours, sj)
    print(f"  C4 mediator: corr(ours F1, basis_r2) = {c_r2:+.3f}   "
          f"corr(ours F1, seen_joint) = {c_sj:+.3f}   -> "
          f"{'R^2 is the better predictor' if c_r2 > c_sj else 'raw competence predicts as well or better'}")
    if len(r2) > 2:
        pk = int(np.nanargmax(r2))
        print(f"  C5 non-monotone tail: basis_r2 peaks at epoch {eps[pk]} "
              f"({r2[pk]:.3f}); final {r2[-1]:.3f} -> "
              f"{'PEAKS AND DECLINES' if pk < len(r2) - 1 else 'still rising at the top rung'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
