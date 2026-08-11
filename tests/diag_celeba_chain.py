"""Where does CelebA actually break?  Walk the chain and find the first stage that is already bad.

CelebA results here look far worse than the companion paper's, including for the paper's OWN method,
and that has been noted repeatedly without being localized.  "poe-guided looks bad" is not a
diagnosis: poe-guided is the last stage of a four-stage chain, and every stage before it can be the
cause.  This measures each stage separately so the first broken one is visible.

    0. MEASUREMENT   do the attribute probes work at all?  If a probe cannot read its own attribute
                     off REAL images, every downstream number is capped and nothing generative is
                     being measured.  §diag_celeba_attrs found 5 of 14 unreadable at 64px.
    1. BACKBONE      condition on a combination the backbone was TRAINED on and score it.  If that is
                     bad, nothing downstream can be good and the fault is not compositional.
    2. DISCOVERY     are the modes mode ascent returns actual faces, and do they carry the attributes
                     they are supposed to?  Reported against the backbone's own trained-class samples,
                     so a bad score here means ASCENT is failing, not the model.
    3. TEACHER       is mu_T -- the per-dimension PoE mean the whole method guides toward -- a face?
                     A blurry average of misaligned modes cannot be guided toward successfully.
    4. GUIDANCE      poe-guided itself.

Also prints the backbone's conditioning MARGIN over unconditional, which is the number §5o used to
condemn the first CelebA backbone, and the ascent settings actually in use -- n20 runs weaker
discovery than the ColorMNIST default (4 noise levels / 12 starts / 80 iters against 6 / 16 / 150)
on a dataset with 4x the dimensionality.

    python tests/diag_celeba_chain.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.celeba import ATTRS, as_float, load_celeba_comp              # noqa: E402
from lifelong_ood.compose import candidates_ascent, poe_teacher                # noqa: E402
from lifelong_ood.evals import attr_accuracy, fit_attr_probes, probe_usable    # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402

CK = os.path.join(CACHE, "ddpm_celeba_factored.pt")
PAPER_POE = dict(schedule="variance", var_scale=1.0, w0=1.2, wmax=2.0, wmin=0.0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    img = int(ck.get("img", 64))
    print(f"[chain] backbone {os.path.basename(CK)}  img {img}  "
          f"cond {ck['arch'].get('cond_mode')}  attrs {len(ck.get('attr_names', ATTRS))}\n")
    data = load_celeba_comp(img=img, max_rows=25000, min_count=ck.get("min_count", 60),
                            n_holdout=ck.get("n_holdout", 24))
    field = load_field(CK, device=dev)
    rng = np.random.default_rng(0)

    # ── 0. MEASUREMENT ───────────────────────────────────────────────────────
    idx = rng.permutation(len(data["X8"]))[:6000]
    d = dict(X=as_float(data["X8"][idx]), attrs=data["attrs"][idx], img=img,
             seen=np.ones(len(idx), bool))
    probes = fit_attr_probes(d, seed=0, epochs=30)
    names = probes["_attrs"]
    usable = [a for a in names if probe_usable(probes, a)]
    print(f"[chain] 0. MEASUREMENT: {len(usable)}/{len(names)} attribute probes can read their own "
          f"attribute off REAL images")
    print(f"          unusable: {[a for a in names if a not in usable]}")

    combos = data["all_combos"]
    seen_ids = np.setdiff1d(np.arange(len(combos)), np.asarray(data["hold_ids"]))
    pick = seen_ids[rng.integers(len(seen_ids), size=4)]
    R = as_float(data["X8"][:400])

    def score(S, combo, tag):
        a = attr_accuracy(probes, S, combo, attrs=usable)
        m = float(np.mean([v for k, v in a.items() if k in usable])) if usable else float("nan")
        print(f"          {tag:34s} mean-attr over READABLE probes {m:.3f}")
        return m

    # real images of those combos: the ceiling this backbone is being asked to reach
    print(f"\n[chain] 1. BACKBONE on combinations it was TRAINED on")
    remap = ck["remap"]
    base_scores, real_scores = [], []
    for c in pick:
        if int(c) not in remap:
            continue
        rows = np.where(data["combo_of"] == c)[0][:64]
        if len(rows) < 8:
            continue
        real_scores.append(score(as_float(data["X8"][rows]), combos[c], f"REAL images of combo {c}"))
        S = to_data(field.sample(n=16, steps=50, seed=0, branch=int(remap[int(c)]),
                                 clip=(-1, 1))).astype(np.float32)
        base_scores.append(score(S, combos[c], f"backbone conditioned on combo {c}"))
    print(f"       -> real {np.mean(real_scores):.3f}   backbone {np.mean(base_scores):.3f}   "
          f"MARGIN {np.mean(base_scores) - np.mean(real_scores):+.3f}")

    # unconditional floor: how much does conditioning move anything at all?
    U = to_data(field.sample(n=16, steps=50, seed=0, clip=(-1, 1))).astype(np.float32)
    unc = float(np.mean([attr_accuracy(probes, U, combos[c], attrs=usable).get(a, np.nan)
                         for c in pick[:1] for a in usable]))
    print(f"       -> unconditional floor {unc:.3f}: conditioning is worth "
          f"{np.mean(base_scores) - unc:+.3f} over drawing at random")

    # ── 2. DISCOVERY ─────────────────────────────────────────────────────────
    print(f"\n[chain] 2. DISCOVERY -- are the ascent modes faces, and do they carry attributes?")
    hq = int(np.asarray(data["hold_ids"])[0])
    qrows = np.where(data["combo_of"] == hq)[0]
    xq = to_model(as_float(data["X8"][qrows[:1]])).ravel()
    for tag, kw in (("n20 settings  (4 lvl / 12 starts / 80 it)",
                     dict(u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=12, iters=80)),
                    ("ColorMNIST default (6 lvl / 16 / 150)",
                     dict(u_grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), n_starts=16, iters=150))):
        M, V, meta = candidates_ascent(field, xq, seed=0, **kw)
        Md = to_data(M).astype(np.float32)
        m = score(Md, combos[hq], f"{tag}: {len(M)} modes")
        t = poe_teacher(xq, M, V, K=3)
        mu = to_data(np.asarray(t["mu"])[None]).astype(np.float32)
        # 3. TEACHER: is mu_T itself a plausible image, or a blur?
        tv = float(np.mean(np.abs(np.diff(mu.reshape(1, img, img, -1), axis=1))))
        rtv = float(np.mean(np.abs(np.diff(R[:64].reshape(-1, img, img, 3), axis=1))))
        print(f"          {'-> teacher mu_T':34s} total-variation {tv:.4f} vs real {rtv:.4f} "
              f"({tv / max(rtv, 1e-9):.2f}x -- <<1 means a BLUR, not a face)")
        # 4. GUIDANCE
        from lifelong_ood.neural import NeuralLifelongLearner
        L = NeuralLifelongLearner(field, steps=50, seed=0, device=dev)
        S = to_data(L.sample_teacher(t, n=16, seed=0, **PAPER_POE)).astype(np.float32)
        score(S, combos[hq], "-> poe-guided samples")
        del L
        torch.cuda.empty_cache()

    print("\n[chain] read the chain top-down: the FIRST stage that is already bad is the cause; "
          "everything after it is a symptom.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
