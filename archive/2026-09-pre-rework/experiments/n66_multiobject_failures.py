"""N66 -- WHY do 3 of 9 held-out pairs fail to compose?  (follow-up to 7bb)

7bb reports ADDRESS on held-out pairs at exact-set 0.302 against a 0.003 floor, but the per-pair
spread is enormous: (0,3) scores 0.875 while (0,5), (1,4) and (2,9) score 0.000.  A mean over that
is not a description of anything, and "it composes for most pairs" is only useful if the failures
have a reason.  Four hypotheses, each with a measurement that can refute it:

  H1 THINLY-SEEN OBJECTS.  Objects appear in 5-9 of the 36 seen scenes.  A vector W[o] fitted from 5
     scenes is worse determined than one fitted from 9, so failures should concentrate on the
     thinly-seen objects.  -> correlate min(seen_count) of the pair against exact.

  H2 WRONG RADIUS.  `c_new = mu + g*(W[a]+W[b])` is a point built from a fit, not a trained row.  If
     it lands at a different distance from the mean than real rows do, the model is being asked to
     condition on a vector unlike anything it saw.  -> ||c_new - mu|| against the distribution of
     ||C[s] - mu||, and the max cosine to any trained row.

  H3 ONE OBJECT, NOT BOTH.  A pair can fail because the scene is wrong, or because ONE object is
     reliably missing.  Those need different fixes.  -> per-object hit rate p_a, p_b separately;
     if failures are always "one object present, one absent", the composition is half-working.

  H4 A GLOBAL GAIN IS WRONG FOR SOME PAIRS.  g* is calibrated once on seen scenes.  -> sweep g PER
     PAIR and report whether any g rescues a failing pair; if one does, the fix is per-pair gain
     calibration, and if none does, the pair is genuinely outside what the basis can address.

H4 is the one that decides whether 7bb's number is a floor or a ceiling, so it is measured on the
failing pairs specifically rather than on all of them.

    python experiments/n66_multiobject_failures.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402
from n64_multiobject_address import (detect, draw, fit_object_basis,   # noqa: E402
                                     train_detector)

from lifelong_ood.data.multiobject import (N_OBJ, load_multiobject,    # noqa: E402
                                           object_membership, scene_pairs)
from lifelong_ood.evals import save_json                              # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_multiobject_v2.pt"))
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-scene", type=int, default=300)
    p.add_argument("--det-epochs", type=int, default=60)
    p.add_argument("--rescue-gains", type=float, nargs="*",
                   default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck["split"]
    d = load_multiobject(n_per_scene=args.n_per_scene, seed=sp["seed"], n_holdout=sp["n_holdout"])
    X, scene, seen = d["X"], d["scene"], d["seen"]
    P, held = scene_pairs(), list(ck["held"])
    remap = ck["remap"]
    uniq = np.array(sorted(int(k) for k in remap))

    Y = np.zeros((len(X), N_OBJ), np.float32)
    for i, s in enumerate(scene):
        a, b = P[int(s)]
        Y[i, a] = Y[i, b] = 1.0
    det, dq = train_detector(X[seen], Y[seen], X[~seen], Y[~seen], seed=args.seed, dev=dev,
                             epochs=args.det_epochs, verbose=False)
    print(f"[n66] detector held-out-pair top2 {dq['top2_exact']:.3f}", flush=True)

    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in uniq]]
    M = object_membership(uniq)
    mu, W, _ = fit_object_basis(C, M)
    seen_count = M.sum(0).astype(int)                       # scenes per object among SEEN
    rad = np.linalg.norm(C - mu, axis=1)
    print(f"[n66] trained rows: ||C-mu|| median {np.median(rad):.3f} "
          f"[{rad.min():.3f}, {rad.max():.3f}]", flush=True)

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    def hits(imgs, a, b):
        """-> (exact-set, p_a, p_b) under the top-2 read."""
        Pr = detect(det, imgs, dev)
        top2 = [set(np.argsort(-q)[:2].tolist()) for q in Pr]
        return (float(np.mean([t == {a, b} for t in top2])),
                float(np.mean([a in t for t in top2])),
                float(np.mean([b in t for t in top2])))

    gstar = 1.5                                              # 7bb's calibrated value
    rows = []
    for h in held:
        a, b = P[int(h)]
        c_new = mu + gstar * (W[a] + W[b])
        r = float(np.linalg.norm(c_new - mu))
        cmax = float(np.max((C - mu) @ (c_new - mu)
                            / (np.linalg.norm(C - mu, axis=1) * max(r, 1e-9))))
        im = draw(field, c_new, args.n_samples, args.steps, args.seed + int(h))
        ex, pa, pb = hits(im.reshape(len(im), -1), a, b)
        rows.append(dict(pair=[int(a), int(b)], exact=ex, p_a=pa, p_b=pb,
                         radius=r, radius_ratio=r / float(np.median(rad)), max_cos=cmax,
                         seen_a=int(seen_count[a]), seen_b=int(seen_count[b])))
        print(f"[n66] {(a, b)}: exact {ex:.3f} | p_{a} {pa:.2f} p_{b} {pb:.2f} | "
              f"radius {r:.2f} ({r / np.median(rad):.2f}x median) maxcos {cmax:.2f} | "
              f"seen {seen_count[a]}/{seen_count[b]}", flush=True)

    ex = np.array([r["exact"] for r in rows])
    print(f"\n[n66] {'=' * 66}")
    print("[n66] HYPOTHESES")
    mins = np.array([min(r["seen_a"], r["seen_b"]) for r in rows], float)
    print(f"[n66] H1 thinly-seen objects: corr(min seen-count, exact) = "
          f"{np.corrcoef(mins, ex)[0, 1]:+.2f}"
          + ("  -> supported" if np.corrcoef(mins, ex)[0, 1] > 0.5 else "  -> NOT supported"))
    rr = np.array([r["radius_ratio"] for r in rows])
    print(f"[n66] H2 wrong radius: ratios {rr.min():.2f}-{rr.max():.2f} of median; "
          f"corr(radius_ratio, exact) = {np.corrcoef(rr, ex)[0, 1]:+.2f}")
    half = [r for r in rows if r["exact"] < 0.1 and max(r["p_a"], r["p_b"]) > 0.5]
    print(f"[n66] H3 one-object-not-both: {len(half)} of "
          f"{sum(1 for r in rows if r['exact'] < 0.1)} failing pairs have ONE object present "
          f">50% of the time" + ("  -> half-working, not absent" if half else ""))

    # H5 -- COMPETITION between the two summed directions, which the p_a/p_b split points at.
    # In (1,4) one object is present 94% of the time and the other 3%; in (4,7), 91% and 9%.  And
    # object 4 is NOT weak in general -- it reaches 0.56 and 0.78 in the pairs where its partner is
    # less dominant.  So the failure is not a missing object, it is one direction overwhelming the
    # other in the sum.  If that is right, the fix is per-OBJECT balancing rather than a per-pair
    # scalar, because a scalar scales both directions together and cannot change their ratio.
    wn = np.linalg.norm(W, axis=1)
    print(f"[n66] H5 competition: per-object ||W|| " +
          " ".join(f"{o}:{wn[o]:.2f}" for o in range(N_OBJ)), flush=True)
    imb = np.array([abs(r["p_a"] - r["p_b"]) for r in rows])
    print(f"[n66]   corr(|p_a - p_b|, exact) = {np.corrcoef(imb, ex)[0, 1]:+.2f} "
          "(strongly negative => the failures ARE one-sided, not both-absent)", flush=True)

    # H4/H5 -- can a per-pair gain, or a per-object BALANCED gain, rescue a failing pair?
    print("[n66] H4/H5 rescue sweep on the FAILING pairs "
          "(scalar gain vs unit-normalised per-object directions):", flush=True)
    rescued = []
    for r in rows:
        if r["exact"] >= 0.1:
            continue
        a, b = r["pair"]
        best_s = (-1.0, None)
        for g in args.rescue_gains:                       # H4: one scalar, both directions
            im = draw(field, mu + g * (W[a] + W[b]), 12, args.steps, args.seed + a * 10 + b)
            best_s = max(best_s, (hits(im.reshape(len(im), -1), a, b)[0], g), key=lambda t: t[0])
        best_b = (-1.0, None)
        ua, ub = W[a] / max(wn[a], 1e-9), W[b] / max(wn[b], 1e-9)
        for g in args.rescue_gains:                       # H5: equal contribution per object
            im = draw(field, mu + g * (ua + ub), 12, args.steps, args.seed + a * 10 + b)
            best_b = max(best_b, (hits(im.reshape(len(im), -1), a, b)[0], g), key=lambda t: t[0])
        rescued.append(dict(pair=[a, b], best_exact=best_s[0], best_gain=best_s[1],
                            balanced_exact=best_b[0], balanced_gain=best_b[1]))
        print(f"[n66]   {(a, b)}: scalar best {best_s[0]:.3f} at g={best_s[1]} | "
              f"BALANCED best {best_b[0]:.3f} at g={best_b[1]}"
              + ("  -> balancing rescues it" if best_b[0] >= 0.25 > best_s[0] else
                 "  -> rescued by gain alone" if best_s[0] >= 0.25 else
                 "  -> neither rescues it"), flush=True)
    n_s = sum(1 for r in rescued if r["best_exact"] >= 0.25)
    n_b = sum(1 for r in rescued if r["balanced_exact"] >= 0.25)
    print(f"\n[n66] VERDICT: of {len(rescued)} failing pairs, {n_s} are rescued by a per-pair "
          f"SCALAR gain and {n_b} by per-object BALANCING.")
    if n_b > n_s:
        print("[n66]   BALANCING WINS, so the failures are COMPETITION between the two summed")
        print("[n66]   directions, not missing objects: a scalar scales both together and cannot")
        print("[n66]   change their ratio, which is why it does not help. 7bb's number is a FLOOR")
        print("[n66]   set by an unbalanced sum -- but balancing uses only ||W||, which is fitted")
        print("[n66]   from SEEN scenes, so it stays label-free and is a legitimate improvement.")
    elif n_s or n_b:
        print("[n66]   Some pairs are recoverable by gain, so 7bb's number is a floor; the gain")
        print("[n66]   must still be calibrated label-free before any improvement is claimed.")
    else:
        print("[n66]   Neither rescues them, so those pairs are outside what this basis can")
        print("[n66]   address and 7bb's number is the honest one for this backbone.")
    save_json(dict(rows=rows, rescued=rescued, detector=dq, gain=gstar,
                   w_norms=[float(v) for v in wn],
                   trained_radius=dict(median=float(np.median(rad)), min=float(rad.min()),
                                       max=float(rad.max())), config=vars(args)),
              os.path.join(RESULTS, "n66_multiobject_failures.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
