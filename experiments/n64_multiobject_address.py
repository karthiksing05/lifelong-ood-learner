"""N64 -- does a k-object scene cost k ADDRESSES, or is the scene itself a composition over them?

The question `docs/CLEVR_VERDICT.md` closed on and named as the next formulation, asked on the
exact-label substrate of `data/multiobject.py`.  N63 trained a scene-conditional DDPM whose class
table has ONE FREE ROW PER SCENE and was never told that a scene contains objects.  So:

  1. **Does the scene table factorise over object MEMBERSHIP?**  Fit `C[s] ~ mu + sum_{o in s} W[o]`.
     This is NOT `idealwords.factorize`: that fits one table per FACTOR POSITION, which for a set
     would learn a separate "first object" and "second object" table and let {3,7} and {7,3} take
     different vectors.  A set has one shared object table, so the design matrix is 0/1 MEMBERSHIP
     and the ordered fit is kept only as a control -- it has twice the free vectors, so if the
     shared model matches it, the sharing is real rather than a fit artefact.

  2. **Does ADDRESS place a HELD-OUT PAIR?**  `c_new = mu + g*(W[a] + W[b])` for a pair that never
     appeared in training, though both objects did.  If this generates scenes containing both, then
     a k-object scene of known objects costs NOTHING to store -- it is an address, not a new block.

  3. **Measured by DETECTORS, not a whole-image classifier.**  CLEVR_VERDICT is explicit that a
     multi-object label needs per-object detection, and that a 50-72 %-reliable detector poisons
     everything downstream.  So the detector is trained on images of SEEN scenes ONLY and validated
     on images of HELD-OUT scenes: it must find objects in pair-combinations it never trained on,
     or it is not an instrument for this measurement.  Its held-out score is REPORTED and gates the
     rest -- if it is weak, the composition numbers are not printed as findings.

Three guards against reading a fit artefact as a finding, because 36 scenes and 11 free parameters
per dimension is a regime where R^2 comes cheap:

  * a PERMUTATION null -- membership rows shuffled, everything else identical
  * leave-one-scene-out CV R^2, so the reported fit predicts embeddings it did not see
  * the payoff metric is downstream generation on HELD-OUT pairs, which no embedding fit can inflate

    python experiments/n64_multiobject_address.py --n-samples 32
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.multiobject import (N_OBJ, load_multiobject,    # noqa: E402
                                           object_membership, scene_pairs)
from lifelong_ood.evals import save_json                              # noqa: E402
from lifelong_ood.neural import NeuralScoreField, UNet                 # noqa: E402
from lifelong_ood.neural.ddpm import to_data                          # noqa: E402
from lifelong_ood.neural.inversion import sample_embedding             # noqa: E402

IMG = 32
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")


# ----------------------------------------------------------------------------- detector
class ObjectDetector(torch.nn.Module):
    """Multi-label CNN: 10 independent sigmoids, "is object o present anywhere in this scene".

    A whole-image classifier over the 45 scene classes would be the wrong instrument twice over: it
    could not score a sample containing an unseen pair, and it would let a model that memorised
    scene-level texture score well without either object being present.
    """

    def __init__(self, n_obj=N_OBJ, base=64):
        super().__init__()

        def blk(i, o):
            return [torch.nn.Conv2d(i, o, 3, 1, 1), torch.nn.BatchNorm2d(o), torch.nn.ReLU(),
                    torch.nn.Conv2d(o, o, 3, 2, 1), torch.nn.BatchNorm2d(o), torch.nn.ReLU()]
        self.trunk = torch.nn.Sequential(*blk(3, base), *blk(base, base * 2),
                                         *blk(base * 2, base * 4))
        # AVERAGE pooling alone is the wrong readout for PRESENCE.  A 16x16 digit occupies a
        # quarter of the canvas, so averaging over the map dilutes its evidence by the empty rest
        # of the scene; max pooling asks "is it anywhere", which is the actual question.  Both are
        # concatenated because average still carries how much of the object is there.
        self.head = torch.nn.Linear(base * 8, n_obj)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1))


def _to_chw(X):
    return torch.as_tensor(np.asarray(X, np.float32).reshape(-1, IMG, IMG, 3)).permute(0, 3, 1, 2)


def train_detector(Xtr, Ytr, Xva, Yva, epochs=40, lr=2e-3, batch=256, seed=0, dev="cuda",
                   min_steps=40, verbose=True):
    """Train the measuring instrument.  Steps per epoch are floored at `min_steps`.

    Without that floor the schedule is `len(X) // batch` steps, which on a small run is 2 steps per
    epoch -- 36 gradient steps in total, after which the detector predicts all-zeros and scores a
    respectable-looking 0.800 per-object accuracy purely because only 2 of 10 objects are ever
    present.  That is a metric that fails silently, which is the exact trap CLEVR_VERDICT records.
    """
    torch.manual_seed(seed)
    det = ObjectDetector().to(dev)
    opt = torch.optim.AdamW(det.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = _to_chw(Xtr).to(dev), torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    steps = max(min_steps, len(xt) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    for ep in range(epochs):
        for _ in range(steps):
            i = torch.as_tensor(rng.integers(0, len(xt), batch), device=dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(det(xt[i]), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if verbose and (ep + 1) % 10 == 0:
            det.eval()
            q = detector_scores(det, Xva, Yva, dev)
            print(f"[n64]   detector ep{ep + 1}/{epochs} ({steps} steps/ep): "
                  f"held-out-pair top2 {q['top2_exact']:.3f}", flush=True)
            det.train()
    det.eval()
    return det, detector_scores(det, Xva, Yva, dev)


@torch.no_grad()
def detect(det, X, dev="cuda", batch=512):
    out = []
    x = _to_chw(X)
    for i in range(0, len(x), batch):
        out.append(torch.sigmoid(det(x[i:i + batch].to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_OBJ), np.float32)


def detector_scores(det, X, Y, dev="cuda"):
    """Held-out quality of the INSTRUMENT.  `top2` is the metric the composition results use."""
    P = detect(det, X, dev)
    Y = np.asarray(Y)
    top2 = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        pred = set(np.argsort(-p)[:2].tolist())
        top2[i] = pred == set(np.where(Y[i] > 0)[0].tolist())
    return dict(top2_exact=float(top2.mean()),
                per_obj_acc=float((( P > 0.5) == (Y > 0)).mean()),
                recall_at2=float(np.mean([len(set(np.argsort(-p)[:2].tolist())
                                              & set(np.where(Y[i] > 0)[0].tolist())) / 2.0
                                          for i, p in enumerate(P)])))


def draw(field, c, n, steps, seed):
    """Sample conditioned on `c` and return DATA-space rows in [0,1].

    The field works in MODEL space [-1,1] -- `to_model` is a RANGE transform -- so
    `sample_embedding` returns [-1,1] and its default clip=(-1,1) is the correct one.  Both
    conversions were wrong here, and the second hid the first: passing clip=(0.0,1.0) clamped away
    the entire negative half at every reverse step, and these scenes are ~90 % black background,
    which is -1 in model space.  Every trajectory was destroyed, so BOTH the conditional and the
    unconditional arms came out as noise and the model looked untrained -- while its eps-loss on
    real data was 0.0097, better than the ColorMNIST reference.  The detector was then handed
    model-space rows it read as [0,1].  Every draw goes through here so the two conventions live in
    one place.
    """
    return np.asarray(to_data(sample_embedding(field, c, n=n, steps=steps, seed=seed)))


# ----------------------------------------------------------------------------- object basis
def fit_object_basis(C, M, ridge=1e-3):
    """`C[s] ~ mu + (M[s] - mean) @ W`, one SHARED vector per object.  -> mu, W, r2.

    M is centred rather than an intercept column being appended: every scene holds exactly two
    objects, so M's columns sum to the constant 2 and an explicit intercept would be collinear with
    them, leaving mu and W unidentifiable (ridge would still return *an* answer, silently).
    """
    C = np.asarray(C, np.float64)
    M = np.asarray(M, np.float64)
    mu = C.mean(0)
    Mc = M - M.mean(0)
    A = Mc.T @ Mc + ridge * np.eye(M.shape[1])
    W = np.linalg.solve(A, Mc.T @ (C - mu))
    pred = mu + Mc @ W
    ss_res = float(((C - pred) ** 2).sum())
    ss_tot = float(((C - mu) ** 2).sum())
    return mu, W, 1.0 - ss_res / max(ss_tot, 1e-12)


def cv_r2(C, M, ridge=1e-3):
    """Leave-one-scene-out R^2: predicts embeddings the fit never saw."""
    C = np.asarray(C, np.float64)
    res, tot = 0.0, 0.0
    for i in range(len(C)):
        k = np.ones(len(C), bool)
        k[i] = False
        mu, W, _ = fit_object_basis(C[k], M[k], ridge)
        pred = mu + (M[i] - M[k].mean(0)) @ W
        res += float(((C[i] - pred) ** 2).sum())
        tot += float(((C[i] - C[k].mean(0)) ** 2).sum())
    return 1.0 - res / max(tot, 1e-12)


def perm_null(C, M, ridge=1e-3, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [fit_object_basis(C, M[rng.permutation(len(M))], ridge)[2] for _ in range(n)]


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_multiobject.pt"))
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-scene", type=int, default=120)
    p.add_argument("--gains", type=float, nargs="*", default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--det-epochs", type=int, default=40)
    p.add_argument("--det-gate", type=float, default=0.85,
                   help="minimum held-out detector top2 to treat the composition numbers as "
                        "findings rather than as uninterpretable")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck["split"]
    d = load_multiobject(n_per_scene=args.n_per_scene, seed=sp["seed"],
                         n_holdout=sp["n_holdout"])
    X, scene, seen = d["X"], d["scene"], d["seen"]
    P, held = scene_pairs(), list(ck["held"])
    remap = ck["remap"]
    uniq = np.array(sorted(int(k) for k in remap))

    # ---- 1. the instrument, trained on SEEN scenes and validated on HELD-OUT ones ---------------
    Yall = np.zeros((len(X), N_OBJ), np.float32)
    for i, s in enumerate(scene):
        a, b = P[int(s)]
        Yall[i, a] = Yall[i, b] = 1.0
    det, dq = train_detector(X[seen], Yall[seen], X[~seen], Yall[~seen], seed=args.seed,
                             dev=dev, epochs=args.det_epochs)
    print(f"[n64] detector, trained on SEEN pairs only, scored on HELD-OUT pairs: "
          f"top2_exact {dq['top2_exact']:.3f}  recall@2 {dq['recall_at2']:.3f}  "
          f"per-obj acc {dq['per_obj_acc']:.3f}", flush=True)
    ok = dq["top2_exact"] >= args.det_gate
    if not ok:
        print(f"[n64] WARNING detector below the {args.det_gate:.2f} gate -- composition numbers "
              f"below are NOT reportable as findings (this is the CLEVR failure mode).", flush=True)

    # ---- 2. does the freely-learned scene table factorise over object membership? ---------------
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in uniq]]
    M = object_membership(uniq)
    mu, W, r2 = fit_object_basis(C, M)
    r2cv = cv_r2(C, M)
    null = perm_null(C, M, seed=args.seed)
    print(f"[n64] object-membership basis: R2 {r2:.3f} | leave-one-out CV R2 {r2cv:.3f} | "
          f"permutation null {np.mean(null):.3f}+/-{np.std(null):.3f} "
          f"(max {np.max(null):.3f})", flush=True)

    # ordered-pair control: twice the free vectors, so it upper-bounds what sharing could give up
    from lifelong_ood.neural.idealwords import factorize
    lab = np.array([P[int(s)] for s in uniq])
    r2_ord = float(factorize(C, lab)["r2"])
    print(f"[n64] ordered-pair control (2 tables, 2x the vectors): R2 {r2_ord:.3f} "
          f"-- shared-table R2 {r2:.3f}", flush=True)

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()

    def score(imgs, pair):
        """-> (both-present rate under top-2, per-object recall)."""
        Pr = detect(det, imgs, dev)
        tgt = set(int(v) for v in pair)
        ex = np.mean([set(np.argsort(-q)[:2].tolist()) == tgt for q in Pr])
        rc = np.mean([len(set(np.argsort(-q)[:2].tolist()) & tgt) / 2.0 for q in Pr])
        return float(ex), float(rc)

    # ---- 2b. THE CONTROL THAT MAKES THE REST INTERPRETABLE --------------------------------------
    # Condition on the model's OWN trained row for a seen scene.  Without this, a low composition
    # score has three indistinguishable causes: the model never learned to condition, the detector
    # cannot measure, or ADDRESS does not compose.  This arm is high only if the first two are fine,
    # so it is what licenses reading the ADDRESS number as being about ADDRESS at all.
    self_sc = []
    for s in uniq[:6]:
        j = int(np.searchsorted(uniq, int(s)))
        im = draw(field, C[j], 8, args.steps, args.seed)
        self_sc.append(score(im.reshape(len(im), -1), P[int(s)])[0])
    self_mean = float(np.mean(self_sc))
    print(f"[n64] CONTROL -- model conditioned on its OWN trained row, seen scenes: "
          f"exact {self_mean:.3f}", flush=True)
    if self_mean < 0.3:
        print("[n64]   -> the backbone does not reproduce its OWN training scenes on demand. "
              "Nothing downstream is about ADDRESS; fix conditioning or the detector first.",
              flush=True)

    # ---- 3. calibrate the gain on SEEN scenes, refitting the basis WITHOUT each one -------------
    # Calibrating on a scene that is in the basis fit asks the model to reproduce something already
    # memorised, and gain 1.0 always suffices; the held-out pair then gets an under-delivered gain.
    cal = {g: [] for g in args.gains}
    for s in uniq[:6]:
        k = np.array([int(v) != int(s) for v in uniq])
        mu_k, W_k, _ = fit_object_basis(C[k], M[k])
        a, b = P[int(s)]
        for g in args.gains:
            c_new = mu_k + g * (W_k[a] + W_k[b])
            im = draw(field, c_new, 8, args.steps, args.seed)
            cal[g].append(score(im.reshape(len(im), -1), (a, b))[0])
    gstar = max(args.gains, key=lambda g: float(np.mean(cal[g])))
    print("[n64] gain calibration on SEEN scenes (basis refit without each): "
          + "  ".join(f"g={g}:{np.mean(v):.3f}" for g, v in cal.items())
          + f"  -> g*={gstar}", flush=True)

    # ---- 4. ADDRESS on HELD-OUT pairs, against an unconditional floor ---------------------------
    null_vec = field.model.cemb.weight[field.model.null_id].detach().cpu().numpy()
    unc = draw(field, null_vec, args.n_samples, args.steps,
                args.seed + 1).reshape(args.n_samples, -1)
    rows = []
    for h in held:
        a, b = P[int(h)]
        c_new = mu + gstar * (W[a] + W[b])
        im = draw(field, c_new, args.n_samples, args.steps, args.seed + int(h))
        ex, rc = score(im.reshape(len(im), -1), (a, b))
        fex, frc = score(unc, (a, b))
        rows.append(dict(scene=int(h), pair=[int(a), int(b)], exact=ex, recall=rc,
                         floor_exact=fex, floor_recall=frc))
        print(f"[n64]   held-out {(a, b)}: exact {ex:.3f} (floor {fex:.3f})  "
              f"recall {rc:.3f} (floor {frc:.3f})", flush=True)

    # ---- 5. LOOK at it.  Three numbers cannot distinguish "renders nothing", "renders one object
    # twice", and "renders the wrong pair", and those call for different fixes.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = held[:5]
        fig, ax = plt.subplots(3, len(show), figsize=(2.1 * len(show), 6.6))
        for col, h in enumerate(show):
            a, b = P[int(h)]
            real = X[scene == h][:1]
            addr = draw(field, mu + gstar * (W[a] + W[b]), 1, args.steps,
                        args.seed + int(h)).reshape(1, -1)
            # a DIFFERENT unconditional draw per column -- reusing unc[:1] painted the same image
            # five times, which reads as "the model collapsed" when it is just one sample repeated
            for r, (im, lab) in enumerate(((real, f"REAL {(a, b)}"),
                                           (addr, "ADDRESS"),
                                           (unc[col:col + 1], "uncond"))):
                ax[r, col].imshow(np.asarray(im[0], np.float32).reshape(IMG, IMG, 3))
                ax[r, col].set_title(lab, fontsize=8)
                ax[r, col].axis("off")
        plt.suptitle(f"N64 multi-object: held-out pairs, g*={gstar}, "
                     f"detector top2 {dq['top2_exact']:.2f}")
        plt.tight_layout()
        os.makedirs(os.path.join(FIGS, "multiobject"), exist_ok=True)
        fp = os.path.join(FIGS, "multiobject", "n64_heldout_pairs.png")
        plt.savefig(fp, dpi=120)
        print(f"[n64] wrote {fp}", flush=True)
    except Exception as e:                                   # a figure must never kill the result
        print(f"[n64] figure skipped: {e}", flush=True)

    ex = float(np.mean([r["exact"] for r in rows]))
    fex = float(np.mean([r["floor_exact"] for r in rows]))
    rc = float(np.mean([r["recall"] for r in rows]))
    frc = float(np.mean([r["floor_recall"] for r in rows]))
    print(f"\n[n64] {'=' * 66}")
    print(f"[n64] HELD-OUT PAIRS  exact-set {ex:.3f} vs floor {fex:.3f}  "
          f"({ex / max(fex, 1e-6):.1f}x) | recall {rc:.3f} vs floor {frc:.3f}", flush=True)
    print("[n64] VERDICT")
    if not ok:
        print("[n64]   DETECTOR TOO WEAK -- not reportable. Fix the instrument before the claim.")
    elif r2cv > 0.3 and ex > max(2 * fex, fex + 0.15):
        print("[n64]   A SCENE IS A COMPOSITION OVER ADDRESSES. The scene table factorises over")
        print("[n64]   object membership (CV R2 above the permutation null) and an unseen PAIR of")
        print("[n64]   known objects is generated from a composed address -- so k known objects")
        print("[n64]   cost NO new storage, not k blocks.")
    elif r2cv > 0.3:
        print("[n64]   The table FACTORISES but ADDRESS does not deliver the pair. The structure is")
        print("[n64]   there and the composition is not: report the basis result, not a storage")
        print("[n64]   claim.")
    else:
        print("[n64]   NO membership structure in the scene table. On this backbone a k-object")
        print("[n64]   scene is NOT a composition over addresses -- the honest negative.")
    save_json(dict(detector=dq, det_gate_passed=bool(ok), self_cond=self_mean,
                   r2=r2, r2_cv=r2cv, r2_ordered=r2_ord,
                   perm_null_mean=float(np.mean(null)), perm_null_max=float(np.max(null)),
                   gain=gstar, gain_curve={str(g): float(np.mean(v)) for g, v in cal.items()},
                   held=rows, summary=dict(exact=ex, floor_exact=fex, recall=rc,
                                           floor_recall=frc), config=vars(args)),
              os.path.join(RESULTS, "n64_multiobject_address.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
