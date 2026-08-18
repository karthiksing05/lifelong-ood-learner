"""Does CelebA conditioning MOVE the distribution between categories, or only within one?

Raised from the case study: the `ours (ADDRESS)` rows for DIFFERENT held-out attribute
combinations look like the same population of faces, and individual samples do not obviously carry
the combination they were conditioned on.  Both are measurable, and the second is the sharper test:
if conditioning on combination A and on combination B produce the same distribution, the
conditioning is inert regardless of what any per-attribute accuracy says.

Two quantities, each against the scale bar that makes it interpretable:

  BETWEEN vs WITHIN.  Mean distance between samples drawn from DIFFERENT combinations, over the
  mean distance between samples drawn from the SAME combination.  A ratio near 1.0 means the
  conditioning does not separate categories at all -- the samples are one population wearing
  different labels.  The same ratio computed on REAL images of those combinations is the ceiling:
  it is what "as separated as the data itself" looks like.

  ATTRIBUTE MARGIN.  Per-attribute accuracy of the samples against their own target, minus the same
  for the UNCONDITIONAL floor.  CelebA attributes are near-constant (No_Beard almost always 1), so a
  base-rate guess already scores ~0.66; only the MARGIN over that floor is evidence.

The comparison is run on `ddpm_celeba128.pt` (ours, with a real holdout) rather than the paper's
fine-tuned backbone, because the question is whether OUR conditioning separates classes.

    python tests/diag_celeba_between_class.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.data.celeba import ATTRS, load_celeba                # noqa: E402
from lifelong_ood.evals import fit_attr_probes                         # noqa: E402
from lifelong_ood.neural import load_field, sample_embedding, to_data  # noqa: E402


def pair_dist(A, B=None, cap=12):
    A = np.asarray(A, np.float32)[:cap]
    if B is None:
        return float(np.mean([np.linalg.norm(A[i] - A[j])
                              for i in range(len(A)) for j in range(i + 1, len(A))]))
    B = np.asarray(B, np.float32)[:cap]
    return float(np.mean([np.linalg.norm(a - b) for a in A for b in B]))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = os.path.join(CACHE, "ddpm_celeba128.pt")
    if not os.path.exists(p):
        print(f"SKIP: no backbone at {p}")
        return 0
    ck = torch.load(p, map_location="cpu", weights_only=False)
    field = load_field(p, device=dev)
    d = load_celeba(img=int(ck.get("img", 128)), min_count=int(ck.get("min_count", 120)),
                    seed=int(ck.get("seed", 0)))
    cls = np.asarray(d["cls"] if "cls" in d else d["slot"])
    X = np.asarray(d["X"], np.float32)
    A = np.asarray(d["attrs"])
    probes = fit_attr_probes(d, seed=0)
    names = [n for n in ATTRS if n in probes]

    # TRAINED classes, not held-out ones: the question is whether conditioning separates AT ALL,
    # so it must first be asked where the model was actually taught the answer.  If it fails here
    # it cannot be expected to work on a composed address.
    remap = ck["remap"]
    trained = [int(k) for k in list(remap)[:4]]
    trained = [c for c in trained if (cls == c).any()][:4]
    print(f"[cbc] {len(trained)} TRAINED classes: {trained}", flush=True)

    # This backbone conditions on an ATTRIBUTE VECTOR (`attr_embedding`), not on a class row --
    # `cemb` does not exist on it.  Each class IS an attribute combination, so the conditioning
    # vector for a class is its own attribute vector pushed through that embedding.
    S, R = {}, {}
    for c in trained:
        rows = np.where(cls == c)[0]
        combo = A[int(rows[0])][:len(ATTRS)]
        emb = field.model.attr_embedding(
            torch.as_tensor(np.array([[int(v) for v in combo]]), dtype=torch.long)
        ).detach().cpu().numpy().ravel()
        S[c] = to_data(sample_embedding(field, emb, n=12, steps=200, seed=0))
        R[c] = X[rows[:12]]

    within_s = float(np.mean([pair_dist(S[c]) for c in trained]))
    within_r = float(np.mean([pair_dist(R[c]) for c in trained]))
    between_s, between_r = [], []
    for i, a in enumerate(trained):
        for b in trained[i + 1:]:
            between_s.append(pair_dist(S[a], S[b]))
            between_r.append(pair_dist(R[a], R[b]))
    bs, br = float(np.mean(between_s)), float(np.mean(between_r))

    print(f"\n[cbc] {'':22s}{'within':>10s}{'between':>10s}{'ratio':>9s}")
    print(f"[cbc] {'SAMPLES (conditioned)':22s}{within_s:>10.2f}{bs:>10.2f}"
          f"{bs / max(within_s, 1e-9):>9.3f}")
    print(f"[cbc] {'REAL images (ceiling)':22s}{within_r:>10.2f}{br:>10.2f}"
          f"{br / max(within_r, 1e-9):>9.3f}")

    # attribute margin over the unconditional floor
    U = to_data(field.sample(n=12, steps=200, seed=7, clip=(-1, 1), branch=None))
    marg = []
    for c in trained:
        tgt = A[int(np.where(cls == c)[0][0])]
        hit = np.mean([float((probes[n].predict(np.asarray(S[c], np.float32))
                              == int(tgt[ATTRS.index(n)])).mean()) for n in names])
        flo = np.mean([float((probes[n].predict(np.asarray(U, np.float32))
                              == int(tgt[ATTRS.index(n)])).mean()) for n in names])
        marg.append(hit - flo)
        print(f"[cbc]   class {c}: attr {hit:.3f} vs unconditional {flo:.3f} "
              f"-> margin {hit - flo:+.3f}")
    m = float(np.mean(marg))

    print(f"\n[cbc] {'=' * 66}")
    sep = bs / max(within_s, 1e-9)
    ceil = br / max(within_r, 1e-9)
    print(f"[cbc] between/within = {sep:.3f} for samples, {ceil:.3f} for real images")
    print(f"[cbc] mean attribute margin over the floor = {m:+.3f}")
    # GATE ON THE CEILING FIRST.  Measured, the REAL-image ratio is 1.001: genuine CelebA photos of
    # different attribute combinations are no further apart in pixel space than photos of the same
    # combination.  Pixel distance therefore cannot discriminate CelebA classes AT ALL, and a
    # sample ratio near 1.0 is what a perfect conditioner would also produce.  Reading it as
    # "conditioning is inert" would be reading a blind instrument -- the same fault N69 (floor too
    # high) and N70 (unreadable probe) hit, and the third time a verdict has fired on a metric that
    # could not discriminate.  The ATTRIBUTE MARGIN is the informative quantity here.
    if ceil < 1.05:
        print(f"[cbc] PIXEL SEPARATION IS UNINFORMATIVE HERE: the REAL-image ceiling is "
              f"{ceil:.3f}, so this metric does not separate CelebA classes even on real data.")
        print(f"[cbc]   The sample ratio ({sep:.3f}) is therefore not evidence either way.")
        print(f"[cbc]   Judge on the ATTRIBUTE MARGIN: {m:+.3f} over the unconditional floor "
              f"({'weak' if m < 0.15 else 'substantial'}), positive on "
              f"{sum(1 for x in marg if x > 0)}/{len(marg)} classes.")
        if m < 0.15:
            print("[cbc]   A margin this small is consistent with samples that do not visibly")
            print("[cbc]   carry their combination. CelebA composition numbers must be reported")
            print("[cbc]   with it, and CelebA must not be the dataset a composition claim rests on.")
    elif sep < 1.05:
        print("[cbc] VERDICT: CONDITIONING DOES NOT SEPARATE CLASSES. Samples from different")
        print("[cbc]   combinations are no further apart than samples from the SAME one -- one")
        print("[cbc]   population wearing different labels. Any composed address inherits this,")
        print("[cbc]   so a CelebA ADDRESS number cannot be evidence of composition.")
    elif sep < 0.5 * ceil:
        print(f"[cbc] VERDICT: WEAK. Samples separate at {sep:.3f} against a real-image ceiling of")
        print(f"[cbc]   {ceil:.3f}, so the conditioning moves the distribution far less than the")
        print("[cbc]   data does. Report CelebA composition with that stated.")
    else:
        print("[cbc] VERDICT: conditioning separates classes comparably to the data itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
