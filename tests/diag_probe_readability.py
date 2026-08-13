"""The CelebA attribute probes are linear-on-raw-pixels, and that is why 9 of 14 are unreadable.

RESULTS 5e-sexies had to withdraw the whole `joint` accuracy column on CelebA: conditioning on a
class's OWN trained embedding scored 0.100 against 0.142 for no conditioning at all, so the metric's
ceiling sat at its floor and could not rank anything.  The proximate cause is that `attr_accuracy`
needs every readable probe correct at once and only **5 of 14** pass `probe_usable`.

The root cause is the probe itself.  `fit_attr_probes` trains a `SoftmaxProbe` -- a linear softmax
classifier on RAW PIXELS.  On ColorMNIST that is the right instrument: the factors are colour and
digit shape, both close to linear in pixel space.  On CelebA the attributes are `Big_Lips`,
`Oval_Face`, `Wavy_Hair` -- properties no linear function of raw pixels can express.  Nine failing
probes is not a statement about the generator; it is the wrong measuring device.

The fix is the same instrument the composition metrics already use.  `lifelong_ood.inception`
extracts 2048-d pool3 features, and a linear probe on top of those is the standard
representation-quality protocol.  This compares the two, per attribute, on REAL images only -- no
generator involved, so it measures the probe and nothing else:

    pixel      linear probe on raw pixels             (what the repository uses today)
    inception  linear probe on Inception-V3 pool3     (the proposal)

Reported per attribute: balanced accuracy, TPR, and whether `probe_usable` admits it.  What matters
is the COUNT of usable probes, because `joint` needs all of them simultaneously and each unreadable
attribute is one the benchmark cannot see.

    python tests/diag_probe_readability.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from lifelong_ood import inception                                             # noqa: E402
from lifelong_ood.celeba import ATTRS, load_celeba                             # noqa: E402
from lifelong_ood.evals import SoftmaxProbe, probe_usable                      # noqa: E402


def fit_report(X, A, attrs, seed=0, epochs=60):
    """-> per-attribute (bal, tpr, usable) for a linear probe on whatever features X holds."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.9 * n)
    tr, va = idx[:cut], idx[cut:]
    out = {}
    for k, a in enumerate(attrs):
        y = A[:, k].astype(int)
        pr = SoftmaxProbe(2, seed=seed, epochs=epochs).fit(X[tr], y[tr])
        p, yv = pr.predict(X[va]), y[va]
        tpr = float((p[yv == 1] == 1).mean()) if (yv == 1).any() else float("nan")
        tnr = float((p[yv == 0] == 0).mean()) if (yv == 0).any() else float("nan")
        out[a] = dict(bal=0.5 * (tpr + tnr), tpr=tpr, tnr=tnr, base=float(yv.mean()))
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = 128                                     # the paper's CelebA resolution (their Table 5)
    data = load_celeba(n_per_class=120, img=img, seed=0, min_count=25, max_rows=25000)
    X = np.asarray(data["X"], np.float32)
    A = np.asarray(data["attrs"])
    print(f"[probe] {len(X)} real CelebA images at {img}px, {len(ATTRS)} attributes\n")

    feats = {"pixel": X}
    if inception.available():
        print("[probe] extracting Inception-V3 pool3 features ...", flush=True)
        feats["inception"] = inception.features(X, img=img, device=dev)
    else:
        print("[probe] inception unavailable -- pixel only")

    rep = {k: fit_report(F, A, ATTRS) for k, F in feats.items()}
    keys = list(rep)
    print(f"  {'attribute':16s} {'base':>6s}" +
          "".join(f" | {k:>9s} bal  tpr  ok" for k in keys))
    counts = {k: 0 for k in keys}
    for a in ATTRS:
        line = f"  {a:16s} {rep[keys[0]][a]['base']:6.3f}"
        for k in keys:
            r = rep[k][a]
            ok = probe_usable({f"{a}_bal": r["bal"], f"{a}_tpr": r["tpr"]}, a)
            counts[k] += bool(ok)
            line += f" | {'':>9s}{r['bal']:.2f} {r['tpr']:.2f}  {'Y' if ok else '.'}"
        print(line)
    print()
    for k in keys:
        print(f"  {k:>10s}: {counts[k]}/{len(ATTRS)} attributes readable")
    print("\n[probe] `joint` requires every readable attribute correct at once, so the count is what\n"
          "        decides whether a CelebA accuracy number can rank anything at all. A large gap\n"
          "        here means the withdrawn column was measuring the probe, not the generator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
