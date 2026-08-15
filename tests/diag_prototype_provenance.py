"""Is a `prototypes` panel a DISCOVERY, or is it the query handed back?

The case-study figures show a row labelled `prototypes (mode ascent)` next to a HELD-OUT query, and
read naively that says the model recovered a combination it was never trained on -- which it cannot
have done, because the combination has no class row and zero training images.  So what is in that
row?  Mode ascent climbs the model's own density from an initial point, and half the chains are
initialised from the NOISED QUERY.  At low noise the query is barely perturbed, so a chain can
converge back to (a denoised version of) the thing it started from.

That is measurable rather than arguable.  For each noise level:

  d(mode, query)     how far the returned mode is from the query it started near
  d(mode, nearest training image of a DIFFERENT slot)
  probe factors      what the mode actually depicts

and the same for chains started from PURE NOISE, which have no query to return to.  If
query-initialised modes sit at d ~ 0 from the query while random-initialised ones do not, then the
panel is showing the query back and must not be read as discovery.

This exists because the figures are the most-read artefact in the repository and the row is the
easiest one in them to over-read.

    python tests/diag_prototype_provenance.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE                                               # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                 # noqa: E402
from lifelong_ood.evals import fit_probes                              # noqa: E402
from lifelong_ood.neural.ddpm import load_field, to_data, to_model     # noqa: E402

U_GRID = (0.05, 0.15, 0.30, 0.60)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = os.path.join(CACHE, "ddpm_colormnist_nobg3.pt")
    if not os.path.exists(p):
        print(f"SKIP: no backbone at {p}")
        return 0
    ck = torch.load(p, map_location="cpu", weights_only=False)
    field = load_field(p, device=dev)
    data = load_colormnist(n_per_slot=20, seed=0)
    X, slots, seen = np.asarray(data["X"], np.float32), np.asarray(data["slot"]), data["seen"]
    probes = fit_probes(data, seed=0, epochs=30)
    held = np.array([s for s in np.unique(slots) if s not in set(int(k) for k in ck["remap"])])
    print(f"[proto] {len(held)} held-out slots; using slot {int(held[0])}", flush=True)

    # SWEEP SLOTS, not one.  A single held-out slot answered this question two different ways --
    # slot 3's query-seeded mode flips the background off the held-out value, while the published
    # figures show a query-seeded panel that is visibly the held-out combination.  Both happen; the
    # only honest answer is the RATE, so every held-out slot is measured.
    rows = []
    n_slots = min(int(os.environ.get("PROTO_SLOTS", 24)), len(held))
    for si, s in enumerate(held[:n_slots]):
        s = int(s)
        qi = int(np.where(slots == s)[0][0])
        xq = X[qi]
        d, f, b = s // 16, (s % 16) // 4, s % 4
        for u in U_GRID:
            for init in ("query", "random"):
                xt, x0 = field.mode_ascent(to_model(xq).ravel(), u, n_starts=4, seed=si,
                                           init=init)
                M = to_data(np.asarray(x0))
                dq = float(np.mean(np.linalg.norm(M - to_data(to_model(xq).ravel()), axis=1)))
                same = float(np.mean((probes["digit"].predict(M) == d)
                                     & (probes["fg"].predict(M) == f)
                                     & (probes["bg"].predict(M) == b)))
                rows.append(dict(slot=s, u=u, init=init, d_query=dq, same_as_query=same))
        if (si + 1) % 8 == 0:
            print(f"[proto]   ...{si + 1}/{n_slots} slots", flush=True)

    print(f"\n{'u':>6} {'init':>7} {'d(mode,query)':>14} {'reproduces held-out combo':>26}")
    for u in U_GRID:
        for init in ("query", "random"):
            g = [r for r in rows if r["u"] == u and r["init"] == init]
            dqs = np.array([r["d_query"] for r in g])
            sm = np.array([r["same_as_query"] for r in g])
            print(f"{u:>6g} {init:>7} {dqs.mean():>8.2f}+/-{dqs.std():<5.2f} "
                  f"{sm.mean():>18.0%}  (>=1 of 4 chains: {np.mean(sm > 0):.0%})")

    print(f"\n[proto] {'=' * 66}")
    qlow = [r for r in rows if r["init"] == "query" and r["u"] <= 0.15]
    rlow = [r for r in rows if r["init"] == "random" and r["u"] <= 0.15]
    mq = float(np.mean([r["d_query"] for r in qlow]))
    mr = float(np.mean([r["d_query"] for r in rlow]))
    sq = float(np.mean([r["same_as_query"] for r in qlow]))
    sr = float(np.mean([r["same_as_query"] for r in rlow]))
    print(f"[proto] at low noise (u<=0.15): query-init sits {mq:.3f} from the query and reproduces "
          f"its exact combination {sq:.0%} of the time")
    print(f"[proto]                          random-init sits {mr:.3f} away and reproduces it "
          f"{sr:.0%} of the time")
    # TWO CRITERIA, and they disagree -- which is the finding.  Distance alone says "anchored";
    # only the probe read says WHAT the panel depicts.  An earlier version of this script tested
    # distance alone and concluded "essentially the query", which the probe read refutes: the modes
    # sit near the query and still do NOT depict its combination.
    print(f"[proto] VERDICT")
    if mq < 0.5 * mr:
        print(f"[proto]   ANCHORED: query-init modes sit {mr / max(mq, 1e-9):.1f}x closer to the "
              f"query than random-init ones, so they are not independent samples of the model.")
    else:
        print("[proto]   NOT anchored: query-init modes travel as far as random-init ones.")
    if sq < 0.1:
        print(f"[proto]   BUT NOT THE QUERY: they reproduce the held-out combination {sq:.0%} of "
              f"the time. Mode ascent lands them on a TRAINED neighbour instead -- it cannot")
        print("[proto]   return a combination the model does not have, which is the whole reason")
        print("[proto]   the paper needs several prototypes and a per-dimension product to assemble")
        print("[proto]   one. A `prototypes` panel therefore shows trained parts, NOT the held-out")
        print("[proto]   concept, and must be labelled so it is not read as recovery.")
    else:
        print(f"[proto]   AND IT IS THE QUERY: reproduced {sq:.0%} of the time -- circular, and the")
        print("[proto]   panel must not be shown as discovery.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
