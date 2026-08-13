"""Accumulate, then promote: the router decision taken over a STREAM instead of on arrival.

RESULTS 6c-ter's verdict is that description-length routing loses to the threshold cascade, and its
diagnosis is why: under

    L(tier) = C_tier + N * (bits to code one example under that tier)

the fit term scales with the number of examples and the storage cost does not, so the tier a query
warrants is not a property of the query at all -- it is a property of the query AND how many
instances have arrived.  Measured, ~15 instances justify a 72 B address and ~36 justify a 1 KB
vocabulary extension.  n18 hands the router ONE.  So the rule was being asked a question it cannot
answer, and its "store nothing" answer at N=1 is arithmetically correct rather than broken.

This is the regime the rule is actually for, and the one Cobweb operates in: an instance does not
force a restructure on sight: it updates counts, and structure changes when the accumulated evidence
justifies it.  Concepts here RECUR.  The system keeps a buffer keyed by the basis row a query infers,
accumulates that concept's coding gain across its instances, and promotes a concept up the ladder the
moment its accrued gain exceeds the tier's cost.  Nothing is thresholded and nothing is calibrated;
the only quantities are bits.

What this measures, against the cascade run on the same stream:

  * FINAL accuracy once the stream has delivered enough of each concept
  * TIME-TO-PROMOTION -- how many instances each population needs before it is correctly placed
  * TOTAL BYTES spent over the stream against the oracle, which is the number 5z/N24 charge back
    into the growth curve and the one the over-spend was costing

The cascade is batch-blind by construction: it decides on the first instance and never revisits, so
its curve is flat by definition.  That is the comparison.

Reuses the codelengths `tests/diag_mdl_router.py` already recorded -- the per-concept per-N bits are
exactly what a stream accumulates -- so this needs no GPU.

    python tests/diag_mdl_stream.py [results/diag_mdl_router.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.neural.router import TIER_BITS, TIER_ROUTE  # noqa: E402

ROUTES = ["in-distribution", "new-combination", "new-vocabulary"]
RANK = {r: i for i, r in enumerate(ROUTES)}
ROUTE_BYTES = {"in-distribution": 0.0, "new-combination": 72.0, "new-vocabulary": 1024.0}
TIERS = ["none", "address", "extend"]


def promote(code, n):
    """Cheapest-to-dearest: the dearest tier whose cost is repaid by n instances.  -> tier.

    Identical arithmetic to `mdl_route_bits`, written as a promotion so the intent is visible: a
    concept sits at `none` until its accrued coding gain covers ADDRESS's 576 bits, then sits there
    until it covers EXTEND's 8192.  A concept never *demotes*, because accrued gain only grows.
    """
    L = {t: TIER_BITS[t] + n * code[t] for t in TIERS}
    return min(L, key=L.get)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results",
                                                              "diag_mdl_router.json")
    if not os.path.exists(path):
        print(f"[stream] {path} not found -- run tests/diag_mdl_router.py first")
        return 1
    d = json.load(open(path, encoding="utf-8"))
    rows = [r for r in d["rows"] if "code" in r]
    if not rows:
        print("[stream] no recorded codelengths")
        return 1

    Ns = sorted({r["N"] for r in rows})
    # per concept: its codelength per tier at each batch size actually measured
    by_slot = {}
    for r in rows:
        by_slot.setdefault(r["slot"], {"truth": r["truth"], "cascade": r.get("cascade"),
                                       "code": {}})["code"][r["N"]] = r["code"]
    print(f"[stream] {len(by_slot)} concepts, instance counts measured at {Ns}\n")

    # ── time to correct placement ────────────────────────────────────────────────────────
    print("[stream] instances needed before the concept is placed CORRECTLY "
          "(-- = not reached by N=%d):" % Ns[-1])
    hdr = f"{'population':>18} {'n concepts':>11} {'median n':>10} {'reached':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    ttp = {}
    for t in ROUTES:
        need = []
        for s, v in by_slot.items():
            if v["truth"] != t:
                continue
            hit = None
            for n in Ns:
                if n in v["code"] and TIER_ROUTE[promote(v["code"][n], n)] == t:
                    hit = n
                    break
            need.append(hit)
        ttp[t] = need
        got = [x for x in need if x is not None]
        med = f"{np.median(got):.0f}" if got else "--"
        print(f"{t:>18} {len(need):>11} {med:>10} {len(got):>4}/{len(need):<4}")

    # ── accuracy and bytes as the stream deepens ─────────────────────────────────────────
    print("\n[stream] accuracy and spend as each concept's instance count grows:")
    hdr = (f"{'n':>4} {'mdl acc':>9} {'mdl bytes':>11} {'cascade acc':>13} "
           f"{'cascade bytes':>15}")
    print(hdr + "\n" + "-" * len(hdr))
    for n in Ns:
        sl = [v for v in by_slot.values() if n in v["code"]]
        if not sl:
            continue
        mdl = [TIER_ROUTE[promote(v["code"][n], n)] for v in sl]
        truth = [v["truth"] for v in sl]
        casc = [v["cascade"] for v in sl]
        oracle = sum(ROUTE_BYTES[t] for t in truth)
        ma = float(np.mean([a == b for a, b in zip(mdl, truth)]))
        mb = sum(ROUTE_BYTES[a] for a in mdl) / oracle if oracle else float("nan")
        if all(c is not None for c in casc):
            ca = float(np.mean([a == b for a, b in zip(casc, truth)]))
            cb = sum(ROUTE_BYTES[a] for a in casc) / oracle if oracle else float("nan")
            print(f"{n:>4} {ma:>9.3f} {mb:>10.2f}x {ca:>13.3f} {cb:>14.2f}x")
        else:
            print(f"{n:>4} {ma:>9.3f} {mb:>10.2f}x {'--':>13} {'--':>15}")

    # ── the one thing a stream can do that a single decision cannot ──────────────────────
    print("\n[stream] does the ladder stay ORDERED as evidence accumulates?  "
          "(mean position, 0 nothing / 1 ADDRESS / 2 EXTEND)")
    print(f"{'population':>18} " + "".join(f"{('n=' + str(n)):>8}" for n in Ns))
    for t in ROUTES:
        cells = []
        for n in Ns:
            v = [RANK[TIER_ROUTE[promote(x["code"][n], n)]]
                 for x in by_slot.values() if x["truth"] == t and n in x["code"]]
            cells.append(f"{np.mean(v):>8.2f}" if v else f"{'--':>8}")
        print(f"{t:>18} " + "".join(cells))

    out = os.path.join(HERE, "results", "diag_mdl_stream.json")
    json.dump(dict(time_to_promotion={k: v for k, v in ttp.items()},
                   measured_at=Ns, source=os.path.basename(path)),
              open(out, "w", encoding="utf-8"), indent=1)
    print(f"\n[stream] wrote {out}")
    print("[stream] NOTE: instance counts stop at N=%d, and 6c-ter measures the median concept "
          "needing\n[stream]       ~15 (ADDRESS) and ~36 (EXTEND). A stream that stops at %d "
          "CANNOT show most\n[stream]       promotions -- read 'reached' above as a lower bound, "
          "not as a failure rate." % (Ns[-1], Ns[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
