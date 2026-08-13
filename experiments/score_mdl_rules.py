"""Score every routing rule that the MDL sweep's recorded codelengths can support, offline.

`tests/diag_mdl_router.py` records, per query and per batch size, the codelength in bits of the query
under each tier's best conditioning.  That is enough to evaluate several DIFFERENT decision rules
without touching a GPU again, which matters because the expensive part -- fitting a basis extension
per factor per query -- is already paid for.

Four rules, in increasing order of how much calibration they need:

  SIGN      Does each tier code the query BETTER than doing nothing, yes or no?  Route to the most
            expensive tier whose coding gain is positive.  No exchange rate, no threshold, no batch
            size -- only the sign of a difference.  Motivated by `tests/diag_elbo_stability.py`,
            which found the sign of that difference stable across seeds (100% agreement on 4 of 6
            query/band combinations, and stable-but-opposite-to-the-simple-loss on the vocabulary
            query, where the bound is the one that is semantically right: the basis has no direction
            for the missing value, so composing codes WORSE than the nearest trained slot).

  MDL       C_tier + N * code_tier, argmin.  The full rule.  Needs the codelength to be on a real
            scale, which the variational bound provides.

  CROSSOVER Route by which tier has the smallest N* (examples before it pays for itself), among
            those with N* <= N.  Equivalent to MDL, kept because it is the form that makes the
            exchange-rate reading explicit.

  UTILITY   Cobweb's category utility literally: coding gain PER BIT SPENT, argmax.  Scale-free but
            batch-blind.

Scored against the same three columns 6c-bis uses, so the rows are comparable to the cascade.

    python experiments/score_mdl_rules.py [results/diag_mdl_router.json]
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


def rule_sign(code, N):
    """Most expensive tier that codes better than storing nothing.  No calibration at all."""
    gain = {t: code["none"] - code[t] for t in ("address", "extend")}
    best = "none"
    for t in ("address", "extend"):
        if gain[t] > 0:
            best = t
    return best


def rule_mdl(code, N):
    L = {t: TIER_BITS[t] + N * code[t] for t in TIERS}
    return min(L, key=L.get)


def rule_crossover(code, N):
    star = {}
    for t in ("address", "extend"):
        g = code["none"] - code[t]
        star[t] = (TIER_BITS[t] - TIER_BITS["none"]) / g if g > 0 else float("inf")
    afford = [t for t in ("address", "extend") if star[t] <= N]
    return max(afford, key=lambda t: TIER_BITS[t]) if afford else "none"


def rule_utility(code, N):
    U = {t: (code["none"] - code[t]) / TIER_BITS[t] for t in ("address", "extend")}
    return max(U, key=U.get) if max(U.values()) > 0 else "none"


RULES = {"sign": rule_sign, "mdl": rule_mdl, "crossover": rule_crossover, "utility": rule_utility}


def score(rows, key):
    """-> dict of the columns 6c-bis reports."""
    comb = [r for r in rows if r["truth"] == "new-combination"]
    voc = [r for r in rows if r["truth"] == "new-vocabulary"]
    ind = [r for r in rows if r["truth"] == "in-distribution"]
    over = np.mean([RANK[r[key]] > RANK[r["truth"]] for r in comb + ind]) if comb + ind else np.nan
    under = np.mean([RANK[r[key]] < RANK[r["truth"]] for r in voc + comb]) if voc + comb else np.nan
    spent = sum(ROUTE_BYTES[r[key]] for r in rows)
    oracle = sum(ROUTE_BYTES[r["truth"]] for r in rows)
    return dict(acc=float(np.mean([r[key] == r["truth"] for r in rows])),
                over=float(over), under=float(under),
                bytes=spent / oracle if oracle else float("nan"))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results",
                                                              "diag_mdl_router.json")
    if not os.path.exists(path):
        print(f"[score] {path} not found")
        return 1
    d = json.load(open(path, encoding="utf-8"))
    rows = [r for r in d["rows"] if "code" in r]
    if not rows:
        print("[score] rows carry no recorded codelength -- rerun diag_mdl_router.py")
        return 1
    for r in rows:
        for name, fn in RULES.items():
            r[name] = TIER_ROUTE[fn(r["code"], r["N"])]

    Ns = sorted({r["N"] for r in rows})
    print(f"[score] {len(rows)} decisions, batch sizes {Ns}, "
          f"{len({r['slot'] for r in rows})} distinct concepts\n")
    hdr = f"{'N':>3} {'rule':>10} {'acc':>6} {'overspend':>10} {'underspend':>11} {'bytes/oracle':>13}"
    print(hdr)
    print("-" * len(hdr))
    for N in Ns:
        R = [r for r in rows if r["N"] == N]
        for name in list(RULES) + ["cascade"]:
            if name == "cascade" and "cascade" not in R[0]:
                continue
            s = score(R, name)
            print(f"{N:>3} {name:>10} {s['acc']:>6.3f} {s['over']:>10.2f} {s['under']:>11.2f} "
                  f"{s['bytes']:>13.2f}")
        print()

    # the promotion curve, as a table (f32 draws it)
    print("[score] promotion -- mean ladder position by MDL (0 nothing / 1 ADDRESS / 2 EXTEND):")
    print(f"{'truth':>18} " + "".join(f"{('N=' + str(n)):>8}" for n in Ns))
    for t in ROUTES:
        cells = []
        for N in Ns:
            v = [RANK[r["mdl"]] for r in rows if r["N"] == N and r["truth"] == t]
            cells.append(f"{np.mean(v):>8.2f}" if v else f"{'-':>8}")
        print(f"{t:>18} " + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
