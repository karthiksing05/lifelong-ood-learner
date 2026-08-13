"""Does the PoE teacher find the COMPOSITION, or does it find the QUERY?

RESULTS 5e-quater: `poe-guided` and `query-only` render identically in every block of the case-study
figure and score within noise of each other in N49 (F1 0.569/0.363 vs 0.612/0.316).  The mechanism is
in the selection objective -- Eq. 8-9 scores each candidate by how well it explains the QUERY, so a
candidate whose ascent chain STARTED at the query is favoured by construction, and the first greedy
pick lands closer to the query than two real images of the class lie to each other.

That is a hypothesis about a mechanism, and it makes a testable prediction: take the query-seeded
candidates OUT of the eligible set -- leaving them in the ascent, so the query still seeds the search
-- and the teacher must be built from query-agnostic modes.  Then either

    the composition SURVIVES   -> discovery really did find it, and the query-seeded candidate was
                                  a shortcut that selection preferred but did not need
    the composition COLLAPSES  -> what the teacher knew about the OOD class came from the query, and
                                  `poe-guided` is `query-only` with extra steps

Three arms, one shared candidate pool per query so the comparison is paired and the only difference
is which candidates selection may pick:

    all        selection over the whole pool                       (the method as it runs today)
    random     selection over query-agnostic candidates only       (the control)
    query      selection over query-seeded candidates only         (the bias, isolated)

Reported per arm: composition accuracy of the teacher MEAN read by the probes (does mu_T carry the
right digit/fg/bg?), and its distance to the query relative to the class's own spread.

    python tests/diag_selection_bias.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE, ood_queries                                          # noqa: E402

from lifelong_ood.compose import PAPER_COLORMNIST, candidates_ascent, poe_teacher  # noqa: E402
from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes                 # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_colormnist(n_per_slot=100, seed=0)
    probes = fit_probes(data, seed=0, epochs=30)
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    qs = ood_queries(data, n_slots=6, seed=0)

    print("[selbias] the teacher MEAN, read by the attribute probes, under three eligible sets.\n"
          "          `d(query)/spread` < 1 means mu_T sits closer to the query than two real images\n"
          "          of its class sit to each other.\n")
    print(f"  {'slot':>5s} {'eligible':>9s} {'n_elig':>7s} {'joint':>6s} {'digit':>6s} {'fg':>5s} "
          f"{'bg':>5s} {'d(query)/spread':>16s}")
    agg = {}
    for q in qs:
        xq = to_model(data["X"][q["query"]]).ravel()
        M, V, meta = candidates_ascent(field, xq, seed=0, **PAPER_COLORMNIST)
        src = np.asarray(meta["src"])
        rest = data["X"][q["rest"]]
        spread = float(np.median(np.sqrt(((rest[:40, None] - rest[None, 40:80]) ** 2).mean(2))))
        for name, mask in (("all", None), ("random", src == "random"), ("query", src == "query")):
            t = poe_teacher(xq, M, V, K=3, eligible=mask)
            mu01 = to_data(np.asarray(t["mu"])[None]).astype(np.float32)
            a = composition_accuracy(probes, mu01, q["d"], q["f"], q["b"])
            dq = float(np.sqrt(((mu01[0] - data["X"][q["query"]]) ** 2).mean())) / max(spread, 1e-8)
            n_el = len(M) if mask is None else int(mask.sum())
            print(f"  {q['slot']:5d} {name:>9s} {n_el:7d} {a['joint']:6.2f} {a['digit']:6.2f} "
                  f"{a['fg']:5.2f} {a['bg']:5.2f} {dq:16.3f}")
            agg.setdefault(name, []).append((a["joint"], a["digit"], a["fg"], a["bg"], dq))

    print(f"\n  {'MEAN over slots':>15s} {'joint':>6s} {'digit':>6s} {'fg':>5s} {'bg':>5s} "
          f"{'d(query)/spread':>16s}")
    for name, rows in agg.items():
        m = np.mean(np.array(rows), 0)
        print(f"  {name:>15s} {m[0]:6.2f} {m[1]:6.2f} {m[2]:5.2f} {m[3]:5.2f} {m[4]:16.3f}")

    print("\n[selbias] if `random` holds its composition accuracy while its d(query)/spread rises,\n"
          "          discovery genuinely found the concept and selection was merely taking the\n"
          "          shortcut. If `random` collapses, the teacher's knowledge came from the query.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
