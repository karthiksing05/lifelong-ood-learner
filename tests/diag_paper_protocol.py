"""Is `poe-guided` recall 0.001 a property of their method, or of how we measure it?

RESULTS §5e concludes the first: "the PoE teacher is a unimodal Gaussian, guiding toward it
collapses every trajectory onto its mean (recall 0.00 in every configuration tried)".  Seven
guidance schedules, three covariance scales, product-vs-mixture extent, ancestral sampling and
distillation were all ruled out by measurement, and `tests/diag_teacher_variance.py` has since ruled
out the `var_scale` asymmetry too (recall 0.000 at both 1.0 and 30.0).  So the collapse is real
**per query** -- vendi 1.06 out of 24 samples is one effective image.

But the paper never measures one query.  Its §4.2 protocol is:

    "For each OOD class, we run the full framework ... independently on each of the 100 queries"
    Table 8:  "Saved gens per query per teacher: 16"

so their generated set for ONE OOD class is 100 x 16 = 1600 samples drawn from **100 different
teachers**, and their two reference sets are

    Faithfulness    the 100 query images themselves          <- where PoE scores recall 80.7
    Generalization  100 other held-out images of the class   <- where PoE scores recall 69.8

We generate 24 samples from ONE query and score them against held-out images -- that is the
Generalization reference at 1/100th of the query diversity.  If each query collapses to a near-copy
of ITS OWN query, then pooling 100 such collapses gives a set of 100 distinct images that covers the
class, and recall is high without the per-query collapse ever being fixed.  Both facts would then be
true at once, and the number we publish as theirs would be answering a question they did not ask.

`n6_diversity.py` already pools, but against the Generalization reference only and at 12-24 queries,
which is the one combination that cannot distinguish these.  This sweeps the query count against
BOTH reference sets, generating once at the full budget and subsampling for the smaller counts so
the curve is paired.

    python tests/diag_paper_protocol.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
from common import CACHE                                                       # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.evals import paper_metrics, vendi_score                      # noqa: E402
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402
from lifelong_ood.neural.lifelong_neural import NeuralLifelongLearner          # noqa: E402

N_QUERIES = 100          # their §4.2
PER_QUERY = 16           # their Table 8, "saved gens per query per teacher"
PAPER_POE = dict(schedule="variance", var_scale=1.0, w0=1.2, wmax=2.0, wmin=0.0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 100 queries + 100 reference images of the SAME OOD class means the slot needs >= 200 rows
    data = load_colormnist(n_per_slot=260, seed=0)
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    L = NeuralLifelongLearner(field, device=dev)

    slots, counts = np.unique(data["slot"][~data["seen"]], return_counts=True)
    slot = int(slots[np.argmax(counts)])
    idx = np.where(data["slot"] == slot)[0]
    n_q = min(N_QUERIES, (len(idx) - 100) if len(idx) > 150 else len(idx) // 2)
    q_idx, r_idx = idx[:n_q], idx[n_q:n_q + 100]
    print(f"[proto] slot {slot}: {len(idx)} images -> {n_q} queries, {len(r_idx)} reference\n")

    xq_many = [to_model(data["X"][i]).ravel() for i in q_idx]
    print(f"[proto] running the framework independently on each of {n_q} queries "
          f"(their §4.2), {PER_QUERY} gens each = {n_q * PER_QUERY} samples")
    T = L.compose_batch(xq_many, u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=16, iters=120)
    P = L.sample_teacher_batch(T, n=PER_QUERY, seed=0, u_start=1.0, **PAPER_POE)
    P = [to_data(p).astype(np.float32) for p in P]

    REF = {"Faithfulness (the queries)": data["X"][q_idx],
           "Generalization (other held-out)": data["X"][r_idx]}

    # BOTH feature spaces.  Pixel is what every published number here used; Inception is what the
    # paper defines its metrics in, and the two disagree about what "a different instance of the
    # same class" means -- which is the whole content of a recall measurement.
    rows = []
    for space in ("pixel", "inception"):
        print(f"\n  === {space} features ===")
        print(f"  {'n queries':>10s} {'n samples':>10s} {'vendi':>7s}", end="")
        for name in REF:
            print(f" | {name[:15]:>15s} {'prec':>6s} {'rec':>6s} {'F1':>6s}", end="")
        print()
        for nq in (1, 4, 16, 50, n_q):
            if nq > n_q:
                continue
            S = np.concatenate(P[:nq])
            print(f"  {nq:10d} {len(S):10d} {vendi_score(S[:256]):7.2f}", end="")
            for name, R in REF.items():
                # against Faithfulness, score only the queries actually used to generate
                Rn = R[:nq] if name.startswith("Faith") else R
                m = (paper_metrics(S, Rn, space=space, device=dev) if len(Rn) >= 2
                     else dict(precision=float("nan"), recall=float("nan"), f1=float("nan")))
                print(f" | {'':>15s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}",
                      end="")
                rows.append(dict(space=space, n_queries=int(nq), n_samples=int(len(S)),
                                 reference=name.split()[0], precision=m["precision"],
                                 recall=m["recall"], f1=m["f1"]))
            print()

    print(f"\n[proto] the paper reports PoE = prec 75.5 / rec 80.7 / F1 75.3 (Faithfulness)\n"
          f"        and prec 67.0 / rec 69.8 / F1 63.0 (Generalization), in Inception-V3 features.\n"
          f"        Recall RISING with the query count shows the 0.001 we publish is the protocol\n"
          f"        and not the method; whether Generalization follows decides whether the\n"
          f"        remaining gap is the metric SPACE or a real limit of their teacher.")
    from lifelong_ood.evals import save_json
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "protocol_sweep.json")
    save_json(dict(rows=rows, slot=int(slot), per_query=PER_QUERY,
                   paper=dict(PoE_faithful=dict(precision=0.755, recall=0.807, f1=0.753),
                              PoE_general=dict(precision=0.670, recall=0.698, f1=0.630))), out)
    print(f"[proto] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
