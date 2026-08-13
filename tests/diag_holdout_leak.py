"""Are the recovered prototypes MEMORIZED held-out images, or composed from seen parts?

Some case-study prototypes look exactly like instances of the OOD slot.  If the backbone had ever
seen those images the whole result would be retrieval wearing composition's clothes, so this checks
it rather than assuming the split is honest.

The split itself is verifiable by reading: `n1_train` fits on `data["X"][seen]` and its class table
remaps only the 120 seen slots, so the 40 OOD slots have no images and no embedding row.  That is
asserted here directly against the checkpoint, not taken on trust.

But a clean split does not settle the appearance, because two innocent mechanisms produce prototypes
that look like OOD instances:

  1. MODE ASCENT IS INITIALIZED FROM THE QUERY (`init="query"` / `"hybrid"`).  A chain that starts at
     the query and moves a short distance still looks like the query, and the query IS an instance of
     the OOD slot.  That is initialization, not knowledge.
  2. ColorMNIST is factored.  A held-out slot shares its digit with seen slots and both its colours
     with other seen slots, so a mode of the TRAINED distribution can carry the right digit in the
     right colours without any held-out image ever existing.  That is the phenomenon under study.

and one guilty one: the backbone memorized held-out images.

The three are separable by asking WHAT each prototype is nearest to:

    d(query)      the one held-out image the method was given -- initialization reproduces this
    d(held-out)   OTHER images of the same OOD slot, which the model never saw
    d(train)      the images the model was actually fitted on
    d(real-real)  how far apart two real held-out images of the slot are -- the scale bar

Memorization means a prototype sits closer to an UNSEEN held-out image than real images of that slot
sit to each other, while being far from anything in the training set.

The sharpest cut is the RANDOM-initialized half of the pool.  With `init="hybrid"` half the chains
start from noise and never see the query, so "it looks like the query because it started there"
cannot apply to them.  If a random-init mode lands on a held-out image, that is memorization; the
diagnostic reports the two halves separately for exactly that reason.

    python tests/diag_holdout_leak.py
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
from lifelong_ood.neural import load_field, to_data, to_model                  # noqa: E402


def nn_rmse(P, R):
    """-> for each row of P, the RMSE to its nearest row of R."""
    P, R = np.atleast_2d(P), np.atleast_2d(R)
    d = np.sqrt(((P[:, None] - R[None]) ** 2).mean(2))
    return d.min(1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(CACHE, "ddpm_colormnist.pt")
    data = load_colormnist(n_per_slot=100, seed=0)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]

    # ── 1. the split, asserted against the checkpoint ────────────────────────
    seen_slots = set(int(s) for s in np.unique(data["slot"][data["seen"]]))
    ood_slots = set(int(s) for s in np.unique(data["slot"][~data["seen"]]))
    trained = set(int(s) for s in remap)
    print(f"[leak] slots: {len(seen_slots)} seen, {len(ood_slots)} held out")
    print(f"[leak] checkpoint class table covers {len(trained)} slots")
    bad = trained & ood_slots
    print(f"[leak] held-out slots WITH a trained embedding row: {len(bad)}"
          f"{'  <-- LEAK' if bad else '  (none -- the split is clean)'}")
    print(f"[leak] trained rows that are not seen slots: {len(trained - seen_slots)}\n")

    field = load_field(ckpt, device=dev)
    TRAIN = data["X"][data["seen"]]                    # what the model actually saw
    qs = ood_queries(data, n_slots=4, seed=0)

    # PER PROTOTYPE, not a median over the K.  The concern is about the ONE prototype that looks
    # like a clean instance of the OOD slot; the other two are visibly junk, and a median over all
    # three is dominated by them -- it would answer a question nobody asked.  `k` is the selection
    # ORDER, so k=0 is the first pick of the submodular greedy.
    print("  Nearest-neighbour RMSE per SELECTED PROTOTYPE. `real-real` is the scale bar: how far\n"
          "  apart two REAL held-out images of the same slot are.\n")
    print(f"  {'slot':>5s} {'init':>7s} {'k':>2s} {'d(query)':>9s} {'d(held-out)':>12s} "
          f"{'d(train)':>9s} {'real-real':>10s} {'verdict':>24s}")
    for q in qs:
        xq = to_model(data["X"][q["query"]]).ravel()
        rest = data["X"][q["rest"]]                     # same slot, NEVER seen by the model
        rr = float(np.median(nn_rmse(rest[:40], rest[40:80])))
        # query-initialized and random-initialized halves, run separately so the two mechanisms
        # can be told apart rather than pooled
        for init in ("query", "random"):
            kw = dict(PAPER_COLORMNIST)
            kw["init"] = init
            M, V, meta = candidates_ascent(field, xq, seed=0, **kw)
            t = poe_teacher(xq, M, V, K=3)
            P = to_data(np.asarray(M)[t["sel"]]).astype(np.float32)
            for k in range(len(P)):
                dq = float(nn_rmse(P[k:k + 1], data["X"][q["query"]][None])[0])
                dh = float(nn_rmse(P[k:k + 1], rest)[0])
                dt = float(nn_rmse(P[k:k + 1], TRAIN)[0])
                if dh < 0.5 * rr and dt > dh:
                    v = "MEMORIZED held-out"
                elif dq < 0.5 * rr:
                    v = "IS the query (init)"
                elif dt <= dh:
                    v = "nearest TRAIN (healthy)"
                else:
                    v = "near held-out"
                print(f"  {q['slot']:5d} {init:>7s} {k:2d} {dq:9.4f} {dh:12.4f} {dt:9.4f} "
                      f"{rr:10.4f} {v:>24s}")

    print("\n[leak] MEMORIZED requires a prototype closer to an image the model never saw than two\n"
          "       real images of that slot are to each other, AND further from everything it was\n"
          "       trained on. `tracks the query` on the query-init rows is expected and benign --\n"
          "       the chain starts there; the RANDOM rows are the ones that carry evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
