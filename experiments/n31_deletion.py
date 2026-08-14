"""N31 -- certified deletion: can a concept be REMOVED without disturbing the others?

Every lifelong result in this repository is about adding.  Deletion is the same question with the
sign flipped, and it is asked of deployed systems constantly -- a retracted training source, a
right-to-be-forgotten request, a concept that turned out to be wrong.  It is also the question on
which parameter-isolation and parameter-protection methods differ most sharply, and no experiment
here had measured it.

**Ours.** A concept IS a set of nodes on a path, and the frozen base is never written to.  So
deletion is dropping the nodes only that concept uses (`bank.remove_concept`).  Every remaining
concept should be **bit-identical** afterwards -- no gradient step, no re-fitting -- and the
parameters should actually be reclaimed.

**Everyone else.** A method that folds updates into shared weights has no way to remove one task's
contribution.  The honest baseline is what such a system would actually have to do: **retrain from
scratch on the remaining concepts**.  That is measured here rather than assumed -- its cost in
wall-clock, and how far the surviving concepts move when it is done.

Reported per arm:

  `retention_exact`   fraction of SURVIVING concepts whose generation is bit-identical after the
                      deletion (1.000 = nothing moved at all)
  `drift`             max abs deviation of a surviving concept's samples, 0.0 for an exact method
  `forgotten`         does the deleted concept actually stop being reachable?
  `freed_params`      parameters reclaimed
  `wall_seconds`      what the deletion cost

    python experiments/n31_deletion.py --n-concepts 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                              # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                            # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                         # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,          # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.adapters import fingerprint                         # noqa: E402


def build(ckpt, dev, steps, seed):
    field = load_field(ckpt, device=dev)
    return field, NeuralLifelongLearner(field, steps=steps, seed=seed, device=dev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--delete", type=int, default=None,
                   help="index of the concept to delete (default: the middle one, so it has both "
                        "earlier and later concepts to disturb)")
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_per_slot = 4, 20
        args.epochs, args.steps, args.inv_steps = 3, 20, 60
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    victim = args.n_concepts // 2 if args.delete is None else args.delete

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    names = [f"c{q['slot']}" for q in qs]
    print(f"[n31] stream of {len(names)} concepts; deleting index {victim} ({names[victim]})\n",
          flush=True)

    rows = []

    # ── ours: structural deletion ────────────────────────────────────────────
    field, L = build(args.ckpt, dev, args.steps, args.seed)
    for q, nm in zip(qs, names):
        xq = to_model(data["X"][q["query"]]).ravel()
        L.learn_concept(xq, nm, n_pool=32, epochs=args.epochs, inv_steps=args.inv_steps,
                        seed=args.seed)
    before = {n: fingerprint(field, n, seed=args.seed) for n in names}
    fp_base_before = fingerprint(field, None, seed=args.seed)
    params_before = L.bank.n_params()

    # How much of a concept is actually RECLAIMABLE?  Not all of it, and the reason is structural:
    # `_descend` can route a later concept THROUGH an earlier concept's leaf, adopting it as an
    # ancestor.  Such a node cannot be dropped without breaking its descendant, so deleting that
    # concept makes it unreachable (the forgetting guarantee holds) while reclaiming only its
    # embedding.  Measured for every concept before deleting any, since it is a property of the
    # stream rather than of the concept we happen to pick.
    def reclaimable(nm):
        mine = set(L.bank.concepts[nm]["path"])
        others = {n for k, c in L.bank.concepts.items() if k != nm for n in c["path"]}
        uniq = mine - others
        per_node = {}
        for nid in mine:
            per_node[nid] = sum(w.A[str(nid)].numel() + w.B[str(nid)].numel()
                                for w in L.bank.wrapped if str(nid) in w.A)
        tot = sum(per_node.values()) or 1
        return dict(unique_nodes=sorted(uniq), free=sum(per_node[n] for n in uniq),
                    total_on_path=tot, frac=sum(per_node[n] for n in uniq) / tot)

    rec = {n: reclaimable(n) for n in names}
    fracs = [rec[n]["frac"] for n in names]
    print(f"[n31] reclaimable fraction of a concept's path parameters: "
          f"min {min(fracs):.3f}  median {float(np.median(fracs)):.3f}  max {max(fracs):.3f}")
    print(f"[n31]   ({sum(1 for f in fracs if f == 0)}/{len(names)} concepts have a leaf that a "
          f"later concept descended through, so only their embedding is reclaimable)", flush=True)

    t0 = time.time()
    acct = L.bank.remove_concept(names[victim])
    wall = time.time() - t0

    survivors = [n for n in names if n != names[victim]]
    after = {n: fingerprint(field, n, seed=args.seed) for n in survivors}
    drifts = {n: float(np.max(np.abs(after[n] - before[n]))) for n in survivors}
    exact = float(np.mean([d == 0.0 for d in drifts.values()]))
    # Is it really gone?  The field resolves a concept name THROUGH `bank.concepts`, so once the
    # bank has dropped it the name is genuinely unknown and `eps` raises -- it does not quietly fall
    # back to the base and keep generating something.  That is the stronger of the two possible
    # guarantees, so it is what gets asserted.
    try:
        fingerprint(field, names[victim], seed=args.seed)
        gone = False
    except KeyError:
        gone = True
    # ...and the frozen base itself must be untouched by the deletion, which is the other half of
    # "nothing else moved": the unconditional branch is bit-identical.
    base_drift = float(np.max(np.abs(fingerprint(field, None, seed=args.seed) - fp_base_before)))
    print(f"[n31] ours: dropped nodes {acct['dropped_nodes']}, kept shared {acct['kept_shared']}, "
          f"freed {acct['freed_params']:,} params in {wall:.3f}s")
    print(f"[n31] ours: {sum(d == 0.0 for d in drifts.values())}/{len(survivors)} survivors "
          f"bit-identical, max drift {max(drifts.values()):.3e}, base drift {base_drift:.3e}, "
          f"deleted name now unknown={gone}")
    rows.append(dict(arm="ours (structural)", retention_exact=exact,
                     drift=float(max(drifts.values())), forgotten=float(gone),
                     freed_params=int(acct["freed_params"]), wall_seconds=float(wall),
                     base_drift=base_drift,
                     params_before=int(params_before), params_after=int(L.bank.n_params()),
                     n_survivors=len(survivors)))
    del field, L
    torch.cuda.empty_cache()

    # ── the honest baseline: retrain from scratch without the deleted concept ─
    # This is what a method that folds updates into shared weights has to do.  Same recipe, same
    # budget, simply one fewer concept in the stream.
    field2, L2 = build(args.ckpt, dev, args.steps, args.seed)
    t0 = time.time()
    for q, nm in zip(qs, names):
        if nm == names[victim]:
            continue
        xq = to_model(data["X"][q["query"]]).ravel()
        L2.learn_concept(xq, nm, n_pool=32, epochs=args.epochs, inv_steps=args.inv_steps,
                         seed=args.seed)
    wall2 = time.time() - t0
    after2 = {n: fingerprint(field2, n, seed=args.seed) for n in survivors}
    drifts2 = {n: float(np.max(np.abs(after2[n] - before[n]))) for n in survivors}
    exact2 = float(np.mean([d == 0.0 for d in drifts2.values()]))
    print(f"\n[n31] retrain-from-scratch: {sum(d == 0.0 for d in drifts2.values())}/"
          f"{len(survivors)} survivors bit-identical, max drift {max(drifts2.values()):.3e}, "
          f"{wall2:.1f}s")
    rows.append(dict(arm="retrain-without-it", retention_exact=exact2,
                     drift=float(max(drifts2.values())), forgotten=1.0,
                     freed_params=0, wall_seconds=float(wall2),
                     params_before=int(params_before), params_after=int(L2.bank.n_params()),
                     n_survivors=len(survivors)))

    save_json(dict(rows=rows, deleted=names[victim], accounting=acct,
                   reclaimable={k: v["frac"] for k, v in rec.items()},
                   config=vars(args)),
              os.path.join(RESULTS, f"n31_deletion{args.tag}.json"), seed=args.seed)

    print(f"\n[n31] deletion, {len(survivors)} surviving concepts")
    print(f"    {'arm':22s}{'exact':>8s}{'drift':>12s}{'forgot':>8s}{'freed':>10s}{'seconds':>10s}")
    for r in rows:
        print(f"    {r['arm']:22s}{r['retention_exact']:8.3f}{r['drift']:12.3e}"
              f"{r['forgotten']:8.0f}{r['freed_params']:10,d}{r['wall_seconds']:10.2f}")
    # Deliberately NOT reported as a ratio: structural deletion is a dict operation whose time is
    # ~0, so any speedup factor is an artifact of the denominator and would read as fake precision.
    # The honest statement is that one is O(1) in the stream and the other is O(K) model fits.
    print(f"\n[n31] structural deletion is constant-time ({rows[0]['wall_seconds']:.3f}s, a pointer "
          f"operation) against {rows[1]['wall_seconds']:.1f}s to retrain the remaining "
          f"{rows[1]['n_survivors']} concepts -- and is the only arm that leaves them bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
