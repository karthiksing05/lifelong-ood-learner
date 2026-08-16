"""N71 -- the whole claim in one experiment: one example in, the concept INTO the manifold, novel
class-matching samples out of it afterwards, nothing already learned disturbed.

Every other experiment here measures a slice.  This measures the loop the VLA translation actually
needs, in the order a policy would need it, and refuses to let any stage be assumed:

    ONE held-out example
      -> READ the composition from it        (no ground-truth factors -- the case studies use those)
      -> COMPOSE an address in the basis      (hard argmax, N67: joint 0.493 -> 0.642)
      -> generate a pool from the model's own conditional
      -> CONSOLIDATE the pool into the model  (a written adapter + embedding row, not a transient
                                               `emb_override` -- see below)
      -> SAMPLE THE STORED CONCEPT BY NAME    (the step that proves it is in the manifold)
      -> and check nothing else moved         (SHA-256 over every pre-existing parameter)

WHY THE STORED-SAMPLE STEP IS THE POINT.  `n59` (Shapes3D) and `n60` (DiT) report a "lifelong"
result by recomposing `compose(fac, TRUE factors)` each round and sampling through a transient
`emb_override`.  Nothing is written; the concept exists only for the duration of one call, and the
factors were handed over rather than read.  That measures whether the BASIS stays usable as more
coordinates are added -- a real question -- but it is not storage, and a policy that has to
re-derive a skill from its ground-truth description at every use has not learned it.

So the two numbers that matter here are:

    stored     joint accuracy and recall of samples drawn from the CONSOLIDATED concept
    address    the same, drawn from the composed address directly, before consolidation

`stored` is the claim.  `address` is its ceiling: consolidation distils a pool into weights, so the
gap between them is what the storage step COSTS.  Reporting `address` alone -- which is what the
case studies do -- reports a concept that was never stored.

And the third, which makes it lifelong rather than one-shot: the same measurements after a STREAM of
concepts has been absorbed, with preservation hashed at every step.

    python experiments/n71_absorb_and_generate.py --n-concepts 5
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries          # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                 # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,      # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,    # noqa: E402
                                 sample_embedding, to_data, to_model)
from lifelong_ood.neural.idealwords import compose, slot_factors       # noqa: E402


def weight_hash(model, keys=None):
    """SHA-256 over the parameters named in `keys`, so 'unchanged' is bit-identical, not close.

    `keys` is captured BEFORE any concept is absorbed and reused afterwards, because consolidation
    ADDS parameters -- an adapter path and an embedding row per concept.  Hashing the whole
    state_dict then compares different key sets and reports "changed" for a model whose original
    weights never moved, which is the opposite of the claim being tested.  The claim is that
    absorbing a concept leaves everything ALREADY LEARNED untouched; new parameters appearing is
    the mechanism, not a violation.
    """
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(keys if keys is not None else sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-concepts", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        (args.n_concepts, args.n_samples, args.steps, args.n_per_slot,
         args.epochs, args.inv_steps, args.n_pool) = 2, 8, 40, 20, 4, 40, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    ckp = os.path.join(CACHE, "ddpm_colormnist.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots]]
    field = load_field(ckp, device=dev)
    gains = backbone_gains(field, C, slots, probes, data, seed=args.seed, steps=args.steps)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, slot_factors(slots), gains=gains)

    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)
    base_keys = sorted(field.model.state_dict().keys())
    base_hash = weight_hash(field.model, base_keys)
    print(f"[n71] base weight hash {base_hash[:16]}...  {len(qs)} concepts to absorb", flush=True)

    rows, stream = [], []
    for ci, q in enumerate(qs):
        name = f"c{int(q['slot'])}"
        xq = to_model(data["X"][q["query"]]).ravel()
        R = np.asarray(data["X"], np.float32)[q["rest"][:100]]
        tgt = (q["d"], q["f"], q["b"])

        # READ from the one example, then compose HARD (N67's better composer)
        c_soft, ws, read = L.compose_factored(xq, inv_steps=args.inv_steps, seed=args.seed)
        c_hard = compose(L.fac, tuple(int(v) for v in read), gains=gains)
        read_ok = tuple(int(v) for v in read) == tgt

        # the ADDRESS arm: sample the composed point directly, nothing stored yet
        S_addr = to_data(sample_embedding(field, c_hard, n=args.n_samples, steps=args.steps,
                                          seed=args.seed + ci))

        # ABSORB: distil a pool from that address into a written adapter + embedding row
        pool = to_data(sample_embedding(field, c_hard, n=args.n_pool, steps=args.steps,
                                        seed=args.seed + 500 + ci))
        # The sequence `learn_concept_factored` uses, with the composed address PINNED into the
        # concept's embedding before distillation -- otherwise consolidation is free to drift the
        # embedding off the address that was just read, and "stored" would no longer be storing the
        # thing that was composed.
        pool = np.asarray(pool, np.float32)
        L.bank.add_concept(name, pool.mean(0))
        L.bank._emb_extra[name].data = torch.as_tensor(
            np.asarray(c_hard, np.float32), device=dev).clone()
        L.consolidate(pool, name, mu=pool.mean(0), epochs=args.epochs, seed=args.seed,
                      freeze_embedding=True, already_registered=True)

        # THE STEP THAT PROVES STORAGE: sample the concept BY NAME
        S_store = to_data(L.sample_concept(name, n=args.n_samples, steps=args.steps,
                                           seed=args.seed + ci))

        for tag, S in (("address", S_addr), ("stored", S_store)):
            S = np.asarray(S, np.float32)
            a = composition_accuracy(probes, S, *tgt)
            pr = precision_recall(S, R)
            rows.append(dict(concept=name, arm=tag, joint=a["joint"], digit=a["digit"],
                             fg=a["fg"], bg=a["bg"], recall=pr["recall"],
                             precision=pr["precision"], vendi=float(vendi_score(S)),
                             read_ok=bool(read_ok)))
        j_a = [r for r in rows if r["concept"] == name and r["arm"] == "address"][0]
        j_s = [r for r in rows if r["concept"] == name and r["arm"] == "stored"][0]
        print(f"[n71] {name} read {tuple(int(v) for v in read)} vs {tgt} "
              f"{'OK' if read_ok else 'WRONG'} | address joint {j_a['joint']:.3f} "
              f"recall {j_a['recall']:.3f} -> STORED joint {j_s['joint']:.3f} "
              f"recall {j_s['recall']:.3f} vendi {j_s['vendi']:.2f}", flush=True)
        stream.append(name)

    # ── does the whole stream survive, and did the base move? ────────────────
    print(f"\n[n71] re-sampling every stored concept AFTER all {len(stream)} were absorbed",
          flush=True)
    after = []
    for ci, q in enumerate(qs):
        name = f"c{int(q['slot'])}"
        tgt = (q["d"], q["f"], q["b"])
        S = np.asarray(to_data(L.sample_concept(name, n=args.n_samples, steps=args.steps,
                                                seed=args.seed + ci)), np.float32)
        a = composition_accuracy(probes, S, *tgt)
        R = np.asarray(data["X"], np.float32)[q["rest"][:100]]
        pr = precision_recall(S, R)
        after.append(dict(concept=name, joint=a["joint"], recall=pr["recall"]))
        first = [r for r in rows if r["concept"] == name and r["arm"] == "stored"][0]
        print(f"[n71]   {name}: joint {first['joint']:.3f} -> {a['joint']:.3f}  "
              f"recall {first['recall']:.3f} -> {pr['recall']:.3f}", flush=True)

    final_hash = weight_hash(field.model, base_keys)
    n_new = len(field.model.state_dict()) - len(base_keys)
    frozen_ok = base_hash == final_hash

    def mean(arm, k):
        return float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    print(f"\n[n71] {'=' * 72}")
    print(f"{'arm':10s}{'joint':>9s}{'recall':>9s}{'precision':>11s}{'vendi':>8s}")
    for arm in ("address", "stored"):
        print(f"{arm:10s}{mean(arm, 'joint'):>9.3f}{mean(arm, 'recall'):>9.3f}"
              f"{mean(arm, 'precision'):>11.3f}{mean(arm, 'vendi'):>8.2f}")
    fgt = float(np.mean([[r for r in rows if r["concept"] == a_["concept"]
                          and r["arm"] == "stored"][0]["joint"] - a_["joint"] for a_ in after]))
    n_read = sum(1 for r in rows if r["arm"] == "stored" and r["read_ok"])
    print(f"\n[n71] read accuracy {n_read}/{len(qs)} | forgetting over the stream {fgt:+.3f}")
    print(f"[n71] pre-existing weights bit-identical after absorbing {len(stream)}: "
          f"{frozen_ok}  (+{n_new} new parameter tensors added by consolidation)")
    print("[n71] VERDICT")
    st, ad = mean("stored", "joint"), mean("address", "joint")
    if st >= 0.5 * ad and mean("stored", "recall") > 0.3 and abs(fgt) < 0.1:
        print("[n71]   THE LOOP CLOSES END TO END. One example is read, composed, absorbed as a")
        print("[n71]   WRITTEN concept, and sampled back out as novel class-matching images that")
        print("[n71]   cover the class -- with the stream intact and the base bit-identical.")
        print("[n71]   This is the form the VLA claim has to take: a skill stored, not re-derived.")
    elif st < 0.5 * ad:
        print(f"[n71]   STORAGE LOSES THE CONCEPT: stored {st:.3f} against an address ceiling of")
        print(f"[n71]   {ad:.3f}. Consolidation is the weak stage, not the read or the compose --")
        print("[n71]   report the address number as a COMPOSITION result and not as lifelong.")
    else:
        print("[n71]   The stream degrades or coverage collapses; see the per-concept lines.")
    save_json(dict(rows=rows, after=after, forgetting=fgt, frozen=bool(frozen_ok),
                   n_new_tensors=int(n_new),
                   base_hash=base_hash, final_hash=final_hash, config=vars(args)),
              os.path.join(RESULTS, "n71_absorb_and_generate.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
