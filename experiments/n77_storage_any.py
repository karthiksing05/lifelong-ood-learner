"""N77 -- the storage loop on ANY factored backbone, gated. The generalisation of N71.

N71 closes the loop that carries the thesis -- one example in, the concept WRITTEN into the model,
novel class-matching samples back out by name, every pre-existing weight bit-identical -- and it is
hardcoded to ColorMNIST in twelve lines of preamble: `load_colormnist`, a positional `(digit, fg,
bg)` target, `fit_probes`/`composition_accuracy`, `ck["ema"]["cemb.weight"]`, and
`backbone_gains(..., factor=2)` where 2 literally means "background".

This is the same loop written against the contract in `lifelong_ood/data/factored.py`, so it runs on
ColorMNIST, Shapes3D or any future dataset that satisfies it. N71 is left untouched: its numbers are
pinned, and rewriting the experiment that produced them to prove a different point is how claims
quietly drift.

WHAT IS DELIBERATELY DIFFERENT FROM N71, and each is a lesson from 2026-08-19:

  * THE INSTRUMENT GATE RUNS FIRST. Targets whose real images cannot be read are not sampled, not
    scored, and not counted -- with the reason in the JSON. On the CelebA grid that removed 20 of 36
    measurements after the fact; here it happens before.
  * NO GAIN CALIBRATION. `backbone_gains` cannot run off ColorMNIST without rewriting three
    functions, and n59 already skips it and passes `gains=None`. Whether it is needed at all is
    reported (`oracle` vs `address`) rather than assumed.
  * THE ORACLE ARM SEPARATES READ COST FROM STORAGE COST. `oracle` composes from TRUE factors and
    never touches the read, so oracle -> address is what the READ costs and address -> stored is what
    STORAGE costs. Without it a bad read reads as a weak composer, which is exactly how N74's first
    run was misread.

    python experiments/n77_storage_any.py --dataset colormnist
    python experiments/n77_storage_any.py --dataset shapes3d --ckpt results/cache/ddpm_s3d_full.pt
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                     # noqa: E402

from lifelong_ood.data.factored import decode_slot, holdout_mask, load_factored  # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.instrument import (check_instrument, sample_conditioned,  # noqa: E402
                                     to_data_space)
from lifelong_ood.neural import NeuralLifelongLearner, to_model    # noqa: E402
from lifelong_ood.neural.idealwords import (compose, cv_r2, factorize,  # noqa: E402
                                            read_by_enumeration)


def _raw_sample(field, n, steps, seed, emb):
    """Sample WITHOUT converting to data space -- the pool `consolidate` distils against.

    `consolidate` trains the adapter on these rows as diffusion targets, so they must be in the
    space the field TRAINS in, not the space the probes read. Converting them cost a whole
    ColorMNIST result once (the stored concept came back at joint 0.000 with every sample
    identical). Signature dispatch is shared with `instrument.sample_conditioned`.
    """
    import inspect

    sig = inspect.signature(field.sample).parameters
    call = dict(n=n, steps=steps, seed=seed)
    for name in ("emb_override", "emb"):
        if name in sig:
            call[name] = emb
            break
    else:
        # our own UNets condition through `guide_eps`, and `sample_embedding` is that wrapper
        from lifelong_ood.neural import sample_embedding
        return sample_embedding(field, emb, n=n, steps=steps, seed=seed)
    return field.sample(**call)


def weight_hash(model, keys):
    """SHA-256 over the parameters named in `keys` only.

    `keys` is captured BEFORE the first absorption, because consolidation ADDS tensors. Hashing the
    whole state_dict afterwards compares different key sets and reports "changed" for a model whose
    original weights never moved -- the opposite of the claim.
    """
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_backbone(dataset, ckpt, dev):
    """-> (field, C, labels_of_C, seen_slots, remap). The one dataset-specific hop."""
    from lifelong_ood.neural import load_field
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    seen_slots = np.array(sorted(int(k) for k in remap))
    if "ema_cemb" in ck:                       # the VE-bridge / pretrained-table form (n57)
        from lifelong_ood.neural.ve_bridge import load_ve_field
        field, _ = load_ve_field(ckpt, device=dev)
        C = np.asarray(ck["ema_cemb"])[[remap[int(s)] for s in seen_slots]]
    else:                                      # our own UNet checkpoints
        field = load_field(ckpt, device=dev)
        C = ck["ema"]["cemb.weight"].numpy()[[remap[int(s)] for s in seen_slots]]
    return field, C, seen_slots, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="colormnist")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n-concepts", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--n-holdout", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reader", choices=("enumerate", "invert"), default="enumerate",
                   help="how the factors are READ off the one example. `enumerate` scores every "
                        "combination on common random draws -- exact where the vocabulary fits in "
                        "memory. `invert` is the gradient path `learn_factored` uses; it reads 0/5 "
                        "on the pretrained Shapes3D table (oracle 0.958 -> address 0.017), which "
                        "the oracle arm is what revealed.")
    p.add_argument("--n-draws", type=int, default=64)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        (args.n_concepts, args.n_samples, args.steps, args.n_per_slot,
         args.epochs, args.n_pool) = 2, 8, 40, 40, 4, 8
        # A SMOKE RUN MUST NOT BE ABLE TO OVERWRITE A CANONICAL RESULT. It just did: a --quick
        # regression check replaced the 5-concept ColorMNIST numbers and test_claims went 92/92 ->
        # 88/92 with a storage cost of -0.312 where the real run gives +0.008. The repository has an
        # earlier incident of exactly this shape (a 7 KB smoke run silently replacing a 3-seed
        # canonical table), which is why `aggregate.py` tracks provenance -- but provenance cannot
        # help when both runs write the same key. Forcing the tag is what makes it impossible.
        args.tag = args.tag or "_quick"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(CACHE, f"ddpm_{args.dataset}.pt")

    # ── data, through the generic contract, ON THE CHECKPOINT'S OWN SPLIT ───
    # The split is READ FROM THE CHECKPOINT, never defaulted. This line defaulted and it produced a
    # false negative that survived into RESULTS 7kk: the Shapes3D backbone is trained with
    # `wall_hue` PINNED, and loading without that gives a reference set carrying ten times the
    # wall-hue variation the model was ever shown. Recall is measured against images the model
    # cannot produce, so it collapsed to 0.110 while the identical samples score 0.958 in n78 --
    # vendi and precision matched to three decimals between the two runs, which is what proved the
    # samples were the same and the REFERENCE was the difference.
    #
    # It is the same fault class as 7ee's global ceiling: a denominator that does not match the
    # thing being measured.
    ck_split = {}
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            ck_split = torch.load(args.ckpt, map_location="cpu",
                                  weights_only=False).get("split", {}) or {}
        except Exception:                                             # noqa: BLE001
            ck_split = {}
    kw = dict(n_per_slot=int(ck_split.get("n_per_slot") or args.n_per_slot), seed=args.seed)
    for k in ("value_stride", "fix_nuisance", "max_rows", "n_holdout"):
        if ck_split.get(k) is not None and args.dataset in ("shapes3d",):
            kw[k] = ck_split[k]
    if kw.get("n_holdout") is not None:
        kw["n_holdout"] = 0            # the holdout comes from `remap`, not from the loader
    print(f"[n77] split from checkpoint: {kw}", flush=True)
    d = load_factored(args.dataset, **kw)
    sizes = [int(s) for s in d["sizes"]]
    X, F, slot = (np.asarray(d["X"], np.float32), np.asarray(d["factors"], int),
                  np.asarray(d["slot"], int))

    # THE BACKBONE FIRST, so `seen` comes from its own `remap` rather than from a holdout invented
    # here. Inventing one is how an experiment ends up scoring a checkpoint against a different
    # split than it was trained on -- n73 needed exactly this fix after a hardcoded n_holdout.
    field, C, seen_slots, ck = load_backbone(args.dataset, ckpt, dev)
    if len(seen_slots):
        d["seen"] = np.isin(slot, seen_slots)
    elif "seen" not in d or bool(np.all(np.asarray(d["seen"]))):
        seen, _ = holdout_mask(np.asarray(d["factors"]), sizes, args.n_holdout, seed=args.seed)
        d["seen"] = np.asarray(seen, bool)
    seen_mask = np.asarray(d["seen"], bool)
    probes = fit_factor_probes(d, seed=args.seed)
    lab = np.asarray(decode_slot(seen_slots, sizes), int)
    fac = factorize(C, lab)
    r2cv, n_cv = cv_r2(C, lab)
    print(f"[n77] {args.dataset}: {len(seen_slots)} seen classes, factors {d['names']} {sizes}; "
          f"basis R2 {fac['r2']:.3f} in-sample, {r2cv:.3f} leave-one-out", flush=True)

    held = [s for s in sorted(set(slot.tolist())) if s not in set(seen_slots.tolist())]
    held = [h for h in held if (slot == h).sum() >= 8]
    # SPREAD THE CONCEPTS ACROSS THE FIRST FACTOR'S VALUES rather than taking the first N by slot id.
    # Taking the first N gave five targets that ALL had floor_hue=0 -- so when the read failed on
    # floor_hue 5 times out of 5, the design could not distinguish "this factor is unreadable" from
    # "the value 0 is unreadable" from "the read always answers something else". A confounded
    # holdout cannot support either conclusion.
    by_first = {}
    for h in held:
        by_first.setdefault(int(F[slot == h][0][0]), []).append(h)
    spread, pool = [], sorted(by_first)
    while len(spread) < args.n_concepts and any(by_first.values()):
        for v in pool:
            if by_first[v] and len(spread) < args.n_concepts:
                spread.append(by_first[v].pop(0))
    held = spread or held[:args.n_concepts]
    if not held:
        print("[n77] no held-out class with enough images")
        return 0

    # ── the gate, before anything is sampled ────────────────────────────────
    gate = check_instrument(tag="n77", real=d, probes=probes, field=field,
                            targets={int(h): tuple(int(v) for v in F[slot == h][0]) for h in held},
                            n_arm=args.n_samples, table=C, labels=lab, seed=args.seed, ck=ck)
    print(gate.summary("n77"), flush=True)
    gate.require("n77")
    held = [h for h in held if gate.readable(int(h))]
    print(f"[n77] {len(held)} readable concepts to absorb: {held}", flush=True)

    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, lab)
    base_keys = sorted(field.model.state_dict().keys())
    base_hash = weight_hash(field.model, base_keys)

    rows, order = [], []
    rng = np.random.default_rng(args.seed)
    for ci, h in enumerate(held):
        idx = np.where(slot == int(h))[0]
        name = f"c{int(h)}"
        tgt = tuple(int(v) for v in F[idx[0]])
        qi = int(rng.choice(idx))
        R = X[[j for j in idx if j != qi][:60]]
        xq = to_model(X[qi]).ravel()

        if args.reader == "enumerate":
            read_t, _ = read_by_enumeration(field, to_model(X[qi]) if
                                            getattr(field, "data_space", "model") == "model"
                                            else X[qi],
                                            fac, n_draws=args.n_draws, seed=args.seed,
                                            max_candidates=4096)
        else:
            _, _, read = L.compose_factored(xq, seed=args.seed, hard=True)
            read_t = tuple(int(v) for v in read)
        c_read = compose(fac, read_t)

        S = {}
        # `oracle` never touches the read, so oracle -> address is the READ cost and
        # address -> stored is the STORAGE cost. Conflating them misread N74's first run.
        #
        # `sample_conditioned` rather than `sample_embedding`: the latter routes the conditional
        # through `guide_eps` and passes `eta`, neither of which the VE bridge's sampler accepts,
        # and that TypeError cost this experiment a full run on Shapes3D.
        S["oracle"] = sample_conditioned(field, n=args.n_samples, steps=args.steps,
                                         seed=args.seed + ci, emb=compose(fac, tgt))
        S["address"] = sample_conditioned(field, n=args.n_samples, steps=args.steps,
                                          seed=args.seed + ci, emb=c_read)

        # ABSORB. The pool stays in MODEL space -- `consolidate` trains the adapter against these
        # rows as diffusion targets; converting them cost a whole result on ColorMNIST once.
        # The pool is what `consolidate` distils against, so it must be in the space the field
        # TRAINS in -- model space for our UNets, data space for the VE bridge. `field.sample`
        # returns exactly that, before any `to_data` conversion.
        pool = np.asarray(_raw_sample(field, n=args.n_pool, steps=args.steps,
                                      seed=args.seed + 500 + ci, emb=c_read), np.float32)
        L.bank.add_concept(name, pool.mean(0))
        L.bank._emb_extra[name].data = torch.as_tensor(
            np.asarray(c_read, np.float32), device=dev).clone()
        L.consolidate(pool, name, mu=pool.mean(0), epochs=args.epochs, seed=args.seed,
                      freeze_embedding=True, already_registered=True)
        S["stored"] = to_data_space(field, L.sample_concept(
            name, n=args.n_samples, steps=args.steps, seed=args.seed + ci))

        for arm, Sx in S.items():
            Sx = np.asarray(Sx, np.float32)
            a = factor_accuracy(probes, Sx, tgt)
            pr = precision_recall(Sx, R)
            j = float(a["joint"])
            rows.append(dict(concept=name, slot=int(h), arm=arm, target=list(tgt),
                             joint=j if j == j else 0.0, recall=pr["recall"],
                             precision=pr["precision"], vendi=float(vendi_score(Sx)),
                             read=list(read_t), read_ok=bool(read_t == tgt), space=pr.get("space")))
        g = {r["arm"]: r for r in rows if r["concept"] == name}
        print(f"[n77] {name} {tgt} read {read_t} {'OK' if read_t == tgt else 'WRONG'} | "
              f"oracle {g['oracle']['joint']:.3f} -> address {g['address']['joint']:.3f} "
              f"-> STORED {g['stored']['joint']:.3f} (recall {g['stored']['recall']:.3f})",
              flush=True)
        order.append((name, int(h), tgt))

    print(f"\n[n77] re-sampling all {len(order)} stored concepts AFTER the stream", flush=True)
    after = []
    for ci, (name, h, tgt) in enumerate(order):
        idx = np.where(slot == int(h))[0]
        Sx = np.asarray(to_data_space(field, L.sample_concept(
            name, n=args.n_samples, steps=args.steps, seed=args.seed + ci)), np.float32)
        a = factor_accuracy(probes, Sx, tgt)
        j = float(a["joint"])
        pr = precision_recall(Sx, X[idx[:60]])
        after.append(dict(concept=name, joint=j if j == j else 0.0, recall=pr["recall"]))
        first = [r for r in rows if r["concept"] == name and r["arm"] == "stored"][0]
        print(f"[n77]   {name}: {first['joint']:.3f} -> {after[-1]['joint']:.3f}", flush=True)

    final_hash = weight_hash(field.model, base_keys)
    frozen = base_hash == final_hash
    n_new = len(field.model.state_dict()) - len(base_keys)

    def m(arm, k="joint"):
        v = [r[k] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    fgt = float(np.mean([[r for r in rows if r["concept"] == a_["concept"]
                          and r["arm"] == "stored"][0]["joint"] - a_["joint"] for a_ in after])) \
        if after else float("nan")
    n_read = sum(1 for r in rows if r["arm"] == "stored" and r["read_ok"])

    print(f"\n[n77] {'=' * 70}")
    print(f"{'arm':10s}{'joint':>9s}{'recall':>9s}{'vendi':>8s}")
    for arm in ("oracle", "address", "stored"):
        print(f"{arm:10s}{m(arm):>9.3f}{m(arm, 'recall'):>9.3f}{m(arm, 'vendi'):>8.2f}")
    print(f"\n[n77] read {n_read}/{len(order)} | forgetting {fgt:+.3f} | "
          f"base bit-identical {frozen} (+{n_new} new tensors)")
    print(f"[n77] read cost (oracle - address) {m('oracle') - m('address'):+.3f} | "
          f"storage cost (address - stored) {m('address') - m('stored'):+.3f}")

    print("[n77] VERDICT")
    if m("stored") >= 0.5 * m("oracle") and m("stored", "recall") > 0.3 and abs(fgt) < 0.1:
        print(f"[n77]   THE LOOP CLOSES ON {args.dataset.upper()}. One example is read, composed,")
        print("[n77]   WRITTEN, and sampled back by name -- stream intact, base bit-identical.")
    else:
        # NAME THE CONDITION THAT FAILED. The first version printed "does not close: stored 0.967
        # against an oracle of 0.958", which reads as an accuracy failure when accuracy passed
        # comfortably and COVERAGE is what fell short. A verdict that does not say which of its
        # clauses fired is the same defect this repository spent a day removing from its metrics.
        why = []
        if m("stored") < 0.5 * m("oracle"):
            why.append(f"accuracy: stored {m('stored'):.3f} < half the oracle's {m('oracle'):.3f}")
        if not m("stored", "recall") > 0.3:
            why.append(f"COVERAGE: stored recall {m('stored', 'recall'):.3f} <= 0.30 at vendi "
                       f"{m('stored', 'vendi'):.2f} -- the concept is stored and reads correctly, "
                       f"but its samples do not cover the class")
        if not abs(fgt) < 0.1:
            why.append(f"forgetting {fgt:+.3f}")
        print(f"[n77]   Does not close on {len(why)} of 3 conditions:")
        for w in why:
            print(f"[n77]     - {w}")
    save_json(dict(rows=rows, after=after, forgetting=fgt, frozen=bool(frozen),
                   n_new_tensors=int(n_new), instrument=gate.as_dict(), dataset=args.dataset,
                   basis_r2=float(fac["r2"]), basis_r2_cv=float(r2cv), read_ok=int(n_read),
                   base_hash=base_hash, final_hash=final_hash, config=vars(args)),
              os.path.join(RESULTS, f"n77_storage_any{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
