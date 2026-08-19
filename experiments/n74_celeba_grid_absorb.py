"""N74 -- the storage loop (N71) on the CelebA grid, i.e. on a DENSITY holdout outside ColorMNIST.

N71 closes the whole loop -- one example in, the concept written into the model, novel
class-matching samples out by name, nothing already learned disturbed -- but it has only ever run on
ColorMNIST.  Every other backbone here was pretrained (`google/ddpm-celebahq-256`,
`eurecom-ds/scoresdeve-ema-shapes3d-64`), so a "held-out" combination was held out from the
CONDITIONING while its images sat in the density the whole time; storing a concept the model can
already draw proves nothing about storage.

`ddpm_celeba_grid.pt` is the first non-ColorMNIST backbone here without that defect: trained from
scratch on the seen cells of a hair(3) x Male(2) x Smiling(2) grid, with the held-out cells' images
removed from training entirely.  So this is the first place the storage claim can be asked a second
time, on a different dataset, at 64x64 in colour, with faces rather than digits.

Same three numbers as N71, and the same reason for each:

    address    sample the composed point directly -- the CEILING, nothing stored
    stored     sample the CONSOLIDATED concept by name -- the claim
    after      the same, once every concept in the stream has been absorbed -- forgetting

plus SHA-256 over every pre-existing parameter, so "the base did not move" is bit-identical rather
than close.

READ AGAINST THE PROBE CEILING, NOT 1.0.  Factor probes score ~0.414 joint on REAL images here
(hair is the weak one at 0.568), and chance is 1/n_cells.  A stored joint of 0.2 is half of what the
instrument can register, not a fifth of the truth.  Three verdicts in this repository have already
fired on a metric that could not discriminate; the verdict below gates on the ceiling first.

    python experiments/n74_celeba_grid_absorb.py
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                       # noqa: E402

from lifelong_ood.data.celeba import load_celeba_grid                   # noqa: E402
from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,     # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, NeuralScoreField,  # noqa: E402
                                 UNet, sample_embedding, to_data, to_model)
from lifelong_ood.neural.idealwords import (compose, cv_r2, factorize,   # noqa: E402
                                             read_by_enumeration)


def weight_hash(model, keys):
    """SHA-256 over the parameters named in `keys` only.

    `keys` is captured BEFORE the first absorption, because consolidation ADDS tensors (an adapter
    path and an embedding row per concept).  Hashing the whole state_dict afterwards compares
    different key sets and reports "changed" for a model whose original weights never moved -- the
    opposite of the claim.  New parameters appearing is the mechanism, not a violation.
    """
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_grid.pt"))
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="",
                   help="suffix for the result filename. REQUIRED when pointing --ckpt at a "
                        "different backbone: without it every backbone writes the same "
                        "`n74_celeba_grid_absorb.json` and the last run silently replaces the others, which pools "
                        "as though it were one experiment.")
    p.add_argument("--reader", choices=("enumerate", "invert"), default="enumerate",
                   help="how the factors are read off the single query. `enumerate` scores all "
                        "combinations on common random draws -- exact on a 12-cell grid; `invert` "
                        "is the gradient path `learn_factored` uses, kept for comparison and for "
                        "vocabularies too large to enumerate.")
    p.add_argument("--n-draws", type=int, default=64)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        (args.n_samples, args.steps, args.epochs, args.inv_steps,
         args.n_pool, args.n_draws) = 8, 40, 4, 40, 8, 16
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck.get("split", {})

    def grid(n_holdout):
        return load_celeba_grid(img=int(ck.get("img", 64)), seed=int(sp.get("seed", 0)),
                                n_per_cell=int(sp.get("n_per_slot", 3000)),
                                n_holdout=int(n_holdout), max_rows=sp.get("max_rows"))

    # The split is DERIVED from the checkpoint and then ASSERTED, never assumed -- see the same
    # block in n73.  A wrong holdout still runs every arm and reports a plausible number.
    d = grid(sp.get("n_holdout") or 3)
    want = sorted(int(k) for k in ck["remap"])
    if sorted(set(np.asarray(d["slot"])[np.asarray(d["seen"])].tolist())) != want:
        d = grid(len(d["live"]) - len(want))
    got = sorted(set(np.asarray(d["slot"])[np.asarray(d["seen"])].tolist()))
    if got != want:
        raise SystemExit(f"[n74] split mismatch: checkpoint trained on {want}, loader gives {got}")

    X, F, slot = np.asarray(d["X"], np.float32), np.asarray(d["factors"]), np.asarray(d["slot"])
    probes = fit_factor_probes(d, seed=args.seed)
    ceil = float(np.prod([probes[f"{n}_val"] for n in d["names"] if f"{n}_val" in probes]))
    chance = 1.0 / len(want)
    print(f"[n74] {len(want)} seen cells, held out {d['held']} | probe ceiling {ceil:.3f}, "
          f"chance {chance:.3f}", flush=True)

    seen = np.array(want)
    C = ck["ema"]["cemb.weight"].numpy()[[ck["remap"][int(s)] for s in seen]]
    lab = np.stack([F[slot == int(s)][0] for s in seen])
    fac = factorize(C, lab)
    # BOTH R^2s.  In-sample is near-arithmetic once the table is small (7 cells against 5 free
    # vectors), so the leave-one-cell-out number is the one that says anything about structure.
    r2cv, n_cv = cv_r2(C, lab)
    print(f"[n74] ideal-words basis R^2 {fac['r2']:.3f} in-sample | {r2cv:.3f} "
          f"leave-one-cell-out over {n_cv} of {len(want)} cells", flush=True)

    field = NeuralScoreField(UNet(n_classes=ck["n_classes"], **ck["arch"]), device=dev,
                             layout=ck.get("layout"))
    field.model.load_state_dict(ck["ema"])
    field.model.eval()
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.attach_factorization(C, lab)

    base_keys = sorted(field.model.state_dict().keys())
    base_hash = weight_hash(field.model, base_keys)
    print(f"[n74] base weight hash {base_hash[:16]}...", flush=True)

    rng = np.random.default_rng(args.seed)
    rows, order = [], []
    for ci, h in enumerate(d["held"]):
        idx = np.where(slot == int(h))[0]
        if not len(idx):
            continue
        name = f"cell{int(h)}"
        tgt = tuple(int(v) for v in F[idx[0]])
        # ONE example, drawn from images the backbone has never seen, and the REST of that cell is
        # the reference population `precision_recall` is read against -- never the query itself.
        qi = int(rng.choice(idx))
        R = X[[j for j in idx if j != qi][:60]]
        xq = to_model(X[qi]).ravel()

        if args.reader == "enumerate":
            read_t, _ = read_by_enumeration(field, to_model(X[qi]), fac, n_draws=args.n_draws,
                                            seed=args.seed)
        else:
            _, _, read = L.compose_factored(xq, inv_steps=args.inv_steps, seed=args.seed, hard=True)
            read_t = tuple(int(v) for v in read)
        c_hard = compose(fac, read_t)
        read_ok = read_t == tgt

        # THREE arms, so a bad number is ATTRIBUTABLE. `oracle` composes from the TRUE factors and
        # never touches the read, so oracle -> address is what the READ costs and address -> stored
        # is what STORAGE costs. Without it the two are confounded: the first full run scored
        # address 0.167 where n73's true-factor composition scores 0.375 on the same cell, which
        # reads as a weak composer and was a wrong read.
        S_oracle = to_data(sample_embedding(field, compose(fac, tgt), n=args.n_samples,
                                            steps=args.steps, seed=args.seed + ci))
        S_addr = to_data(sample_embedding(field, c_hard, n=args.n_samples, steps=args.steps,
                                          seed=args.seed + ci))
        # THE POOL STAYS IN MODEL SPACE -- `consolidate` trains the adapter against these rows as
        # diffusion targets and the field works in [-1,1].  Wrapping this in `to_data` once cost a
        # whole result on ColorMNIST: the stored concept came back at joint 0.000 with every sample
        # identical, which reads as "consolidation loses the concept" and was a range mismatch.
        pool = np.asarray(sample_embedding(field, c_hard, n=args.n_pool, steps=args.steps,
                                           seed=args.seed + 500 + ci), np.float32)
        L.bank.add_concept(name, pool.mean(0))
        # PIN the composed address into the concept's embedding before distillation, or
        # consolidation is free to drift off the point that was just read and `stored` would no
        # longer be storing the thing that was composed.
        L.bank._emb_extra[name].data = torch.as_tensor(
            np.asarray(c_hard, np.float32), device=dev).clone()
        L.consolidate(pool, name, mu=pool.mean(0), epochs=args.epochs, seed=args.seed,
                      freeze_embedding=True, already_registered=True)

        S_store = to_data(L.sample_concept(name, n=args.n_samples, steps=args.steps,
                                           seed=args.seed + ci))
        for tag, S in (("oracle", S_oracle), ("address", S_addr), ("stored", S_store)):
            S = np.asarray(S, np.float32)
            a = factor_accuracy(probes, S, tgt)
            pr = precision_recall(S, R)
            j = float(a["joint"])
            rows.append(dict(concept=name, cell=int(h), arm=tag, target=list(tgt),
                             joint=j if j == j else 0.0, recall=pr["recall"],
                             precision=pr["precision"], vendi=float(vendi_score(S)),
                             read=list(read_t), read_ok=bool(read_ok)))
        ra = [r for r in rows if r["concept"] == name and r["arm"] == "address"][0]
        rs = [r for r in rows if r["concept"] == name and r["arm"] == "stored"][0]
        print(f"[n74] {name} {tgt} read {read_t} {'OK' if read_ok else 'WRONG'} | "
              f"address {ra['joint']:.3f} -> STORED {rs['joint']:.3f} "
              f"(recall {rs['recall']:.3f}, vendi {rs['vendi']:.2f})", flush=True)
        order.append((name, int(h), tgt))

    print(f"\n[n74] re-sampling all {len(order)} stored concepts AFTER the whole stream", flush=True)
    after = []
    for ci, (name, h, tgt) in enumerate(order):
        idx = np.where(slot == int(h))[0]
        S = np.asarray(to_data(L.sample_concept(name, n=args.n_samples, steps=args.steps,
                                                seed=args.seed + ci)), np.float32)
        a = factor_accuracy(probes, S, tgt)
        j = float(a["joint"])
        pr = precision_recall(S, X[idx[:60]])
        after.append(dict(concept=name, joint=j if j == j else 0.0, recall=pr["recall"]))
        first = [r for r in rows if r["concept"] == name and r["arm"] == "stored"][0]
        print(f"[n74]   {name}: joint {first['joint']:.3f} -> {after[-1]['joint']:.3f} | "
              f"recall {first['recall']:.3f} -> {pr['recall']:.3f}", flush=True)

    final_hash = weight_hash(field.model, base_keys)
    frozen_ok = base_hash == final_hash
    n_new = len(field.model.state_dict()) - len(base_keys)

    def m(arm, k):
        v = [r[k] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    fgt = float(np.mean([[r for r in rows if r["concept"] == a_["concept"]
                          and r["arm"] == "stored"][0]["joint"] - a_["joint"] for a_ in after])) \
        if after else float("nan")
    n_read = sum(1 for r in rows if r["arm"] == "stored" and r["read_ok"])

    print(f"\n[n74] {'=' * 70}")
    print(f"{'arm':10s}{'joint':>9s}{'/ceiling':>10s}{'recall':>9s}{'vendi':>8s}")
    for arm in ("oracle", "address", "stored"):
        print(f"{arm:10s}{m(arm, 'joint'):>9.3f}{m(arm, 'joint') / max(ceil, 1e-9):>10.2f}"
              f"{m(arm, 'recall'):>9.3f}{m(arm, 'vendi'):>8.2f}")
    print(f"\n[n74] read accuracy {n_read}/{len(order)} | forgetting over the stream {fgt:+.3f}")
    print(f"[n74] pre-existing weights bit-identical: {frozen_ok} (+{n_new} new tensors)")

    st, ad = m("stored", "joint"), m("address", "joint")
    print("[n74] VERDICT")
    # THE CEILING ARM GATES EVERYTHING.  `stored >= 0.5 * address` is trivially satisfiable when
    # `address` is itself at chance, and a --quick run demonstrated exactly that: address 0.125
    # against a chance of 0.111 printed "the loop closes".  Storage can only be judged relative to a
    # composition that actually happened, so require the ceiling arm to clear chance by a margin
    # rather than by a hair -- half way to the probe ceiling, which is the same bar N73 uses.
    floor = chance + 0.5 * max(ceil - chance, 0.0)
    if ad < floor:
        print(f"[n74]   THE CEILING ARM IS TOO WEAK TO JUDGE STORAGE: address {ad:.3f} against a")
        print(f"[n74]   chance of {chance:.3f} and a probe ceiling of {ceil:.3f} (bar {floor:.3f}).")
        print("[n74]   Nothing downstream of composition can be read from this run -- a low")
        print("[n74]   `stored` here would look like a storage failure when composition is what")
        print("[n74]   never happened. Check --steps and --inv-steps before rerunning.")
    elif st >= 0.5 * ad and m("stored", "recall") > 0.3 and abs(fgt) < 0.1:
        print("[n74]   THE LOOP CLOSES ON A SECOND DATASET. One held-out CelebA example is read,")
        print("[n74]   composed, WRITTEN into the model, and sampled back as novel class-matching")
        print("[n74]   faces -- with the stream intact and every pre-existing weight bit-identical.")
        print("[n74]   ColorMNIST is no longer the only place the storage claim holds.")
    elif st < 0.5 * ad:
        print(f"[n74]   STORAGE LOSES THE CONCEPT: stored {st:.3f} against an address ceiling of")
        print(f"[n74]   {ad:.3f}. Consolidation is the weak stage, not the read or the compose.")
        print("[n74]   Report the address number as COMPOSITION and not as lifelong.")
    else:
        print("[n74]   The stream degrades or coverage collapses; see the per-concept lines.")

    save_json(dict(rows=rows, after=after, forgetting=fgt, frozen=bool(frozen_ok),
                   n_new_tensors=int(n_new), ceiling=ceil, chance=chance,
                   basis_r2=float(fac["r2"]), basis_r2_cv=float(r2cv), n_cv=int(n_cv),
                   reader=args.reader, held=d["held"], read_ok=int(n_read),
                   base_hash=base_hash, final_hash=final_hash, config=vars(args)),
              os.path.join(RESULTS, f"n74_celeba_grid_absorb{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
