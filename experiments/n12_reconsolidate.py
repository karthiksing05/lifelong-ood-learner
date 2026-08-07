"""N12 — absorb concepts into the EXISTING weights instead of accumulating adapters.

Same protocol as N5 (12 tasks of real held-out images, batches of 3, ACC/BWT/FGT) so the numbers sit
directly beside that table.  What changes is where the update ends up.

  `adapter-keep`   the current method: one frozen adapter per concept, carried forever.  Preservation
                   is bit-exact and the parameter count grows without bound.
  `merge`          learn the same delta, then FOLD it into the base weights and discard the adapter.
                   Parameter count is the base model's forever.  `tests/test_merge.py` shows the fold
                   is exact for the concept just learned (relative error 1.1e-06) -- but the NEXT
                   concept then writes over the same shared weights, and drift on an earlier concept
                   measured 1.086 after a single further merge.  This arm quantifies that cost on
                   real tasks rather than on a synthetic pair.
  `merge+project`  the same, with the update's input-side factor projected out of the activation
                   subspace that already-known conditionings produce, so the write lands where
                   nothing already known reads.  On the synthetic pair this cut drift 1.086 -> 0.237.
                   The subspace is SAMPLED from the model, so no exemplars are stored.

  `merge+rehearse` the same fold, but the delta is trained to be a NO-OP off-concept: every step
                   also sees a batch of known compositions under their own conditionings, with the
                   loss being to reproduce what the model predicted for them BEFORE this concept
                   started.  A LoRA delta is conditional by construction -- it only applies when the
                   concept's path is active -- so merging makes it unconditionally on, which it was
                   never trained to be.  This arm trains it for the job it is about to be given.
  `finetune-direct` no adapter at all: update the base weights directly, conditioned on the
                   concept's embedding.  Parameter count is flat by construction.  This is the
                   honest floor for "just reconsolidate the weights", and N5 already measured its
                   sibling at ACC 0.725 / BWT -0.082.

The question is whether "reconsolidate rather than accumulate" is a real option or a nice idea that
costs too much: does any zero-growth arm hold accuracy and forgetting near the frozen-adapter arm?
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                            # noqa: E402
from n4_batch import cl_metrics                                           # noqa: E402

from lifelong_ood.datasets import load_colormnist                         # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes, save_json  # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, activation_subspace,  # noqa: E402
                                 fingerprint, load_field, to_data, to_model)

ARMS = ("adapter-keep", "merge", "merge+project", "merge+rehearse", "finetune-direct")


def direct_finetune(L, field, X, name, C, epochs=40, batch=16, lr=1e-4, seed=0):
    """Update the BASE weights, with no adapter anywhere.  The floor for zero-growth absorption.

    The concept still needs a name to be sampled by, so it gets an embedding -- 256 floats, which is
    what actually identifies a concept (RESULTS 5n) -- but nothing else is added.  A lower learning
    rate than the adapter arms because the whole model is being written to, not a low-rank slice.
    """
    from lifelong_ood.score import cosine_abar
    L.bank.add_concept(name, np.ravel(X.mean(0)))
    emb = L.bank._emb_extra[name]
    model = field.model
    for p_ in model.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.Adam(list(model.parameters()) + [emb], lr=lr)
    Xt = torch.from_numpy(np.asarray(X, np.float32)).to(L.device)
    Xt = Xt.reshape(-1, 3, field.img, field.img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i0 in range(0, len(Xt), batch):
            x0 = Xt[perm[i0:i0 + batch].to(L.device)]
            u = torch.rand(len(x0), device=L.device).clamp_(1e-3, 1.0)
            a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(L.device)
            a = a.float().view(-1, 1, 1, 1)
            e = torch.randn_like(x0)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            loss = torch.nn.functional.mse_loss(model(xt, u, emb_override=emb), e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    # `add_concept` allocates LoRA nodes even though this arm never trains them; B is zero-init so
    # folding them in is an exact no-op, and it stops the arm reporting parameters it does not use.
    L.bank.merge_concept(name)
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-tasks", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--energy", type=float, default=0.999,
                   help="fraction of known-activation energy projected out of the update")
    p.add_argument("--n-cond", type=int, default=12, help="known conditionings sampled for the subspace")
    p.add_argument("--rehearse-w", type=float, default=5.0,
                   help="weight on the off-concept no-op loss.  Swept at seed 0: ACC / BWT "
                        "0.339/-0.244 at 1, 0.479/-0.085 at 5, 0.375/-0.068 at 15, "
                        "0.245/-0.097 at 50 -- the usual stability-plasticity curve, "
                        "peaking for accuracy at 5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_tasks, args.batch_size, args.n_train = 4, 2, 24
        args.n_samples, args.n_per_slot, args.epochs, args.steps = 8, 20, 4, 10
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    C = ck["ema"]["cemb.weight"].numpy()
    qs = ood_queries(data, n_slots=args.n_tasks, seed=args.seed)
    tasks = []
    for q in qs:
        idx = np.concatenate([[q["query"]], q["rest"]])[:args.n_train]
        tasks.append(dict(name=f"task_{q['slot']}", q=q, X=to_model(data["X"][idx])))
    seen_idx = np.where(data["seen"])[0]
    REAL = to_model(data["X"][seen_idx[:256]])        # real images of KNOWN compositions
    print(f"[n12] {len(tasks)} tasks x {args.n_train} real images, arms {args.arms}, "
          f"device {dev}", flush=True)

    rows = []
    for arm in args.arms:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev, flat=True)
        acc_rows = {t["name"]: [] for t in tasks}
        learned, marks = [], {}
        P = None
        t0 = time.time()
        for i0 in range(0, len(tasks), args.batch_size):
            grp = tasks[i0:i0 + args.batch_size]
            if arm == "merge+project":
                # re-estimate before every batch: what counts as "already known" grows as we absorb
                conds = [C[i] for i in range(0, min(len(C), 4 * args.n_cond), 4)][:args.n_cond]
                conds += [L.bank.embedding(nm).detach().cpu().numpy() for nm, _ in learned[-4:]]
                P = activation_subspace(L.bank, field, conds, n=4, energy=args.energy,
                                        seed=args.seed, x0=REAL)
            for t in grp:
                Xf = t["X"].reshape(len(t["X"]), -1)
                if arm == "finetune-direct":
                    direct_finetune(L, field, Xf, t["name"], C, epochs=args.epochs,
                                    seed=args.seed)
                else:
                    rehearse = None
                    if arm == "merge+rehearse":
                        # protect the base classes AND every concept already absorbed.  Leaving the
                        # absorbed ones out was a real hole: their behaviour lives in the merged
                        # weights, so nothing was defending it while the next delta was written.
                        cs = [C[i] for i in range(0, min(len(C), 4 * args.n_cond), 4)][:args.n_cond]
                        cs += [L.bank.embedding(nm).detach().cpu().numpy() for nm, _ in learned]
                        rehearse = dict(x0=REAL, conds=cs, w=args.rehearse_w)
                    L.consolidate(Xf, t["name"], mu=Xf.mean(0), epochs=args.epochs,
                                  project=P, rehearse=rehearse)
                if arm not in ("adapter-keep", "finetune-direct"):
                    L.bank.merge_concept(t["name"])      # fold in, drop the adapter
                learned.append((t["name"], t["q"]))
                marks[t["name"]] = fingerprint(field, t["name"], seed=args.seed)
            for si, (nm, q) in enumerate(learned):
                S = to_data(L.sample_concept(nm, n=args.n_samples,
                                             seed=args.seed + 500 + si)).astype(np.float32)
                acc_rows[nm].append(composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"])
            exact = float(np.mean([float(np.array_equal(fingerprint(field, nm, seed=args.seed),
                                                        marks[nm])) for nm, _ in learned]))
            print(f"[n12:{arm}] {len(learned)}/{len(tasks)}: "
                  f"mean acc {np.mean([acc_rows[nm][-1] for nm, _ in learned]):.2f}, "
                  f"exact {exact:.2f}, adapter {L.bank.n_params() / 1e6:.2f}M, "
                  f"{time.time() - t0:.0f}s", flush=True)
        acc = [acc_rows[t["name"]] for t in tasks if acc_rows[t["name"]]]
        m = cl_metrics(acc)
        rows.append(dict(arm=arm, wall_seconds=time.time() - t0, params=L.bank.n_params(),
                         retention_exact=exact, acc_matrix=acc, **m))
        print(f"[n12] -> {arm}: ACC {m['ACC']:.3f}, BWT {m['BWT']:+.3f}, FGT {m['FGT']:.3f}, "
              f"adapter params {L.bank.n_params() / 1e6:.2f}M\n", flush=True)

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, "n12_reconsolidate.json"), seed=args.seed)
    print(f"\n[n12] reconsolidation vs accumulation, {args.n_tasks} tasks of real held-out images")
    print(f"    {'arm':16s}{'ACC':>7s}{'BWT':>9s}{'FGT':>8s}{'exact':>8s}"
          f"{'adapter params':>16s}{'wall(s)':>9s}")
    for r in rows:
        print(f"    {r['arm']:16s}{r['ACC']:7.3f}{r['BWT']:+9.3f}{r['FGT']:8.3f}"
              f"{r['retention_exact']:8.2f}{r['params'] / 1e6:15.2f}M{r['wall_seconds']:9.0f}")
    print("\n  adapter params is what is CARRIED after learning: the merge arms fold the delta into")
    print("  the base weights, so they end at 0.00M no matter how many concepts were absorbed.")
    print(f"[n12] wrote {RESULTS}/n12_reconsolidate.json")


if __name__ == "__main__":
    main()
