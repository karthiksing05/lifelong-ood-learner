"""N17 — orthogonal deltas, merged: the one route left to zero growth WITH preservation.

Where the previous experiments leave it:

  5r  Folding a LoRA delta into the base is exact for the concept just learned, but repeated folds
      destroy the model (ACC 0.837 -> 0.073), because a delta trained while one concept's path was
      active becomes UNCONDITIONAL once merged -- a job it was never trained for.  Rehearsal recovers
      part of it (0.380) and nothing closes the gap.
  5v  Orthogonal (Cayley) deltas match LoRA on the composition task at a third of the parameters.

The reason to put them together is specific rather than hopeful.  A rotation cannot change the norm
of what it acts on, and norm distortion is the destructive part of a merge; rotations are also closed
under composition, so merging a second one yields another rotation rather than an unconstrained
matrix.  `tests/test_ortho_merge.py` confirms the fold is exact (relative error 1.6e-06).

What is NOT solved by construction, and is the actual question: merging still makes a
concept-specific transform unconditional.  Norm preservation might make that survivable or might
not.  Same protocol as N5/N12 -- 12 tasks of real held-out images -- so the numbers sit beside both.
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

from lifelong_ood.data.datasets import load_colormnist                         # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes, save_json  # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, to_data, to_model)  # noqa: E402
from lifelong_ood.neural.alternatives import OrthoDelta, OrthoHook        # noqa: E402
from lifelong_ood.score import cosine_abar                                # noqa: E402

ARMS = ("lora-keep", "lora-merge+rehearse", "ortho-keep", "ortho-merge", "ortho-merge+rehearse")


def train_ortho(field, bank, od, emb, X, epochs, batch, lr, seed, rehearse=None, reg=None):
    """Fit one rotation to a concept, optionally holding it to a no-op off-concept."""
    opt = torch.optim.Adam(list(od.parameters()) + [emb], lr=lr)
    dev = field.device
    Xt = torch.from_numpy(np.asarray(X, np.float32)).to(dev).reshape(-1, 3, field.img, field.img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    reh = None
    if rehearse is not None:
        Xr = torch.from_numpy(np.asarray(rehearse["x0"], np.float32)).to(dev)
        Xr = Xr.reshape(-1, 3, field.img, field.img)
        Cr = [torch.as_tensor(np.asarray(c, np.float32), device=dev) for c in rehearse["conds"]]
        reh = dict(X=Xr, C=Cr, w=float(rehearse.get("w", 5.0)))
    field.model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i0 in range(0, len(Xt), batch):
            x0 = Xt[perm[i0:i0 + batch].to(dev)]
            u = torch.rand(len(x0), device=dev).clamp_(1e-3, 1.0)
            ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float()
            ab = ab.view(-1, 1, 1, 1)
            e = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
            reg["active"] = od
            loss = torch.nn.functional.mse_loss(field.model(xt, u, emb_override=emb), e)
            if reh is not None:
                j = torch.randint(0, len(reh["X"]), (min(batch, len(reh["X"])),),
                                  generator=g).to(dev)
                xr0 = reh["X"][j]
                cr = reh["C"][int(torch.randint(0, len(reh["C"]), (1,), generator=g))]
                ur = torch.rand(len(xr0), device=dev).clamp_(1e-3, 1.0)
                ar = torch.from_numpy(cosine_abar(ur.detach().cpu().numpy())).to(dev).float()
                ar = ar.view(-1, 1, 1, 1)
                er = torch.randn_like(xr0)
                xrt = ar.sqrt() * xr0 + (1 - ar).sqrt() * er
                reg["active"] = None                 # the reference is the base, rotation OFF
                with torch.no_grad():
                    ref = field.model(xrt, ur, emb_override=cr)
                reg["active"] = od                   # what the rotation does to a conditioning
                loss = loss + reh["w"] * torch.nn.functional.mse_loss(
                    field.model(xrt, ur, emb_override=cr), ref)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            reg["active"] = None
    field.model.eval()


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
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--rehearse-w", type=float, default=5.0)
    p.add_argument("--n-cond", type=int, default=12)
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
    REAL = to_model(data["X"][seen_idx[:256]])
    print(f"[n17] {len(tasks)} tasks of real held-out images, arms {args.arms}, device {dev}",
          flush=True)

    rows = []
    for arm in args.arms:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev, flat=True)
        is_ortho = arm.startswith("ortho")
        reg = {"active": None}
        if is_ortho:
            for i, w in enumerate(L.bank.wrapped):
                w.register_forward_hook(OrthoHook(reg, i))
        objs, acc_rows, learned = {}, {t["name"]: [] for t in tasks}, []
        t0 = time.time()
        for i0 in range(0, len(tasks), args.batch_size):
            grp = tasks[i0:i0 + args.batch_size]
            for t in grp:
                X = t["X"].reshape(len(t["X"]), -1)
                reh = None
                if "rehearse" in arm:
                    cs = [C[i] for i in range(0, len(C), 4)][:args.n_cond]
                    cs += [L.bank.embedding(nm).detach().cpu().numpy() for nm, _ in learned]
                    reh = dict(x0=REAL, conds=cs, w=args.rehearse_w)
                if is_ortho:
                    L.bank.add_concept(t["name"], X.mean(0))
                    od = OrthoDelta(L.bank.wrapped, block=args.block, device=dev)
                    train_ortho(field, L.bank, od, L.bank._emb_extra[t["name"]], X,
                                args.epochs, 16, 1e-3, args.seed, rehearse=reh, reg=reg)
                    if "merge" in arm:
                        od.merge_into(L.bank.wrapped)     # exact fold; nothing carried
                    else:
                        objs[t["name"]] = od
                    L.bank.merge_concept(t["name"])       # drop the unused LoRA node it allocated
                else:
                    L.consolidate(X, t["name"], mu=X.mean(0), epochs=args.epochs, rehearse=reh)
                    if "merge" in arm:
                        L.bank.merge_concept(t["name"])
                learned.append((t["name"], t["q"]))

            for si, (nm, q) in enumerate(learned):
                # `sample_concept` activates the concept's adapter path, which the LoRA arms need
                # and the merged arms simply find empty.  Sampling the base directly with only the
                # embedding would silently omit the adapter and make `lora-keep` look broken.
                if is_ortho and nm in objs:
                    reg["active"] = objs[nm]
                S = to_data(L.sample_concept(nm, n=args.n_samples,
                                             seed=args.seed + 500 + si)).astype(np.float32)
                reg["active"] = None
                acc_rows[nm].append(composition_accuracy(probes, S, q["d"], q["f"],
                                                         q["b"])["joint"])
            carried = L.bank.n_params() + sum(o.n_params() for o in objs.values())
            print(f"[n17:{arm}] {len(learned)}/{len(tasks)}: "
                  f"mean acc {np.mean([acc_rows[nm][-1] for nm, _ in learned]):.2f}, "
                  f"carried {carried / 1e6:.2f}M, {time.time() - t0:.0f}s", flush=True)
        acc = [acc_rows[t["name"]] for t in tasks if acc_rows[t["name"]]]
        m = cl_metrics(acc)
        carried = L.bank.n_params() + sum(o.n_params() for o in objs.values())
        rows.append(dict(arm=arm, wall_seconds=time.time() - t0, params=carried,
                         acc_matrix=acc, **m))
        print(f"[n17] -> {arm}: ACC {m['ACC']:.3f}, BWT {m['BWT']:+.3f}, "
              f"carried {carried / 1e6:.2f}M\n", flush=True)

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, "n17_ortho_merge.json"), seed=args.seed)
    print(f"\n[n17] orthogonal vs low-rank, kept vs merged, {args.n_tasks} real tasks")
    print(f"    {'arm':24s}{'ACC':>7s}{'BWT':>9s}{'FGT':>8s}{'carried':>12s}{'wall(s)':>9s}")
    for r in rows:
        print(f"    {r['arm']:24s}{r['ACC']:7.3f}{r['BWT']:+9.3f}{r['FGT']:8.3f}"
              f"{r['params'] / 1e6:11.2f}M{r['wall_seconds']:9.0f}")
    print(f"[n17] wrote {RESULTS}/n17_ortho_merge.json")


if __name__ == "__main__":
    main()
