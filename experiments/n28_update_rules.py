"""N28 — which update rule adds a concept while DISTURBING THE EXISTING SCORE FIELD LEAST?

An insert-only bank makes preservation trivial in one sense and leaves it open in another.  When a
concept's path is INACTIVE the model is bit-identical, by construction, and that is what N5/N22
certify.  But every time concepts are composed -- score addition or path union -- an adapter is
evaluated on inputs it was never trained for, and there its behaviour is not certified by anything.
An update that changes the frozen field only where it needs to should compose better than one that
changes it everywhere, and that is a measurable claim rather than a stylistic preference.

So this measures two things for each update rule, not one:

  `factor accuracy`  does the concept actually get learned?
  `off-concept drift`  with the path ACTIVE, how far has the model moved on conditionings this
                       concept has nothing to do with?  ||eps_path(x,u,c_seen) - eps_frozen(x,u,c_seen)||
                       averaged over trained classes.  Zero means the delta is a pure conditional
                       refinement; large means it is a general perturbation that happens to help.
  `composed joint`     and then: does low drift actually buy better composition?

Rules compared, all on identical data, budget and architecture:

  `uniform`      sample the noise level uniformly.  What every experiment here has done.
  `high-t`       train only at t >= t_min.  The pretrained-backbone finding (RESULTS 12b) promoted
                 to a hypothesis: the semantics of a factor are settled at high noise, and low-t
                 capacity is spent on texture the model already renders correctly.
  `need-weighted` sample t in proportion to how badly the FROZEN model already does on this
                 concept's data.  The direct statement of "reflect the existing field": measure
                 where the field is already right, and do not spend capacity there.
  `residual`     regress the delta on (e - eps_frozen) rather than on e, so the adapter is fit to
                 the part of the target the existing field does not already explain.
  `rehearse`     uniform, plus a penalty for changing predictions on SEEN conditionings -- drift
                 suppressed by an explicit objective rather than by choosing where to train.

    python experiments/n28_update_rules.py --n-cases 2
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                          # noqa: E402
from n27_factor_conditionals import score_add                           # noqa: E402

from lifelong_ood.datasets import load_colormnist                       # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,       # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,     # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.ddpm import CH                                 # noqa: E402
from lifelong_ood.neural.idealwords import slot_factors                 # noqa: E402
from lifelong_ood.score import cosine_abar                              # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RULES = ("uniform", "high-t", "need-weighted", "residual", "rehearse")


def frozen_need(field, X01, img, grid=np.linspace(0.05, 0.95, 10), n=32, seed=0):
    """How badly does the FROZEN model already predict this concept's noise, per noise level?

    -> (grid, need) with need normalized to sum 1.  This is the empirical version of "where is the
    existing score field already right", and it is measured on the concept's own data with no
    conditioning, so it costs one forward pass per grid point and needs no labels.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(to_model(X01), np.float64)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    x0 = X[idx]
    need = []
    for u in grid:
        a = float(cosine_abar(np.array([u]))[0])
        e = rng.standard_normal(x0.shape)
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * e
        pred = field.eps(xt, float(u), None)
        need.append(float(np.mean((pred - e) ** 2)))
    need = np.asarray(need, np.float64)
    need = np.clip(need - need.min(), 1e-8, None)
    return grid, need / need.sum()


def train_rule(field, bank, name, X01, img, rule, epochs=40, batch=16, lr=1e-3, seed=0,
               t_min=0.5, seen_conds=None, seen_X=None, rehearse_w=5.0, need=None):
    """One concept, one update rule.  Returns wall-clock seconds."""
    params = bank.trainable(name)
    opt = torch.optim.Adam(params, lr=lr)
    dev = field.device
    X = torch.from_numpy(np.asarray(to_model(X01), np.float32)).to(dev).reshape(-1, CH, img, img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model = field.model
    model.train()
    Xr = Cr = None
    if rule == "rehearse" and seen_X is not None:
        Xr = torch.from_numpy(np.asarray(to_model(seen_X), np.float32)).to(dev)
        Xr = Xr.reshape(-1, CH, img, img)
        Cr = [c if torch.is_tensor(c) else torch.as_tensor(np.asarray(c, np.float32),
                                                           device=dev)
              for c in seen_conds]
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i0 in range(0, len(X), batch):
            x0 = X[perm[i0:i0 + batch].to(dev)]
            m = len(x0)
            if rule == "high-t":
                u = (torch.rand(m, generator=g) * (1.0 - t_min) + t_min).to(dev)
            elif rule == "need-weighted" and need is not None:
                grid, w = need
                pick = torch.from_numpy(
                    np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=g))
                                          ).choice(len(grid), size=m, p=w))
                u = torch.as_tensor(np.asarray(grid)[pick.numpy()], dtype=torch.float32, device=dev)
            else:
                u = torch.rand(m, generator=g).to(dev).clamp_(1e-3, 1.0)
            a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float()
            a = a.view(-1, 1, 1, 1)
            e = torch.randn(x0.shape, generator=g).to(dev)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            tgt = e
            if rule == "residual":
                # fit the part of the target the frozen field does NOT already explain
                with torch.no_grad(), bank.activate(None):
                    base = model(xt, u, emb_override=None)
                tgt = e - base.detach()
            with bank.activate(name):
                pred = model(xt, u, emb_override=bank.embedding(name))
                if rule == "residual":
                    with torch.no_grad(), bank.activate(None):
                        base2 = model(xt, u, emb_override=None)
                    pred = pred - base2.detach()
            loss = torch.nn.functional.mse_loss(pred, tgt)
            if rule == "rehearse" and Xr is not None:
                j = torch.randint(0, len(Xr), (min(batch, len(Xr)),), generator=g).to(dev)
                xr0 = Xr[j]
                cr = Cr[int(torch.randint(0, len(Cr), (1,), generator=g))]
                ur = torch.rand(len(xr0), generator=g).to(dev).clamp_(1e-3, 1.0)
                ar = torch.from_numpy(cosine_abar(ur.detach().cpu().numpy())).to(dev).float()
                ar = ar.view(-1, 1, 1, 1)
                er = torch.randn(xr0.shape, generator=g).to(dev)
                xrt = ar.sqrt() * xr0 + (1 - ar).sqrt() * er
                with torch.no_grad(), bank.activate(None):
                    ref = model(xrt, ur, emb_override=cr)
                with bank.activate(name):
                    cur = model(xrt, ur, emb_override=cr)
                loss = loss + rehearse_w * torch.nn.functional.mse_loss(cur, ref)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return time.time() - t0


def off_concept_drift(field, bank, name, seen_conds, X01, img, n=16, grid=(0.2, 0.5, 0.8), seed=0):
    """With `name`'s path ACTIVE, how far has the model moved on conditionings it is not for?

    This is the number the insert-only guarantee does NOT cover: preservation is certified for an
    inactive path, but composition evaluates adapters on inputs they never saw.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(to_model(X01), np.float64)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    x0 = X[idx]
    tot, cnt = 0.0, 0
    for c in seen_conds:
        for u in grid:
            a = float(cosine_abar(np.array([u]))[0])
            e = rng.standard_normal(x0.shape)
            xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * e
            # SAME conditioning `c` both times; the only difference is whether this concept's
            # adapter path is active.  Setting bank.state["path"] by hand does not work -- eps()
            # calls bank.activate(branch) itself, which overwrites it, and the two calls then come
            # out bit-identical and the metric reads a constant 0.0000.  Passing branch=name with an
            # explicit `emb` activates the path while keeping the foreign conditioning.
            base = field.eps(xt, float(u), None, emb=c)
            got = field.eps(xt, float(u), name, emb=c)
            tot += float(np.sqrt(np.mean((got - base) ** 2)))
            cnt += 1
    return tot / max(cnt, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--t-min", type=float, default=0.5)
    p.add_argument("--rehearse-w", type=float, default=5.0)
    p.add_argument("--rules", nargs="+", default=list(RULES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.n_per_slot = 1, 8, 20
        args.n_train, args.epochs, args.steps = 24, 3, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    seen = data["seen"]
    img = data.get("img", 32)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # a fixed reference set for drift: real seen images + a few trained conditionals
    sidx = np.where(seen)[0]
    seen_X = data["X"][rng.choice(sidx, size=min(32, len(sidx)), replace=False)]
    # torch tensors, not numpy: these go straight into `emb_override`, which is a module input
    seen_conds = [torch.as_tensor(np.asarray(C[i], np.float32), device=dev)
                  for i in rng.permutation(len(C))[:4]]

    print(f"[n28] {len(qs)} held-out compositions x {len(args.rules)} update rules", flush=True)
    rows = []
    for qi, q in enumerate(qs):
        d, f, b = int(q["d"]), int(q["f"]), int(q["b"])
        R = data["X"][q["rest"][:100]].astype(np.float32)
        specs = [(f"fg{f}", seen & (data["fg"] == f)), (f"bg{b}", seen & (data["bg"] == b)),
                 (f"digit{d}", seen & (data["digit"] == d))]
        for rule in args.rules:
            field = load_field(args.ckpt, device=dev)
            L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev,
                                      flat=True)
            L.attach_factorization(C, slot_factors(slots_seen), gains=[1.0, 1.0, 1.0])
            names, wall, drifts = [], 0.0, []
            for nm, mask in specs:
                idx = np.where(mask)[0]
                rng.shuffle(idx)
                Xc = data["X"][idx[:args.n_train]]
                need = (frozen_need(field, Xc, img, seed=args.seed)
                        if rule == "need-weighted" else None)
                L.bank.add_concept(nm, to_model(Xc).mean(0))
                wall += train_rule(field, L.bank, nm, Xc, img, rule, epochs=args.epochs,
                                   lr=1e-3, seed=args.seed, t_min=args.t_min,
                                   seen_conds=seen_conds, seen_X=seen_X,
                                   rehearse_w=args.rehearse_w, need=need)
                names.append(nm)
                drifts.append(off_concept_drift(field, L.bank, nm, seen_conds, seen_X, img,
                                                seed=args.seed))
            # each factor alone
            per = {}
            for nm in names:
                S = to_data(L.sample_concept(nm, n=args.n_samples, steps=args.steps,
                                             seed=args.seed)).astype(np.float32)
                a = composition_accuracy(probes, S, d, f, b)
                key = "fg" if nm.startswith("fg") else ("bg" if nm.startswith("bg") else "digit")
                per[key] = a[key]
            # and composed, by score addition (the operator RESULTS 12e recommends)
            S = to_data(score_add(field, names, args.n_samples, args.steps,
                                  args.seed, scale=1.0)).astype(np.float32)
            a = composition_accuracy(probes, S, d, f, b)
            pr = precision_recall(S, R)
            rows.append(dict(rule=rule, slot=int(q["slot"]), wall_seconds=wall,
                             drift=float(np.mean(drifts)), joint=a["joint"], digit=a["digit"],
                             fg=a["fg"], bg=a["bg"], recall=pr["recall"], f1=pr["f1"],
                             vendi=vendi_score(S),
                             **{f"alone_{k}": v for k, v in per.items()}))
            print(f"[n28] slot {q['slot']} {rule:14s} drift {np.mean(drifts):.4f}  "
                  f"alone(fg/bg/digit) {per.get('fg', float('nan')):.2f}/"
                  f"{per.get('bg', float('nan')):.2f}/{per.get('digit', float('nan')):.2f}  "
                  f"composed joint {a['joint']:.3f}  ({wall:.0f}s)", flush=True)
            del field, L
            torch.cuda.empty_cache()

    summary = {r: {k: float(np.nanmean([x[k] for x in rows if x["rule"] == r]))
                   for k in ("drift", "joint", "digit", "fg", "bg", "recall", "f1", "vendi",
                             "wall_seconds", "alone_fg", "alone_bg", "alone_digit")}
               for r in args.rules if any(x["rule"] == r for x in rows)}
    save_json(dict(rows=rows, summary=summary, config=vars(args)),
              os.path.join(RESULTS, f"n28_update_rules{args.tag}.json"), seed=args.seed)

    print(f"\n[n28] update rules, {len(qs)} compositions, {args.n_samples} samples")
    print(f"    {'rule':15s}{'drift':>9s}{'aloneFG':>9s}{'aloneBG':>9s}{'aloneD':>8s}"
          f"{'joint':>8s}{'recall':>8s}{'F1':>7s}{'wall':>7s}")
    for r in args.rules:
        if r not in summary:
            continue
        s = summary[r]
        print(f"    {r:15s}{s['drift']:9.4f}{s['alone_fg']:9.2f}{s['alone_bg']:9.2f}"
              f"{s['alone_digit']:8.2f}{s['joint']:8.3f}{s['recall']:8.3f}{s['f1']:7.3f}"
              f"{s['wall_seconds']:7.0f}")
    print("\n  drift = how far the model moves on conditionings this concept is NOT for, with its")
    print("  path active.  The insert-only guarantee does not cover this, and composition depends")
    print("  on it: an adapter evaluated outside its own conditioning is exactly what composing does.")
    if len(summary) > 1:
        best_d = min(summary, key=lambda r: summary[r]["drift"])
        best_j = max(summary, key=lambda r: summary[r]["joint"])
        print(f"\n[n28] lowest drift: {best_d} ({summary[best_d]['drift']:.4f})")
        print(f"[n28] best composed joint: {best_j} ({summary[best_j]['joint']:.3f})")
        print(f"[n28] {'SAME rule wins both -- less disturbance composes better' if best_d == best_j else 'DIFFERENT rules win -- low drift does not by itself buy composition'}")


if __name__ == "__main__":
    main()
