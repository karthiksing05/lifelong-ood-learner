"""N16 — four ways to add an OOD composition to the manifold, on the goal's own loop.

The loop this repo exists to do: compose prototypes into an OUT-OF-DISTRIBUTION prototype, sample
instances from it, add those instances to the manifold, and have nothing already there degrade.
Every arm below does exactly that; what differs is how the "add" step is parameterized.

  `lora`            the current method.  A frozen low-rank delta per concept.  Preservation is
                    bit-exact and the parameter count grows forever.
  `hypernet`        the delta is GENERATED from the concept embedding by a shared network, so
                    nothing is stored per concept.  Follows from 5n: if the embedding carries the
                    semantics, generate the correction from it.  The risk it must answer for is that
                    the shared trunk moves for every earlier concept too.
  `ortho`           the delta is an orthogonal (Cayley) rotation, which cannot change the norm of
                    what it acts on.  Aimed at 5r, where merged deltas corrupted the base.
  `guidance-pool`   the OOD prototype is produced by CLASSIFIER GUIDANCE -- the composition is a sum
                    of attribute log-probability gradients -- and those samples are then consolidated
                    normally.  Tests whether a better composer makes a better manifold update.
  `guidance-only`   the same guidance with no consolidation at all.  Zero parameters, nothing added.
                    The control for whether the manifold update is doing any work.

Reported per arm: accuracy and coverage of the OOD composition, content diversity, what is retained
after the whole stream, and the parameter cost of getting there.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, backbone_gains, ood_queries              # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                           # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                sample_geometry, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, to_data, to_model)  # noqa: E402
from lifelong_ood.neural.alternatives import (GuidanceHead, HyperDelta, HyperHook,  # noqa: E402
                                              OrthoDelta, OrthoHook, sample_guided)
from lifelong_ood.neural.idealwords import sample_factored_pool, slot_factors  # noqa: E402
from lifelong_ood.score import cosine_abar                                   # noqa: E402

ARMS = ("lora", "hypernet", "ortho", "guidance-pool", "guidance-only")


def train_delta(field, params, emb, X, epochs, batch, lr, seed, pre=None):
    """One consolidation step against an arbitrary delta parameterization."""
    opt = torch.optim.Adam(params, lr=lr)
    dev = field.device
    Xt = torch.from_numpy(np.asarray(X, np.float32)).to(dev).reshape(-1, 3, field.img, field.img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    field.model.train()
    curve = []
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        tot, nb = 0.0, 0
        for i0 in range(0, len(Xt), batch):
            x0 = Xt[perm[i0:i0 + batch].to(dev)]
            u = torch.rand(len(x0), device=dev).clamp_(1e-3, 1.0)
            ab = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float()
            ab = ab.view(-1, 1, 1, 1)
            e = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
            if pre is not None:
                pre()
            loss = torch.nn.functional.mse_loss(field.model(xt, u, emb_override=emb), e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        curve.append(tot / max(nb, 1))
    field.model.eval()
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=6)
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--inv-steps", type=int, default=400)
    p.add_argument("--guide-scale", type=float, default=2.0)
    p.add_argument("--guide-epochs", type=int, default=10)
    p.add_argument("--hyper-hidden", type=int, default=64)
    p.add_argument("--ortho-block", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_concepts, args.n_pool, args.n_samples = 2, 12, 8
        args.epochs, args.steps, args.inv_steps = 4, 10, 60
        args.n_per_slot, args.guide_epochs = 20, 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)
    print(f"[n16] {len(qs)} OOD compositions, arms {args.arms}, device {dev}", flush=True)

    # the guidance head is trained ONCE on seen data and reused for every composition; its cost is
    # per-attribute, not per-concept, which is the whole point of the arm
    head = None
    if any(a.startswith("guidance") for a in args.arms):
        f0 = load_field(args.ckpt, device=dev)
        seen = np.where(data["seen"])[0]
        lab = np.stack([data["digit"][seen], data["fg"][seen], data["bg"][seen]], 1)
        head = GuidanceHead(img=f0.img, device=dev).fit(
            to_model(data["X"][seen]), lab, epochs=args.guide_epochs, seed=args.seed,
            verbose=not args.quick)
        print(f"[n16] guidance head: {sum(q.numel() for q in head.parameters()) / 1e6:.2f}M params "
              f"-- fixed, shared by every composition", flush=True)

    rows = []
    for arm in args.arms:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev, flat=True)
        gains = backbone_gains(field, C, slots_seen, probes, data, seed=args.seed,
                               steps=args.steps, n=8 if args.quick else 16, verbose=False)
        L.attach_factorization(C, slot_factors(slots_seen), gains=gains)
        hyper = hooks = ortho_objs = None
        if arm == "hypernet":
            hyper = HyperDelta(L.bank.wrapped, emb_dim=C.shape[1], hidden=args.hyper_hidden,
                               device=dev)
            hooks = [w.register_forward_hook(HyperHook(hyper, i))
                     for i, w in enumerate(L.bank.wrapped)]
            print(f"[n16:{arm}] hypernetwork {hyper.n_params() / 1e6:.2f}M params -- FIXED, "
                  f"shared by every concept", flush=True)
        if arm == "ortho":
            ortho_objs, ortho_reg = {}, {"active": None}
            hooks = [w.register_forward_hook(OrthoHook(ortho_reg, i))
                     for i, w in enumerate(L.bank.wrapped)]
        embs, learned = {}, []
        t0 = time.time()

        for qi, q in enumerate(qs):
            name = f"c{q['slot']}"
            xq = to_model(data["X"][q["query"]]).ravel()
            # ── 1-2. compose the OOD prototype and sample instances from it ──
            if arm.startswith("guidance"):
                pool = sample_guided(field, head, (q["d"], q["f"], q["b"]), n=args.n_pool,
                                     steps=args.steps, seed=args.seed + qi, scale=args.guide_scale)
                c_new = None
            else:
                c_new, _, _ = L.compose_factored(xq, inv_steps=args.inv_steps, seed=args.seed + qi)
                pool = sample_factored_pool(field, c_new, n=args.n_pool, steps=args.steps,
                                            seed=args.seed + qi)
            # ── 3. add those instances to the manifold ───────────────────────
            if arm == "guidance-only":
                pass                                    # nothing is added; the control
            elif arm == "lora" or arm == "guidance-pool":
                if c_new is None:                       # guidance pool -> learn a conditioning too
                    c_new, _, _ = L.compose_factored(xq, inv_steps=args.inv_steps,
                                                     seed=args.seed + qi)
                L.bank.add_concept(name, np.asarray(pool).mean(0))
                L.bank._emb_extra[name].data = c_new.detach().clone()
                L.consolidate(pool, name, epochs=args.epochs, freeze_embedding=True,
                              already_registered=True)
            elif arm == "hypernet":
                hyper.set_concept(c_new.detach())
                train_delta(field, list(hyper.parameters()), c_new.detach(), pool, args.epochs,
                            16, 1e-3, args.seed + qi, pre=lambda: hyper.set_concept(c_new.detach()))
            elif arm == "ortho":
                od = OrthoDelta(L.bank.wrapped, block=args.ortho_block, device=dev)
                ortho_objs[name] = od
                ortho_reg["active"] = od
                train_delta(field, list(od.parameters()), c_new.detach(), pool, args.epochs,
                            16, 1e-3, args.seed + qi)
                ortho_reg["active"] = None
            embs[name] = c_new
            learned.append((name, q))

        # ── 4. what survived?  score every concept after the whole stream ────
        def sample_for(name, q):
            if arm == "guidance-only":
                return to_data(sample_guided(field, head, (q["d"], q["f"], q["b"]),
                                             n=args.n_samples, steps=args.steps, seed=args.seed,
                                             scale=args.guide_scale)).astype(np.float32)
            if arm == "hypernet":
                hyper.set_concept(embs[name].detach())
                out = to_data(sample_factored_pool(field, embs[name], n=args.n_samples,
                                                   steps=args.steps, seed=args.seed))
                hyper.set_concept(None)
                return out.astype(np.float32)
            if arm == "ortho":
                ortho_reg["active"] = ortho_objs[name]
                out = to_data(sample_factored_pool(field, embs[name], n=args.n_samples,
                                                   steps=args.steps, seed=args.seed))
                ortho_reg["active"] = None
                return out.astype(np.float32)
            return to_data(L.sample_concept(name, n=args.n_samples,
                                            seed=args.seed)).astype(np.float32)

        for (name, q) in learned:
            S = sample_for(name, q)
            R = data["X"][q["rest"][:100]]
            a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
            pr = precision_recall(S, R)
            g = sample_geometry(S, R[0], R)
            rows.append(dict(arm=arm, slot=int(q["slot"]), joint=a["joint"], digit=a["digit"],
                             fg=a["fg"], bg=a["bg"], f1=pr["f1"], precision=pr["precision"],
                             recall=pr["recall"],
                             content_ratio=g.get("content_ratio", float("nan"))))
        extra = 0
        if arm == "hypernet":
            extra = hyper.n_params()
        elif arm == "ortho":
            extra = sum(o.n_params() for o in ortho_objs.values())
        elif arm.startswith("guidance"):
            extra = sum(q_.numel() for q_ in head.parameters())
        params = L.bank.n_params() + extra
        m = [r for r in rows if r["arm"] == arm]
        print(f"[n16] -> {arm}: acc {np.mean([r['joint'] for r in m]):.3f}, "
              f"F1 {np.mean([r['f1'] for r in m]):.3f}, "
              f"div {np.nanmean([r['content_ratio'] for r in m]):.2f}, "
              f"{params / 1e6:.2f}M, {time.time() - t0:.0f}s\n", flush=True)
        for r in m:
            r["params"] = params
            r["wall_seconds"] = time.time() - t0
        if hooks:
            for h_ in (hooks if isinstance(hooks, list) else []):
                try:
                    h_.remove()
                except Exception:
                    pass

    summary = {a: {k: float(np.nanmean([x[k] for x in rows if x["arm"] == a]))
                   for k in ("joint", "digit", "fg", "bg", "precision", "recall", "f1",
                             "content_ratio", "params")}
               for a in args.arms if any(x["arm"] == a for x in rows)}
    save_json(dict(rows=rows, summary=summary, config=vars(args)),
              os.path.join(RESULTS, "n16_alternatives.json"), seed=args.seed)
    print(f"\n[n16] adding {args.n_concepts} OOD compositions to the manifold, four ways")
    print(f"    {'arm':16s}{'acc':>6s}{'digit':>7s}{'fg':>5s}{'bg':>5s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}{'div':>7s}{'params':>10s}")
    for a in args.arms:
        if a not in summary:
            continue
        v = summary[a]
        print(f"    {a:16s}{v['joint']:6.2f}{v['digit']:7.2f}{v['fg']:5.2f}{v['bg']:5.2f}"
              f"{v['precision']:7.2f}{v['recall']:8.2f}{v['f1']:7.2f}{v['content_ratio']:7.2f}"
              f"{v['params'] / 1e6:9.2f}M")
    print(f"[n16] wrote {RESULTS}/n16_alternatives.json")


if __name__ == "__main__":
    main()
