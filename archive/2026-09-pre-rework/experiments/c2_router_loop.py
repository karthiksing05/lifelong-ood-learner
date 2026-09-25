"""C2 -- the three-tier loop, run END TO END with the router deciding what is stored.

Every cost number in this repository before this came from ground-truth tiers: n24 is a no-model
simulation, n18 counts the paid route instead of running it, n41 addresses every query.  C2 is the
first run in which the router's decision determines what is actually written, and the written thing
is then sampled and scored.  It runs on either benchmark through `c_bench.py`.

A stream of concepts arrives, each as a small batch of real instances (RESULTS 6c-ter puts the
break-even for storing an address at ~15 instances, so a batch, not one image), drawn from three
populations the router is NOT told about:

  in-distribution   a class the backbone was trained on             -> should store nothing
  new-combination   an untrained combination of known values        -> should ADDRESS
  new-value         the value the backbone never rendered           -> WEIGHT once, then ADDRESS

The cascade, every threshold calibrated on the backbone's OWN trained classes (no held-out label):

  1. IN-DIST   every noise level of the best trained row is at the trained-class level (q90)
  2. ADDRESS   every level of the best basis composition is at the level a trained class reaches
               through a basis refit WITHOUT it -- leave-one-class-out addresses are exactly what a
               held-out combination looks like, so the basis's generalisation gap is priced in
  3. EXTEND    attribute the novel FACTOR by sharpness (a known factor's loss curve over its values
               has a sharp minimum; a factor whose value is new is flat, relative to how sharp that
               factor is on trained classes), then trial REFINE (host in an existing extended value,
               Cobweb's 'host') against NEW (a new value, Cobweb's 'create'); keep the better
  4. WEIGHT    the Cobweb-form update with that direction, until every level is absorbed (or a cap)

After 3 or 4 the direction JOINS THE BASIS, so later concepts using it are read and addressed.

    python experiments/c2_router_loop.py --bench colormnist --seed 0
    python experiments/c2_router_loop.py --bench shapes3d --weight-mode lora
"""
import argparse
import copy
import itertools
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c_bench import load_bench                                        # noqa: E402
from common import RESULTS                                            # noqa: E402

from lifelong_ood.evals import density_coverage, save_json, vendi_score  # noqa: E402
from lifelong_ood.neural.cobweb_update import (GatedLoRA, anchor_loss,  # noqa: E402
                                               candidate_losses, ddim_sample, delta_stats,
                                               eps_loss, gate_on, inject_gated_lora, sq_grad,
                                               to_img, topk_mask)
from lifelong_ood.neural.idealwords import factorize                  # noqa: E402

U_GRID = np.round(np.arange(0.1, 1.0, 0.1), 2)
CRN = 4242


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="colormnist", choices=("colormnist", "shapes3d"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-in", type=int, default=4)
    p.add_argument("--n-comb", type=int, default=8)
    p.add_argument("--n-new", type=int, default=8, help="new-value concepts")
    p.add_argument("--n-train", type=int, default=None, help="default 64 (colormnist) / 48")
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-ref", type=int, default=None, help="default 100 (colormnist) / 36")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=None, help="DDIM steps; 50 colormnist / 200 shapes3d")
    p.add_argument("--u-min-crit", type=float, default=0.3,
                   help="every threshold test reads only levels u >= this (low levels test pixel "
                        "fidelity the input already carries, not the concept)")
    p.add_argument("--ext-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--anchor-batch", type=int, default=32)
    p.add_argument("--n-anchor-per-class", type=int, default=2)
    p.add_argument("--n-anchor-new", type=int, default=8)
    p.add_argument("--n-fisher", type=int, default=256)
    p.add_argument("--n-gnew", type=int, default=96)
    p.add_argument("--n-cal-sharp", type=int, default=24, help="trained classes for informativeness")
    p.add_argument("--n-cal", type=int, default=120,
                   help="trained classes the thresholds are calibrated on (all of them if fewer)")
    p.add_argument("--weight-mode", choices=("mask", "lora", "all"), default="mask")
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--lora-exclude", default=None,
                   help="colon-separated module-name substrings LoRA must not wrap "
                        "(default: none on colormnist, the time embedding on shapes3d)")
    p.add_argument("--delta-frac", type=float, default=0.01)
    p.add_argument("--lr-v", type=float, default=0.05)
    p.add_argument("--lr-v-w", type=float, default=0.005)
    p.add_argument("--tol", type=float, default=0.0,
                   help="per-level margin: absorbed iff every level <= threshold * (1 + tol). The "
                        "threshold is a q90 over trained classes and carries its own estimation "
                        "noise; without a margin a 1-6%% excess at u=0.1 alone forced every task "
                        "to the step cap")
    p.add_argument("--fast", action="store_true",
                   help="for large tables: the IN-DIST check scores only trained classes among the "
                        "top-10 basis compositions, and every scan reads only the decision levels "
                        "(u >= --u-min-crit) with one noise draw")
    p.add_argument("--routing", choices=("cascade", "oracle"), default="cascade",
                   help="cascade: the label-free router decides every tier.  oracle: the tier comes "
                        "from the concept's TRUE population (in-dist -> nothing; new-comb -> ADDRESS; "
                        "the first new-value concept CREATES the value node and later ones UPDATE it "
                        "-- Cobweb's host-and-update), while reads, absorption and storage stay "
                        "label-free.  Separates what the loop achieves from how well the router routes")
    p.add_argument("--attribution", choices=("inform", "oracle"), default="inform",
                   help="which factor an escalated concept's new direction attaches to: label-free "
                        "informativeness, or the TRUE novel factor (tier decisions stay label-free "
                        "either way).  Oracle isolates the rest of the loop from attribution, which a "
                        "new value near a known one defeats from one concept's evidence")
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    sh = args.bench == "shapes3d"
    args.n_train = args.n_train or (48 if sh else 64)
    args.n_ref = args.n_ref or (36 if sh else 100)
    if args.lora_exclude is None:
        args.lora_exclude = "time_emb" if sh else ""
    if args.steps is None:
        args.steps = 200 if sh else 50
    global U_GRID
    if args.fast:
        U_GRID = U_GRID[U_GRID >= args.u_min_crit - 1e-9]
    n_noise = 1 if args.fast else 2
    if args.quick:
        args.n_in, args.n_comb, args.n_new = 1, 2, 3
        args.n_samples, args.ext_steps, args.max_steps = 12, 60, 150
        args.n_fisher, args.n_gnew, args.n_cal_sharp = 48, 32, 8
        args.tag = args.tag or "_quick"
    if not torch.cuda.is_available():
        raise SystemExit("[c2] refusing to run on CPU")
    dev = "cuda"
    torch.manual_seed(args.seed)
    t0 = time.time()

    B = load_bench(args.bench, seed=args.seed, device=dev)
    base = B.model
    for q in base.parameters():
        q.requires_grad_(False)
    model = copy.deepcopy(base).eval()
    kw = dict(layout=B.layout, img=B.img, device=dev)
    cl_batch = 256 if sh else 1024          # candidate batch: the 64px VE UNet runs out of memory at 1024
    fac = B.fac
    mu = np.asarray(fac["mu"], np.float32)
    V = [np.asarray(v, np.float32).copy() for v in fac["V"]]
    nF = len(V)
    base_norm = [float(np.linalg.norm(np.asarray(v), axis=1).mean()) for v in fac["V"]]
    extended = {}                                               # (f, j) -> dict(count, lora)
    print(f"[c2] {B.name}: basis sizes {[len(v) for v in V]} R2 {fac['r2']:.3f}; new value "
          f"{B.names[B.new_f]}={B.new_j}", flush=True)

    def split(X):
        n_ref = min(args.n_ref, len(X) - args.n_train - args.n_val)
        assert n_ref >= 8, f"only {len(X)} images"
        return (X[:args.n_train], B.to_m(X[args.n_train:args.n_train + args.n_val]), X[-n_ref:])

    # ── the stream ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    seen_t = [t for _, t in B.seen]
    pick = lambda pool, n: [pool[i] for i in rng.permutation(len(pool))[:n]]   # noqa: E731
    stream = ([(t, "in-dist") for t in pick(seen_t, args.n_in)]
              + [(t, "new-comb") for t in pick(B.held_comb, args.n_comb)]
              + [(t, "new-value") for t in pick(B.new_pairs, args.n_new)])
    stream = [stream[i] for i in rng.permutation(len(stream))]
    print(f"[c2] stream of {len(stream)}: {stream}", flush=True)

    # ── calibration on TRAINED classes only ─────────────────────────────────
    cal_cls = [B.seen[i] for i in rng.permutation(len(B.seen))[:args.n_cal]]
    cal = [(c, t, split(B.images(t))[1]) for c, t in cal_cls]
    P_row = np.stack([candidate_losses(base, xv, U_GRID, cls=[c], seed=CRN, per_u=True, batch=cl_batch, n_noise=n_noise, **kw)[0]
                      for c, t, xv in cal])
    # leave-one-class-out addresses: the basis refit on ALL trained classes but this one
    tab = B.model.cemb.weight.detach().cpu().numpy()
    all_c = np.array([c for c, _ in B.seen])
    all_t = np.array([t for _, t in B.seen])
    C_all = tab[all_c]
    P_loo = []
    for c, t, xv in cal:
        keep = all_c != c
        fi = factorize(C_all[keep], all_t[keep])
        e = np.asarray(fi["mu"]) + sum(np.asarray(fi["V"][f])[t[f]] for f in range(nF))
        P_loo.append(candidate_losses(base, xv, U_GRID, embs=e[None], seed=CRN, per_u=True,
                                      batch=cl_batch, n_noise=n_noise, **kw)[0])
    P_loo = np.stack(P_loo)
    t_row, t_addr = np.quantile(P_row, 0.9, axis=0), np.quantile(P_loo, 0.9, axis=0)
    crit = U_GRID >= args.u_min_crit - 1e-9

    def within(pr, t):
        return bool(np.all(np.asarray(pr)[crit] <= t[crit] * (1.0 + args.tol)))
    print(f"[c2] thresholds per u: t_row {np.round(t_row, 4).tolist()}\n"
          f"[c2]                   t_addr(LOO) {np.round(t_addr, 4).tolist()}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    def emb_np(coords):
        return mu + sum(V[f][j] for f, j in enumerate(coords))

    def enumerate_basis(Xv, gate_fn):
        combos = list(itertools.product(*[range(len(v)) for v in V]))
        E = np.stack([emb_np(c) for c in combos]).astype(np.float32)
        gates = [gate_fn(c) for c in combos]
        L = np.empty((len(combos), len(U_GRID)))
        for use_gate in (False, True):
            ix = [i for i, gt in enumerate(gates) if (gt is not None) == use_gate]
            if ix:
                L[ix] = candidate_losses(model, Xv, U_GRID, embs=E[ix], seed=CRN, per_u=True,
                                         gate=gate_on(lora_state) if use_gate else None,
                                         batch=cl_batch, n_noise=n_noise, **kw)
        return combos, L

    lora_state = {"on": False}
    lora_params = []

    def gate_for(coords):
        if args.weight_mode == "lora" and any(extended.get((f, j), {}).get("lora")
                                              for f, j in enumerate(coords)):
            return gate_on(lora_state)
        return None

    def losses_for(coord_list, Xv):
        """Mean eps-loss (over u >= u_min_crit) of Xv under each coordinate tuple; entries may be
        None to leave a factor OUT of the composition.  Splits candidates by LoRA gate."""
        E = np.stack([mu + sum(V[f][j] for f, j in enumerate(c) if j is not None)
                      for c in coord_list]).astype(np.float32)
        gates = [gate_for(tuple(j if j is not None else -1 for j in c)) for c in coord_list]
        out = np.empty(len(coord_list))
        for use_gate in (False, True):
            ix = [i for i, gt in enumerate(gates) if (gt is not None) == use_gate]
            if ix:
                P = candidate_losses(model, Xv, U_GRID, embs=E[ix], seed=CRN, per_u=True,
                                     gate=gate_on(lora_state) if use_gate else None,
                                     batch=cl_batch, n_noise=n_noise, **kw)
                out[ix] = P[:, crit].mean(1)
        return out

    def informativeness(Xv):
        """Per factor: read the OTHER factors with this one left out, then the relative improvement
        its best known value adds.  A factor whose value is new adds little, whatever the other
        reads -- unlike slice sharpness, which a compensating misread of another factor fools."""
        g_out = []
        for f in range(nF):
            others = [h for h in range(nF) if h != f]
            cl = []
            for vals in itertools.product(*[range(len(V[h])) for h in others]):
                c = [None] * nF
                for h, j in zip(others, vals):
                    c[h] = j
                cl.append(c)
            lo = losses_for(cl, Xv)
            best = cl[int(np.argmin(lo))]
            wf = []
            for j in range(len(V[f])):
                c = list(best)
                c[f] = j
                wf.append(c)
            lf = losses_for(wf, Xv)
            g_out.append(float((lo.min() - lf.min()) / lo.min()))
        return np.asarray(g_out)

    inf_cal = np.stack([informativeness(xv) for _, _, xv in
                        [cal[i] for i in rng.permutation(len(cal))[:args.n_cal_sharp]]])
    med_gain = np.median(inf_cal, 0)
    gamma = float(np.quantile((inf_cal / med_gain).min(1), 0.10))
    print(f"[c2] typical factor informativeness on trained classes "
          f"{dict(zip(B.names, np.round(med_gain, 3)))}; ADDRESS iff every factor >= {gamma:.3f} x "
          f"typical ({time.time() - t0:.0f}s)", flush=True)

    # ── anchors (the model's own samples of trained classes) and F_old ──────
    a_cls = np.concatenate([np.repeat([c for c, _ in B.seen], args.n_anchor_per_class),
                            np.full(4 * args.n_anchor_per_class, base.null_id)])
    a_x0 = ddim_sample(base, len(a_cls), steps=args.steps, seed=777 + args.seed, cls=a_cls,
                       clip=B.clip, **kw)
    groups = [dict(x0=to_img(torch.from_numpy(a_x0).to(dev), B.img, B.layout),
                   cls=torch.as_tensor(a_cls, device=dev), coords=None, teacher=base, tgate=None,
                   tag="orig")]
    W0 = {n: q for n, q in model.named_parameters() if not n.startswith("cemb")}
    names_w = list(W0)
    theta0 = {n: q.detach().clone() for n, q in W0.items()}
    P0 = dict(base.named_parameters())
    for n in names_w:
        P0[n].requires_grad_(True)
    gf = torch.Generator(device="cpu").manual_seed(args.seed + 99)
    order = torch.randint(0, len(a_cls), (args.n_fisher,), generator=gf)
    F_old = sq_grad([P0[n] for n in names_w],
                    lambda i: eps_loss(base, groups[0]["x0"][int(order[i]):int(order[i]) + 1],
                                       cls=groups[0]["cls"][int(order[i]):int(order[i]) + 1],
                                       gen=gf), args.n_fisher)
    for n in names_w:
        P0[n].requires_grad_(False)
    F_old = dict(zip(names_w, F_old))
    F_mean = float(torch.cat([f.reshape(-1) for f in F_old.values()]).mean())
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    print(f"[c2] anchors {len(a_cls)}, F_old over {args.n_fisher} ({time.time() - t0:.0f}s)",
          flush=True)

    def emb_t(coords, vdict=None):
        out = torch.as_tensor(mu, device=dev)
        for f, j in enumerate(coords):
            out = out + (vdict[(f, j)] if vdict is not None and (f, j) in vdict
                         else torch.as_tensor(V[f][j], device=dev))
        return out

    def anchor_term(n_tot, vdict=None, new_only=False):
        orig = [] if (args.weight_mode == "lora" or new_only) \
            else [gr for gr in groups if gr["tag"] == "orig"]
        new = [gr for gr in groups if gr["tag"] != "orig"]
        if not (orig or new):
            return torch.zeros((), device=dev)
        plan = [(orig, n_tot // 2), (new, n_tot - n_tot // 2)] if (orig and new) \
            else [(orig or new, n_tot)]
        tot, cnt = torch.zeros((), device=dev), 0
        for grs, k in plan:
            gi = torch.randint(0, len(grs), (k,), generator=g)
            for gidx in torch.unique(gi):
                gr = grs[int(gidx)]
                m = int((gi == gidx).sum())
                j = torch.randint(0, len(gr["x0"]), (m,), generator=g).to(dev)
                if gr["coords"] is None:
                    tot = tot + m * anchor_loss(model, gr["teacher"], gr["x0"][j], cls=gr["cls"][j],
                                                gen=g)
                else:
                    tot = tot + m * anchor_loss(
                        model, gr["teacher"], gr["x0"][j],
                        emb=emb_t(gr["coords"], vdict)[None].expand(m, -1),
                        t_emb=gr["emb_then"][None].expand(m, -1), gen=g,
                        teacher_gate=gr["tgate"], gate=gate_for(gr["coords"]))
                cnt += m
        return tot / max(cnt, 1)

    def prof(Xv, coords, vdict=None):
        e = emb_t(coords, vdict).detach().cpu().numpy()[None]
        return candidate_losses(model, Xv, U_GRID, embs=e, seed=CRN, per_u=True,
                                gate=gate_for(coords), n_noise=n_noise, **kw)[0]

    def absorbed(pr):
        return within(pr, t_addr)

    def train_direction(Xtr_t, Xv, coords, key, v_init, steps, lr, weights=None):
        v = torch.as_tensor(v_init, device=dev).clone().requires_grad_(True)
        groups_opt = [{"params": [v], "lr": lr}]
        masks = ref = None
        if weights is not None:
            groups_opt.append({"params": weights["params"], "lr": args.lr})
            masks, ref = weights.get("masks"), weights.get("ref")
            for q in weights["params"]:
                q.requires_grad_(True)
        opt = torch.optim.Adam(groups_opt)
        gate = gate_for(coords)
        # with the weights frozen, only stored concepts that SHARE this value can be disturbed
        shares = any(gr["coords"] is not None and gr["coords"][key[0]] == key[1] for gr in groups)
        used, ok = 0, False
        for step in range(steps):
            j = torch.randint(0, len(Xtr_t), (args.batch,), generator=g).to(dev)
            e = emb_t(coords, {key: v})[None].expand(args.batch, -1)
            if gate is None:
                loss = eps_loss(model, Xtr_t[j], emb=e, gen=g)
            else:
                with gate():
                    loss = eps_loss(model, Xtr_t[j], emb=e, gen=g)
            if weights is not None or shares:
                loss = loss + args.lam * anchor_term(args.anchor_batch, {key: v},
                                                     new_only=weights is None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if masks is not None:
                for q in weights["params"]:
                    if q.grad is not None:
                        q.grad.mul_(masks[id(q)])
            opt.step()
            with torch.no_grad():
                nv = float(v.norm())
                if nv > 1e-8:
                    v.mul_(base_norm[key[0]] / nv)
                if masks is not None:
                    for q in weights["params"]:
                        q.copy_(masks[id(q)] * q + (1 - masks[id(q)]) * ref[id(q)])
            used = step + 1
            if weights is not None and used % args.check_every == 0:
                if absorbed(prof(Xv, coords, {key: v.detach()})):
                    ok = True
                    break
        if weights is not None:
            for q in weights["params"]:
                q.requires_grad_(False)
        return v.detach(), used, ok

    def sample_rec(rec, n, seed):
        if rec["tier"] == "IN-DIST":
            return B.to_d(ddim_sample(model, n, steps=args.steps, seed=seed, cls=rec["row"],
                                      clip=B.clip, **kw))
        return B.to_d(ddim_sample(model, n, steps=args.steps, seed=seed,
                                  emb=emb_np(rec["coords"]).astype(np.float32),
                                  gate=gate_for(rec["coords"]), clip=B.clip, **kw))

    records, total, version, snap_cache = [], 0, 0, {}
    for ci, (tup, pop) in enumerate(stream):
        tc = time.time()
        Xtr, Xv, ref = split(B.images(tup))
        Xtr_t = to_img(torch.as_tensor(B.to_m(Xtr), dtype=torch.float32, device=dev), B.img, B.layout)
        rec = dict(i=ci, true=list(tup), pop=pop)
        combos, La = enumerate_basis(Xv, gate_for)
        ka = int(np.argmin(La.mean(1)))
        read = combos[ka]
        rows = np.array([c for c, _ in B.seen])
        if args.fast:
            # the concept's own trained row, if it has one, sits among its best compositions
            seen_of = {t: c for c, t in B.seen}
            top = [combos[i] for i in np.argsort(La.mean(1))[:10]]
            rows = np.array([seen_of[t] for t in top if t in seen_of] or [rows[0]])
        Lr = candidate_losses(model, Xv, U_GRID, cls=rows, seed=CRN, per_u=True, batch=cl_batch,
                              n_noise=n_noise, **kw)
        kr = int(np.argmin(Lr.mean(1)))
        rec.update(row=int(rows[kr]), s_row=float(Lr[kr].mean()), s_addr=float(La[ka].mean()),
                   read=list(read), row_ratio=(Lr[kr] / t_row).tolist(),
                   addr_ratio=(La[ka] / t_addr).tolist())
        steps_w, cost, coords, trials = 0, 0, read, []
        gains = informativeness(Xv) / med_gain
        rec["informativeness"] = gains.tolist()
        oracle = args.routing == "oracle"
        if (oracle and pop == "in-dist") or (not oracle and within(Lr[kr], t_row)):
            tier = "IN-DIST"
            if oracle:
                rec["row"] = int({t: c for c, t in B.seen}[tuple(tup)])
        elif (oracle and pop == "new-comb") or (not oracle and gains.min() >= gamma):
            # every factor has a KNOWN value that explains its part of the concept: a new
            # arrangement of known parts, whatever the absolute fit of the basis costs it
            tier, cost = "ADDRESS", 4 * sum(len(v) for v in V)
        else:
            ff = int(np.argmin(gains))
            rec["factor_inform"] = B.names[ff]
            if (args.attribution == "oracle" or oracle) and tup[B.new_f] == B.new_j:
                ff = B.new_f
            rec.update(factor=B.names[ff])
            options = [("new", None)] + [("refine", jj) for (f2, jj) in extended if f2 == ff]
            if oracle and any(f2 == ff for (f2, _) in extended):
                # the value node exists: host the concept in it and update it, never duplicate it
                options = [("refine", jj) for (f2, jj) in extended if f2 == ff]
            for kind, jj in options:
                cand = list(read)
                if kind == "new":
                    key = (ff, len(V[ff]))
                    init = V[ff].mean(0)
                    V[ff] = np.vstack([V[ff], init[None]])
                    cand[ff] = key[1]
                    v_try, _, _ = train_direction(Xtr_t, Xv, tuple(cand), key, init,
                                                  args.ext_steps, args.lr_v)
                    V[ff][key[1]] = v_try.cpu().numpy()
                    pr = prof(Xv, tuple(cand))
                    V[ff] = V[ff][:-1]
                else:
                    key = (ff, jj)
                    cand[ff] = jj
                    keep = V[ff][jj].copy()
                    v_try, _, _ = train_direction(Xtr_t, Xv, tuple(cand), key, keep,
                                                  args.ext_steps,
                                                  args.lr_v / (1 + extended[key]["count"]))
                    V[ff][jj] = v_try.cpu().numpy()
                    pr = prof(Xv, tuple(cand))
                    V[ff][jj] = keep
                trials.append(dict(kind=kind, key=key, coords=tuple(cand), v=v_try,
                                   s=float(pr.mean()), ok=absorbed(pr)))
            # prefer an absorbing trial; among equals, prefer REFINE (it costs nothing)
            best = min(trials, key=lambda tr: (not tr["ok"], tr["s"] * (1.0 if tr["kind"] == "new"
                                                                        else 0.99)))
            kind, key, coords = best["kind"], best["key"], best["coords"]
            if kind == "new":
                V[ff] = np.vstack([V[ff], best["v"].cpu().numpy()[None]])
                extended[key] = dict(count=0, lora=False)
            else:
                V[ff][key[1]] = best["v"].cpu().numpy()
            extended[key]["count"] += 1
            if best["ok"]:
                tier, cost = ("EXTEND" if kind == "new" else "REFINE"), (1024 if kind == "new" else 0)
            else:
                if args.weight_mode == "lora":
                    if not lora_params:
                        lora_params.extend(inject_gated_lora(
                            model, args.rank, lora_state,
                            exclude=tuple(x for x in args.lora_exclude.split(":") if x)))
                        for q in lora_params:
                            q.requires_grad_(False)
                    first = not extended[key]["lora"]
                    extended[key]["lora"] = True
                    wts = dict(params=lora_params)
                elif args.weight_mode == "all":
                    wts = dict(params=[W0[n] for n in names_w])
                else:
                    cp = [W0[n] for n in names_w]
                    for q in cp:
                        q.requires_grad_(True)
                    gtr = torch.Generator(device="cpu").manual_seed(args.seed + 7 + ci)

                    def _one(i):
                        j = int(torch.randint(0, len(Xtr_t), (1,), generator=gtr))
                        return eps_loss(model, Xtr_t[j:j + 1], emb=emb_t(coords)[None], gen=gtr)

                    gnew = sq_grad(cp, _one, args.n_gnew)
                    for q in cp:
                        q.requires_grad_(False)
                    mk = topk_mask([gn / (F_old[n] + args.delta_frac * F_mean)
                                    for gn, n in zip(gnew, names_w)], args.rho)
                    sel = [(n, m) for n, m in zip(names_w, mk) if bool(m.any())]
                    wts = dict(params=[W0[n] for n, _ in sel],
                               masks={id(W0[n]): m.float() for n, m in sel},
                               ref={id(W0[n]): W0[n].detach().clone() for n, _ in sel})
                before = {n: W0[n].detach().clone() for n in names_w}
                v_w, steps_w, ok = train_direction(Xtr_t, Xv, coords, key, V[ff][key[1]],
                                                   args.max_steps, args.lr_v_w, weights=wts)
                V[ff][key[1]] = v_w.cpu().numpy()
                tier = "WEIGHT" if ok else "WEIGHT-cap"
                version += 1
                dsn = delta_stats([(n, W0[n]) for n in names_w], before)
                rec.update(w_changed=dsn["changed"], w_dtheta=dsn["l2"])
                cost = (1024 if kind == "new" else 0) + (
                    4 * sum(q.numel() for q in lora_params)
                    if args.weight_mode == "lora" and first else 0)
        coords = tuple(coords)
        rec.update(tier=tier, cost_bytes=int(cost), steps_weight=int(steps_w),
                   trials=[dict(kind=t["kind"], s=t["s"], ok=t["ok"]) for t in trials])
        if tier != "IN-DIST":
            rec["coords"] = list(coords)
            rec["read_ok"] = bool(tuple(read) == tuple(tup))
            rec["stored_ok"] = bool(coords == tuple(tup))
        total += cost
        rec["bytes_total"] = int(total)
        S = sample_rec(rec, args.n_samples, seed=3000 + ci).astype(np.float32)
        a = B.acc(S, tup)
        dc = density_coverage(S, ref)
        rec.update(acc=a, joint=a["joint"], coverage=dc["coverage"], density=dc["density"],
                   vendi=float(vendi_score(S)), secs=time.time() - tc)
        records.append(rec)
        if tier != "IN-DIST":
            if version not in snap_cache:                        # weights moved: new teacher
                snap_cache = {version: copy.deepcopy(model).eval()}
            snap = snap_cache[version]
            sst = next((m.state for m in snap.modules() if isinstance(m, GatedLoRA)), None)
            e_then = emb_t(coords).detach().clone()
            ax = ddim_sample(model, args.n_anchor_new, steps=args.steps, seed=900 + ci,
                             emb=e_then.cpu().numpy(), gate=gate_for(coords), clip=B.clip, **kw)
            groups.append(dict(x0=to_img(torch.from_numpy(ax).to(dev), B.img, B.layout), cls=None,
                               coords=coords, emb_then=e_then, teacher=snap,
                               tgate=gate_on(sst) if (sst is not None and gate_for(coords)) else None,
                               tag=f"c{ci}"))
        extra = f" factor {rec.get('factor', '-')}" if "factor" in rec else ""
        print(f"[c2] {ci:2d} {pop:9s} {tup} -> {tier:10s} read {tuple(read)}{extra} | row "
              f"{max(rec['row_ratio']):.2f} addr {max(rec['addr_ratio']):.2f} (max ratio) | "
              f"{cost:>7d} B (total {total}) steps {steps_w} | joint {a['joint']:.3f} "
              f"cov {dc['coverage']:.3f}", flush=True)

    # ── retention: everything stored, re-sampled at the end; old classes ────
    for rec in records:
        S = sample_rec(rec, args.n_samples, seed=3000 + rec["i"]).astype(np.float32)
        rec["joint_end"] = B.acc(S, tuple(rec["true"]))["joint"]
    ro = [B.seen[i] for i in rng.permutation(len(B.seen))[:12]]
    cids = np.repeat([c for c, _ in ro], args.n_samples)

    def old_acc(m):
        S = B.to_d(ddim_sample(m, len(cids), steps=args.steps, seed=5000, cls=cids, clip=B.clip,
                               **kw))
        return float(np.mean([B.acc(S[i * args.n_samples:(i + 1) * args.n_samples], t)["joint"]
                              for i, (_, t) in enumerate(ro)]))

    ds = delta_stats([(n, W0[n]) for n in names_w], theta0)
    summary = dict(old_joint_start=old_acc(base), old_joint_end=old_acc(model),
                   changed=ds["changed"], changed_frac=ds["frac"], dtheta=ds["l2"],
                   lora_params=int(sum(q.numel() for q in lora_params)), bytes_total=int(total),
                   basis_sizes=[len(v) for v in V], minutes=(time.time() - t0) / 60)
    for pop in ("in-dist", "new-comb", "new-value"):
        rs = [r for r in records if r["pop"] == pop]
        if rs:
            tiers = {}
            for r in rs:
                tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
            summary[pop] = dict(n=len(rs), tiers=tiers,
                                joint=float(np.mean([r["joint"] for r in rs])),
                                joint_end=float(np.mean([r["joint_end"] for r in rs])),
                                stored_ok=float(np.mean([r.get("stored_ok", True) for r in rs])),
                                cost=int(sum(r["cost_bytes"] for r in rs)))
    print(f"[c2] SUMMARY {summary}", flush=True)
    save_json(dict(records=records, summary=summary, config=vars(args), stream=stream,
                   t_row=t_row.tolist(), t_addr=t_addr.tolist(), med_gain=med_gain.tolist(),
                   gamma=gamma),
              os.path.join(RESULTS, f"c2_router_loop_{args.bench}{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
