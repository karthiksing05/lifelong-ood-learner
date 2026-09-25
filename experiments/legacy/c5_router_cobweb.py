"""C5 -- the three-tier loop with Cobweb's operators.  A concept that no node explains becomes a LEAF
whose factor is left open; a later concept that the leaf explains MERGES with it, and the leaf becomes
a value node on the factor that concept revealed.

C2 attributes a novel concept's new value to one factor from that concept alone, and every node it
creates shares ONE adapter.  Both fail (C2 FLI seed 0: the new background put on the foreground; FLO
seed 0: a new combination's node trained over the new value's adapter).  The first is not fixable by
a better score: with an additive conditioning c = mu + sum_f V_f[v_f], a new direction on ANY factor
fits the concept that created it equally well (C4).  C5 changes exactly those two things:

  * every node owns its adapter (`KeyedLoRA`): a concept's forward pass switches on the nodes in its
    coordinates and nothing else, so trained classes -- and every unrelated node -- stay exact;
  * a concept that is neither IN-DIST nor ADDRESS creates a LEAF with one HYPOTHESIS per factor f:
    the concept's other factors read with f LEFT OUT (so a new value on f cannot bias them), and a
    value vector on f plus its own adapter, trained on the concept.  Every later concept is offered
    each open hypothesis on its factor.  When one explains a concept better than every basis
    composition, the leaf MERGES: that hypothesis becomes a value node (its row joins the basis), the
    others are dropped, the leaf's own concept moves to the node, and the new concept is stored there
    -- ADDRESS if the node already absorbs it, else the node is updated on it (REFINE, Cobweb's
    incorporate: 0 B).

Costs: IN-DIST 0 B | ADDRESS 4 x (basis size) B | LEAF nF x (1 KB + adapter) while open, refunded to
one hypothesis at the merge -- or at the end of the stream, where an unmerged leaf stays as a
concept-level BLOCK | REFINE 0 B.  The stream, the calibration and every threshold are C2's, drawn in
the same order from the same seed, so C2 and C5 see identical concepts.

    python experiments/c5_router_cobweb.py --bench colormnist --seed 0
    python experiments/c5_router_cobweb.py --bench shapes3d --fast --n-cal 60 --n-cal-sharp 12
"""
import argparse
import copy
import itertools
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))  # c_bench lives there until streams/bench.py
from c_bench import load_bench                                        # noqa: E402
from lifelong_ood.ledger import RESULTS                               # noqa: E402

from lifelong_ood.evals import density_coverage, save_json, vendi_score  # noqa: E402
from lifelong_ood.neural.cobweb_update import (KeyedLoRA, add_node, anchor_loss,  # noqa: E402
                                               candidate_losses, ddim_sample, drop_node,
                                               eps_loss, inject_keyed_lora, keys_on, to_img)
from lifelong_ood.neural.idealwords import factorize                  # noqa: E402

U_GRID = np.round(np.arange(0.1, 1.0, 0.1), 2)
CRN = 4242
LIVE = ("base", "node")                 # rows the basis read enumerates
KEYED = ("node", "hyp", "block")        # rows that switch an adapter on


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="colormnist", choices=("colormnist", "shapes3d"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-in", type=int, default=4)
    p.add_argument("--n-comb", type=int, default=8)
    p.add_argument("--n-new", type=int, default=8)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-ref", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--u-min-crit", type=float, default=0.3)
    p.add_argument("--max-steps", type=int, default=800, help="leaf / refine training cap")
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--anchor-batch", type=int, default=32)
    p.add_argument("--n-anchor-new", type=int, default=8)
    p.add_argument("--n-cal", type=int, default=120)
    p.add_argument("--n-cal-sharp", type=int, default=24)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--lora-exclude", default=None)
    p.add_argument("--lam", type=float, default=10.0, help="anchor weight for concepts sharing a node")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-v-w", type=float, default=0.005)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--address-rule", choices=("inform", "absorb", "retrieve"), default="inform",
                   help="inform: C2's rule (every factor's best known value adds >= gamma x its "
                        "typical gain); absorb: the best address passes the t_addr test; retrieve: "
                        "COMPOSING beats RETRIEVING -- the best address's worst decision level is "
                        "below --rho-cr x the best trained class's.  A new arrangement of known parts "
                        "is explained better by its parts than by any class the backbone was trained "
                        "on; a concept with a part the basis lacks is not, since both fall back on "
                        "the nearest known value")
    p.add_argument("--rho-cr", type=float, default=0.9)
    p.add_argument("--q-row", type=float, default=0.9,
                   help="IN-DIST threshold = this quantile of trained classes' best-row loss per "
                        "level.  0.9 (C2's) sends one trained class in ten onward by construction, "
                        "and under the retrieve rule such a class -- composing cannot beat retrieving "
                        "a class that exists -- became a LEAF; 1.0 = as well explained as the worst "
                        "trained class")
    p.add_argument("--incorporate", action="store_true",
                   help="Cobweb's incorporate: a concept ADDRESSED through a value node that does not "
                        "yet absorb it updates that node on its real images (0 B, anchored to the "
                        "node's earlier members) -- every instance updates the node it joins")
    p.add_argument("--no-refine", action="store_true",
                   help="store a merged concept by ADDRESS even when the node does not absorb it")
    p.add_argument("--flat", action="store_true",
                   help="the FLAT ablation: every escalated concept gets its own concept-level "
                        "adapter (a BLOCK) and nothing ever merges -- one adapter per OOD concept, "
                        "the layout the hierarchy is meant to beat")
    p.add_argument("--read-band", action="store_true",
                   help="read (every argmin over coordinates) at the noise levels where the backbone "
                        "reads its OWN trained classes exactly, each level divided by the trained-class "
                        "level.  Raw losses are largest at low noise, so a pooled read was decided "
                        "there -- where a new background's pixels swamp the conditioning: foreground "
                        "read 0.00-0.03 at u <= 0.5 and 1.00 at u = 0.8-0.95 (tests/diag_read_by_u.py)")
    p.add_argument("--n-band", type=int, default=40, help="trained classes that calibrate the band")
    p.add_argument("--n-untaught", type=int, default=8,
                   help="never-streamed new-value combinations addressed at the end through the "
                        "new factor's node: what the hierarchy can render that a flat store cannot")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--no-u-high", action="store_true",
                   help="drop u = 0.95 / 0.99 from every scan.  With them (C1's --u-high) the "
                        "absorption test also asks the CONDITIONING to carry the concept, where the "
                        "input no longer can; without them a merged concept passed it at joint 0.000")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    sh = args.bench == "shapes3d"
    args.n_train = args.n_train or (48 if sh else 64)
    args.n_ref = args.n_ref or (36 if sh else 100)
    args.steps = args.steps or (200 if sh else 50)
    if args.lora_exclude is None:
        args.lora_exclude = "time_emb" if sh else ""
    global U_GRID
    if args.fast:
        U_GRID = U_GRID[U_GRID >= args.u_min_crit - 1e-9]
    if not args.no_u_high:
        U_GRID = np.concatenate([U_GRID, [0.95, 0.99]])
    n_noise = 1 if args.fast else 2
    if not torch.cuda.is_available():
        raise SystemExit("[c5] refusing to run on CPU")
    dev = "cuda"
    torch.manual_seed(args.seed)
    t0 = time.time()

    B = load_bench(args.bench, seed=args.seed, device=dev)
    base = B.model
    for q in base.parameters():
        q.requires_grad_(False)
    model = copy.deepcopy(base).eval()
    kstate = {"on": frozenset()}
    banks = inject_keyed_lora(model, kstate, exclude=tuple(x for x in args.lora_exclude.split(":")
                                                           if x))
    kw = dict(layout=B.layout, img=B.img, device=dev)
    cl_batch = 256 if sh else 1024
    mu = np.asarray(B.fac["mu"], np.float32)
    V = [np.asarray(v, np.float32).copy() for v in B.fac["V"]]
    nF = len(V)
    n_base = [len(v) for v in V]
    base_norm = [float(np.linalg.norm(v, axis=1).mean()) for v in V]
    status = {(f, j): "base" for f in range(nF) for j in range(len(V[f]))}
    node_key, origin = {}, {}                  # (f, j) -> adapter key / the leaf it came from
    print(f"[c5] {B.name}: basis sizes {n_base} R2 {B.fac['r2']:.3f}; new value "
          f"{B.names[B.new_f]}={B.new_j}", flush=True)

    def split(X):
        n_ref = min(args.n_ref, len(X) - args.n_train - args.n_val)
        assert n_ref >= 8, f"only {len(X)} images"
        return (X[:args.n_train], B.to_m(X[args.n_train:args.n_train + args.n_val]), X[-n_ref:])

    # ── the stream: C2's, drawn in the same order ───────────────────────────
    rng = np.random.default_rng(args.seed)
    seen_t = [t for _, t in B.seen]
    pick = lambda pool, n: [pool[i] for i in rng.permutation(len(pool))[:n]]   # noqa: E731
    stream = ([(t, "in-dist") for t in pick(seen_t, args.n_in)]
              + [(t, "new-comb") for t in pick(B.held_comb, args.n_comb)]
              + [(t, "new-value") for t in pick(B.new_pairs, args.n_new)])
    stream = [stream[i] for i in rng.permutation(len(stream))]
    print(f"[c5] stream of {len(stream)}: {stream}", flush=True)

    def live(f):
        return [j for j in range(len(V[f])) if status[(f, j)] in LIVE]

    def keys_of(coords):
        return frozenset(node_key[(f, j)] for f, j in enumerate(coords)
                         if j is not None and status.get((f, j)) in KEYED and (f, j) in node_key)

    def gate_for(coords):
        ks = keys_of(coords)
        return keys_on(kstate, ks) if ks else None

    def emb_np(coords):
        return mu + sum(V[f][j] for f, j in enumerate(coords) if j is not None)

    def scan(Xv, coord_list, m=None):
        """-> [C, U] eps-loss profiles of Xv under each coordinate tuple (None = factor left out),
        one candidate_losses call per distinct adapter key set; shared noise draws throughout."""
        m = model if m is None else m
        E = np.stack([emb_np(c) for c in coord_list]).astype(np.float32)
        ks = [keys_of(c) for c in coord_list]
        out = np.empty((len(coord_list), len(U_GRID)))
        for kset in set(ks):
            ix = [i for i, k in enumerate(ks) if k == kset]
            out[ix] = candidate_losses(m, Xv, U_GRID, embs=E[ix], seed=CRN, per_u=True,
                                       gate=keys_on(kstate, kset) if kset else None,
                                       batch=cl_batch, n_noise=n_noise, **kw)
        return out

    # ── calibration on TRAINED classes only (C2's, same draws) ──────────────
    cal_cls = [B.seen[i] for i in rng.permutation(len(B.seen))[:args.n_cal]]
    cal = [(c, t, split(B.images(t))[1]) for c, t in cal_cls]
    P_row = np.stack([candidate_losses(base, xv, U_GRID, cls=[c], seed=CRN, per_u=True,
                                       batch=cl_batch, n_noise=n_noise, **kw)[0]
                      for c, t, xv in cal])
    tab = B.model.cemb.weight.detach().cpu().numpy()
    all_c = np.array([c for c, _ in B.seen])
    all_t = np.array([t for _, t in B.seen])
    P_loo = []
    for c, t, xv in cal:
        keep = all_c != c
        fi = factorize(tab[all_c][keep], all_t[keep])
        e = np.asarray(fi["mu"]) + sum(np.asarray(fi["V"][f])[t[f]] for f in range(nF))
        P_loo.append(candidate_losses(base, xv, U_GRID, embs=e[None], seed=CRN, per_u=True,
                                      batch=cl_batch, n_noise=n_noise, **kw)[0])
    t_row = np.quantile(P_row, args.q_row, axis=0)
    t_addr = np.quantile(np.stack(P_loo), 0.9, axis=0)
    crit = U_GRID >= args.u_min_crit - 1e-9

    def within(pr, t):
        return bool(np.all(np.asarray(pr)[crit] <= t[crit] * (1.0 + args.tol)))

    # ── the read band (label-free: the backbone's own trained classes) ──────
    band = np.ones(len(U_GRID), bool)
    band_rate = None
    if args.read_band:
        combos0 = list(itertools.product(*[range(len(v)) for v in V]))
        E0 = np.stack([emb_np(c) for c in combos0]).astype(np.float32)
        hit = np.zeros(len(U_GRID))
        for c, t, xv in cal[:args.n_band]:
            P = candidate_losses(base, xv, U_GRID, embs=E0, seed=CRN, per_u=True, batch=cl_batch,
                                 n_noise=n_noise, **kw)
            hit += np.array([combos0[k] == tuple(t) for k in P.argmin(0)], float)
        band_rate = hit / min(args.n_band, len(cal))
        band = band_rate >= band_rate.max() - 0.02
        print(f"[c5] read band: trained-class exact-read rate per u "
              f"{dict(zip(U_GRID.round(2).tolist(), band_rate.round(2).tolist()))} -> read at u "
              f"{U_GRID[band].round(2).tolist()} ({time.time() - t0:.0f}s)", flush=True)

    def score(P, which):
        """The read's objective for each candidate row of P [C, U]."""
        P = np.atleast_2d(P)
        if args.read_band:
            return (P[:, band] / t_addr[band]).mean(1)
        return P.mean(1) if which == "all" else P[:, crit].mean(1)

    def informativeness(Xv):
        """C2's score over the LIVE basis, plus each factor's leave-one-out read of the others."""
        g_out, reads = [], []
        for f in range(nF):
            others = [h for h in range(nF) if h != f]
            cl = []
            for vals in itertools.product(*[live(h) for h in others]):
                c = [None] * nF
                for h, j in zip(others, vals):
                    c[h] = j
                cl.append(c)
            lo = score(scan(Xv, cl), "crit")
            best = cl[int(np.argmin(lo))]
            wf = []
            for j in live(f):
                c = list(best)
                c[f] = j
                wf.append(c)
            lf = score(scan(Xv, wf), "crit")
            g_out.append(float((lo.min() - lf.min()) / lo.min()))
            reads.append(best)
        return np.asarray(g_out), reads

    inf_cal = np.stack([informativeness(xv)[0] for _, _, xv in
                        [cal[i] for i in rng.permutation(len(cal))[:args.n_cal_sharp]]])
    med_gain = np.median(inf_cal, 0)
    gamma = float(np.quantile((inf_cal / med_gain).min(1), 0.10))
    print(f"[c5] thresholds per u: t_row {np.round(t_row, 4).tolist()} t_addr "
          f"{np.round(t_addr, 4).tolist()}; gamma {gamma:.3f} ({time.time() - t0:.0f}s)",
          flush=True)

    # ── training one node (value vector + its own adapter) on one concept ───
    groups = []                                  # stored concepts' anchors
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    adapter_bytes = [0]
    key_count = [0]

    def new_key():
        key_count[0] += 1
        return f"n{key_count[0]}"

    def emb_t(coords, vdict):
        out = torch.as_tensor(mu, device=dev)
        for f, j in enumerate(coords):
            out = out + (vdict[(f, j)] if (f, j) in vdict else torch.as_tensor(V[f][j], device=dev))
        return out

    def prof(Xv, coords):
        return scan(Xv, [list(coords)])[0]

    def train_node(Xtr_t, Xv, coords, f, j, params, lr_v):
        """The row (f, j)'s vector and the adapter `params`, on this concept's REAL images, with
        every stored concept that shares (f, j) held where it was (the r ~ 0 half); stops when the
        concept is absorbed at every decision level."""
        key = node_key[(f, j)]
        v = torch.as_tensor(V[f][j], device=dev).clone().requires_grad_(True)
        for q in params:
            q.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [v], "lr": lr_v}, {"params": params, "lr": args.lr}])
        share = [gr for gr in groups if gr["coords"][f] == j]
        gate = keys_on(kstate, keys_of(coords) | {key})
        used, ok = 0, False
        for step in range(args.max_steps):
            ix = torch.randint(0, len(Xtr_t), (args.batch,), generator=g).to(dev)
            e = emb_t(coords, {(f, j): v})[None].expand(args.batch, -1)
            with gate():
                loss = eps_loss(model, Xtr_t[ix], emb=e, gen=g)
            if share:
                # up to anchor_batch / 4 sharing concepts, 4 of each one's own samples
                gis = torch.randint(0, len(share), (max(1, args.anchor_batch // 4),),
                                    generator=g).unique()
                tot = 0.0
                for gi in gis:
                    gr = share[int(gi)]
                    jj = torch.randint(0, len(gr["x0"]), (4,), generator=g).to(dev)
                    tot = tot + anchor_loss(
                        model, gr["teacher"], gr["x0"][jj],
                        emb=emb_t(gr["coords"], {(f, j): v})[None].expand(4, -1),
                        t_emb=gr["emb_then"][None].expand(4, -1), gen=g,
                        teacher_gate=gr["tgate"], gate=gate_for(gr["coords"]))
                loss = loss + args.lam * tot / len(gis)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                v.mul_(base_norm[f] / max(float(v.norm()), 1e-8))
            V[f][j] = v.detach().cpu().numpy()
            used = step + 1
            if used % args.check_every == 0 and within(prof(Xv, coords), t_addr):
                ok = True
                break
        for q in params:
            q.requires_grad_(False)
        return used, ok

    node_params = {}                             # key -> its adapter parameters

    def snapshot():
        snap = copy.deepcopy(model).eval()
        sst = next(m.state for m in snap.modules() if isinstance(m, KeyedLoRA))
        return snap, sst

    def store_anchor(ci, coords, snap_sst):
        snap, sst = snap_sst
        e_then = emb_t(coords, {}).detach().clone()
        ax = ddim_sample(model, args.n_anchor_new, steps=args.steps, seed=900 + ci,
                         emb=e_then.cpu().numpy(), gate=gate_for(coords), clip=B.clip, **kw)
        ks = keys_of(coords)
        gr = dict(ci=ci, x0=to_img(torch.from_numpy(ax).to(dev), B.img, B.layout),
                  coords=tuple(coords), emb_then=e_then, teacher=snap,
                  tgate=keys_on(sst, ks) if ks else None)
        groups[:] = [x for x in groups if x["ci"] != ci] + [gr]

    # ── the stream ──────────────────────────────────────────────────────────
    leaves, records, total = [], [], 0
    rows = np.array([c for c, _ in B.seen])
    seen_of = {t: c for c, t in B.seen}

    def addr_cost():
        return 4 * sum(len(live(f)) for f in range(nF))

    for ci, (tup, pop) in enumerate(stream):
        tc = time.time()
        Xtr, Xv, ref = split(B.images(tup))
        Xtr_t = to_img(torch.as_tensor(B.to_m(Xtr), dtype=torch.float32, device=dev), B.img,
                       B.layout)
        rec = dict(i=ci, true=list(tup), pop=pop)
        combos = list(itertools.product(*[live(f) for f in range(nF)]))
        La = scan(Xv, [list(c) for c in combos])
        sa = score(La, "all")
        ka = int(np.argmin(sa))
        read = combos[ka]
        rr = rows
        if args.fast:
            top = [combos[i] for i in np.argsort(sa)[:10]]
            rr = np.array([seen_of[t] for t in top if t in seen_of] or [rows[0]])
        Lr = candidate_losses(model, Xv, U_GRID, cls=rr, seed=CRN, per_u=True, batch=cl_batch,
                              n_noise=n_noise, **kw)
        kr = int(np.argmin(score(Lr, "all")))
        rec.update(row=int(rr[kr]), read=list(read), row_ratio=(Lr[kr] / t_row).tolist(),
                   addr_ratio=(La[ka] / t_addr).tolist())
        cost, steps_w, coords, tier = 0, 0, tuple(read), None
        if within(Lr[kr], t_row):
            tier = "IN-DIST"
        else:
            # every open hypothesis of every leaf, on its own factor
            best = None
            for li, lf in enumerate(leaves):
                if not lf["open"]:
                    continue
                for f, hy in lf["hyps"].items():
                    others = [h for h in range(nF) if h != f]
                    cl = []
                    for vals in itertools.product(*[live(h) for h in others]):
                        c = [None] * nF
                        c[f] = hy["j"]
                        for h, jv in zip(others, vals):
                            c[h] = jv
                        cl.append(c)
                    Ll = score(scan(Xv, cl), "all")
                    k = int(np.argmin(Ll))
                    if best is None or Ll[k] < best[0]:
                        best = (float(Ll[k]), li, f, tuple(cl[k]))
            rec["leaf_best"] = None if best is None else dict(loss=best[0], leaf=best[1],
                                                               factor=B.names[best[2]])
            if best is not None and best[0] < float(sa[ka]):
                # MERGE: the winning hypothesis becomes a value node; the rest are dropped
                _, li, f, _ = best
                lf = leaves[li]
                # the concept's OTHER factors read with f LEFT OUT and no new node on -- as every
                # hypothesis was built; reading them through the one-concept node misread fg in
                # the smoke run, and a refine would then have taught the node that misreading
                others = [h for h in range(nF) if h != f]
                cl = []
                for vals in itertools.product(*[live(h) for h in others]):
                    c = [None] * nF
                    for h, jv in zip(others, vals):
                        c[h] = jv
                    cl.append(c)
                lo = score(scan(Xv, cl), "crit")
                coords = list(cl[int(np.argmin(lo))])
                coords[f] = lf["hyps"][f]["j"]
                coords = tuple(coords)
                for f2, hy in lf["hyps"].items():
                    if f2 == f:
                        status[(f2, hy["j"])] = "node"
                    else:
                        status[(f2, hy["j"])] = "dead"
                        drop_node(banks, hy["key"])
                        node_params.pop(hy["key"], None)
                        total -= 1024 + adapter_bytes[0]
                lf.update(open=False, merged=dict(factor=f, j=lf["hyps"][f]["j"], by=ci))
                src = lf["ci"]
                records[src]["coords"] = list(lf["hyps"][f]["coords"])
                records[src]["merged_on"] = B.names[f]
                snap = snapshot()
                store_anchor(src, lf["hyps"][f]["coords"], snap)
                rec.update(merge=dict(leaf=li, factor=B.names[f], src=src))
                pr = prof(Xv, coords)
                if within(pr, t_addr) or args.no_refine:
                    tier, cost = "ADDRESS", addr_cost()
                else:
                    key = node_key[(f, lf["hyps"][f]["j"])]
                    steps_w, ok = train_node(Xtr_t, Xv, coords, f, lf["hyps"][f]["j"],
                                             node_params[key], args.lr_v_w)
                    tier = "REFINE" if ok else "REFINE-cap"
            else:
                gains, loo = informativeness(Xv)
                gains = gains / med_gain
                rec["informativeness"] = gains.tolist()
                # one normaliser for both, so the IN-DIST quantile cannot move this rule
                cr = float((La[ka] / t_addr)[crit].max() / (Lr[kr] / t_addr)[crit].max())
                rec["compose_vs_retrieve"] = cr
                ok_addr = {"inform": gains.min() >= gamma, "absorb": within(La[ka], t_addr),
                           "retrieve": cr <= args.rho_cr}[args.address_rule]
                if ok_addr:
                    tier, cost = "ADDRESS", addr_cost()
                    nodes = [(f, j) for f, j in enumerate(read) if status.get((f, j)) == "node"]
                    if args.incorporate and nodes and not within(La[ka], t_addr):
                        f, j = nodes[0]
                        steps_w, ok = train_node(Xtr_t, Xv, tuple(read), f, j,
                                                 node_params[node_key[(f, j)]], args.lr_v_w)
                        tier = "ADDRESS+R" if ok else "ADDRESS+R*"
                else:
                    # LEAF: one hypothesis per factor, each on its own adapter (flat: one BLOCK,
                    # on the factor informativeness names, that no later concept can reach)
                    hyps, own = {}, {}
                    for f in ([int(np.argmin(gains))] if args.flat else range(nF)):
                        j = len(V[f])
                        V[f] = np.vstack([V[f], V[f][live(f)].mean(0)[None]])
                        key = new_key()
                        status[(f, j)], node_key[(f, j)] = "hyp", key
                        node_params[key] = add_node(banks, key, args.rank)
                        if not adapter_bytes[0]:
                            adapter_bytes[0] = 4 * sum(q.numel() for q in node_params[key])
                        c = list(loo[f])
                        c[f] = j
                        used, ok = train_node(Xtr_t, Xv, tuple(c), f, j, node_params[key],
                                              args.lr_v_w)
                        own[f] = float(prof(Xv, c).mean())
                        hyps[f] = dict(j=j, key=key, coords=tuple(c), steps=used, absorbed=ok)
                        origin[(f, j)] = len(leaves)
                        steps_w += used
                    tent = min(own, key=own.get)
                    leaves.append(dict(ci=ci, pop=pop, open=not args.flat, hyps=hyps, own=own,
                                       tent=tent, block=args.flat))
                    if args.flat:
                        status[(tent, hyps[tent]["j"])] = "block"
                    coords = hyps[tent]["coords"]
                    tier = "BLOCK" if args.flat else "LEAF"
                    cost = len(hyps) * (1024 + adapter_bytes[0])
                    rec["leaf"] = dict(idx=len(leaves) - 1, tent=B.names[tent],
                                       own={B.names[f]: v for f, v in own.items()},
                                       absorbed={B.names[f]: h["absorbed"] for f, h in hyps.items()})
        total += cost
        rec.update(tier=tier, cost_bytes=int(cost), steps_weight=int(steps_w),
                   bytes_total=int(total))
        if tier != "IN-DIST":
            rec["coords"] = list(coords)
        S = (B.to_d(ddim_sample(model, args.n_samples, steps=args.steps, seed=3000 + ci,
                                cls=rec["row"], clip=B.clip, **kw)) if tier == "IN-DIST" else
             B.to_d(ddim_sample(model, args.n_samples, steps=args.steps, seed=3000 + ci,
                                emb=emb_np(coords).astype(np.float32), gate=gate_for(coords),
                                clip=B.clip, **kw)))
        a = B.acc(S, tup)
        dc = density_coverage(S, ref)
        rec.update(acc=a, joint=a["joint"], coverage=dc["coverage"], density=dc["density"],
                   vendi=float(vendi_score(S)), secs=time.time() - tc)
        records.append(rec)
        if tier != "IN-DIST":
            store_anchor(ci, coords, snapshot() if tier in ("LEAF", "BLOCK", "REFINE", "REFINE-cap",
                                                            "ADDRESS+R", "ADDRESS+R*")
                         or not groups else (groups[-1]["teacher"], next(
                             m.state for m in groups[-1]["teacher"].modules()
                             if isinstance(m, KeyedLoRA))))
        extra = ""
        if "merge" in rec:
            extra = f" MERGE leaf {rec['merge']['leaf']} on {rec['merge']['factor']}"
        if "leaf" in rec:
            extra = f" LEAF {rec['leaf']['idx']} (tentative {rec['leaf']['tent']})"
        print(f"[c5] {ci:2d} {pop:9s} {tup} -> {tier:10s} read {tuple(read)} coords "
              f"{tuple(coords)}{extra} | row {max(rec['row_ratio']):.2f} addr "
              f"{max(rec['addr_ratio']):.2f} | {cost:>7d} B (total {total}) steps {steps_w} | "
              f"joint {a['joint']:.3f} cov {dc['coverage']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)

    # ── end of stream: an unmerged leaf keeps one hypothesis, as a BLOCK ────
    for li, lf in enumerate(leaves):
        if lf["open"]:
            for f2, hy in lf["hyps"].items():
                if f2 == lf["tent"]:
                    status[(f2, hy["j"])] = "block"
                else:
                    status[(f2, hy["j"])] = "dead"
                    drop_node(banks, hy["key"])
                    total -= 1024 + adapter_bytes[0]
            lf["open"] = False
            lf["block"] = True

    # ── retention and correctness, with every coordinate final ──────────────
    def sample_final(rec, seed):
        if rec["tier"] == "IN-DIST":
            return B.to_d(ddim_sample(model, args.n_samples, steps=args.steps, seed=seed,
                                      cls=rec["row"], clip=B.clip, **kw))
        c = rec["coords"]
        return B.to_d(ddim_sample(model, args.n_samples, steps=args.steps, seed=seed,
                                  emb=emb_np(c).astype(np.float32), gate=gate_for(c), clip=B.clip,
                                  **kw))

    for rec in records:
        rec["joint_end"] = B.acc(sample_final(rec, 3000 + rec["i"]), tuple(rec["true"]))["joint"]
        if rec["tier"] == "IN-DIST":
            continue
        c, t = rec["coords"], rec["true"]
        if rec["pop"] == "new-value":
            nf = B.new_f
            jn = c[nf]
            src_leaf = origin.get((nf, jn))
            rec["stored_ok"] = bool(all(c[h] == t[h] for h in range(nF) if h != nf)
                                    and jn >= n_base[nf] and status.get((nf, jn)) in ("node", "block")
                                    and src_leaf is not None
                                    and leaves[src_leaf]["pop"] == "new-value")
        else:
            rec["stored_ok"] = bool(tuple(c) == tuple(t))
    ro = [B.seen[i] for i in rng.permutation(len(B.seen))[:12]]
    cids = np.repeat([c for c, _ in ro], args.n_samples)

    def old_acc(m):
        S = B.to_d(ddim_sample(m, len(cids), steps=args.steps, seed=5000, cls=cids, clip=B.clip,
                               **kw))
        return float(np.mean([B.acc(S[i * args.n_samples:(i + 1) * args.n_samples], t)["joint"]
                              for i, (_, t) in enumerate(ro)]))

    # ── never-taught combinations through the new value's node ─────────────
    # Evaluation only (labels used to pick WHICH row stands for "the new value": the row on the new
    # factor made from the earliest new-value concept, as a node or a block).  The router's store has
    # no other coordinate for the new value; a store without such a row cannot address these at all.
    nf = B.new_f
    rows_new = sorted((leaves[origin[(nf, j)]]["ci"], j) for j in range(len(V[nf]))
                      if status.get((nf, j)) in ("node", "block") and (nf, j) in origin
                      and leaves[origin[(nf, j)]]["pop"] == "new-value")
    streamed = {tuple(t) for t, _ in stream}
    pool = [t for t in B.new_pairs if tuple(t) not in streamed]
    rng_u = np.random.default_rng(args.seed + 555)
    untaught = [pool[i] for i in rng_u.permutation(len(pool))[:args.n_untaught]]
    ut = []
    for t in untaught:
        if not rows_new:
            ut.append(dict(true=list(t), addressable=False, joint=0.0))
            continue
        c = list(t)
        c[nf] = rows_new[0][1]
        S = B.to_d(ddim_sample(model, args.n_samples, steps=args.steps, seed=7000 + len(ut),
                               emb=emb_np(c).astype(np.float32), gate=gate_for(c), clip=B.clip,
                               **kw))
        ut.append(dict(true=list(t), addressable=True, coords=c, joint=B.acc(S, tuple(t))["joint"],
                       status=status[(nf, c[nf])]))
    print(f"[c5] never-taught new-value combinations: {np.mean([u['addressable'] for u in ut]):.2f} "
          f"addressable, joint {np.mean([u['joint'] for u in ut]):.3f}", flush=True)
    merges = [lf for lf in leaves if lf.get("merged")]
    summary = dict(old_joint_start=old_acc(base), old_joint_end=old_acc(model),
                   bytes_total=int(total), adapter_bytes=int(adapter_bytes[0]),
                   n_leaves=len(leaves), n_merged=len(merges),
                   n_blocks=sum(1 for lf in leaves if lf.get("block")),
                   merges=[dict(src=lf["ci"], src_pop=lf["pop"], by=lf["merged"]["by"],
                                by_pop=records[lf["merged"]["by"]]["pop"],
                                factor=B.names[lf["merged"]["factor"]],
                                right_factor=bool(lf["pop"] == "new-value"
                                                  and lf["merged"]["factor"] == B.new_f))
                           for lf in merges],
                   basis_sizes=[len(live(f)) for f in range(nF)],
                   untaught=dict(n=len(ut), addressable=float(np.mean([u["addressable"] for u in ut])),
                                 joint=float(np.mean([u["joint"] for u in ut])), rows=ut),
                   minutes=(time.time() - t0) / 60)
    for pp in ("in-dist", "new-comb", "new-value"):
        rs = [r for r in records if r["pop"] == pp]
        if rs:
            tiers = {}
            for r in rs:
                tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
            summary[pp] = dict(n=len(rs), tiers=tiers,
                               joint=float(np.mean([r["joint"] for r in rs])),
                               joint_end=float(np.mean([r["joint_end"] for r in rs])),
                               stored_ok=float(np.mean([r.get("stored_ok", True) for r in rs])),
                               cost=int(sum(r["cost_bytes"] for r in rs)))
    print(f"[c5] SUMMARY {summary}", flush=True)
    save_json(dict(records=records, summary=summary, config=vars(args), stream=stream,
                   leaves=[{k: (v if k != "hyps" else {B.names[f]: dict(j=h["j"], coords=list(h["coords"]),
                                                                          steps=h["steps"],
                                                                          absorbed=h["absorbed"])
                                                        for f, h in v.items()})
                            for k, v in lf.items() if k != "own"} for lf in leaves],
                   t_row=t_row.tolist(), t_addr=t_addr.tolist(), gamma=gamma,
                   u_grid=U_GRID.tolist(), read_band=U_GRID[band].tolist(),
                   band_rate=None if band_rate is None else band_rate.tolist()),
              os.path.join(RESULTS, f"c5_router_cobweb_{args.bench}{args.tag}.json"),
              seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
