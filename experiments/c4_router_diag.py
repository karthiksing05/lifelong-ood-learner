"""C4 -- the router's decisions measured one at a time, on the untouched backbone.

C2 runs the loop end to end, so one wrong decision propagates: in FLI seed 0 the first new-value
concept was attributed to the FOREGROUND, and six later concepts reused that wrong node.  C4 scores
each decision on every concept of each population, with no state carried between concepts:

  D1  novel?            the best trained row's loss against t_row           in-dist vs the rest
  D2  address or node?  the best address's loss against t_addr, and the
                        smallest per-factor informativeness                  new-comb vs new-value
  D3  which factor?     (a) informativeness on the concept alone: the factor whose best known value
                            adds least is declared new
                        (b) the Cobweb LEAF: a node trained on ONE other new-value concept with its
                            factor left open, offered to the read on EVERY factor.  The factor the
                            read puts it on is the attribution; a new combination or a trained class
                            that reads it at all is a false merge

Two leaf forms (`--leaf-mode`):
  free  one leaf: the source's full read plus a free 1 KB delta, and one adapter.  If the novel value
        confounds the source's read of its OTHER factors, the delta absorbs those errors too.
  hyp   one leaf PER FACTOR HYPOTHESIS f: the source's other factors read with f LEFT OUT (so a novel
        value on f cannot bias them), a new value vector on f (norm of that factor's values), and its
        own adapter.  The second concept is offered each hypothesis on its own factor.

Why (b).  With an additive conditioning c = mu + sum_f V_f[v_f], a new direction on ANY factor fits
the concept that created it equally well: one concept cannot say which of its factors is new.  A
second concept that shares the new value can.  Offered on the right factor, the leaf leaves the second
concept's other factors to the known basis; offered on a wrong factor, it imposes the first concept's
value there.  Cobweb resolves the same ambiguity the same way: it creates a leaf, and merges when a
later instance shows what the leaf shares with it.

    python experiments/c4_router_diag.py --bench colormnist
    python experiments/c4_router_diag.py --bench shapes3d --fast --n-in 12 --n-comb 24 --n-new 24
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

from lifelong_ood.evals import save_json                              # noqa: E402
from lifelong_ood.neural.cobweb_update import (candidate_losses, ddim_sample,  # noqa: E402
                                               eps_loss, gate_on, inject_gated_lora, to_img)
from lifelong_ood.neural.idealwords import factorize                  # noqa: E402

U_GRID = np.round(np.arange(0.1, 1.0, 0.1), 2)
CRN = 4242


def auroc(neg, pos):
    """P(a positive scores above a negative), ties counted half."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if not len(neg) or not len(pos):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="colormnist", choices=("colormnist", "shapes3d"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-in", type=int, default=24)
    p.add_argument("--n-comb", type=int, default=30)
    p.add_argument("--n-new", type=int, default=40)
    p.add_argument("--n-sources", type=int, default=3, help="leaves, each from one new-value concept")
    p.add_argument("--leaf-steps", type=int, default=800)
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-v-w", type=float, default=0.005)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-ref", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--u-min-crit", type=float, default=0.3)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--n-cal", type=int, default=120)
    p.add_argument("--n-cal-sharp", type=int, default=24)
    p.add_argument("--lora-exclude", default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--leaf-mode", choices=("free", "hyp"), default="free")
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
    n_noise = 1 if args.fast else 2
    if not torch.cuda.is_available():
        raise SystemExit("[c4] refusing to run on CPU")
    dev = "cuda"
    torch.manual_seed(args.seed)
    t0 = time.time()

    B = load_bench(args.bench, seed=args.seed, device=dev)
    base = B.model
    for q in base.parameters():
        q.requires_grad_(False)
    kw = dict(layout=B.layout, img=B.img, device=dev)
    cl_batch = 256 if sh else 1024
    mu = np.asarray(B.fac["mu"], np.float32)
    V = [np.asarray(v, np.float32) for v in B.fac["V"]]
    nF = len(V)
    crit = U_GRID >= args.u_min_crit - 1e-9

    def split(X):
        n_ref = min(args.n_ref, len(X) - args.n_train - args.n_val)
        assert n_ref >= 8, f"only {len(X)} images"
        return X[:args.n_train], B.to_m(X[args.n_train:args.n_train + args.n_val]), X[-n_ref:]

    def emb_np(coords):
        return mu + sum(V[f][j] for f, j in enumerate(coords))

    def losses(model, Xv, E, gate=None):
        return candidate_losses(model, Xv, U_GRID, embs=np.asarray(E, np.float32), seed=CRN,
                                per_u=True, gate=gate, batch=cl_batch, n_noise=n_noise, **kw)

    def within(pr, t):
        return bool(np.all(np.asarray(pr)[crit] <= t[crit] * (1.0 + args.tol)))

    # ── calibration on trained classes, exactly as C2 ───────────────────────
    rng = np.random.default_rng(args.seed)
    cal = [B.seen[i] for i in rng.permutation(len(B.seen))[:args.n_cal]]
    cal = [(c, t, split(B.images(t))[1]) for c, t in cal]
    P_row = np.stack([candidate_losses(base, xv, U_GRID, cls=[c], seed=CRN, per_u=True,
                                       batch=cl_batch, n_noise=n_noise, **kw)[0] for c, t, xv in cal])
    tab = B.model.cemb.weight.detach().cpu().numpy()
    all_c = np.array([c for c, _ in B.seen])
    all_t = np.array([t for _, t in B.seen])
    P_loo = []
    for c, t, xv in cal:
        keep = all_c != c
        fi = factorize(tab[all_c][keep], all_t[keep])
        e = np.asarray(fi["mu"]) + sum(np.asarray(fi["V"][f])[t[f]] for f in range(nF))
        P_loo.append(losses(base, xv, e[None])[0])
    t_row = np.quantile(P_row, 0.9, axis=0)
    t_addr = np.quantile(np.stack(P_loo), 0.9, axis=0)

    def informativeness(Xv, with_reads=False):
        g_out, reads = [], []
        for f in range(nF):
            others = [h for h in range(nF) if h != f]
            cl = list(itertools.product(*[range(len(V[h])) for h in others]))
            E = [mu + sum(V[h][j] for h, j in zip(others, vals)) for vals in cl]
            lo = losses(base, Xv, E)[:, crit].mean(1)
            best = dict(zip(others, cl[int(np.argmin(lo))]))
            Ef = [mu + V[f][j] + sum(V[h][best[h]] for h in others) for j in range(len(V[f]))]
            lf = losses(base, Xv, Ef)[:, crit].mean(1)
            g_out.append(float((lo.min() - lf.min()) / lo.min()))
            reads.append(best)
        return (np.asarray(g_out), reads) if with_reads else np.asarray(g_out)

    inf_cal = np.stack([informativeness(xv) for _, _, xv in
                        [cal[i] for i in rng.permutation(len(cal))[:args.n_cal_sharp]]])
    med_gain = np.median(inf_cal, 0)
    gamma = float(np.quantile((inf_cal / med_gain).min(1), 0.10))
    print(f"[c4] {B.name}: basis {[len(v) for v in V]}; calibrated on {len(cal)} trained classes; "
          f"gamma {gamma:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # ── D1 / D2 / D3a: every concept, alone ─────────────────────────────────
    combos = list(itertools.product(*[range(len(v)) for v in V]))
    E_all = np.stack([emb_np(c) for c in combos])
    rows = np.array([c for c, _ in B.seen])
    seen_of = {t: c for c, t in B.seen}
    pops = ([("in-dist", t) for t in [B.seen[i][1] for i in rng.permutation(len(B.seen))[:args.n_in]]]
            + [("new-comb", B.held_comb[i]) for i in rng.permutation(len(B.held_comb))[:args.n_comb]]
            + [("new-value", B.new_pairs[i]) for i in rng.permutation(len(B.new_pairs))[:args.n_new]])
    recs, data = [], []
    for k, (pop, tup) in enumerate(pops):
        Xtr, Xv, _ = split(B.images(tup))
        La = losses(base, Xv, E_all)
        ka = int(np.argmin(La.mean(1)))
        if args.fast:
            top = [combos[i] for i in np.argsort(La.mean(1))[:10]]
            rr = np.array([seen_of[t] for t in top if t in seen_of] or [rows[0]])
        else:
            rr = rows
        Lr = candidate_losses(base, Xv, U_GRID, cls=rr, seed=CRN, per_u=True, batch=cl_batch,
                              n_noise=n_noise, **kw)
        kr = int(np.argmin(Lr.mean(1)))
        gains, loo = informativeness(Xv, with_reads=True)
        gains = gains / med_gain
        rec = dict(k=k, pop=pop, true=list(tup), read=list(combos[ka]),
                   loo_reads=[{int(h): int(j) for h, j in r.items()} for r in loo],
                   read_ok=bool(tuple(combos[ka]) == tuple(tup)),
                   row_ratio=float((Lr[kr] / t_row)[crit].max()),
                   addr_ratio=float((La[ka] / t_addr)[crit].max()),
                   addr_mean_ratio=float((La[ka] / t_addr)[crit].mean()),
                   s_addr=float(La[ka].mean()), in_dist=within(Lr[kr], t_row),
                   gains=gains.tolist(), inf_min=float(gains.min()),
                   inf_argmin=B.names[int(np.argmin(gains))])
        recs.append(rec)
        data.append((Xtr, Xv, La))
        print(f"[c4] {k:3d} {pop:9s} {tuple(tup)} read {combos[ka]} | row {rec['row_ratio']:6.2f} "
              f"addr {rec['addr_ratio']:6.2f} | inform min {rec['inf_min']:.2f} on "
              f"{rec['inf_argmin']} ({time.time() - t0:.0f}s)", flush=True)

    # ── D3b: Cobweb leaves ──────────────────────────────────────────────────
    excl = tuple(x for x in args.lora_exclude.split(":") if x)
    base_norm = [float(np.linalg.norm(v, axis=1).mean()) for v in V]

    def train_leaf(Xtr, Xv, e_base, f_norm=None, v_init=None, seed_=0):
        """A fresh gated adapter plus one trainable vector added to `e_base`: a free delta (init 0)
        when `f_norm` is None, else a value vector on factor `f_norm` (init `v_init`) held at that
        factor's value norm, as C2 holds every new value."""
        leaf = copy.deepcopy(base).eval()
        st = {"on": False}
        lp = inject_gated_lora(leaf, args.rank, st, exclude=excl)
        for q in lp:
            q.requires_grad_(True)
        gate = gate_on(st)
        eb = torch.as_tensor(np.asarray(e_base, np.float32), device=dev)
        v0 = np.zeros_like(e_base) if v_init is None else v_init
        v = torch.as_tensor(np.asarray(v0, np.float32), device=dev).clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [v], "lr": args.lr_v_w}, {"params": lp, "lr": args.lr}])
        Xtr_t = to_img(torch.as_tensor(B.to_m(Xtr), dtype=torch.float32, device=dev), B.img,
                       B.layout)
        g = torch.Generator(device="cpu").manual_seed(seed_)
        used, ok = 0, False
        for step in range(args.leaf_steps):
            j = torch.randint(0, len(Xtr_t), (args.batch,), generator=g).to(dev)
            with gate():
                loss = eps_loss(leaf, Xtr_t[j], emb=(eb + v)[None].expand(args.batch, -1), gen=g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if f_norm is not None:
                with torch.no_grad():
                    v.mul_(base_norm[f_norm] / max(float(v.norm()), 1e-8))
            used = step + 1
            if used % args.check_every == 0:
                pr = losses(leaf, Xv, (eb + v).detach().cpu().numpy()[None], gate=gate)[0]
                if within(pr, t_addr):
                    ok = True
                    break
        for q in lp:
            q.requires_grad_(False)
        return leaf, gate, v.detach().cpu().numpy(), used, ok

    def cands_on(f, e_f):
        """The leaf offered on factor f, whose value there is `e_f`; every other factor from the
        known basis.  -> [(f, coords, emb)]"""
        out = []
        others = [h for h in range(nF) if h != f]
        for vals in itertools.product(*[range(len(V[h])) for h in others]):
            c = [None] * nF
            c[f] = "L"
            for h, jv in zip(others, vals):
                c[h] = jv
            out.append((f, c, mu + e_f + sum(V[h][jv] for h, jv in zip(others, vals))))
        return out

    src_ix = [r["k"] for r in recs if r["pop"] == "new-value"][:args.n_sources]
    leaves, pairs = [], []
    for s in src_ix:
        Xtr, Xv, _ = data[s]
        true_s = tuple(recs[s]["true"])
        # entries: (model, gate, candidates); one for 'free', one per factor for 'hyp'
        entries, info = [], []
        if args.leaf_mode == "free":
            read1 = tuple(recs[s]["read"])
            leaf, gate, dL, used, ok = train_leaf(Xtr, Xv, emb_np(read1), seed_=args.seed + 5 + s)
            S = B.to_d(ddim_sample(leaf, args.n_samples, steps=args.steps, seed=11,
                                   emb=(emb_np(read1) + dL).astype(np.float32), gate=gate,
                                   clip=B.clip, **kw))
            own = B.acc(S, true_s)["joint"]
            cc = [c for f in range(nF) for c in cands_on(f, V[f][read1[f]] + dL)]
            entries.append((leaf, gate, cc))
            info.append(dict(f=None, read=list(read1), steps=used, absorbed=ok, own_joint=own,
                             vnorm=float(np.linalg.norm(dL))))
        else:
            for f in range(nF):
                loo = {int(h): int(jv) for h, jv in recs[s]["loo_reads"][f].items()}
                e_oth = mu + sum(V[h][loo[h]] for h in loo)
                leaf, gate, vf, used, ok = train_leaf(Xtr, Xv, e_oth, f_norm=f,
                                                      v_init=V[f].mean(0),
                                                      seed_=args.seed + 5 + s + 101 * f)
                S = B.to_d(ddim_sample(leaf, args.n_samples, steps=args.steps, seed=11,
                                       emb=(e_oth + vf).astype(np.float32), gate=gate,
                                       clip=B.clip, **kw))
                own = B.acc(S, true_s)["joint"]
                entries.append((leaf, gate, cands_on(f, vf)))
                info.append(dict(f=B.names[f], loo_read=loo, steps=used, absorbed=ok,
                                 own_joint=own))
        leaves.append(dict(src=s, true=list(true_s), mode=args.leaf_mode, parts=info))
        print(f"[c4] LEAF ({args.leaf_mode}) from {s} {true_s}: " + "; ".join(
            f"{p_['f'] or 'all'}: {p_['steps']} steps absorbed {p_['absorbed']} own joint "
            f"{p_['own_joint']:.3f}" for p_ in info) + f" ({time.time() - t0:.0f}s)", flush=True)
        for r, (Xtr_k, Xv_k, La_k) in zip(recs, data):
            if r["k"] == s:
                continue
            allc = []                                  # (f, coords, emb, loss, entry)
            for ei, (lm, gt, cc) in enumerate(entries):
                Lg = losses(lm, Xv_k, np.stack([c[2] for c in cc]), gate=gt).mean(1)
                allc += [(c[0], c[1], c[2], float(lv), ei) for c, lv in zip(cc, Lg)]
            best_f = [min(a[3] for a in allc if a[0] == f) for f in range(nF)]
            bi = min(range(len(allc)), key=lambda i: allc[i][3])
            fstar = allc[bi][0]
            s_un = float(La_k.mean(1).min())
            pr = dict(src=s, k=r["k"], pop=r["pop"], true=r["true"], best_ungated=s_un,
                      best_gated=best_f, fstar=B.names[fstar], merge=bool(allc[bi][3] < s_un),
                      ratio=float(allc[bi][3] / s_un), coords=allc[bi][1])
            if r["pop"] == "new-value":
                nf = B.new_f
                ti = min((i for i, a in enumerate(allc) if a[0] == nf), key=lambda i: allc[i][3])
                pr["coords_true_f"] = allc[ti][1]
                for tagk, ii in (("best", bi), ("true_f", ti)):
                    if tagk == "true_f" and ii == bi:
                        pr["joint_true_f"] = pr["joint_best"]
                        continue
                    lm, gt, _ = entries[allc[ii][4]]
                    S = B.to_d(ddim_sample(lm, args.n_samples, steps=args.steps, seed=13,
                                           emb=allc[ii][2].astype(np.float32), gate=gt,
                                           clip=B.clip, **kw))
                    pr[f"joint_{tagk}"] = B.acc(S, tuple(r["true"]))["joint"]
                # the right other values, too: is the leaf's error the factor or the read?
                pr["true_f_other_read_ok"] = bool(all(allc[ti][1][h] == r["true"][h]
                                                      for h in range(nF) if h != nf))
            pairs.append(pr)
        print(f"[c4]   leaf {s}: new-value targets read it on the true factor "
              f"{np.mean([q['fstar'] == B.names[B.new_f] for q in pairs if q['src'] == s and q['pop'] == 'new-value']):.2f}"
              f", merge {np.mean([q['merge'] for q in pairs if q['src'] == s and q['pop'] == 'new-value']):.2f}"
              f"; false merge new-comb "
              f"{np.mean([q['merge'] for q in pairs if q['src'] == s and q['pop'] == 'new-comb']):.2f}"
              f" in-dist {np.mean([q['merge'] for q in pairs if q['src'] == s and q['pop'] == 'in-dist']):.2f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        del entries
        torch.cuda.empty_cache()

    # ── summary ─────────────────────────────────────────────────────────────
    by = {pp: [r for r in recs if r["pop"] == pp] for pp in ("in-dist", "new-comb", "new-value")}
    nov = by["new-comb"] + by["new-value"]
    summ = dict(
        gamma=gamma, med_gain=med_gain.tolist(),
        D1_auroc_row=auroc([r["row_ratio"] for r in by["in-dist"]], [r["row_ratio"] for r in nov]),
        D1_indist_rate={pp: float(np.mean([r["in_dist"] for r in rs])) for pp, rs in by.items()},
        D2_auroc_addr=auroc([r["addr_ratio"] for r in by["new-comb"]],
                            [r["addr_ratio"] for r in by["new-value"]]),
        D2_auroc_inform=auroc([-r["inf_min"] for r in by["new-comb"]],
                              [-r["inf_min"] for r in by["new-value"]]),
        D2_gamma_rule={pp: float(np.mean([r["inf_min"] >= gamma for r in rs]))
                       for pp, rs in by.items()},
        D2_kappa_rule={f"{kap:g}": {pp: float(np.mean([r["addr_ratio"] <= kap for r in rs]))
                                    for pp, rs in by.items()} for kap in (1.5, 2, 3, 4, 6)},
        D2_read_ok={pp: float(np.mean([r["read_ok"] for r in rs])) for pp, rs in by.items()},
        D3a_inform_acc=float(np.mean([r["inf_argmin"] == B.names[B.new_f] for r in by["new-value"]])),
        leaves=leaves)
    nv = [q for q in pairs if q["pop"] == "new-value"]
    if nv:
        summ["D3b"] = dict(
            merge_rate=float(np.mean([q["merge"] for q in nv])),
            factor_acc_given_merge=float(np.mean([q["fstar"] == B.names[B.new_f]
                                                  for q in nv if q["merge"]] or [np.nan])),
            factor_acc_any=float(np.mean([q["fstar"] == B.names[B.new_f] for q in nv])),
            other_read_ok=float(np.mean([q["true_f_other_read_ok"] for q in nv])),
            joint_best=float(np.mean([q["joint_best"] for q in nv])),
            joint_true_f=float(np.mean([q["joint_true_f"] for q in nv])),
            false_merge={pp: float(np.mean([q["merge"] for q in pairs if q["pop"] == pp]))
                         for pp in ("in-dist", "new-comb")})
    print("[c4] SUMMARY", {k: v for k, v in summ.items() if k != "leaves"}, flush=True)
    save_json(dict(records=recs, pairs=pairs, summary=summ, config=vars(args),
                   t_row=t_row.tolist(), t_addr=t_addr.tolist(), u_grid=U_GRID.tolist()),
              os.path.join(RESULTS, f"c4_router_diag_{args.bench}_{args.leaf_mode}{args.tag}.json"), seed=args.seed)
    print(f"[c4] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
