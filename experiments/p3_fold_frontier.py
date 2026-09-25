"""GPU probe p3 (Phase 3): the FOLD frontier on the C1 stream.

Same protocol as experiments/legacy/c1_cobweb_update.py (ColorMNIST nobg3, 4 tasks x 3 concepts x 64
images of the withheld background, identity through the factor basis + one EXTEND direction, C1's
absorption criterion, taught / never-taught / old-class / drift evaluation), with two changes:

  * the WEIGHT phase trains an EXPERT from lifelong_ood.experts behind one interface
    (lora-all = E1, lora-b32 / lora-b16 = E2, film-all / film-512 / film-128 = E3, token-emb = E4,
    nonparam = E5), and every byte is charged by lifelong_ood.accounting;
  * the gate is either C1's LABEL gate (the control: on for new-value concepts, off elsewhere) or the
    ANALYTIC gate (the tree's responsibility for the new value's node, label-free), and sampling with
    the analytic gate uses the NATURAL gate at every step, so a sample enters the new value's basin
    only when the tree says so.

    sbatch --export=ALL,SCRIPT="experiments/p3_fold_frontier.py --seed 0 --arms lora-all@gate=label:lora-all" slurm/jobs/c_gpu.sbatch

Arms take knobs with '@': gate=label|analytic, lam=<anchor weight>, lr=<expert lr>.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.accounting import charge                             # noqa: E402
from lifelong_ood.backbones.layout import to_flat, to_img              # noqa: E402
from lifelong_ood.backbones.loader import Backbone                     # noqa: E402
from lifelong_ood.backbones.reads import eps_profile                   # noqa: E402
from lifelong_ood.backbones.sampler import _cond, sample               # noqa: E402
from lifelong_ood.evals import density_coverage, memorization, vendi_score  # noqa: E402
from lifelong_ood.experts.base import FiLMExpert, LoRAExpert, NonparamExpert, TokenExpert  # noqa: E402
from lifelong_ood.operators.address import calibrate, explained        # noqa: E402
from lifelong_ood.operators.anchor import sq_grad                      # noqa: E402
from lifelong_ood.operators.fold import (Gate, arrival_loss, anchor_loss_natural, base_profile,  # noqa: E402
                                         create_nodes, incorporate_rows, stop_rule)
from lifelong_ood.streams.bench import load_bench                      # noqa: E402
from lifelong_ood.tree.build import DEFAULT_GRID, build                # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402

CRN_SEED = 4242
ARMS = {
    "lora-all": dict(expert="lora", rank=1, band=None),
    "lora-all-c1": dict(expert="lora", rank=1, band=None, exclude=()),     # wraps the time embedding too: C1's 68,894 params
    "lora-b32": dict(expert="lora", rank=1, band=32),
    "lora-b16": dict(expert="lora", rank=1, band=16),
    "lora-b8": dict(expert="lora", rank=1, band=8),
    "film-all": dict(expert="film", channels="all"),
    "film-512": dict(expert="film", channels=512),
    "film-128": dict(expert="film", channels=128),
    "token-emb": dict(expert="token", rank=1),
    "token-only": dict(expert="token", rank=0),
    "nonparam": dict(expert="nonparam"),
}


def parse_arm(name, args):
    base, *toks = name.split("@")
    if base not in ARMS:
        raise SystemExit(f"unknown arm {base!r}; known: {sorted(ARMS)}")
    cfg = dict(ARMS[base], name=name, gate=args.gate, lam=args.lam, lr=args.lr_expert)
    for tok in toks:
        k, _, v = tok.partition("=")
        if k == "gate":
            cfg["gate"] = v
        elif k == "lam":
            cfg["lam"] = float(v)
        elif k == "lr":
            cfg["lr"] = float(v)
        else:
            raise SystemExit(f"unknown knob {tok!r} (gate/lam/lr)")
    return cfg


def pick_stream(rng, pairs, new_f, n_tasks, per_task, n_untaught):
    """C1's stream: tasks whose other-factor values are distinct inside each task."""
    pool = [tuple(p) for p in pairs]
    pool = [pool[i] for i in rng.permutation(len(pool))]
    others = [f for f in range(len(pool[0])) if f != new_f]
    tasks = []
    for _ in range(n_tasks):
        task, used = [], {f: set() for f in others}
        for c in list(pool):
            if len(task) >= per_task:
                break
            if any(c[f] in used[f] for f in others):
                continue
            task.append(c); pool.remove(c)
            for f in others:
                used[f].add(c[f])
        tasks.append(task)
    untaught, used = [], {f: set() for f in others}
    for c in list(pool):
        if len(untaught) >= n_untaught:
            break
        if all(c[f] in used[f] for f in others):
            continue
        untaught.append(c); pool.remove(c)
        for f in others:
            used[f].add(c[f])
    while len(untaught) < n_untaught and pool:
        untaught.append(pool.pop(0))
    return tasks, untaught


def make_expert(cfg, bk, key, Xtr_t, emb_rows, gate_label, gen):
    """Build the arm's expert; FiLM top-k picks channels by the arrival's gradient energy."""
    kind = cfg["expert"]
    if kind == "lora":
        kw = {} if "exclude" not in cfg else {"exclude": tuple(cfg["exclude"])}
        return LoRAExpert.create(bk, key, rank=cfg["rank"], band=cfg["band"], **kw)
    if kind == "token":
        return TokenExpert.create(bk, key, rank=cfg["rank"])
    if kind == "film":
        if cfg["channels"] == "all":
            return FiLMExpert.create(bk, key, channels="all")
        tmp = FiLMExpert.create(bk, key + "#probe", channels="all")
        params = tmp.parameters()

        def _one(i):
            j = int(torch.randint(0, len(Xtr_t), (1,), generator=gen))
            with tmp.on(1.0)():
                return arrival_loss(bk.model, tmp, gate_label, Xtr_t[j:j + 1], bk.layout, emb=emb_rows[j:j + 1],
                                    gen=gen, flag=np.ones(1), weight_by_r=False)
        for q in params:
            q.requires_grad_(True)
        g2 = sq_grad(params, _one, 48)
        for q in params:
            q.requires_grad_(False)
        energy = []
        for w, (gg, gb) in zip(tmp.wrappers, zip(g2[0::2], g2[1::2])):
            e = (gg + gb).detach().cpu().numpy()
            energy += [(float(v), w.name, int(c)) for c, v in enumerate(e)]
        energy.sort(reverse=True)
        chosen = {}
        for _, name, c in energy[:int(cfg["channels"])]:
            chosen.setdefault(name, []).append(c)
        tmp.drop()
        return FiLMExpert.create(bk, key, channels={k: np.asarray(sorted(v)) for k, v in chosen.items()})
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="colormnist", choices=("colormnist",))
    p.add_argument("--arms", default="lora-all@gate=label:lora-all", help="COLON-separated; knobs with '@'")
    p.add_argument("--gate", default="analytic", choices=("label", "analytic"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--per-task", type=int, default=3)
    p.add_argument("--n-untaught", type=int, default=8)
    p.add_argument("--n-old", type=int, default=12)
    p.add_argument("--n-drift-classes", type=int, default=120)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-ref", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ext-steps", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--anchor-batch", type=int, default=32)
    p.add_argument("--n-anchor-per-class", type=int, default=4)
    p.add_argument("--crn-noise", type=int, default=2)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--lr-expert", type=float, default=1e-3)
    p.add_argument("--lr-v", type=float, default=0.05)
    p.add_argument("--lr-v-w", type=float, default=0.005)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--u-min-crit", type=float, default=0.3)
    p.add_argument("--stop", default="c1", choices=("c1", "new"), help="stopping criterion: C1's loss rule or S1-S3")
    p.add_argument("--tree", default=None, help="tree .npz to reuse (default: build or load results/trees/tree_colormnist_s{seed}.npz)")
    p.add_argument("--n-tree-samples", type=int, default=512)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tasks, args.per_task, args.n_untaught = 2, 3, 4
        args.n_samples, args.ext_steps, args.max_steps, args.check_every = 12, 60, 150, 50
        args.n_ref, args.n_drift_classes, args.n_tree_samples = 60, 30, 128
        args.tag = args.tag or "_quick"
    u_grid = np.concatenate([np.round(np.arange(0.1, 1.0, 0.1), 2), [0.95, 0.99]])
    crit = u_grid >= args.u_min_crit - 1e-9
    if not torch.cuda.is_available():
        raise SystemExit("[p3] refusing to run on CPU")
    dev = "cuda"
    torch.manual_seed(args.seed)
    t_start = time.time()

    B = load_bench(args.bench, seed=args.seed, device=dev, probe_epochs=10 if args.quick else 30)
    base = B.model.eval()
    for q in base.parameters():
        q.requires_grad_(False)
    bk = Backbone(model=base, kind="unet", layout=B.layout, img=B.img, clip=B.clip, space="model",
                  null_id=int(base.null_id), emb_dim=int(base.emb_dim),
                  n_params=int(sum(q.numel() for q in base.parameters())), device=dev)
    kw = dict(layout=B.layout, img=B.img, device=dev)
    fac, nF, nf = B.fac, len(B.fac["V"]), B.new_f
    mu = np.asarray(fac["mu"], np.float32)
    Vs = [np.asarray(v, np.float32) for v in fac["V"]]
    v_norm = float(np.linalg.norm(Vs[nf], axis=1).mean())
    print(f"[p3] {B.name}: {len(B.seen)} classes, layout {B.layout}; new value {B.names[nf]}={B.new_j}; "
          f"probes {[(k, round(v, 3)) for k, v in B.probes.items() if k.endswith('val_acc')]}", flush=True)

    # ── the tree (label-free) and its calibration ───────────────────────────
    tpath = args.tree or os.path.join(ROOT, "results", "trees", f"tree_{args.bench}_s{args.seed}.npz")
    own = sample(bk, args.n_tree_samples, steps=args.steps, seed=args.seed)
    if os.path.exists(tpath):
        tree = Tree.load(tpath)
        print(f"[p3] tree loaded {tpath}: {tree.summary()}", flush=True)
    else:
        tree = build(bk, u_grid=DEFAULT_GRID, samples=own, seed=args.seed, iters=150, lr=0.1, steps=args.steps)
        os.makedirs(os.path.dirname(tpath), exist_ok=True)
        tree.save(tpath)
    tree_bytes0 = tree.bytes()
    own_cal = sample(bk, 256, steps=args.steps, seed=args.seed + 1)      # HELD-OUT own samples: never the build set
    cal = calibrate(tree, own_cal, q=0.95, seed=args.seed)
    print(f"[p3] calibration {cal.to_dict()}", flush=True)
    u_band = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    ref_profile = base_profile(bk, own_cal[:128], u_band, seed=CRN_SEED)

    # ── the stream, as C1 ───────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    tasks, untaught = pick_stream(rng, B.new_pairs, nf, args.tasks, args.per_task, args.n_untaught)
    old = [B.seen[i] for i in rng.permutation(len(B.seen))[:args.n_old]]
    print(f"[p3] stream {tasks}\n[p3] untaught {untaught}\n[p3] old {[t for _, t in old]}", flush=True)

    def row(tup):
        return mu + sum(Vs[f][tup[f]] for f in range(nF) if f != nf)

    split = {}
    for c in [c for t in tasks for c in t] + untaught:
        X = B.images(c)
        n_ref = min(args.n_ref, len(X) - args.n_train - args.n_val)
        split[c] = dict(train=X[:args.n_train], val=X[args.n_train:args.n_train + args.n_val], ref=X[-n_ref:])

    drift_cls = [B.seen[i] for i in rng.permutation(len(B.seen))[:args.n_drift_classes]]
    old_X, old_cls, old_comp = [], [], []
    for cid, tup in drift_cls:
        old_X.append(B.images(tup)[:4])
        old_cls += [cid] * 4
        old_comp.append(np.repeat((mu + sum(Vs[f][tup[f]] for f in range(nF)))[None], 4, 0))
    old_X = B.to_m(np.concatenate(old_X))
    old_cls = np.asarray(old_cls)
    old_comp = np.concatenate(old_comp).astype(np.float32)
    ncls = len(drift_cls)
    P_old0 = eps_profile(base, old_X, u_grid, cls=old_cls, n_noise=args.crn_noise, seed=CRN_SEED, per_image=True, **kw)
    P_comp0 = eps_profile(base, old_X, u_grid, emb=old_comp, n_noise=args.crn_noise, seed=CRN_SEED, per_image=True, **kw)
    t_u = np.quantile(P_comp0.reshape(ncls, 4, -1).mean(1), 0.9, axis=0)

    # anchors: the model's OWN samples of the original classes (+ the null branch), as C1
    a_cls = np.concatenate([np.repeat([c for c, _ in B.seen], args.n_anchor_per_class), np.full(8, base.null_id)])
    a_x0 = sample(bk, len(a_cls), steps=args.steps, seed=777 + args.seed, cls=a_cls)
    A_x0 = to_img(torch.from_numpy(a_x0).to(dev), B.img, B.layout)
    A_cls = torch.as_tensor(a_cls, device=dev)

    def sample_old(gate_fn):
        cids = np.repeat([c for c, _ in old], args.n_samples)
        S = B.to_d(sample(bk, len(cids), steps=args.steps, seed=5000 + args.seed, cls=cids, step_gate=gate_fn))
        out = []
        for i, (_, tup) in enumerate(old):
            a = B.acc(S[i * args.n_samples:(i + 1) * args.n_samples], tup)
            out.append([a["joint"]] + [a[nm] for nm in B.names])
        return np.asarray(out)

    old_acc0 = sample_old(None)
    print(f"[p3] old classes at theta_0: joint {old_acc0[:, 0].mean():.3f} ({time.time() - t_start:.0f}s)", flush=True)

    def is_in(prof):
        return bool(np.all(np.asarray(prof)[crit] <= t_u[crit] * (1.0 + args.tol)))

    for arm_name in args.arms.split(":"):
        cfg = parse_arm(arm_name, args)
        run_arm(cfg, args, dict(B=B, bk=bk, base=base, tree=tree, cal=cal, ref_profile=ref_profile, u_band=u_band,
                                tasks=tasks, untaught=untaught, split=split, row=row, v_norm=v_norm, t_u=t_u,
                                crit=crit, u_grid=u_grid, is_in=is_in, sample_old=sample_old, old_acc0=old_acc0,
                                old=old, old_X=old_X, old_cls=old_cls, P_old0=P_old0, A_x0=A_x0, A_cls=A_cls,
                                own=own_cal, tree_bytes0=tree_bytes0, kw=kw, dev=dev, nf=nf))
    print(f"[p3] done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


def run_arm(cfg, args, X_):
    B, bk, base, tree, cal = X_["B"], X_["bk"], X_["base"], X_["tree"], X_["cal"]
    tasks, untaught, split, row = X_["tasks"], X_["untaught"], X_["split"], X_["row"]
    kw, dev, u_grid, nf, crit = X_["kw"], X_["dev"], X_["u_grid"], X_["nf"], X_["crit"]
    name, mode = cfg["name"], cfg["gate"]
    print(f"\n[p3] ===== {name}  {cfg}", flush=True)
    t_arm = time.time()
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    v = torch.zeros(base.emb_dim, device=dev, requires_grad=True)
    key = f"fold:{name}"
    expert, node_ids, gate = None, [], None
    taught, anchors_new = [], []          # anchors_new: (x0 rows, targets) of absorbed concepts, charged
    base_sd = {k: t.detach().clone() for k, t in base.state_dict().items()
               if not any(s in k for s in (".A.", ".B.", "dgamma", "dbeta"))}
    t_u = X_["t_u"]

    def norm_match():
        with torch.no_grad():
            nv = float(v.norm())
            if nv > 1e-8:
                v.mul_(X_["v_norm"] / nv)

    def gate_ctx_for_eval():
        if expert is None:
            return None
        return (Gate(tree, node_ids, "forced") if mode == "label" else gate).step_gate(expert)

    def s_val(X_val_m, rows_np):
        emb = rows_np + v.detach().cpu().numpy()[None]
        if expert is None:
            return eps_profile(base, X_val_m, u_grid, emb=emb, n_noise=args.crn_noise, seed=CRN_SEED, **kw)
        from lifelong_ood.operators.fold import eps_profile_gated
        gg = Gate(tree, node_ids, "forced") if mode == "label" else gate
        return eps_profile_gated(bk, expert, gg, X_val_m, u_grid, emb=emb, seed=CRN_SEED, n_noise=args.crn_noise)

    for t, task in enumerate(tasks):
        t_task = time.time()
        rows_t = np.stack([row(c) for c in task]).astype(np.float32)
        Xtr = np.concatenate([split[c]["train"] for c in task])
        ridx = np.concatenate([np.full(len(split[c]["train"]), i) for i, c in enumerate(task)])
        Xval = B.to_m(np.concatenate([split[c]["val"] for c in task]))
        rval = np.concatenate([np.full(len(split[c]["val"]), i) for i, c in enumerate(task)])
        Xtr_m = B.to_m(Xtr)
        Xtr_t = to_img(torch.as_tensor(Xtr_m, dtype=torch.float32, device=dev), kw["img"], kw["layout"])
        rows_tt = torch.as_tensor(rows_t, device=dev)
        ridx_t = torch.as_tensor(ridx, device=dev)

        def new_loss_v(bs):
            j = torch.randint(0, len(Xtr_t), (bs,), generator=g).to(dev)
            emb = rows_tt[ridx_t[j]] + v[None]
            from lifelong_ood.operators.anchor import eps_loss
            return eps_loss(base, Xtr_t[j], emb=emb, gen=g)

        s_arr = s_val(Xval, rows_t[rval])
        decay = 1.0 / (1.0 + t)
        opt_v = torch.optim.Adam([v], lr=args.lr_v * decay)
        for _ in range(args.ext_steps):
            loss = new_loss_v(args.batch)
            opt_v.zero_grad(set_to_none=True)
            loss.backward()
            opt_v.step()
            norm_match()
        s_ext = s_val(Xval, rows_t[rval])
        absorbed = X_["is_in"](s_ext)
        tier, steps_w, stop_last = ("EXTEND" if absorbed else "NONE"), 0, None

        # ── the nodes and the expert ────────────────────────────────────────
        if node_ids:
            incorporate_rows(tree, node_ids, Xtr_m)
        else:
            node_ids = create_nodes(tree, cal, Xtr_m, key=key)
            print(f"[p3] {name}: new nodes {node_ids} at levels {[tree.nodes[i].level for i in node_ids]} "
                  f"(u {[tree.u_grid[tree.nodes[i].level] for i in node_ids]})", flush=True)
            gate = Gate(tree, node_ids, "analytic" if mode == "analytic" else "label")
        gate_label = Gate(tree, node_ids, "label")

        if not absorbed and cfg["expert"] == "nonparam":
            if expert is None:
                nid = node_ids[0]
                expert = NonparamExpert(key, tree.nodes[nid].mode, tree.nodes[nid].var, model=base)
            tier = "NONPARAM"
        elif not absorbed:
            if expert is None:
                emb_rows = rows_tt[ridx_t] + v.detach()[None]
                expert = make_expert(cfg, bk, key, Xtr_t, emb_rows, gate_label, g)
                print(f"[p3] {name}: expert {expert.describe()}", flush=True)
            params = expert.parameters()
            for q in params:
                q.requires_grad_(True)
            opt = torch.optim.Adam([{"params": params, "lr": cfg["lr"]}, {"params": [v], "lr": args.lr_v_w * decay}])
            flag_all = np.ones(args.batch, np.float32)
            for step in range(args.max_steps):
                j = torch.randint(0, len(Xtr_t), (args.batch,), generator=g).to(dev)
                emb = rows_tt[ridx_t[j]] + v[None]
                loss = arrival_loss(base, expert, gate, Xtr_t[j], kw["layout"], emb=emb, gen=g, flag=flag_all)
                if mode == "analytic" and cfg["lam"] > 0:
                    k = torch.randint(0, len(X_["A_x0"]), (args.anchor_batch,), generator=g).to(dev)
                    loss = loss + cfg["lam"] * anchor_loss_natural(base, expert, gate, X_["A_x0"][k], kw["layout"],
                                                                   cls=X_["A_cls"][k], gen=g)
                    for x0a, ea, tga in anchors_new:
                        kk = torch.randint(0, len(x0a), (max(4, args.anchor_batch // 4),), generator=g).to(dev)
                        loss = loss + cfg["lam"] * anchor_loss_natural(base, expert, gate, x0a[kk], kw["layout"],
                                                                       emb=ea[kk], gen=g)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                norm_match()
                steps_w = step + 1
                if steps_w % args.check_every == 0:
                    if args.stop == "c1":
                        if X_["is_in"](s_val(Xval, rows_t[rval])):
                            absorbed = True
                            break
                    else:
                        stop_last, _ = stop_rule(bk, tree, cal, expert, gate, node_ids, Xval, X_["own"][:64],
                                                 X_["ref_profile"], X_["u_band"], steps=args.steps, seed=args.seed,
                                                 X_train=Xtr_m, emb_val=rows_t[rval] + v.detach().cpu().numpy()[None])
                        if stop_last.absorbed:
                            absorbed = True
                            break
            for q in params:
                q.requires_grad_(False)
            tier = "WEIGHT" if absorbed else "WEIGHT-cap"
        taught += task
        s_fin = s_val(Xval, rows_t[rval])

        # the stop-rule diagnostics, always
        if expert is not None and cfg["expert"] != "nonparam":
            stop_last, S_forced = stop_rule(bk, tree, cal, expert, gate, node_ids, Xval, X_["own"][:64],
                                            X_["ref_profile"], X_["u_band"], steps=args.steps, seed=args.seed,
                                            X_train=Xtr_m, emb_val=rows_t[rval] + v.detach().cpu().numpy()[None])
            # anchors for this task's concepts: the gated model's own samples + their targets (charged)
            arows = np.repeat(rows_t + v.detach().cpu().numpy()[None], 8, 0)
            ax = sample(bk, len(arows), steps=args.steps, seed=900 + 10 * t + args.seed, emb=arows,
                        step_gate=gate_ctx_for_eval())
            anchors_new.append((to_img(torch.from_numpy(ax).to(dev), kw["img"], kw["layout"]),
                                torch.as_tensor(arows, device=dev), None))

        # ── evaluation (as C1) ──────────────────────────────────────────────
        vv = v.detach().cpu().numpy()
        conc = [("taught", c) for c in taught] + [("untaught", c) for c in untaught]
        E = np.concatenate([np.repeat((row(c) + vv)[None], args.n_samples, 0) for _, c in conc]).astype(np.float32)
        guide = None
        if cfg["expert"] == "nonparam" and expert is not None:
            guide = expert.guide(lambda xr, u: torch.as_tensor(gate.weights(xr, u), device=dev))
        S = B.to_d(sample(bk, len(E), steps=args.steps, seed=3000 + args.seed, emb=E, step_gate=None if guide else gate_ctx_for_eval(), guide=guide))
        per = []
        for i, (kind, c) in enumerate(conc):
            Si = S[i * args.n_samples:(i + 1) * args.n_samples]
            a = B.acc(Si, c)
            dc = density_coverage(Si, split[c]["ref"], space="inception", img=B.img, device=dev)
            dcp = density_coverage(Si, split[c]["ref"])
            rec = dict(kind=kind, concept=list(c), joint=a["joint"], new=a[B.names[nf]], acc={nm: a[nm] for nm in B.names},
                       coverage=dc["coverage"], density=dc["density"], coverage_pixel=dcp["coverage"],
                       vendi=float(vendi_score(Si)))
            if kind == "taught":
                rec["memorized"] = memorization(Si, split[c]["train"])["frac_memorized"]
            per.append(rec)
        oacc = X_["sample_old"](None if mode == "label" else gate_ctx_for_eval())
        oacc_off = oacc if mode == "label" else X_["sample_old"](None)
        # drift of the old classes' denoiser under the natural gate
        if expert is not None and mode == "analytic":
            from lifelong_ood.operators.fold import eps_profile_gated
            P = np.stack([eps_profile_gated(bk, expert, gate, X_["old_X"][i0:i0 + 64], u_grid, seed=CRN_SEED,
                                            n_noise=args.crn_noise) for i0 in range(0, len(X_["old_X"]), 64)]).mean(0)
            drift_u = ((P - X_["P_old0"].mean(0)) / X_["P_old0"].mean(0))
        else:
            drift_u = np.zeros(len(u_grid))
        # gate false-positive rate on old classes' real images, at the new nodes' levels
        fp = {}
        if gate is not None and mode == "analytic":
            for nid in node_ids:
                l = tree.nodes[nid].level
                from lifelong_ood.backbones.schedule import cosine_abar
                a = float(cosine_abar(tree.u_grid[l]))
                rr = np.random.default_rng(1)
                xt = np.sqrt(a) * X_["old_X"][:128] + np.sqrt(1 - a) * rr.standard_normal((min(128, len(X_["old_X"])), bk.dim)).astype(np.float32)
                fp[str(l)] = float(gate.weights(torch.as_tensor(xt), tree.u_grid[l]).mean())
        base_exact = all(torch.equal(base.state_dict()[k], t0) for k, t0 in base_sd.items())

        def mean(kind, k2):
            vals = [r[k2] for r in per if r["kind"] == kind and r.get(k2) == r.get(k2)]
            return float(np.mean(vals)) if vals else float("nan")

        o0 = X_["old_acc0"]
        bill = charge(experts=[expert] if (expert is not None and cfg["expert"] != "nonparam") else [],
                      tree=tree.bytes() - X_["tree_bytes0"] if node_ids else 0,
                      samples=[x0a for x0a, _, _ in anchors_new] + [ea for _, ea, _ in anchors_new],
                      extras=[("v_new", 4 * base.emb_dim), ("tree_shared", X_["tree_bytes0"])])
        metrics = dict(task=t, tier=tier, steps_weight=steps_w, absorbed=bool(absorbed),
                       s_arrival=float(s_arr.mean()), s_after_extend=float(s_ext.mean()), s_final=float(s_fin.mean()),
                       ratio_final_u=(s_fin / t_u).tolist(),
                       taught_joint=mean("taught", "joint"), taught_new=mean("taught", "new"),
                       taught_cov=mean("taught", "coverage"), taught_cov_pixel=mean("taught", "coverage_pixel"),
                       taught_vendi=mean("taught", "vendi"), taught_mem=mean("taught", "memorized"),
                       untaught_joint=mean("untaught", "joint"), untaught_new=mean("untaught", "new"),
                       untaught_cov=mean("untaught", "coverage"),
                       old_joint=float(oacc[:, 0].mean()), old_drop=float(o0[:, 0].mean() - oacc[:, 0].mean()),
                       old_joint_gate_off=float(oacc_off[:, 0].mean()),
                       old_factors={nm: float(oacc[:, 1 + k].mean()) for k, nm in enumerate(B.names)},
                       drift_u=drift_u.tolist(), drift_mean=float(np.mean(drift_u)),
                       drift_hi=float(np.mean(np.abs(drift_u[u_grid >= 0.9]))),
                       gate_fp_old=fp, base_exact=bool(base_exact), v_norm=float(v.detach().norm()),
                       nodes=[dict(id=i, level=tree.nodes[i].level, count=tree.nodes[i].count) for i in node_ids],
                       stop=None if stop_last is None else stop_last.__dict__,
                       expert_bytes=0 if expert is None or cfg["expert"] == "nonparam" else expert.bytes(),
                       secs_task=time.time() - t_task, per=per)
        ledger.record("p3_fold_frontier", bench=B.name, tag=args.tag, seed=args.seed, keys={"arm": name, "task": t},
                      metrics=metrics, bytes=bill, config=dict(vars(args), arm=cfg), script="experiments/p3_fold_frontier.py",
                      command=" ".join(sys.argv), seconds=time.time() - t_task,
                      backbone=dict(bench=B.name, n_params=bk.n_params, layout=bk.layout))
        print(f"[p3] {name} task {t}: {tier} steps {steps_w} | TAUGHT joint {metrics['taught_joint']:.3f} new "
              f"{metrics['taught_new']:.3f} cov {metrics['taught_cov']:.3f} | UNTAUGHT joint {metrics['untaught_joint']:.3f} "
              f"new {metrics['untaught_new']:.3f} | OLD {metrics['old_joint']:.3f} (drop {metrics['old_drop']:+.3f}; "
              f"gate off {metrics['old_joint_gate_off']:.3f}) drift {metrics['drift_mean']:+.4f} hi {metrics['drift_hi']:.4f} "
              f"| gate fp {fp} | bytes {bill['total']:,} (expert {metrics['expert_bytes']:,}) | base exact {base_exact} "
              f"| stop {None if stop_last is None else (stop_last.s1_ok, stop_last.s2_ok, stop_last.s3_ok, stop_last.mem_ok)}",
              flush=True)
    if expert is not None and cfg["expert"] != "nonparam":
        expert.drop()
    for nid in list(node_ids):
        tree.remove(nid)
    print(f"[p3] {name} finished in {(time.time() - t_arm) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
