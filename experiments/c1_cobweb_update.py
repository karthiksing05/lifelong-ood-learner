"""C1 -- can a Cobweb-form SPARSE WEIGHT UPDATE teach a backbone a factor value it has never rendered,
with no added structure and without disturbing what it already knows?  And once taught, does the
value become ADDRESSABLE -- do never-taught combinations that use it cost only their coordinates?

Runs on either benchmark through `c_bench.py`:
  colormnist  nobg3 UNet, new value bg=3 -- RESULTS 5q Phase B measured the wall: conditioning alone
              renders it at 0.000, and only an adapter on real images worked (0.862, 551K/concept)
  shapes3d    the pretrained eurecom backbone + 360-class table; new value = a GREY wall, a colour no
              Shapes3D image contains (`data/novel_value.py`)

Stream: `--tasks` x `--per-task` concepts using the new value, each taught from `--n-train` REAL images,
with `--n-val` more held back for the absorption check and a disjoint reference set for coverage.
Every concept is conditioned IN THE BASIS, c = mu + sum_{f != new} V_f[v_f] + v_new, with ONE learned
direction v_new for the new value, shared by every concept that uses it (the EXTEND tier), so after
each task `--n-untaught` never-taught combinations are ADDRESSED -- sampled from their coordinates
alone -- which tests whether the value entered the model's vocabulary.

Every arm shares the data, the conditioning, the sampler, the anchors and the stop rule; they differ
only in WHAT may move:

  emb-only          v_new only, weights frozen (the Phase B control)
  ft-full           + every backbone parameter
  ewc               + every parameter, diagonal-Fisher EWC penalty        (ewc@ewcLAMBDA)
  anchor-full       + every parameter, old concepts anchored              (anchor-full@lamLAMBDA)
  cw-sparse         + top-rho parameters by g_new^2 / (F_old + delta), anchored   (@rhoR@lamL@lrX)
  cw-sparse-noanc   same mask, no anchoring
  rand-mask         a random rho mask, anchored        -- is the SELECTION what matters?
  grad-mask         top-rho by g_new^2 alone, anchored -- is the F_old denominator what matters?
  cw-cond           cw-sparse confined to the conditioning projections (`.emb.` / `time_emb_proj`)
  cw-condout        ... plus the output convolution
  value-lora-r1/r8  one gated LoRA for the new VALUE: added structure, exact for old classes
  lora-consol       the value's LoRA as a FAST store, then CONSOLIDATED into the backbone by sparse,
                    anchored distillation from the gated model itself, and zeroed -- the Cobweb
                    restructure operator; zero added parameters after every task

The stop rule is the Cobweb one: train until the concept's denoising loss on its own held-back images
is at the level a trained class reaches through the basis, at EVERY noise level (`--criterion all`;
the mean over levels is dominated by low noise, where the input still carries the new colour, and
declared concepts absorbed that did not render it).  `--u-high` adds u = 0.95 and 0.99, where the
signal left in the input is too weak to carry the colour and only the conditioning can.

    python experiments/c1_cobweb_update.py --bench colormnist --seed 0
    python experiments/c1_cobweb_update.py --bench shapes3d --arms emb-only:cw-sparse:value-lora-r1
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c_bench import load_bench                                        # noqa: E402
from common import RESULTS                                            # noqa: E402

from lifelong_ood.evals import density_coverage, memorization, save_json, vendi_score  # noqa: E402
from lifelong_ood.neural.cobweb_update import (GatedLoRA, anchor_loss,  # noqa: E402
                                               ddim_sample, delta_stats, eps_loss, eps_profile,
                                               gate_on, inject_gated_lora, sq_grad, to_img,
                                               topk_mask)

CRN_SEED = 12345               # common random numbers for every eps profile, every arm, every task

ARMS = {
    "emb-only":        dict(weights="none"),
    "ft-full":         dict(weights="all", anchor=0.0),
    "ewc":             dict(weights="all", anchor=0.0, ewc=True),
    "anchor-full":     dict(weights="all", anchor=None),
    "cw-sparse":       dict(weights="mask", mask="ratio", anchor=None),
    "cw-sparse-noanc": dict(weights="mask", mask="ratio", anchor=0.0),
    "rand-mask":       dict(weights="mask", mask="random", anchor=None),
    "grad-mask":       dict(weights="mask", mask="grad", anchor=None),
    "cw-cond":         dict(weights="mask", mask="ratio", anchor=None, scope="cond"),
    "cw-condout":      dict(weights="mask", mask="ratio", anchor=None, scope="condout"),
    "value-lora-r1":   dict(weights="lora", rank=1, anchor=None),
    "value-lora-r8":   dict(weights="lora", rank=8, anchor=None),
    "lora-consol":     dict(weights="lora", rank=8, anchor=None, consolidate=True),
}


def parse_arm(name, args):
    """'cw-sparse@rho0.001@lam10@lr3e-4' -> the base config with those knobs overridden.

    Keys: rho (mask density), lam (anchor weight), lr (weight-phase lr), ewc (EWC lambda).  '@'
    rather than ',' because sbatch --export splits on commas.
    """
    base, *toks = name.split("@")
    if base not in ARMS:
        raise SystemExit(f"unknown arm {base!r}; known: {sorted(ARMS)}")
    cfg = dict(ARMS[base])
    cfg["name"] = name
    cfg["rho"] = args.rho
    cfg["ewc_lambda"] = args.ewc_lambda
    if cfg.get("anchor", 0.0) is None:
        cfg["anchor"] = args.lam
    cfg["lr"] = {"all": args.lr_w, "mask": args.lr_mask, "lora": args.lr_lora}.get(cfg["weights"])
    for tok in toks:
        k = next((kk for kk in ("rho", "lam", "lr", "ewc") if tok.startswith(kk)), None)
        if k is None:
            raise SystemExit(f"unknown knob {tok!r} in arm {name!r} (rho/lam/lr/ewc)")
        cfg[{"rho": "rho", "lam": "anchor", "lr": "lr", "ewc": "ewc_lambda"}[k]] = float(tok[len(k):])
    return cfg


def pick_stream(rng, pairs, new_f, n_tasks, per_task, n_untaught):
    """Tasks of new-value concepts whose OTHER factor values are distinct inside each task, so the
    shared direction is never taught against a constant context (n11's first failure: a direction
    taught under one foreground absorbed it).  Untaught concepts are spread over the other factors."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="colormnist", choices=("colormnist", "shapes3d"))
    p.add_argument("--arms", default="emb-only:ft-full:anchor-full:cw-sparse:value-lora-r1",
                   help="COLON-separated (sbatch --export splits on commas)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--per-task", type=int, default=3)
    p.add_argument("--n-untaught", type=int, default=8)
    p.add_argument("--n-old", type=int, default=12, help="old classes sampled for retention")
    p.add_argument("--n-drift-classes", type=int, default=120, help="old classes in the drift map")
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-ref", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--ext-steps", type=int, default=300, help="EXTEND (v_new only) steps per task")
    p.add_argument("--max-steps", type=int, default=800, help="weight-phase cap per task")
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--anchor-batch", type=int, default=32)
    p.add_argument("--n-anchor-per-class", type=int, default=None,
                   help="default 4 (colormnist, 90 classes) / 2 (shapes3d, 360)")
    p.add_argument("--n-anchor-new", type=int, default=8, help="anchors per absorbed concept")
    p.add_argument("--n-fisher", type=int, default=256)
    p.add_argument("--n-gnew", type=int, default=96)
    p.add_argument("--crn-noise", type=int, default=2)
    p.add_argument("--lam", type=float, default=1.0, help="anchor weight")
    p.add_argument("--rho", type=float, default=0.01, help="mask density")
    p.add_argument("--delta-frac", type=float, default=0.01,
                   help="delta in g^2/(F+delta), as a fraction of mean F_old")
    p.add_argument("--ewc-lambda", type=float, default=1.0)
    p.add_argument("--lr-w", type=float, default=1e-4, help="dense weight arms")
    p.add_argument("--lr-mask", type=float, default=1e-3, help="masked arms")
    p.add_argument("--lr-lora", type=float, default=1e-3)
    p.add_argument("--lr-v", type=float, default=0.05, help="EXTEND lr for v_new")
    p.add_argument("--lr-v-w", type=float, default=0.005, help="v_new lr during the weight phase")
    p.add_argument("--consol-steps", type=int, default=600)
    p.add_argument("--consol-per", type=int, default=16, help="teacher samples per taught concept")
    p.add_argument("--lr-consol", type=float, default=1e-3)
    p.add_argument("--lora-exclude", default=None,
                   help="colon-separated module-name substrings LoRA must not wrap "
                        "(default: none on colormnist, the time embedding on shapes3d)")
    p.add_argument("--no-count-decay", action="store_true",
                   help="disable the Cobweb 1/(1+t) decay of v_new's step as the value matures")
    p.add_argument("--criterion", choices=("all", "ratio", "mean"), default="all")
    p.add_argument("--u-high", action="store_true",
                   help="add u = 0.95 and 0.99 to the absorption grid")
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    sh = args.bench == "shapes3d"
    if args.n_anchor_per_class is None:
        args.n_anchor_per_class = 2 if sh else 4
    if args.lora_exclude is None:
        args.lora_exclude = "time_emb" if sh else ""
    if args.quick:
        args.tasks, args.per_task, args.n_untaught = 2, 3, 4
        args.n_samples, args.ext_steps, args.max_steps, args.check_every = 12, 60, 150, 50
        args.n_fisher, args.n_gnew, args.n_ref, args.n_drift_classes = 48, 32, 60, 30
        args.tag = args.tag or "_quick"
    u_grid = np.round(np.arange(0.1, 1.0, 0.1), 2)
    if args.u_high:
        u_grid = np.concatenate([u_grid, [0.95, 0.99]])
    args._u = u_grid
    if not torch.cuda.is_available():
        raise SystemExit("[c1] refusing to run on CPU (a slot can come without a usable GPU)")
    dev = "cuda"
    torch.manual_seed(args.seed)
    t_start = time.time()

    B = load_bench(args.bench, seed=args.seed, device=dev,
                   **({} if sh else {"probe_epochs": 10 if args.quick else 30}))
    base = B.model.eval()
    for q in base.parameters():
        q.requires_grad_(False)
    kw = dict(layout=B.layout, img=B.img, device=dev)
    fac, nF, nf = B.fac, len(B.fac["V"]), B.new_f
    mu = np.asarray(fac["mu"], np.float32)
    Vs = [np.asarray(v, np.float32) for v in fac["V"]]
    v_norm = float(np.linalg.norm(Vs[nf], axis=1).mean())
    print(f"[c1] {B.name}: {len(B.seen)} classes, layout {B.layout}, basis sizes {fac['sizes']} "
          f"R2 {fac['r2']:.3f}; new value {B.names[nf]}={B.new_j}, direction norm {v_norm:.3f}; "
          f"probes {[(k, round(v, 3)) for k, v in B.probes.items() if k.endswith('val_acc')]}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tasks, untaught = pick_stream(rng, B.new_pairs, nf, args.tasks, args.per_task, args.n_untaught)
    old = [B.seen[i] for i in rng.permutation(len(B.seen))[:args.n_old]]
    print(f"[c1] stream {tasks}\n[c1] untaught (addressed) {untaught}\n[c1] old retention "
          f"{[t for _, t in old]}", flush=True)

    def row(tup):
        return mu + sum(Vs[f][tup[f]] for f in range(nF) if f != nf)

    split = {}
    for c in [c for t in tasks for c in t] + untaught:
        X = B.images(c)
        n_ref = min(args.n_ref, len(X) - args.n_train - args.n_val)
        assert n_ref >= 16, f"only {len(X)} images for {c}"
        split[c] = dict(train=X[:args.n_train], val=X[args.n_train:args.n_train + args.n_val],
                        ref=X[-n_ref:])

    # old classes: 4 real images each (a subset of classes on shapes3d), for drift and threshold
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
    P_old0 = eps_profile(base, old_X, u_grid, cls=old_cls, n_noise=args.crn_noise, seed=CRN_SEED,
                         per_image=True, **kw)
    P_null0 = eps_profile(base, old_X, u_grid, cls=None, n_noise=args.crn_noise, seed=CRN_SEED,
                          per_image=True, **kw)
    P_comp0 = eps_profile(base, old_X, u_grid, emb=old_comp, n_noise=args.crn_noise,
                          seed=CRN_SEED, per_image=True, **kw)
    t90_row = np.quantile(P_old0.reshape(ncls, 4, -1).mean(1), 0.9, axis=0)
    t90_comp = np.quantile(P_comp0.reshape(ncls, 4, -1).mean(1), 0.9, axis=0)
    thr = float(t90_comp.mean())
    args._t_u = t90_comp
    print(f"[c1] absorption threshold (old classes through the BASIS, q90) per u "
          f"{np.round(t90_comp, 4).tolist()} at u {u_grid.tolist()}; criterion {args.criterion}",
          flush=True)

    # anchors for the original classes: the model's OWN samples of them, plus the null branch
    n_apc = args.n_anchor_per_class
    a_cls = np.concatenate([np.repeat([c for c, _ in B.seen], n_apc),
                            np.full(max(8, 2 * n_apc), base.null_id)])
    a_x0 = ddim_sample(base, len(a_cls), steps=args.steps, seed=777 + args.seed, cls=a_cls,
                       clip=B.clip, **kw)
    A0 = dict(x0=to_img(torch.from_numpy(a_x0).to(dev), B.img, B.layout),
              cls=torch.as_tensor(a_cls, device=dev), rows=None, v_old=None, teacher=base,
              tgate=None, sgate=None, tag="original")

    # F_old over the original classes, once, at theta_0
    names_w = [n for n, _ in base.named_parameters() if not n.startswith("cemb")]
    params0 = dict(base.named_parameters())
    for n in names_w:
        params0[n].requires_grad_(True)
    g_f = torch.Generator(device="cpu").manual_seed(args.seed + 99)
    order = torch.randint(0, len(a_cls), (args.n_fisher,), generator=g_f)

    def _old_loss(i):
        j = int(order[i])
        return eps_loss(base, A0["x0"][j:j + 1], cls=A0["cls"][j:j + 1], gen=g_f)

    F_old = sq_grad([params0[n] for n in names_w], _old_loss, args.n_fisher)
    for n in names_w:
        params0[n].requires_grad_(False)
    F_old = dict(zip(names_w, F_old))
    F_mean = float(torch.cat([f.reshape(-1) for f in F_old.values()]).mean())
    print(f"[c1] F_old over {args.n_fisher} anchor examples, mean {F_mean:.3e} "
          f"({time.time() - t_start:.0f}s)", flush=True)

    def sample_old(model):
        cids = np.repeat([c for c, _ in old], args.n_samples)
        S = B.to_d(ddim_sample(model, len(cids), steps=args.steps, seed=5000 + args.seed,
                               cls=cids, clip=B.clip, **kw))
        out = []
        for i, (_, tup) in enumerate(old):
            a = B.acc(S[i * args.n_samples:(i + 1) * args.n_samples], tup)
            out.append([a["joint"]] + [a[nm] for nm in B.names])
        return np.asarray(out)          # [n_old, (joint, factor...)]

    old_acc0 = sample_old(base)
    print(f"[c1] old classes at theta_0: joint {old_acc0[:, 0].mean():.3f} "
          f"{dict(zip(B.names, np.round(old_acc0[:, 1:].mean(0), 3)))}", flush=True)

    ctx = dict(B=B, base=base, A0=A0, F_old=F_old, F_mean=F_mean, names_w=names_w, tasks=tasks,
               untaught=untaught, split=split, row=row, v_norm=v_norm, thr=thr,
               sample_old=sample_old, old_acc0=old_acc0, old_X=old_X, old_cls=old_cls,
               P_old0=P_old0, P_null0=P_null0, kw=kw, dev=dev)
    rows = []
    for arm in args.arms.split(":"):
        rows += run_arm(parse_arm(arm, args), args, ctx)
        save_json(dict(rows=rows, config={k: v for k, v in vars(args).items()
                                          if not k.startswith("_")},
                       bench=B.name, names=list(B.names), new_factor=B.names[nf], tasks=tasks,
                       untaught=untaught, old=[t for _, t in old], u_grid=u_grid.tolist(),
                       basis_r2=float(fac["r2"]), threshold=thr, t90_comp=t90_comp.tolist(),
                       t90_row=t90_row.tolist(), old_acc0=old_acc0.tolist()),
                  os.path.join(RESULTS, f"c1_cobweb_update_{B.name}{args.tag}.json"),
                  seed=args.seed)
    print(f"[c1] done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


def run_arm(cfg, args, X_):
    B, base, A0, F_old, F_mean = X_["B"], X_["base"], X_["A0"], X_["F_old"], X_["F_mean"]
    tasks, untaught, split, row = X_["tasks"], X_["untaught"], X_["split"], X_["row"]
    kw, dev, u_grid, nf = X_["kw"], X_["dev"], args._u, B.new_f
    name, W = cfg["name"], cfg["weights"]
    print(f"\n[c1] ===== {name}  {cfg}", flush=True)
    t_arm = time.time()
    model = copy.deepcopy(base).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    state = {"on": False}
    excl = tuple(x for x in args.lora_exclude.split(":") if x)
    lora = inject_gated_lora(model, cfg["rank"], state, exclude=excl) if W == "lora" else []
    for q in lora:                  # trainable only in the weight phase, never during EXTEND
        q.requires_grad_(False)
    vgate = gate_on(state) if W == "lora" else None
    named = dict(model.named_parameters())
    if W == "lora":
        # base weights live under `.base.` once wrapped; compare them against theta_0 by that name
        base_names = dict(base.named_parameters())
        wmap = {n: n.replace(".base.", ".") for n in named if ".base." in n}
        wmap.update({n: n for n in named if n in base_names and not n.startswith("cemb")})
    else:
        wmap = {n: n for n in X_["names_w"]}
    theta0 = {n: named[n].detach().clone() for n in wmap}
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    v = torch.zeros(base.emb_dim, device=dev, requires_grad=True)
    # anchor groups: a LoRA leaves the old classes exact -- unless it is later folded into them
    groups = [A0] if (W != "lora" or cfg.get("consolidate")) else []
    taught, out = [], []
    t_u = np.asarray(args._t_u)

    def norm_match():
        with torch.no_grad():
            nv = float(v.norm())
            if nv > 1e-8:
                v.mul_(X_["v_norm"] / nv)

    def is_in(prof):
        if args.criterion == "all":
            return bool(np.all(prof <= t_u))
        if args.criterion == "ratio":
            return bool(np.mean(prof / t_u) <= 1.0)
        return float(prof.mean()) <= float(t_u.mean())

    def s_val(X_val_m, rows_np):
        emb = rows_np + v.detach().cpu().numpy()[None]
        return eps_profile(model, X_val_m, u_grid, emb=emb, n_noise=args.crn_noise,
                           seed=CRN_SEED, gate=vgate, **kw)

    def anchor_term(n_tot, new_only=False):
        orig = [] if new_only else [gr for gr in groups if gr["tag"] == "original"]
        new = [gr for gr in groups if gr["tag"] != "original"]
        if not (orig or new) or n_tot <= 0:
            return torch.zeros((), device=dev)
        # half the batch from the original classes (when present), half from absorbed concepts
        plan = [(orig, n_tot // 2), (new, n_tot - n_tot // 2)] if (orig and new) \
            else [(orig or new, n_tot)]
        tot, cnt = torch.zeros((), device=dev), 0
        for grs, k in plan:
            gi = torch.randint(0, len(grs), (k,), generator=g)
            for gidx in torch.unique(gi):
                gr = grs[int(gidx)]
                m = int((gi == gidx).sum())
                j = torch.randint(0, len(gr["x0"]), (m,), generator=g).to(dev)
                if gr["rows"] is not None:
                    # an absorbed concept: CURRENT direction for the student, the one it was
                    # absorbed with for the teacher -- so v_new cannot drift away from it either
                    emb, t_emb = gr["rows"][j] + v[None], gr["rows"][j] + gr["v_old"][None]
                else:
                    emb, t_emb = None, None
                tot = tot + m * anchor_loss(model, gr["teacher"], gr["x0"][j],
                                            cls=None if gr["cls"] is None else gr["cls"][j],
                                            emb=emb, t_emb=t_emb, gen=g,
                                            teacher_gate=gr["tgate"], gate=gr["sgate"])
                cnt += m
        return tot / max(cnt, 1)

    def is_cond(n):
        return ".emb." in n or "time_emb_proj" in n

    def is_out(n):
        return n.startswith("out.") or "conv_out" in n

    for t, task in enumerate(tasks):
        t_task = time.time()
        mask_layers = None
        rows_t = np.stack([row(c) for c in task]).astype(np.float32)
        Xtr = np.concatenate([split[c]["train"] for c in task])
        ridx = np.concatenate([np.full(len(split[c]["train"]), i) for i, c in enumerate(task)])
        Xval = B.to_m(np.concatenate([split[c]["val"] for c in task]))
        rval = np.concatenate([np.full(len(split[c]["val"]), i) for i, c in enumerate(task)])
        Xtr_t = to_img(torch.as_tensor(B.to_m(Xtr), dtype=torch.float32, device=dev),
                       kw["img"], kw["layout"])
        rows_tt = torch.as_tensor(rows_t, device=dev)
        ridx_t = torch.as_tensor(ridx, device=dev)

        def new_loss(bs):
            j = torch.randint(0, len(Xtr_t), (bs,), generator=g).to(dev)
            emb = rows_tt[ridx_t[j]] + v[None]
            if vgate is None:
                return eps_loss(model, Xtr_t[j], emb=emb, gen=g)
            with vgate():
                return eps_loss(model, Xtr_t[j], emb=emb, gen=g)

        s_arr = s_val(Xval, rows_t[rval])
        # ── EXTEND: the new value's direction alone, weights frozen ─────────
        decay = 1.0 if args.no_count_decay else 1.0 / (1.0 + t)
        opt_v = torch.optim.Adam([v], lr=args.lr_v * decay)
        for _ in range(args.ext_steps):
            loss = new_loss(args.batch)
            if cfg.get("anchor", 0.0) > 0:
                # the direction is shared: moving it for this task must not undo earlier ones
                loss = loss + cfg["anchor"] * anchor_term(args.anchor_batch, new_only=True)
            opt_v.zero_grad(set_to_none=True)
            loss.backward()
            opt_v.step()
            norm_match()
        s_ext = s_val(Xval, rows_t[rval])
        absorbed = is_in(s_ext)
        tier, steps_w = ("EXTEND" if absorbed else "NONE"), 0
        prev = {n: named[n].detach().clone() for n in wmap}
        lora_prev = [q.detach().clone() for q in lora]

        # ── WEIGHT: only if the direction alone could not absorb the concepts ─
        if not absorbed and W != "none":
            masks = None
            if W == "all":
                train_p, lr = [named[n] for n in wmap], cfg["lr"]
            elif W == "lora":
                train_p, lr = lora, cfg["lr"]
            else:
                scope = cfg.get("scope")
                cand = [n for n in wmap if scope is None or is_cond(n)
                        or (scope == "condout" and is_out(n))]
                cp = [named[n] for n in cand]
                if cfg["mask"] == "random":
                    sc = [torch.rand(q.shape, generator=g).to(dev) for q in cp]
                else:
                    for q in cp:
                        q.requires_grad_(True)
                    gtr = torch.Generator(device="cpu").manual_seed(args.seed + 7 + t)

                    def _new1(i):
                        j = int(torch.randint(0, len(Xtr_t), (1,), generator=gtr))
                        emb = rows_tt[ridx_t[j:j + 1]] + v.detach()[None]
                        return eps_loss(model, Xtr_t[j:j + 1], emb=emb, gen=gtr)

                    gnew = sq_grad(cp, _new1, args.n_gnew)
                    for q in cp:
                        q.requires_grad_(False)
                    if cfg["mask"] == "grad":
                        sc = gnew
                    else:
                        dl = args.delta_frac * F_mean
                        sc = [gn / (F_old[wmap[n]] + dl) for gn, n in zip(gnew, cand)]
                # rho is a fraction of the WHOLE backbone, whatever the candidate scope, so arms
                # with different scopes move the same number of entries
                k_all = sum(named[n].numel() for n in wmap)
                k_cand = sum(q.numel() for q in cp)
                mk = topk_mask(sc, min(1.0, cfg["rho"] * k_all / max(k_cand, 1)))
                sel = [(n, m) for n, m in zip(cand, mk) if bool(m.any())]
                byl = {}
                for n, m in sel:
                    grp = ("cond" if is_cond(n) else "out" if is_out(n) else
                           "attn" if (".qkv." in n or "attn" in n or n.split(".")[-2] == "out")
                           else "temb" if (n.startswith("temb") or "time_embedding" in n)
                           else "conv/norm")
                    byl[grp] = byl.get(grp, 0) + int(m.sum())
                mask_layers = {k2: v2 / max(1, sum(byl.values())) for k2, v2 in byl.items()}
                train_p = [named[n] for n, _ in sel]
                masks = {id(named[n]): m.to(named[n].dtype) for n, m in sel}
                lr = cfg["lr"]
            for q in train_p:
                q.requires_grad_(True)
            opt = torch.optim.Adam([{"params": train_p, "lr": lr},
                                    {"params": [v], "lr": args.lr_v_w * decay}])
            # entries outside THIS task's mask stay where the task found them -- not at theta_0,
            # which would silently undo whatever earlier tasks' masks had written there
            ref = {id(named[n]): prev[n] for n in wmap} if masks is not None else None
            for step in range(args.max_steps):
                loss = new_loss(args.batch)
                if cfg.get("anchor", 0.0) > 0:
                    loss = loss + cfg["anchor"] * anchor_term(args.anchor_batch)
                if cfg.get("ewc"):
                    pen = sum((F_old[wmap[n]] / F_mean * (named[n] - theta0[n]) ** 2).sum()
                              for n in wmap)
                    loss = loss + 0.5 * cfg["ewc_lambda"] * pen
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if masks is not None:
                    for q in train_p:
                        if q.grad is not None:
                            q.grad.mul_(masks[id(q)])
                opt.step()
                norm_match()
                if masks is not None:
                    with torch.no_grad():     # entries outside the mask may never move, exactly
                        for q in train_p:
                            m = masks[id(q)]
                            q.copy_(m * q + (1 - m) * ref[id(q)])
                steps_w = step + 1
                if steps_w % args.check_every == 0 and is_in(s_val(Xval, rows_t[rval])):
                    absorbed = True
                    break
            for q in train_p:
                q.requires_grad_(False)
            tier = "WEIGHT" if absorbed else "WEIGHT-cap"
        taught += task

        # ── CONSOLIDATE: fold the value's adapter into the BACKBONE, then drop it ──
        cons = None
        if cfg.get("consolidate") and W == "lora" and tier.startswith("WEIGHT"):
            t_c = time.time()
            teacher = copy.deepcopy(model).eval()               # the gated model: the fast store
            tst = next(m_.state for m_ in teacher.modules() if isinstance(m_, GatedLoRA))
            tgate = gate_on(tst)
            with torch.no_grad():                               # the student may not lean on it
                for m_ in model.modules():
                    if isinstance(m_, GatedLoRA):
                        m_.B.weight.zero_()
            crow = np.stack([row(c) for c in taught]).astype(np.float32)
            cemb = np.repeat(crow + v.detach().cpu().numpy()[None], args.consol_per, 0)
            cx = ddim_sample(teacher, len(cemb), steps=args.steps, seed=1300 + 10 * t + args.seed,
                             emb=cemb, gate=tgate, clip=B.clip, **kw)
            CX = to_img(torch.from_numpy(cx).to(dev), kw["img"], kw["layout"])
            CE = torch.as_tensor(cemb, device=dev)
            bnames = list(wmap)                                 # the backbone's own weights
            cp = [named[n] for n in bnames]
            for q in cp:
                q.requires_grad_(True)
            gc = torch.Generator(device="cpu").manual_seed(args.seed + 31 + t)

            def _d1(i):
                j = int(torch.randint(0, len(CX), (1,), generator=gc))
                return anchor_loss(model, teacher, CX[j:j + 1], emb=CE[j:j + 1], gen=gc,
                                   teacher_gate=tgate)

            gd = sq_grad(cp, _d1, args.n_gnew)
            for q in cp:
                q.requires_grad_(False)
            dl = args.delta_frac * F_mean
            mk = topk_mask([g_ / (F_old[wmap[n]] + dl) for g_, n in zip(gd, bnames)], cfg["rho"])
            sel = [(n, m_) for n, m_ in zip(bnames, mk) if bool(m_.any())]
            cparams = [named[n] for n, _ in sel]
            cmask = {id(named[n]): m_.to(named[n].dtype) for n, m_ in sel}
            cref = {id(named[n]): named[n].detach().clone() for n, _ in sel}
            for q in cparams:
                q.requires_grad_(True)
            copt = torch.optim.Adam(cparams, lr=args.lr_consol)
            for _ in range(args.consol_steps):
                j = torch.randint(0, len(CX), (args.batch,), generator=gc).to(dev)
                loss = anchor_loss(model, teacher, CX[j], emb=CE[j], gen=gc, teacher_gate=tgate)
                if cfg.get("anchor", 0.0) > 0:
                    loss = loss + cfg["anchor"] * anchor_term(args.anchor_batch)
                copt.zero_grad(set_to_none=True)
                loss.backward()
                for q in cparams:
                    if q.grad is not None:
                        q.grad.mul_(cmask[id(q)])
                copt.step()
                with torch.no_grad():
                    for q in cparams:
                        q.copy_(cmask[id(q)] * q + (1 - cmask[id(q)]) * cref[id(q)])
            for q in cparams:
                q.requires_grad_(False)
            cons = dict(steps=args.consol_steps, secs=time.time() - t_c,
                        s_after=float(np.mean(s_val(Xval, rows_t[rval]) / t_u)))
            del teacher
        s_fin = s_val(Xval, rows_t[rval])

        # ── anchors for what was just absorbed, against THIS snapshot ────────
        snap = copy.deepcopy(model).eval()
        sstate = next((m.state for m in snap.modules() if isinstance(m, GatedLoRA)), None)
        arows = np.repeat(rows_t + v.detach().cpu().numpy()[None], args.n_anchor_new, 0)
        ax = ddim_sample(model, len(arows), steps=args.steps, seed=900 + 10 * t + args.seed,
                         emb=arows, gate=vgate, clip=B.clip, **kw)
        groups.append(dict(x0=to_img(torch.from_numpy(ax).to(dev), kw["img"], kw["layout"]),
                           cls=None, rows=torch.as_tensor(np.repeat(rows_t, args.n_anchor_new, 0),
                                                          device=dev),
                           v_old=v.detach().clone(), teacher=snap,
                           tgate=gate_on(sstate) if sstate is not None else None,
                           sgate=vgate, tag=f"task{t}"))

        # ── evaluation ──────────────────────────────────────────────────────
        vv = v.detach().cpu().numpy()
        conc = [("taught", c) for c in taught] + [("untaught", c) for c in untaught]
        E = np.concatenate([np.repeat((row(c) + vv)[None], args.n_samples, 0)
                            for _, c in conc]).astype(np.float32)
        S = B.to_d(ddim_sample(model, len(E), steps=args.steps, seed=3000 + args.seed, emb=E,
                               gate=vgate, clip=B.clip, **kw))
        per = []
        for i, (kind, c) in enumerate(conc):
            Si = S[i * args.n_samples:(i + 1) * args.n_samples]
            a = B.acc(Si, c)
            dc = density_coverage(Si, split[c]["ref"])
            rec = dict(kind=kind, concept=list(c), joint=a["joint"], new=a[B.names[nf]],
                       acc={nm: a[nm] for nm in B.names}, coverage=dc["coverage"],
                       density=dc["density"], vendi=float(vendi_score(Si)))
            if kind == "taught":
                rec["memorized"] = memorization(Si, split[c]["train"])["frac_memorized"]
            per.append(rec)
        oacc = X_["sample_old"](model)
        P = eps_profile(model, X_["old_X"], u_grid, cls=X_["old_cls"], n_noise=args.crn_noise,
                        seed=CRN_SEED, per_image=True, **kw)
        Pn = eps_profile(model, X_["old_X"], u_grid, cls=None, n_noise=args.crn_noise,
                         seed=CRN_SEED, per_image=True, **kw)
        drift = ((P - X_["P_old0"]) / X_["P_old0"]).mean(0)
        drift_null = ((Pn - X_["P_null0"]) / X_["P_null0"]).mean(0)
        ds = delta_stats([(n, named[n]) for n in wmap], {n: theta0[n] for n in wmap})
        ds_task = delta_stats([(n, named[n]) for n in wmap], prev)
        lora_l2 = float(sum(((q.detach() - q0) ** 2).sum() for q, q0 in zip(lora, lora_prev))) ** 0.5
        base_exact = bool(all(torch.equal(named[n], theta0[n]) for n in wmap)) if W == "lora" \
            else None

        def mean(kind, key):
            vals = [r[key] for r in per if r["kind"] == kind and r.get(key) == r.get(key)]
            return float(np.mean(vals)) if vals else float("nan")

        o0 = X_["old_acc0"]
        rec = dict(arm=name, task=t, tier=tier, steps_weight=steps_w, absorbed=bool(absorbed),
                   s_arrival=float(s_arr.mean()), s_after_extend=float(s_ext.mean()),
                   s_final=float(s_fin.mean()), s_arrival_u=s_arr.tolist(), threshold=X_["thr"],
                   s_ext_u=s_ext.tolist(), s_final_u=s_fin.tolist(),
                   ratio_arrival_u=(s_arr / t_u).tolist(), ratio_final_u=(s_fin / t_u).tolist(),
                   mask_layers=(mask_layers if W == "mask" and tier.startswith("WEIGHT") else None),
                   taught_joint=mean("taught", "joint"), taught_new=mean("taught", "new"),
                   taught_cov=mean("taught", "coverage"), taught_vendi=mean("taught", "vendi"),
                   taught_mem=mean("taught", "memorized"),
                   untaught_joint=mean("untaught", "joint"), untaught_new=mean("untaught", "new"),
                   untaught_cov=mean("untaught", "coverage"),
                   old_joint=float(oacc[:, 0].mean()), old_drop=float(o0[:, 0].mean() - oacc[:, 0].mean()),
                   old_factors={nm: float(oacc[:, 1 + k].mean()) for k, nm in enumerate(B.names)},
                   drift_u=drift.tolist(), drift_mean=float(drift.mean()),
                   drift_null_u=drift_null.tolist(), drift_null_mean=float(drift_null.mean()),
                   changed=ds["changed"], changed_frac=ds["frac"], dtheta_l2=ds["l2"],
                   task_changed=ds_task["changed"], task_dtheta_l2=ds_task["l2"],
                   lora_params=int(sum(q.numel() for q in lora)), lora_task_l2=lora_l2,
                   base_exact=base_exact, v_norm=float(v.detach().norm()), consolidation=cons,
                   secs_task=time.time() - t_task, per=per)
        out.append(rec)
        of = "/".join(f"{rec['old_factors'][nm]:.2f}" for nm in B.names)
        print(f"[c1] {name} task {t}: tier {tier} steps {steps_w} | ratio arr "
              f"{max(rec['ratio_arrival_u']):.2f} -> fin {max(rec['ratio_final_u']):.2f} | TAUGHT "
              f"joint {rec['taught_joint']:.3f} new {rec['taught_new']:.3f} cov {rec['taught_cov']:.3f}"
              f" | UNTAUGHT joint {rec['untaught_joint']:.3f} new {rec['untaught_new']:.3f} | OLD "
              f"{rec['old_joint']:.3f} (drop {rec['old_drop']:+.3f}; {of}) drift "
              f"{rec['drift_mean']:+.4f} null {rec['drift_null_mean']:+.4f} | changed "
              f"{ds['frac']:.4f} |dθ| {ds['l2']:.3f}", flush=True)
    print(f"[c1] {name} finished in {(time.time() - t_arm) / 60:.1f} min", flush=True)
    return out


if __name__ == "__main__":
    sys.exit(main())
