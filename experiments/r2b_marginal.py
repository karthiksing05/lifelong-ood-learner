"""M2, marginal view: agent prototypes = modes of the policy's action marginal over demo observations.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r2b_marginal.py --suite libero_goal" slurm/jobs/robot_gpu.sbatch

Why this view.  Given one observation, pi05_libero's conditional p(a | o) is expected to be narrow (one mode); its
noisy marginals p_t(x | o) are then blurred copies of one bump, and score ascent from the demo chunk returns
~ the conditional mean at every level (r2_prototypes.py measures this).  The hierarchy the thesis needs lives
in the policy's behaviour ACROSS observations: the action marginal of a task or suite,
    p(a) = (1/N) sum_i p(a | o_i),   o_i = the frames of the demos.
Each conditional is summarised by K samples from the policy's own sampler (mean mu_i, per-dim variance v_i,
floored), so p(a) ~ (1/N) sum_i N(mu_i, v_i) and, along the flow path x_t = (1 - t) a + t eps,
    p_t(x) = (1/N) sum_i N(x; (1 - t) mu_i, (1 - t)^2 v_i + t^2 I).
Prototypes at level t = the modes of p_t, found by the Gaussian-mixture fixed point (Carreira-Perpinan 2000)
from every component's centre; a frame's prototype = the mode its centre converges to (its basin).  The same
analysis on the RAW demo chunks (mu_i = demo chunk, v_i = floor) is the no-policy control: it is the ideal
denoiser of the demo set (docs/COBWEB_UPDATE.md Prop. 1).

Validations (ledger `r2b_marginal`):
  unimodality   within-observation spread (mean per-dim std of the K samples) vs between-frame spread; the
                Tweedie mean at t = 1 (x_1 - v(x_1, 1)) vs the sample mean (a check of the flow identity)
  recurrence    per level: #prototypes, prototype x task incidence density, tasks per prototype, shared fraction,
                vs the null of one prototype per task (density 1 / n_tasks)
  semantics     per level: along whole episodes, prototype-id boundaries vs gripper events (hit = an event within
                +-3 frames of a boundary) vs random segmentations with the same number of boundaries (200 draws)
  hierarchy     parent = the coarser level's mode the child's mode converges to; per level the mean symmetric KL
                child||parent and normalised displacement (Zekun's curves)
  control       the same numbers for the raw-demo mixture
Artifacts: results/robot/r2b_<suite>{tag}.npz (mu, var, demo rows, provenance, basins per level),
results/robot/r2b_<suite>{tag}_tree.npz (policy prototypes as a generic Tree), figures/robot/r2b_<suite>{tag}_*.png.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402
from lifelong_ood.tree.update import dedup                             # noqa: E402


def log(s):
    print(s, flush=True)


# ── the policy's conditionals ───────────────────────────────────────────────────────────────
def frames_of(demos, suite, eps_per_task, seed):
    rng = np.random.default_rng(seed)
    out = []
    for tid in D.SUITES[suite]:
        eps = demos.episodes(tid)
        for e in rng.choice(eps, size=min(eps_per_task, len(eps)), replace=False):
            out.append(int(e))
    return out


def conditionals(bk, demos, episodes, K, batch_obs, seed, stride=1):
    """-> dict(mu, var, tw (Tweedie mean at t=1), demo rows, task, episode, frame, gripper events per episode).
    `stride` keeps every stride-th frame (consecutive chunks share 9 of 10 actions)."""
    norm = D.Norm.load()
    mu, var, tw, demo, task, epi, frame = [], [], [], [], [], [], []
    events = {}
    t0 = time.time()
    for n_ep, e in enumerate(episodes):
        ep = demos.load(e, images=True)
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        rows = D.to_rows(ep["actions"][idx], norm)
        events[e] = D.gripper_events(ep["actions"]).tolist()
        keep = list(range(0, T, stride))
        for b0 in range(0, len(keep), batch_obs):
            sl = keep[b0:b0 + batch_obs]
            pre = bk.prefix(bk.observations([demos.observation(ep, i) for i in sl]))
            preK = bk.repeat(pre, K)
            n = len(sl) * K
            S = bk.sample(n, preK, seed=seed + 7919 * e + b0).reshape(len(sl), K, bk.dim)
            g = torch.Generator(device="cpu").manual_seed(seed + 31 * e + b0)
            x1 = torch.randn((n, bk.dim), generator=g).to(bk.device)
            a1 = bk.tweedie(x1, 1.0, preK).reshape(len(sl), K, bk.dim).mean(1).cpu().numpy()
            mu.append(S.mean(1)); var.append(S.var(1)); tw.append(a1)
            demo.append(rows[list(sl)]); task.append(np.full(len(sl), ep["task_index"])); epi.append(np.full(len(sl), e))
            frame.append(np.asarray(list(sl)))
        if n_ep % 5 == 0:
            log(f"[r2b] conditionals: episode {n_ep + 1}/{len(episodes)} ({T} frames), {time.time() - t0:.0f}s")
    cat = np.concatenate
    return dict(mu=cat(mu).astype(np.float32), var=cat(var).astype(np.float32), tw=cat(tw).astype(np.float32),
                demo=cat(demo).astype(np.float32), task=cat(task).astype(int), episode=cat(epi).astype(int),
                frame=cat(frame).astype(int), events=events, secs=time.time() - t0)


# ── modes of a diagonal Gaussian mixture (the level's p_t) ────────────────────────────────────
@torch.no_grad()
def gmm_modes(C, S, X0=None, iters=400, tol=1e-6, device="cuda", chunk=2048):
    """Fixed point x <- (sum_i w_i(x) / S_i)^-1 (sum_i w_i(x) C_i / S_i) from every centre (or X0).
    C, S: [N, D] centres and variances of the (already diffused) components, equal weights.  -> modes [M, D]."""
    C = torch.as_tensor(C, dtype=torch.float64, device=device)
    S = torch.as_tensor(S, dtype=torch.float64, device=device)
    X = C.clone() if X0 is None else torch.as_tensor(X0, dtype=torch.float64, device=device)
    P = 1.0 / S                                     # [N, D]
    CP = C * P
    c2 = (C * C * P).sum(1)                         # [N]
    logdet = torch.log(S).sum(1)                    # [N]
    active = torch.ones(len(X), dtype=torch.bool, device=device)
    for it in range(iters):
        idx = torch.nonzero(active).squeeze(1)
        if len(idx) == 0:
            break
        for b in range(0, len(idx), chunk):
            j = idx[b:b + chunk]
            x = X[j]
            L = -0.5 * ((x * x) @ P.T - 2.0 * x @ CP.T + c2[None]) - 0.5 * logdet[None]
            W = torch.softmax(L, dim=1)
            xn = (W @ CP) / (W @ P)
            moved = (xn - x).abs().max(1).values
            X[j] = xn
            active[j] = moved > tol
    return X.float().cpu().numpy()


@torch.no_grad()
def nearest(A, B, device="cuda", chunk=2048):
    """Index of the nearest row of B for every row of A (squared distance), in chunks on the device."""
    Bt = torch.as_tensor(np.asarray(B, np.float32), device=device)
    b2 = (Bt * Bt).sum(1)
    out = []
    for i in range(0, len(A), chunk):
        a = torch.as_tensor(np.asarray(A[i:i + chunk], np.float32), device=device)
        d = (a * a).sum(1, keepdim=True) - 2.0 * a @ Bt.T + b2[None]
        out.append(d.argmin(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, int)


def basins(modes, tol):
    """Merge converged modes closer than `tol` (per-dim rms).  -> (kept mode indices, owner per start)."""
    return dedup(modes, tol=tol)


def sym_kl(m1, v1, m2, v2):
    d = m1 - m2
    return float((0.5 * (v1 / v2 + d * d / v2 - 1 + np.log(v2 / v1)) + 0.5 * (v2 / v1 + d * d / v1 - 1 + np.log(v1 / v2))).sum())


def incidence(owner, task):
    protos = sorted(set(owner.tolist()))
    tl = sorted(set(task.tolist()))
    M = np.zeros((len(protos), len(tl)), int)
    pi = {p: i for i, p in enumerate(protos)}
    ti = {t: j for j, t in enumerate(tl)}
    for o, t in zip(owner, task):
        M[pi[o], ti[t]] += 1
    nz = M > 0
    # frame-weighted: the fraction of frames whose prototype is used by >= 2 tasks
    shared_frames = float(M[nz.sum(1) > 1].sum() / max(M.sum(), 1))
    return M, dict(n_prototypes=len(protos), density=float(nz.mean()), tasks_per_prototype=float(nz.sum(1).mean()),
                   prototypes_per_task=float(nz.sum(0).mean()), shared_fraction=float((nz.sum(1) > 1).mean()),
                   shared_frames=shared_frames, top_share=float(M.sum(1).max() / max(M.sum(), 1)))


def semantics(owner, episode, frame, events, rng, window=3, draws=200):
    hits, rhits, nb, ne = [], [], 0, 0
    for e, ev in events.items():
        m = episode == e
        if not m.any() or not ev:
            continue
        order = np.argsort(frame[m])
        seq = owner[m][order]
        fr = frame[m][order]                           # frame numbers (a stride may skip frames)
        b = fr[np.where(np.diff(seq) != 0)[0] + 1]     # boundaries in frame units, like the gripper events
        nb += len(b); ne += len(ev)
        if len(b) == 0:
            hits.append(0.0); rhits.append(0.0)
            continue
        ev = np.asarray(ev)
        hits.append(float(np.mean([np.abs(b - g).min() <= window for g in ev])))
        rr = []
        for _ in range(draws):
            rb = np.sort(rng.choice(fr[1:], size=min(len(b), len(fr) - 1), replace=False))
            rr.append(float(np.mean([np.abs(rb - g).min() <= window for g in ev])))
        rhits.append(float(np.mean(rr)))
    return dict(hit_rate=float(np.mean(hits)) if hits else None, random_hit_rate=float(np.mean(rhits)) if rhits else None,
                boundaries_per_episode=float(nb / max(len(events), 1)), events=int(ne))


def analyse(name, mu, var, task, episode, frame, events, u_grid, floor, merge_tol, device, seed):
    """Modes, basins, incidence, semantics and hierarchy of the mixture N(mu_i, var_i) at every level."""
    rng = np.random.default_rng(seed)
    out, level_modes, level_owner, level_var = {}, {}, {}, {}
    V = np.maximum(var, floor)
    for u in u_grid:
        a, s = 1.0 - u, u
        C, S = a * mu, a * a * V + s * s
        t0 = time.time()
        modes_t = gmm_modes(C, S, device=device)
        x0 = modes_t / a
        kept, owner = basins(x0, merge_tol * max(s / a, 1e-3))
        proto_mode = {k: x0[k] for k in kept}
        # a prototype's extent: moment-matched variance of its basin members' Gaussians (x0 space)
        pv = {}
        for k in kept:
            mm = owner == k
            pv[k] = (V[mm] + (mu[mm] - proto_mode[k][None]) ** 2).mean(0)
        M, inc = incidence(owner, task)
        sem = semantics(owner, episode, frame, events, rng)
        out[str(u)] = dict(incidence=inc, semantics=sem, secs=time.time() - t0, merge_tol=merge_tol * max(s / a, 1e-3))
        level_modes[u], level_owner[u], level_var[u] = proto_mode, owner, pv
        log(f"[r2b] {name} t={u}: {inc['n_prototypes']} prototypes over {len(mu)} frames; density {inc['density']:.2f} "
            f"(null {1 / len(set(task.tolist())):.2f}), tasks/proto {inc['tasks_per_prototype']:.2f}, shared frames {inc['shared_frames']:.2f}; "
            f"gripper events at a boundary {sem['hit_rate']} vs random {sem['random_hit_rate']} ({sem['boundaries_per_episode']:.1f} boundaries/episode)")
    # hierarchy: parent = the coarser level's mode that the child's mode converges to under the coarser p_t
    hier = {}
    for i, u in enumerate(u_grid[:-1]):
        up = u_grid[i + 1]
        a2, s2 = 1.0 - up, up
        kids = list(level_modes[u].keys())
        start = np.stack([level_modes[u][k] for k in kids]) * a2
        m2 = gmm_modes(a2 * mu, a2 * a2 * V + s2 * s2, X0=start, device=device) / a2
        pk = list(level_modes[up].keys())
        P = np.stack([level_modes[up][k] for k in pk])
        par = [pk[j] for j in nearest(m2, P, device)]
        kl = [sym_kl(level_modes[u][c], level_var[u][c], level_modes[up][p], level_var[up][p]) for c, p in zip(kids, par)]
        disp = [float(np.sqrt(((level_modes[u][c] - level_modes[up][p]) ** 2 / level_var[up][p]).mean())) for c, p in zip(kids, par)]
        hier[str(u)] = dict(sym_kl=float(np.mean(kl)), norm_disp=float(np.mean(disp)), n=len(kids), n_parents=len(set(par)))
    return out, hier, level_modes, level_owner, level_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--eps-per-task", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--batch-obs", type=int, default=32)
    ap.add_argument("--levels", default="0.05,0.1,0.2,0.3,0.4,0.55,0.7")
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--merge-tol", type=float, default=0.05, help="merge converged modes within this fraction of the level's x0 blur t/(1-t)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    u_grid = [float(x) for x in a.levels.split(",")]
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    log(f"[r2b] {bk.describe()}")
    demos = D.Demos()
    episodes = frames_of(demos, a.suite, a.eps_per_task, a.seed)
    cpath = f"results/robot/r2b_{a.suite}{a.tag}_conditionals.npz"
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        cond = {k: z[k] for k in ("mu", "var", "tw", "demo", "task", "episode", "frame")}
        cond["events"] = z["events"].item()
        cond["secs"] = float(z["secs"])
        log(f"[r2b] conditionals loaded from {cpath} ({len(cond['mu'])} frames)")
    else:
        cond = conditionals(bk, demos, episodes, a.K, a.batch_obs, a.seed, stride=a.stride)
        os.makedirs("results/robot", exist_ok=True)
        np.savez_compressed(cpath, **{k: cond[k] for k in ("mu", "var", "tw", "demo", "task", "episode", "frame")},
                            events=np.array(cond["events"], dtype=object), secs=cond["secs"], allow_pickle=True)
        log(f"[r2b] conditionals saved to {cpath}")
    N = len(cond["mu"])
    within = float(np.sqrt(cond["var"]).mean())
    sub = np.random.default_rng(a.seed).choice(N, size=min(N, 1500), replace=False)
    between = float(np.sqrt(((cond["mu"][sub][:, None] - cond["mu"][sub][None]) ** 2).mean(-1))[np.triu_indices(len(sub), 1)].mean())
    to_demo = float(np.sqrt(((cond["mu"] - cond["demo"]) ** 2).mean()))
    tw_err = float(np.sqrt(((cond["tw"] - cond["mu"]) ** 2).mean()))
    m = dict(n_frames=N, n_episodes=len(episodes), K=a.K, secs_conditionals=cond["secs"],
             unimodality=dict(within_obs_std=within, between_frame_rms=between, policy_mean_to_demo_rms=to_demo,
                              tweedie_t1_vs_sample_mean_rms=tw_err))
    log(f"[r2b] {N} frames from {len(episodes)} episodes: within-observation std {within:.3f} vs between-frame rms {between:.3f}; "
        f"policy mean {to_demo:.3f} rms from the demo chunk; Tweedie(t=1) vs sample mean {tw_err:.3f} rms")
    res_p, hier_p, modes_p, owner_p, var_p = analyse("policy", cond["mu"], cond["var"], cond["task"], cond["episode"], cond["frame"],
                                                     cond["events"], u_grid, a.floor, a.merge_tol, "cuda", a.seed)
    res_d, hier_d, _, owner_d, _ = analyse("demo", cond["demo"], np.zeros_like(cond["demo"]), cond["task"], cond["episode"], cond["frame"],
                                           cond["events"], u_grid, a.floor, a.merge_tol, "cuda", a.seed)
    m["policy"], m["demo"] = res_p, res_d
    m["hierarchy_policy"], m["hierarchy_demo"] = hier_p, hier_d
    log("[r2b] hierarchy (policy): " + ", ".join(f"t={k}: KL {v['sym_kl']:.1f} disp {v['norm_disp']:.2f} ({v['n']}->{v['n_parents']})" for k, v in hier_p.items()))
    # tree of the policy prototypes
    tree = Tree(u_grid, None, -1e9, 1e9, dim=bk.dim, path="flow")
    nid = {}
    for li, u in enumerate(u_grid):
        for k, mode in modes_p[u].items():
            nid[(u, k)] = tree.add(li, mode, extent_perdim=var_p[u][k], count=float((owner_p[u] == k).sum()), var_floor=1e-4, var_cap=4.0)
    os.makedirs("results/robot", exist_ok=True)
    tree_path = f"results/robot/r2b_{a.suite}{a.tag}_tree.npz"
    tree.save(tree_path)
    np.savez_compressed(f"results/robot/r2b_{a.suite}{a.tag}.npz", mu=cond["mu"], var=cond["var"], tw=cond["tw"], demo=cond["demo"],
                        task=cond["task"], episode=cond["episode"], frame=cond["frame"],
                        **{f"owner_policy_{u}": owner_p[u] for u in u_grid}, **{f"owner_demo_{u}": owner_d[u] for u in u_grid})
    m["tree"] = tree.summary()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures/robot", exist_ok=True)
        fig, axs = plt.subplots(2, len(u_grid), figsize=(2.6 * len(u_grid), 6))
        for j, u in enumerate(u_grid):
            for r, (nm, ow) in enumerate((("policy", owner_p[u]), ("demo", owner_d[u]))):
                M, _ = incidence(ow, cond["task"])
                axs[r, j].imshow(np.log1p(M[np.argsort(-M.sum(1))][:60]), aspect="auto", cmap="Blues")
                axs[r, j].set_title(f"{nm} t={u}: {M.shape[0]}", fontsize=8)
        fig.tight_layout(); fig.savefig(f"figures/robot/r2b_{a.suite}{a.tag}_incidence.png", dpi=110); plt.close(fig)
        # prototype sequences along the first episode of each task at the middle level
        um = u_grid[len(u_grid) // 2]
        fig, ax = plt.subplots(figsize=(12, 5))
        for r, tid in enumerate(D.SUITES[a.suite]):
            e = [x for x in episodes if demos.load(x)["task_index"] == tid][0]
            mm = cond["episode"] == e
            o = np.argsort(cond["frame"][mm])
            seq, fr = owner_p[um][mm][o], cond["frame"][mm][o]
            ax.scatter(fr, np.full(len(seq), r), c=seq % 20, cmap="tab20", s=6, marker="s")
            for g in cond["events"][e]:
                ax.plot([g, g], [r - 0.4, r + 0.4], "k-", lw=1)
        ax.set_yticks(range(len(D.SUITES[a.suite]))); ax.set_yticklabels([demos.tasks[t][:40] for t in D.SUITES[a.suite]], fontsize=7)
        ax.set_xlabel("frame"); ax.set_title(f"policy prototypes at t={um} along one demo per task (black: gripper events)")
        fig.tight_layout(); fig.savefig(f"figures/robot/r2b_{a.suite}{a.tag}_sequences.png", dpi=110); plt.close(fig)
    except Exception as ex:   # noqa: BLE001
        log(f"[r2b] figures skipped: {ex}")
    ledger.record("r2b_marginal", bench=a.suite, tag=a.tag, seed=a.seed, metrics=m,
                  bytes=dict(experts=0, tree=tree.bytes(), codes=0, samples=0, teacher_copies=0, total=tree.bytes()),
                  config=vars(a), script="experiments/r2b_marginal.py", command=" ".join(sys.argv),
                  backbone=dict(config="pi05_libero", dim=bk.dim, H=bk.H), artifacts=[tree_path, f"results/robot/r2b_{a.suite}{a.tag}.npz"])
    log("[r2b] done")


if __name__ == "__main__":
    main()
