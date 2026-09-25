"""M2: isolate agent prototypes in pi05_libero's action expert (Zekun's discovery on action chunks).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r2_prototypes.py --suite libero_goal" slurm/jobs/robot_gpu.sbatch

For E episodes per task and F frames per episode, the demo chunk a (10 x 7, normalised) and its OWN observation
o are the query: at every level t of the grid, noise a to t along the flow path, ascend the conditional score
grad log p_t(x | o) (Adam, Tweedie mode), fit the diagonal Laplace extent, and collect (mode, extent) per row.
Per level the modes are deduplicated (per-dim RMSE < tol collapse) into prototypes with basin counts, a generic
Tree is built (fp16 modes, per-dim extents, flow path), parents linked by the coarser level's Gaussians.

Validations written to the ledger:
  recurrence   prototype x task incidence per level: density, tasks per prototype, prototypes per task, vs the
               one-per-demo null (every chunk its own prototype) and vs dedup of the RAW chunks at the same tol
  unimodality  rms of the conditional mode from its own demo chunk, vs the rms between demo chunks
  hierarchy    symmetric KL child||parent and normalised displacement per level (Zekun's curves)
  semantics    at the semantics level, prototype-id boundaries along whole episodes vs gripper events:
               fraction of gripper events within +-3 frames of a boundary, vs a random segmentation with the
               same number of boundaries (mean over 200 draws)
  timing       forwards per level and seconds
Artifacts: results/robot/r2_<suite>_tree.npz, results/robot/r2_<suite>.npz (modes, extents, provenance),
figures/robot/r2_<suite>_incidence.png, figures/robot/r2_<suite>_prototypes.png.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.backbones.score_ops import laplace_diag, mode_ascent  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy.backbone import PolicyBackbone                # noqa: E402
from lifelong_ood.tree.build import link_parents                       # noqa: E402
from lifelong_ood.tree.nodes import Tree, level_gaussian               # noqa: E402
from lifelong_ood.tree.update import dedup                             # noqa: E402

ZERO = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)


def sample_chunks(demos, suite, eps_per_task, frames_per_ep, H, seed):
    """Stratified: E episodes per task, F evenly spaced frames per episode.  -> (ChunkSet, [(ep, frame)], obs dicts)."""
    rng = np.random.default_rng(seed)
    tasks = D.SUITES[suite]
    rows, raw, task, epi, frame, pad, obs = [], [], [], [], [], [], []
    norm = D.Norm.load()
    for tid in tasks:
        eps = demos.episodes(tid)
        for e in rng.choice(eps, size=min(eps_per_task, len(eps)), replace=False):
            ep = demos.load(int(e), images=True)
            T = len(ep["actions"])
            idx, pm = D.chunk_indices(T, H)
            sel = np.unique(np.linspace(0, T - H, frames_per_ep).round().astype(int))
            c = ep["actions"][idx[sel]]
            rows.append(D.to_rows(c, norm)); raw.append(c)
            task.append(np.full(len(sel), tid)); epi.append(np.full(len(sel), int(e))); frame.append(sel); pad.append(pm[sel].sum(1))
            obs += [demos.observation(ep, int(i)) for i in sel]
    cat = lambda xs, dt: np.concatenate(xs).astype(dt)   # noqa: E731
    cs = D.ChunkSet(cat(rows, np.float32), cat(raw, np.float32), cat(task, int), cat(epi, int), cat(frame, int), cat(pad, int), H)
    return cs, obs


def discover(bk, cs, obs, u_grid, iters, lr, batch, probes, seed, log):
    """Per level: (x_t modes, x0 modes, extents) for every row, ascending under each row's OWN prefix."""
    n = len(cs)
    out = {u: dict(xt=np.zeros((n, bk.dim), np.float32), x0=np.zeros((n, bk.dim), np.float32), ext=np.zeros((n, bk.dim), np.float32)) for u in u_grid}
    fwd = {u: 0 for u in u_grid}
    t_all = time.time()
    for b0 in range(0, n, batch):
        sl = slice(b0, min(n, b0 + batch))
        pre = bk.prefix(bk.observations(obs[sl]))
        bk.bind(pre)
        for u in u_grid:
            xt, x0 = mode_ascent(bk, cs.rows[sl], u, iters=iters, lr=lr, seed=seed + b0 + int(1000 * u), init="query", adam=True, tweedie=True)
            ext = laplace_diag(bk, xt, u, probes=probes, seed=seed + 7, var_floor=1e-4, var_cap=4.0)
            out[u]["xt"][sl], out[u]["x0"][sl], out[u]["ext"][sl] = xt, x0, ext
            fwd[u] += (iters + 1 + 2 * probes) * (sl.stop - sl.start)
        if (b0 // batch) % 4 == 0:
            log(f"[r2] batch {b0 // batch + 1}/{(n + batch - 1) // batch}: {time.time() - t_all:.0f}s")
    return out, fwd


def sym_kl_diag(m1, v1, m2, v2):
    """Symmetric KL between diagonal Gaussians, per pair."""
    d = m1 - m2
    kl12 = 0.5 * ((v1 / v2) + d * d / v2 - 1 + np.log(v2 / v1)).sum(-1)
    kl21 = 0.5 * ((v2 / v1) + d * d / v1 - 1 + np.log(v1 / v2)).sum(-1)
    return kl12 + kl21


def incidence(owner_ids, tasks):
    """prototype x task count matrix and its summaries."""
    protos = sorted(set(owner_ids.tolist()))
    tl = sorted(set(tasks.tolist()))
    M = np.zeros((len(protos), len(tl)), int)
    pi = {p: i for i, p in enumerate(protos)}
    ti = {t: j for j, t in enumerate(tl)}
    for o, t in zip(owner_ids, tasks):
        M[pi[o], ti[t]] += 1
    nz = M > 0
    return M, dict(n_prototypes=len(protos), density=float(nz.mean()), tasks_per_prototype=float(nz.sum(1).mean()),
                   prototypes_per_task=float(nz.sum(0).mean()), shared_fraction=float((nz.sum(1) > 1).mean()),
                   top_share=float(M.sum(1).max() / M.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--eps-per-task", type=int, default=4)
    ap.add_argument("--frames-per-ep", type=int, default=24)
    ap.add_argument("--levels", default="0.1,0.2,0.3,0.4,0.55,0.7")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--sem-level", type=float, default=0.3)
    ap.add_argument("--sem-eps-per-task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    u_grid = [float(x) for x in a.levels.split(",")]
    log = lambda s: print(s, flush=True)   # noqa: E731
    dev = "cuda"
    bk = PolicyBackbone("pi05_libero", device=dev)
    log(f"[r2] {bk.describe()}")
    demos = D.Demos()
    cs, obs = sample_chunks(demos, a.suite, a.eps_per_task, a.frames_per_ep, bk.H, a.seed)
    log(f"[r2] {len(cs)} chunks from {len(set(cs.episode.tolist()))} episodes of {len(set(cs.task.tolist()))} tasks ({a.suite})")
    t0 = time.time()
    found, fwd = discover(bk, cs, obs, u_grid, a.iters, a.lr, a.batch, a.probes, a.seed, log)
    secs_discover = time.time() - t0
    log(f"[r2] discovery {secs_discover:.0f}s; forwards per level {fwd[u_grid[0]]}")

    # per-level selection -> tree
    tree = Tree(u_grid, None, -1e9, 1e9, dim=bk.dim, path="flow")
    m = dict(levels={}, secs_discover=secs_discover, n_chunks=int(len(cs)))
    owner_by_level = {}
    raw_kept, _ = dedup(cs.rows, tol=a.tol)
    between = float(np.sqrt(((cs.rows[:, None, :] - cs.rows[None, :, :]) ** 2).mean(-1)[np.triu_indices(len(cs), 1)].mean())) if len(cs) <= 2048 else float("nan")
    m["raw_distinct"] = int(len(raw_kept))
    m["between_chunk_rms"] = between
    for level, u in enumerate(u_grid):
        x0, ext = found[u]["x0"], found[u]["ext"]
        kept, owner = dedup(x0, tol=a.tol)
        counts = {k: int((owner == k).sum()) for k in kept}
        nid_of = {}
        for k in kept:
            nid_of[k] = tree.add(level, x0[k], extent_perdim=ext[owner == k].mean(0), count=float(counts[k]), var_floor=1e-4, var_cap=4.0)
        owner_ids = np.asarray([nid_of[o] for o in owner])
        owner_by_level[u] = owner_ids
        M, inc = incidence(owner_ids, cs.task)
        disp = float(np.sqrt(((x0 - cs.rows) ** 2).mean()))
        m["levels"][str(u)] = dict(n_distinct=len(kept), incidence=inc, mode_rms_to_own_chunk=disp,
                                   null_density=float(1.0 / len(set(cs.task.tolist()))), forwards=int(fwd[u]))
        log(f"[r2] t={u}: {len(kept)} prototypes from {len(cs)} chunks (raw chunks: {len(raw_kept)} distinct at tol {a.tol}); "
            f"incidence density {inc['density']:.2f} (null {1.0 / len(set(cs.task.tolist())):.2f}), tasks/prototype {inc['tasks_per_prototype']:.2f}, "
            f"shared {inc['shared_fraction']:.2f}; mode is {disp:.3f} rms from its chunk (between chunks {between:.3f})")
    link_parents(tree)
    # hierarchy curves
    hier = {}
    for level in range(len(u_grid) - 1):
        ids_c, Mc, Vc, _ = tree.stats(level)
        vals_kl, vals_disp = [], []
        for i, cid in enumerate(ids_c):
            p = tree.nodes[cid].parent
            if p < 0:
                continue
            mp, vp = tree.nodes[p].mode, tree.nodes[p].var
            vals_kl.append(float(sym_kl_diag(Mc[i], Vc[i], mp, vp)))
            vals_disp.append(float(np.sqrt(((Mc[i] - mp) ** 2 / vp).mean())))
        hier[str(u_grid[level])] = dict(sym_kl=float(np.mean(vals_kl)) if vals_kl else None, norm_disp=float(np.mean(vals_disp)) if vals_disp else None,
                                        n_children=len(vals_kl))
    m["hierarchy"] = hier
    log("[r2] hierarchy (child vs parent): " + ", ".join(f"t={k}: KL {v['sym_kl']:.1f} disp {v['norm_disp']:.2f}" for k, v in hier.items() if v["sym_kl"] is not None))
    m["tree"] = tree.summary()
    os.makedirs("results/robot", exist_ok=True)
    tree.save(f"results/robot/r2_{a.suite}{a.tag}_tree.npz")
    np.savez_compressed(f"results/robot/r2_{a.suite}{a.tag}.npz", rows=cs.rows, task=cs.task, episode=cs.episode, frame=cs.frame,
                        **{f"x0_{u}": found[u]["x0"] for u in u_grid}, **{f"ext_{u}": found[u]["ext"] for u in u_grid},
                        **{f"owner_{u}": owner_by_level[u] for u in u_grid})

    # semantics: prototype boundaries along whole episodes vs gripper events at one level
    us = a.sem_level
    lvl = tree.level_of(us)
    ids, C, S, logpi = level_gaussian(tree, lvl)
    from lifelong_ood.tree.responsibility import _loglik
    hits, rand_hits, n_events, n_bounds = [], [], 0, 0
    rng = np.random.default_rng(a.seed)
    t1 = time.time()
    for tid in D.SUITES[a.suite]:
        eps = demos.episodes(tid)
        for e in rng.choice(eps, size=min(a.sem_eps_per_task, len(eps)), replace=False):
            ep = demos.load(int(e), images=True)
            T = len(ep["actions"])
            idx, _ = D.chunk_indices(T, bk.H)
            rows = D.to_rows(ep["actions"][idx], D.Norm.load())
            seq = np.zeros(T, int)
            for b0 in range(0, T, a.batch):
                sl = slice(b0, min(T, b0 + a.batch))
                bk.bind(bk.prefix(bk.observations([demos.observation(ep, int(i)) for i in range(sl.start, sl.stop)])))
                xt, x0 = mode_ascent(bk, rows[sl], us, iters=a.iters, lr=a.lr, seed=a.seed + 5 + b0, init="query", adam=True, tweedie=True)
                alpha = 1.0 - us
                L = _loglik(alpha * x0, C, S, logpi)          # the mode's node at this level
                seq[sl] = np.asarray(ids)[L.argmax(1)]
            bounds = np.where(np.diff(seq) != 0)[0] + 1
            ev = D.gripper_events(ep["actions"])
            n_events += len(ev); n_bounds += len(bounds)
            if len(ev) and len(bounds):
                hits.append(float(np.mean([np.abs(bounds - g).min() <= 3 for g in ev])))
                rh = []
                for _ in range(200):
                    rb = np.sort(rng.choice(np.arange(1, T), size=len(bounds), replace=False))
                    rh.append(float(np.mean([np.abs(rb - g).min() <= 3 for g in ev])))
                rand_hits.append(float(np.mean(rh)))
    m["semantics"] = dict(level=us, n_episodes=len(hits), n_gripper_events=int(n_events), n_boundaries=int(n_bounds),
                          hit_rate=float(np.mean(hits)) if hits else None, random_hit_rate=float(np.mean(rand_hits)) if rand_hits else None,
                          secs=time.time() - t1)
    log(f"[r2] semantics at t={us}: gripper events within 3 frames of a prototype boundary {m['semantics']['hit_rate']} vs random {m['semantics']['random_hit_rate']} "
        f"({n_bounds} boundaries over {len(hits)} episodes, {n_events} events, {time.time() - t1:.0f}s)")

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures/robot", exist_ok=True)
        fig, axs = plt.subplots(1, len(u_grid), figsize=(3 * len(u_grid), 4))
        for j, u in enumerate(u_grid):
            M, _ = incidence(owner_by_level[u], cs.task)
            order = np.argsort(-M.sum(1))
            axs[j].imshow(M[order], aspect="auto", cmap="Blues")
            axs[j].set_title(f"t={u}: {M.shape[0]} prototypes"); axs[j].set_xlabel("task"); axs[j].set_ylabel("prototype")
        fig.tight_layout(); fig.savefig(f"figures/robot/r2_{a.suite}{a.tag}_incidence.png", dpi=120); plt.close(fig)
        # prototypes as end-effector paths (cumulative xyz deltas of the raw actions) at the semantics level
        ids_l = tree.levels[lvl]
        top = sorted(ids_l, key=lambda i: -tree.nodes[i].count)[:12]
        fig = plt.figure(figsize=(12, 9))
        for k, nid in enumerate(top):
            ax = fig.add_subplot(3, 4, k + 1, projection="3d")
            raw = bk.unnormalize(tree.nodes[nid].mode[None])[0]
            xyz = np.cumsum(raw[:, :3], 0)
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-o", ms=3)
            ax.set_title(f"node {nid} n={int(tree.nodes[nid].count)} grip={raw[:, 6].mean():+.1f}", fontsize=8)
        fig.tight_layout(); fig.savefig(f"figures/robot/r2_{a.suite}{a.tag}_prototypes.png", dpi=110); plt.close(fig)
    except Exception as ex:   # noqa: BLE001
        log(f"[r2] figures skipped: {ex}")
    ledger.record("r2_prototypes", bench=a.suite, tag=a.tag, seed=a.seed, metrics=m,
                  bytes=dict(experts=0, tree=tree.bytes(), codes=0, samples=0, teacher_copies=0, total=tree.bytes()),
                  config=vars(a), script="experiments/r2_prototypes.py", command=" ".join(sys.argv),
                  backbone=dict(config="pi05_libero", dim=bk.dim, H=bk.H),
                  artifacts=[f"results/robot/r2_{a.suite}{a.tag}_tree.npz", f"results/robot/r2_{a.suite}{a.tag}.npz"])
    log("[r2] done")


if __name__ == "__main__":
    main()
