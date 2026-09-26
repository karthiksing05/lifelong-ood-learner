"""M4: FOLD a LIBERO-90 task's unexplained motion into pi0.5 -- new library nodes + a gated action-expert adapter,
anchored on the library's own frames with the base policy's velocity as the target (no teacher copy).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r4_fold.py --task 35 --tree results/robot/r2b_libero_40_tree.npz \
        --library results/robot/r2b_libero_40.npz" slurm/jobs/robot_gpu.sbatch

1. Novelty per demo frame: best-node NLL of the frame's chunk at each band level vs the q95 of HELD-OUT library frames
   (episodes of the 40 trained tasks that did not build the tree).  unexplained = above q95 at any band level.
2. New nodes: the unexplained chunks' mixture modes at every band level (as r2b) are added to the tree (the library
   grows; later tasks can decompose onto them).
3. Expert: KeyedLoRA rank r on the expert, key fold_<task>; arrival = the unexplained frames (their observations and
   chunks, openpi's flow loss); anchor = library frames, target = the base velocity (expert off).
4. Program: the demo's codes (library prototypes) with the fold mask; served by `hybrid:` (compose where explained,
   the expert where not).
Writes results/robot/r4_{task}{tag}_{expert.pt,program.npz,tree.npz}; ledger row r4_fold with bytes.
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
from lifelong_ood.experts.lora import add_node                         # noqa: E402
from lifelong_ood.policy import novelty                              # noqa: E402
from lifelong_ood.policy import adapt                                  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                    # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402


def log(s):
    print(s, flush=True)


def library_obs(demos, episodes, frames):
    cache, out = {}, []
    for e, f in zip(episodes, frames):
        if e not in cache:
            cache[e] = demos.load(int(e), images=True)
        out.append(demos.observation(cache[e], int(f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--library", required=True, help="r2b npz (demo rows, task, episode, frame of the tree's frames)")
    ap.add_argument("--band", default="0,1,2")
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--n-cal", type=int, default=1500)
    ap.add_argument("--n-anchor", type=int, default=384)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arrival", default="novel", choices=["novel", "contact", "all"],
                    help="frames the expert learns and is switched on at: the library-unexplained ones, the contact windows "
                         "around the demo's gripper events (where a grasp needs closed-loop precision), or every frame")
    ap.add_argument("--contact", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    band = [int(x) for x in a.band.split(",")]
    import importlib.util
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    text = texts[a.task]
    tree = Tree.load(a.tree)
    lib = np.load(a.library)
    demos, d90, norm = D.Demos(), D.Demos90(), D.Norm.load()

    # 1. calibration on held-out library frames (episodes of the trained tasks that did not build the tree)
    cal_rows = novelty.heldout_rows(demos, lib, n=a.n_cal, seed=a.seed)
    thr = novelty.calibrate(tree, cal_rows, band, q=a.q)
    log(f"[r4] calibration on {len(cal_rows)} held-out library frames: q{a.q} per band level {thr}")

    # the demo and its novelty
    ep_idx = d90.episodes_of_text(text)[0]
    ep = d90.load(ep_idx)
    T = len(ep["actions"])
    rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
    nll, unexplained, depth = novelty.read(tree, rows, band, thr)
    novel_mask = unexplained.copy()
    if a.arrival == "contact":
        ev = D.gripper_events(ep["actions"])
        unexplained = np.zeros(T, bool)
        for e in ev:
            unexplained[max(0, e - a.contact): e + a.contact + 1] = True
        log(f"[r4] contact windows +-{a.contact} around gripper events {ev.tolist()}: {int(unexplained.sum())} frames")
    elif a.arrival == "all":
        unexplained = np.ones(T, bool)
    log(f"[r4] task {a.task} '{text}': {T} frames, {int(unexplained.sum())} unexplained ({unexplained.mean():.2f}); "
        f"median NLL per band level {[round(float(np.median(nll[l])), 2) for l in band]} vs thresholds {[round(thr[l], 2) for l in band]}")

    # 2. new nodes: modes of the unexplained chunks' mixture at every band level
    import importlib.util as ilu
    s2 = ilu.spec_from_file_location("r2b", os.path.join(ROOT, "experiments", "r2b_marginal.py"))
    r2b = ilu.module_from_spec(s2); s2.loader.exec_module(r2b)
    n_before, b_before = len(tree.nodes), tree.bytes()
    new_nodes = []
    if novel_mask.any():
        U = rows[novel_mask]
        for l in band:
            u = tree.u_grid[l]
            al, sg = 1.0 - u, u
            modes = r2b.gmm_modes(al * U, np.full_like(U, sg * sg) + al * al * 1e-3, device="cuda") / al
            kept, owner = r2b.basins(modes, 0.05 * max(sg / al, 1e-3))
            for k in kept:
                mm = owner == k
                v = (1e-3 + (U[mm] - modes[k][None]) ** 2).mean(0)
                new_nodes.append(tree.add(l, modes[k], extent_perdim=v, count=float(mm.sum()), var_floor=1e-4, var_cap=4.0))
    log(f"[r4] {len(new_nodes)} new nodes ({tree.bytes() - b_before} B) added to the library ({n_before} nodes before)")

    # 3. the gated expert
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    log(f"[r4] {bk.describe()}")
    t0 = time.time()
    fidx = np.where(unexplained)[0] if unexplained.any() else np.arange(T)
    arr_bank = adapt.PrefixBank(bk, [d90.observation(ep, int(i)) for i in fidx])
    arr_chunks = D.pad_rows(rows[fidx], bk.H)
    aj = rng.choice(len(lib["episode"]), size=min(a.n_anchor, len(lib["episode"])), replace=False)
    anc_bank = adapt.PrefixBank(bk, library_obs(demos, lib["episode"][aj], lib["frame"][aj]))
    anc_chunks = D.pad_rows(lib["demo"][aj], bk.H)
    log(f"[r4] prefixes: {arr_bank.n} arrival + {anc_bank.n} anchor observations in {time.time() - t0:.0f}s")
    state, banks = adapt.inject(bk)
    key = f"fold_{a.task}"
    params = add_node(banks, key, a.rank)
    hist = adapt.train(bk, state, key, params, arr_bank, arr_chunks, steps=a.steps, lr=a.lr, seed=a.seed,
                       anchor=(anc_bank, anc_chunks), anchor_weight=a.anchor_weight, log=log)
    os.makedirs("results/robot", exist_ok=True)
    base = f"results/robot/r4_{a.task}{a.tag}"
    eb = adapt.save(banks, key, base + "_expert.pt", meta=dict(task=a.task, text=text, rank=a.rank))

    # 4. the program with its fold mask, and the grown library
    prog = DemoProgram.from_demo(tree, rows, ep["state"], K=a.K, levels=band, variant="code")
    prog.fold = unexplained
    prog.save(base + "_program.npz")
    tree.save(base + "_tree.npz")
    nb = tree.bytes() - b_before
    m = dict(task=a.task, text=text, demo_episode=int(ep_idx), n_frames=int(T), n_unexplained=int(unexplained.sum()),
             thresholds={str(k): v for k, v in thr.items()}, nll_median={str(l): float(np.median(nll[l])) for l in band},
             n_new_nodes=len(new_nodes), expert_params=adapt.n_params(params), curve=hist, anchor_weight=a.anchor_weight)
    ledger.record("r4_fold", bench="libero_90", tag=a.tag, seed=a.seed, keys=dict(task=a.task), metrics=m,
                  bytes=dict(experts=eb, tree=nb, codes=prog.bytes(), samples=0, teacher_copies=0, total=eb + nb + prog.bytes()),
                  config=vars(a), script="experiments/r4_fold.py", command=" ".join(sys.argv),
                  artifacts=[base + "_expert.pt", base + "_program.npz", base + "_tree.npz"])
    log(f"[r4] done: expert {eb / 1e6:.2f} MB, new nodes {nb} B, program {prog.bytes()} B")


if __name__ == "__main__":
    main()
