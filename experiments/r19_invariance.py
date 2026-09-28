"""WHERE in the prototype hierarchy is a task's motion structure invariant across instances yet specific to the task?

The higher levels of the 40-task library lose as TEMPLATES (a mode of a blurred mixture is a slow average: docs/PROTOTYPES.md);
this asks whether they win as CATEGORIES.  Every chunk of an episode (one per policy query, every 5th frame) is labelled at every
level by the tree's own rule -- the mode its diffused copy converges to under that level's mixture (the basin, as r2b_marginal
built the tree) -- and an episode becomes the sequence of its labels with repeats collapsed.  Episodes: up to --demos human
demonstrations and up to --robot successful closed-loop LoRA practice rollouts (m6 rounds 1-2) per held-out task, each at a
different initial state.

Per level: task identification by the nearest other episode (normalised edit distance; leave-one-out; chance 1 / tasks), within
human demos, within robot rollouts, and across (a robot rollout classified by the nearest human demo).  Control: the same
nearest-neighbour read on the raw chunk trajectories, time-normalised to a common length (no prototypes).

    python experiments/r19_invariance.py [--demos 10 --robot 10]
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TASKS = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
FLOOR = 1e-3                                                            # r2b_marginal's variance floor


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def chunks(actions, norm, every=5):
    T = len(actions)
    rows = D.to_rows(np.asarray(actions, np.float32)[D.chunk_indices(T, 10)[0]], norm)
    return rows[::every]


def labels(X, mu, var, modes, u, r2b, device):
    """Basin labels of rows X [n, 70] at level u: converge a diffused copy under the level's mixture, take the nearest mode."""
    a = 1.0 - u
    C, S = a * mu, a * a * np.maximum(var, FLOOR) + u * u
    conv = r2b.gmm_modes(C, S, X0=a * X, device=device) / a
    return r2b.nearest(conv, modes, device)


def collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(int(s))
    return out


def edit(a, b):
    """Levenshtein distance normalised by the longer length."""
    n, m = len(a), len(b)
    if max(n, m) == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[m] / max(n, m)


def resample(rows, L=40):
    idx = np.linspace(0, len(rows) - 1, L)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(rows) - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * rows[lo] + w * rows[hi]).reshape(-1)


def nn_accuracy(Dm, task, rows_idx, ref_idx):
    """Share of `rows_idx` episodes whose nearest `ref_idx` episode (other than itself) has the same task."""
    hit = 0
    for i in rows_idx:
        cand = [j for j in ref_idx if j != i]
        j = cand[int(np.argmin([Dm[i, c] for c in cand]))]
        hit += int(task[i] == task[j])
    return hit / max(len(rows_idx), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=10)
    ap.add_argument("--robot", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    t0 = time.time()
    r2b = load_module("r2b", os.path.join(ROOT, "experiments", "r2b_marginal.py"))
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    from lifelong_ood.tree.nodes import Tree
    lib = np.load(os.path.join(R, "r2b_libero_40.npz"))
    mu, var = lib["mu"], lib["var"]
    tree = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    eps = []                                                              # (task, source, rows)
    for t in TASKS:
        for k, e in enumerate(d90.episodes_of_text(texts[t])[: a.demos]):
            eps.append((t, "human", chunks(d90.load(e)["actions"], norm)))
        recs = sorted(glob.glob(os.path.join(R, f"m6_round1_{t}_*_rec", "*.npz")) + glob.glob(os.path.join(R, f"m6_round2_{t}_*_rec", "*.npz")))
        got = 0
        for f in recs:
            z = np.load(f)
            if bool(z["success"]) and got < a.robot:
                eps.append((t, "robot", chunks(z["action"], norm)))
                got += 1
    task = np.array([e[0] for e in eps])
    src = np.array([e[1] for e in eps])
    print(f"[r19] {len(eps)} episodes ({(src == 'human').sum()} human, {(src == 'robot').sum()} robot) over {len(TASKS)} tasks; "
          f"{sum(len(e[2]) for e in eps)} chunks", flush=True)
    H = np.where(src == "human")[0]
    Rb = np.where(src == "robot")[0]
    allX = np.concatenate([e[2] for e in eps])
    cut = np.cumsum([0] + [len(e[2]) for e in eps])
    res = dict(tasks=TASKS, n_episodes=len(eps), n_human=int(len(H)), n_robot=int(len(Rb)), chance=1.0 / len(TASKS), levels={})
    saved = dict(task=task, source=src)
    for li, u in enumerate(tree.u_grid):
        ids = tree.levels[li]
        modes = np.stack([tree.nodes[i].mode for i in ids]).astype(np.float32)
        lab = labels(allX, mu, var, modes, u, r2b, a.device)
        seqs = [collapse(lab[cut[i]:cut[i + 1]]) for i in range(len(eps))]
        n = len(eps)
        Dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                Dm[i, j] = Dm[j, i] = edit(seqs[i], seqs[j])
        within = [Dm[i, j] for i in range(n) for j in range(i + 1, n) if task[i] == task[j]]
        between = [Dm[i, j] for i in range(n) for j in range(i + 1, n) if task[i] != task[j]]
        r = dict(u=u, prototypes=len(ids), used=int(len(set(lab.tolist()))), seq_len=float(np.mean([len(s) for s in seqs])),
                 within=float(np.mean(within)), between=float(np.mean(between)),
                 acc_human=nn_accuracy(Dm, task, H, H), acc_robot=nn_accuracy(Dm, task, Rb, Rb), acc_robot_by_human=nn_accuracy(Dm, task, Rb, H))
        res["levels"][str(u)] = r
        saved[f"labels_{u}"] = np.array(seqs, dtype=object)
        saved[f"dist_{u}"] = Dm.astype(np.float32)
        print(f"[r19] t={u}: {r['prototypes']} prototypes ({r['used']} used), {r['seq_len']:.1f} labels per episode; edit distance "
              f"within {r['within']:.3f} / between {r['between']:.3f}; task by nearest episode: human {r['acc_human']:.3f}, robot "
              f"{r['acc_robot']:.3f}, robot by human {r['acc_robot_by_human']:.3f}", flush=True)
    # control: the raw chunk trajectories, time-normalised, Euclidean
    Z = np.stack([resample(e[2]) for e in eps])
    Dr = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    res["raw"] = dict(acc_human=nn_accuracy(Dr, task, H, H), acc_robot=nn_accuracy(Dr, task, Rb, Rb), acc_robot_by_human=nn_accuracy(Dr, task, Rb, H))
    print(f"[r19] raw trajectories: human {res['raw']['acc_human']:.3f}, robot {res['raw']['acc_robot']:.3f}, "
          f"robot by human {res['raw']['acc_robot_by_human']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(R, "r19_invariance_sequences.npz"), allow_pickle=True, **saved)
    import json
    json.dump(res, open(os.path.join(R, "r19_invariance.json"), "w"), indent=1)
    if a.record:
        ledger.record("r19_invariance", bench="libero_90", tag="", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/r19_invariance.py", command=" ".join(sys.argv), artifacts=[os.path.join(R, "r19_invariance.json")])


if __name__ == "__main__":
    main()
