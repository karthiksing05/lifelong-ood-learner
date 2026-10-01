"""C1 (archive/2026-10-program/PLAN_TRANSFER.md; ledger m3_prereg tag c1_grammar): do the coarse prototype levels give a task's phase grammar?

Every frame's 10-step chunk of each stream-pool demo (the matched raw demo's kept actions, results/robot/p0_demo/<t>_match.npz) is
labelled at each level of the 40-task prototype tree by r19's basin rule (experiments/r19_invariance.py `labels`).  The change points
of the label sequence are compared with the phase events of the v2 cut (lifelong_ood/policy/phase.py): the close that starts the
carry and the first release after it (demos that move no object: every gripper event).  Recall = events within +-W frames of a
change point; precision = change points within +-W frames of an event; control = random change points, the same count per demo.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/c1_grammar.py [--record]   (GPU)
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import data as D                                # noqa: E402
from lifelong_ood.policy import phase as PH                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def events(z):
    cmd = np.asarray(z["action"])[:, 6] > 0
    T = len(cmd)
    parts = PH.cut(z)
    if parts[0][5] == "obj" and len(parts) > 1:
        c = parts[0][4] - PH.CLOSE_TAIL
        ev = [c]
        rel = [t for t in range(c + 1, T) if not cmd[t] and cmd[t - 1]]
        if rel:
            ev.append(rel[0])
        return ev
    return [t for t in range(1, T) if cmd[t] != cmd[t - 1]]


def score(bounds, ev, T, W):
    rec = np.mean([any(abs(b - e) <= W for b in bounds) for e in ev]) if ev else np.nan
    pre = np.mean([any(abs(b - e) <= W for e in ev) for b in bounds]) if bounds else np.nan
    return rec, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    r19 = load("r19", os.path.join(ROOT, "experiments", "r19_invariance.py"))
    r2b = load("r2b", os.path.join(ROOT, "experiments", "r2b_marginal.py"))
    from lifelong_ood.tree.nodes import Tree
    lib = np.load(os.path.join(R, "r2b_libero_40.npz"))
    tree = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
    norm = D.Norm.load()
    pool = sorted(int(t) for t in json.load(open(os.path.join(R, "p5_pool.json"))))
    demos = {t: PH.at_rest(PH.load_demo(R, t)) for t in pool}
    X = [r19.chunks(demos[t]["action"], norm, every=1) for t in pool]
    cut = np.cumsum([0] + [len(x) for x in X])
    allX = np.concatenate(X)
    ev = {t: events(demos[t]) for t in pool}
    rng = np.random.default_rng(0)
    res = dict(window=a.window, n_demos=len(pool), n_events=int(sum(len(v) for v in ev.values())), levels={})
    for li, u in enumerate(tree.u_grid):
        ids = tree.levels[li]
        modes = np.stack([tree.nodes[i].mode for i in ids]).astype(np.float32)
        lab = r19.labels(allX, lib["mu"], lib["var"], modes, u, r2b, a.device)
        R_, P_, Rr, Pr, nb = [], [], [], [], []
        for k, t in enumerate(pool):
            L = lab[cut[k]:cut[k + 1]]
            b = [i for i in range(1, len(L)) if L[i] != L[i - 1]]
            nb.append(len(b))
            r, p = score(b, ev[t], len(L), a.window)
            R_.append(r); P_.append(p)
            for _ in range(100):
                rb = sorted(rng.choice(np.arange(1, len(L)), size=len(b), replace=False).tolist()) if b else []
                r0, p0 = score(rb, ev[t], len(L), a.window)
                Rr.append(r0); Pr.append(p0)
        d = dict(u=float(u), prototypes=len(ids), change_points_per_demo=float(np.mean(nb)), recall=float(np.nanmean(R_)),
                 precision=float(np.nanmean(P_)), random_recall=float(np.nanmean(Rr)), random_precision=float(np.nanmean(Pr)))
        d["holds"] = bool(d["recall"] >= 0.9 and d["precision"] >= 0.5 and d["recall"] > d["random_recall"] and d["precision"] > d["random_precision"])
        res["levels"][str(u)] = d
        print(f"[c1] t = {u:.2f} ({len(ids)} prototypes): {d['change_points_per_demo']:.1f} change points per demo; recall {d['recall']:.2f} "
              f"(random {d['random_recall']:.2f}), precision {d['precision']:.2f} (random {d['random_precision']:.2f}) -> C1 "
              f"{'holds' if d['holds'] else 'fails'}", flush=True)
    json.dump(res, open(os.path.join(R, "c1_grammar.json"), "w"), indent=1)
    if a.record:
        ledger.record("c1_grammar", bench="libero_90", tag="read", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/c1_grammar.py", command=" ".join(sys.argv), artifacts=[os.path.join(R, "c1_grammar.json")])


if __name__ == "__main__":
    main()
