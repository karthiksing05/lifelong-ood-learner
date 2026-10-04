"""P4 (archive/2026-10-program/PLAN_TRANSFER.md; ledger m3_prereg tag p4): does growing the leaf library with other tasks' practice SUCCESSES change
what a composition executes?  Offline gate before any rollout.

Base = the leaf library (the prior's chunk means at the 40 training tasks' frames; r9_library_ablation's bank_policy).  Grown =
base + every 5th executed 10-step chunk of every successful recorded practice episode (m5_practice, m6_round1, m6_round2) of the 20
OTHER held-out tasks.  Each target's demo is encoded by dimension-wise borrowing (dimcode, K = 8) over both; reported: the rms
difference of the executed teachers, the share of borrowed (frame, dimension) pairs taken from the new leaves, and each program's
rms to the demo.  Gate: median rms difference < 0.01 -> growth is bookkeeping.

    python experiments/p4_growth.py [--record] [--grown-out]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.operators.decompose import candidates                 # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy import dimcode                                 # noqa: E402
from lifelong_ood.tree.nodes import Tree                                # noqa: E402

R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
TARGETS = [36, 48, 52, 53, 58, 62]
BASE_ID = 10 ** 7


def lib_from(ids, M, V):
    ids = np.asarray(ids)
    return types.SimpleNamespace(ids=ids, M=np.asarray(M, np.float64), V=np.asarray(V, np.float64), index={int(i): k for k, i in enumerate(ids)})


def success_rows(task, norm):
    rows = []
    for p in ("m5_practice", "m6_round1", "m6_round2"):
        for f in sorted(glob.glob(f"{R}/{p}_{task}_*_rec/task{task}_trial*.npz")):
            z = np.load(f)
            if not bool(z["success"]):
                continue
            acts = z["action"]
            T = len(acts)
            rows.append(D.to_rows(acts[D.chunk_indices(T, 10)[0]], norm)[::5])
    return np.concatenate(rows) if rows else np.zeros((0, 70), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--save-programs", action="store_true", help="write the grown programs results/robot/r26_grow_<t>_s1.npz")
    a = ap.parse_args()
    os.chdir(ROOT)
    z = np.load(f"{R}/r2b_libero_40.npz")
    tree = Tree.load(f"{R}/r2b_libero_40_tree.npz")
    _, _, _, V2 = candidates(tree, [2])
    v0 = np.median(V2, 0)
    N = len(z["mu"])
    norm = D.Norm.load()
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90 = D.Demos90()
    succ = {t: success_rows(t, norm) for t in HELD}
    print(f"[p4] success leaves per task: { {t: len(v) for t, v in succ.items()} }", flush=True)
    base = lib_from(BASE_ID + np.arange(N), z["mu"], np.tile(v0, (N, 1)))
    res = dict(targets=TARGETS, K=a.K, base_leaves=int(N), per_task={})
    for t in TARGETS:
        new = np.concatenate([succ[s] for s in HELD if s != t])
        M = np.concatenate([z["mu"], new])
        grown = lib_from(BASE_ID + np.arange(len(M)), M, np.tile(v0, (len(M), 1)))
        ep = d90.load(d90.episodes_of_task(t)[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        mb, mg, from_new = [], [], []
        for r in rows:
            ib, cb = dimcode.encode(base, r, a.K)
            ig, cg = dimcode.encode(grown, r, a.K)
            mb.append(dimcode.decode(base, ib, cb)[0]); mg.append(dimcode.decode(grown, ig, cg)[0])
            from_new.append(np.mean([int(ig[c]) - BASE_ID >= N for c in cg]))
        mb, mg = np.stack(mb), np.stack(mg)
        if a.save_programs:
            from lifelong_ood.policy.compose import DemoProgram
            gp = DemoProgram.from_dimcode(grown, rows, ep["state"], K=a.K)
            assert float(np.abs(gp.mu - mg).max()) < 1e-6
            gp.save(f"{R}/r26_grow_{t}_s1.npz")
        prog = np.load(f"{R}/r9_leaf_{t}_s1.npz", allow_pickle=True)["mu"]
        d = dict(new_leaves=int(len(new)), teacher_rms_diff=float(np.sqrt(((mg - mb) ** 2).mean())),
                 share_from_new=float(np.mean(from_new)), rms_to_demo_base=float(np.sqrt(((mb - rows) ** 2).mean())),
                 rms_to_demo_grown=float(np.sqrt(((mg - rows) ** 2).mean())), base_matches_r9=float(np.abs(prog - mb).max()))
        res["per_task"][str(t)] = d
        print(f"[p4] task {t}: {len(new)} new leaves; teacher rms change {d['teacher_rms_diff']:.4f}; {d['share_from_new']:.2f} of borrowed pairs "
              f"from new leaves; rms to the demo {d['rms_to_demo_base']:.4f} -> {d['rms_to_demo_grown']:.4f} (base = r9 program to {d['base_matches_r9']:.1e})", flush=True)
    med = float(np.median([v["teacher_rms_diff"] for v in res["per_task"].values()]))
    res["median_teacher_rms_diff"] = med
    res["gate"] = "growth changes the executed teacher: run the rollouts" if med >= 0.01 else "growth is bookkeeping (< 0.01): rollouts skipped"
    print(f"[p4] median teacher rms change {med:.4f}: {res['gate']}")
    json.dump(res, open(f"{R}/p4_growth.json", "w"), indent=1)
    if a.record:
        ledger.record("p4_growth", bench="libero_90", tag="gate", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p4_growth.py", command=" ".join(sys.argv), artifacts=[f"{R}/p4_growth.json"])


if __name__ == "__main__":
    main()
