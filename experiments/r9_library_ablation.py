"""Does the prototype structure matter for dimension-wise borrowing?  Offline controls on the held-out demos.

The same encoder (dimcode.encode: K lenders chosen by max coverage, one per action dimension) over different libraries:
  band        the tree's levels t = 0.2, 0.3, 0.4 (the method: 8,276 prototypes)
  t0.2 / t0.3 / t0.4   one level alone (t0.4 = the 242 SHARED prototypes, ~3 tasks each)
  t0.3+0.4    the band without its finest level
  bank_demo   every library frame's RAW demo chunk (10,249; no prototype discovery), variance = the t = 0.2 nodes' median
  bank_policy every library frame's policy conditional mean (10,249), same variance
  bank_rand   a random 8,276 of bank_demo (the band's size)
and K = 1, 2, 4, 8 on the band.  Per held-out task: rms of the rebuilt chunks to the demo's, jerk, and the share of
lent pairs from shared (>= 3 tasks) prototypes.

    python experiments/r9_library_ablation.py [--record]
"""
import argparse
import glob
import importlib.util
import os
import re
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


def lib_from(ids, M, V):
    ids = np.asarray(ids)
    return types.SimpleNamespace(ids=ids, M=np.asarray(M, np.float64), V=np.asarray(V, np.float64), index={int(i): k for k, i in enumerate(ids)})


def jerk(rows):
    x = rows.reshape(len(rows), 10, 7)
    return float(np.abs(np.diff(x, 3, axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="results/robot/r2b_libero_40_tree.npz")
    ap.add_argument("--npz", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--programs", default="", help="write sim programs results/robot/r9_{leaf,shared}_<task>_s{1,5}.npz for these tasks "
                    "(leaf = bank_policy: the policy's chunk means, the marginal's components; shared = t0.4 alone) and exit")
    a = ap.parse_args()
    os.chdir(ROOT)
    tree, z = Tree.load(a.tree), np.load(a.npz)
    ntasks, level_of = {}, {}
    for l in (2, 3, 4):
        owner = z[f"owner_policy_{tree.u_grid[l]}"]
        for k, nid in zip(sorted(set(owner.tolist())), tree.levels[l]):
            ntasks[int(nid)] = len(set(z["task"][owner == k].tolist()))
            level_of[int(nid)] = l
    libs = {}
    for name, lv in (("band", (2, 3, 4)), ("t0.2", (2,)), ("t0.3", (3,)), ("t0.4", (4,)), ("t0.3+0.4", (3, 4))):
        ids, _, M, V = candidates(tree, list(lv))
        libs[name] = lib_from(ids, M, V)
    _, _, _, V2 = candidates(tree, [2])
    v0 = np.median(V2, 0)
    N = len(z["demo"])
    base = 10 ** 7                                                       # bank ids: outside the tree's node ids
    libs["bank_demo"] = lib_from(base + np.arange(N), z["demo"], np.tile(v0, (N, 1)))
    libs["bank_policy"] = lib_from(base + np.arange(N), z["mu"], np.tile(v0, (N, 1)))
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(N, len(libs["band"].ids), replace=False))
    libs["bank_rand"] = lib_from(base + pick, z["demo"][pick], np.tile(v0, (len(pick), 1)))
    if a.programs:
        from lifelong_ood.policy.compose import DemoProgram
        spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
        r3p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r3p)
        texts, _ = r3p.libero90_texts()
        d90, norm = D.Demos90(), D.Norm.load()
        for t in [int(x) for x in a.programs.replace("+", ",").split(",")]:
            ep = d90.load(d90.episodes_of_task(t)[0])
            T = len(ep["actions"])
            rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
            for name, lib in (("leaf", libs["bank_policy"]), ("shared", libs["t0.4"])):
                full = DemoProgram.from_dimcode(lib, rows, ep["state"], K=8)
                for st in (1, 5):
                    prog = full.subsample(st) if st > 1 else full
                    prog.save(f"results/robot/r9_{name}_{t}_s{st}.npz")
                rms = float(np.sqrt(((full.mu - rows) ** 2).mean()))
                print(f"[r9] task {t} {name}: rms {rms:.3f}, {full.bytes()} B at stride 1, {full.subsample(5).bytes()} B at stride 5", flush=True)
        return
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    held = sorted(int(re.findall(r"r7_dim_(\d+)_s1", p)[0]) for p in glob.glob("results/robot/r7_dim_*_s1.npz"))
    runs = [(n, 8) for n in libs] + [("band", k) for k in (1, 2, 4)]
    res = {f"{n}_K{k}": {} for n, k in runs}
    for t in held:
        ep = d90.load(d90.episodes_of_task(t)[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        line = []
        for n, k in runs:
            lib = libs[n]
            codes = [dimcode.encode(lib, r, K=k) for r in rows]
            rebuilt = np.stack([dimcode.decode(lib, i, c)[0] for i, c in codes])
            lent = np.concatenate([ids[c] for ids, c in codes])
            shared = float(np.mean([ntasks.get(int(x), 1) >= 3 for x in lent])) if n in ("band", "t0.2", "t0.3", "t0.4", "t0.3+0.4") else float("nan")
            levels = {str(tree.u_grid[l]): float(np.mean([level_of.get(int(x), -1) == l for x in lent])) for l in (2, 3, 4)} \
                if n in ("band", "t0.2", "t0.3", "t0.4", "t0.3+0.4") else None
            m = dict(rms=float(np.sqrt(((rebuilt - rows) ** 2).mean())), jerk=jerk(rebuilt), jerk_demo=jerk(rows), shared_share=shared, levels=levels,
                     library=int(len(lib.ids)), K=k, code_bytes_per_frame=dimcode.code_bytes(k))
            res[f"{n}_K{k}"][str(t)] = m
            line.append(f"{n}/K{k} {m['rms']:.3f}")
        print(f"[r9] {t:3d} '{texts[t][:30]}': " + ", ".join(line), flush=True)
    print("\n[r9] mean over held-out tasks: rms to the demo chunks | jerk (demo = %.4f) | share of lent pairs from >=3-task prototypes"
          % np.mean([res["band_K8"][str(t)]["jerk_demo"] for t in held]))
    summ = {}
    for key, v in res.items():
        lv = [x["levels"] for x in v.values() if x.get("levels")]
        s = dict(rms=float(np.mean([x["rms"] for x in v.values()])), jerk=float(np.mean([x["jerk"] for x in v.values()])),
                 levels={k: float(np.mean([x[k] for x in lv])) for k in lv[0]} if lv else None,
                 shared_share=float(np.nanmean([x["shared_share"] for x in v.values()])) if not all(np.isnan(x["shared_share"]) for x in v.values()) else None,
                 library=next(iter(v.values()))["library"])
        summ[key] = s
        print(f"  {key:16s} library {s['library']:6d}: rms {s['rms']:.3f}, jerk {s['jerk']:.4f}"
              + (f", shared-prototype share {s['shared_share']:.3f}" if s["shared_share"] is not None else ""))
    if a.record:
        ledger.record("r9_library_ablation", bench="libero_90", tag=f"offline_{len(held)}tasks", seed=0,
                      metrics=dict(summary=summ, per_task=res, held=held),
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/r9_library_ablation.py", command="python experiments/r9_library_ablation.py --record")
        print("[r9] ledger row written")


if __name__ == "__main__":
    main()
