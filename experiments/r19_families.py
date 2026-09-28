"""Task FAMILIES that the coarse prototype levels define (after experiments/r19_invariance.py).

At the coarse levels an episode's label sequence barely changes across instances but is shared between tasks: the level names a
task's type.  Per level: the mean edit distance between the human demonstrations of every pair of held-out tasks, an average-
linkage clustering into families at a cut, and, per family, its tasks' instructions -- to compare with the families the
instructions' verbs would give.

    python experiments/r19_families.py [--levels 0.3,0.4,0.55 --cut 0.5]
"""
import argparse
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")


def texts():
    src = open("/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py").read()
    block = src[src.index('"libero_90"'):]
    block = block[block.index("[") + 1: block.index("]")]
    out = {}
    for i, n in enumerate(re.findall(r'"([^"]+)"', block)):
        m = re.match(r"^(?:[A-Z_]+SCENE\d+)_(.*)$", n)
        out[i] = (m.group(1) if m else n).replace("_", " ")
    return out


def average_linkage(Dt, cut):
    """Agglomerate clusters while the mean distance between the closest two stays below `cut`."""
    cl = [[i] for i in range(len(Dt))]
    while len(cl) > 1:
        best, pair = None, None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                d = float(np.mean([Dt[i, j] for i in cl[a] for j in cl[b]]))
                if best is None or d < best:
                    best, pair = d, (a, b)
        if best >= cut:
            break
        a, b = pair
        cl[a] = cl[a] + cl[b]
        del cl[b]
    return cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0.3,0.4,0.55")
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--out", default="", help="write the families of each level to this JSON")
    a = ap.parse_args()
    saved = {}
    z = np.load(os.path.join(R, "r19_invariance_sequences.npz"), allow_pickle=True)
    task, src = z["task"], z["source"]
    T = sorted(set(task.tolist()))
    tx = texts()
    for u in a.levels.split(","):
        Dm = z[f"dist_{u}"]
        H = src == "human"
        Dt = np.zeros((len(T), len(T)))
        for i, ti in enumerate(T):
            for j, tj in enumerate(T):
                ii = np.where(H & (task == ti))[0]
                jj = np.where(H & (task == tj))[0]
                block = Dm[np.ix_(ii, jj)]
                Dt[i, j] = block[~np.eye(len(ii), dtype=bool)].mean() if i == j else block.mean()
        fam = average_linkage(Dt, a.cut)
        saved[u] = [sorted(int(T[i]) for i in f) for f in sorted(fam, key=len, reverse=True)]
        within = np.mean([Dt[i, i] for i in range(len(T))])
        print(f"\nt={u}: within-task {within:.3f}; {len(fam)} families at cut {a.cut}")
        for f in sorted(fam, key=len, reverse=True):
            print(f"  [{len(f)}] " + " | ".join(f"{T[i]}: {tx[T[i]]}" for i in f))
    if a.out:
        import json
        json.dump(dict(cut=a.cut, families=saved), open(a.out, "w"), indent=1)
        print(f"[r19] families -> {a.out}")


if __name__ == "__main__":
    main()
